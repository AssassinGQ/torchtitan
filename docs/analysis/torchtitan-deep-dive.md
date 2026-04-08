# TorchTitan 工程深度分析报告

## 项目全局摘要

TorchTitan 是 Meta 公司开发的 PyTorch 原生大规模生成式 AI 模型训练平台，基于纯 PyTorch 分布式训练技术栈实现。项目定位为"clean-room implementation"，强调不依赖非 PyTorch 库的核心训练基础设施，以最小化代码库提供高度可组合的多维并行训练能力。

**核心技术特点：**
- 多维可组合并行策略：FSDP2 + Tensor Parallel + Pipeline Parallel + Context Parallel + Expert Parallel
- PyTorch 原生：所有并行化均基于 `torch.distributed` / `torch.distributed.tensor` / `torch.distributed._composable` 实现
- 支持多种注意力后端：SDPA、FlexAttention、VarlenAttention
- 量化支持：Float8 (MXFP8)、FP8
- 分布式检查点：基于 PyTorch DCP，支持 Async Checkpointing

**支持的模型：** Llama3/4、Qwen3、DeepSeek V3、GPT-OSS、Flux

---

## 系统架构分析

### 整体架构

TorchTitan 采用模块化分层设计，从上至下分为五层：

```
┌─────────────────────────────────────────────────────────┐
│                     Train Loop (trainer.py)             │
│  batch_generator → forward_backward_step → train_step  │
├─────────────────────────────────────────────────────────┤
│              Components (components/)                   │
│  CheckpointManager / Optimizer / LRScheduler / Metrics  │
├─────────────────────────────────────────────────────────┤
│              Distributed Parallelism (distributed/)    │
│  FSDP / TP / PP / CP / EP / ETP                        │
├─────────────────────────────────────────────────────────┤
│              Model Architecture (models/)               │
│  Decoder / Attention / FeedForward / MoE / RoPE       │
├─────────────────────────────────────────────────────────┤
│              Protocols & Config (protocols/ + config/)   │
│  BaseModel / Module / ModelSpec / Configurable          │
└─────────────────────────────────────────────────────────┘
```

### 关键设计原则

1. **PyTorch-Native**：所有并行化技术均使用 PyTorch 官方 API，不引入外部依赖
2. **配置驱动**：通过 `Configurable` 基类和 dataclass Config 实现声明式配置
3. **模型无关核心**：并行化逻辑与模型架构解耦，共享组件位于 `models/common/`
4. **协议抽象**：通过 `protocols/` 定义抽象接口（`BaseModel`、`Module`、`BaseStateDictAdapter`）

---

## 核心模块代码深度解析

### 1. 训练循环核心：Trainer（`trainer.py`）

Trainer 是整个训练的核心 orchestrator，继承自 `torch.distributed.checkpoint.stateful.Stateful`，实现训练状态的保存与恢复。

**初始化流程（`__init__`）：**
```
1. init_distributed() → ParallelDims → build_mesh()
2. build tokenizer & dataloader
3. build model (meta device init) → apply parallelisms → init_weights()
4. build optimizer & lr_scheduler
5. build checkpointer & metrics_processor
6. build validator (optional)
```

**关键设计点：**
- **Meta Device 初始化**：模型在 meta device 创建以节省 GPU 内存，权重通过 `init_weights()` 在 `to_empty()` 后初始化
- **模型转换器（ModelConverters）**：在并行化之前/之后对模型进行转换（如 FP8 量化）
- **PP vs Non-PP 分支**：PP 时使用 `pipelining_fn` 返回 `model_parts`（多个 stage model chunk），否则返回单个模型
- **TrainContext**：管理 AMP、FSDP context 等分布式上下文

**训练步（`train_step`）：**
```python
# 收集所有 microbatch 到 CPU，计数全局 valid tokens
microbatches = []
for _ in range(gradient_accumulation_steps):
    input_dict, labels = next(data_iterator)
    microbatches.append((input_dict, labels))

# 跨 DP ranks 聚合全局 token 数
global_valid_tokens = dist_sum(local_valid_tokens, batch_mesh)

# 逐个处理 microbatch：移动到 GPU → forward/backward → 释放
for input_dict, labels in microbatches:
    loss = forward_backward_step(input_dict, labels, global_valid_tokens)

# 梯度裁剪 → optimizer step → lr scheduler step
grad_norm = clip_grad_norm_(...)
optimizers.step()
lr_schedulers.step()
```

### 2. 多维并行维度管理：ParallelDims（`distributed/parallel_dims.py`）

ParallelDims 是并行配置的核心数据中心，验证并行维度合法性并构建 DeviceMesh。

**支持的并行维度：**
| 维度 | 名称 | 说明 |
|------|------|------|
| `dp_replicate` | DDP/HSDP 复制维度 | 权重复制 |
| `dp_shard` | FSDP 分片维度 | 权重分片 |
| `cp` | Context Parallel | 序列维度分片 |
| `tp` | Tensor Parallel | 张量切片 |
| `pp` | Pipeline Parallel | 流水线并行 |
| `ep` | Expert Parallel | MoE 专家并行 |
| `etp` | Expert Tensor Parallel | 专家的 TP |

**DeviceMesh 布局：**
```
world_mesh: [dp_replicate, dp_shard, cp, tp, pp]  (dataparallel mesh)
                    ↓ unflatten
dataloading_mesh: ["pp", "batch(=dp_replicate*dp_shard)", "cp", "tp"]
dense_mesh: ["pp", "dp_replicate", "fsdp(=dp_shard*cp)", "tp"]
sparse_mesh: ["pp", "dp_replicate", "efsdp", "ep", "etp"]
```

**验证公式：** `dp_replicate * dp_shard * cp * tp * pp == world_size`

### 3. 注意力机制：Attention（`models/common/attention.py`）

实现了三种注意力后端的统一封装：

**（A）GQAttention — Grouped Query Attention**

支持 GQA（多 query 头共用 KV 头）、QK Norm、RoPE 和三种后端：
- `flex`：基于 `flex_attention` 的编译优化版本
- `sdpa`：基于 `F.scaled_dot_product_attention`
- `varlen`：基于 `varlen_attn` 的变长序列版本

**Forward 流程：**
```python
xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
xq = xq.view(bs, seqlen, -1, self.head_dim)  # 推断实际 local heads
# QK Norm (Qwen3 风格)
if self.q_norm: xq = self.q_norm(xq)
if self.k_norm: xk = self.k_norm(xk)
# RoPE
if self.use_rope: xq, xk = apply_rotary_emb_complex(xq, xk, rope_cache, positions)
# 注意力计算
output = inner_attention(xq, xk, xv, block_mask/attention_masks, enable_gqa=...)
```

**（B）注意力 Mask 工具函数**
- `get_causal_mask_mod()`：因果 mask
- `get_document_mask_mod()`：文档边界 mask（防止跨文档 attention）
- `get_fixed_block_mask_mod()`：块内 causal mask
- `get_sliding_window_mask_mod()`：滑动窗口 causal mask

### 4. 前馈网络：FeedForward（`models/common/feed_forward.py`）

SwiGLU 架构的共享实现：
```python
# w1(x) * sigmoid(w1(x)) * w3(x) — 传说中的 SwiGLU
output = self.w2(F.silu(self.w1(x)) * self.w3(x))
```

### 5. MoE 模块：MoE（`models/common/moe/moe.py`）

**Router（TokenChoiceTopKRouter）：**
- 使用 sigmoid 或 softmax 计算路由分数
- 支持节点限制路由（`num_expert_groups` + `num_limited_groups`）
- `score_before_experts` 控制分数应用时机

**GroupedExperts：**
- `_run_experts_grouped_mm`：使用 `torch._grouped_mm` 高效计算
- `_run_experts_for_loop`：朴素实现（可读性优先）

**前向流程：**
```python
# 1. 路由
top_scores, selected_experts_indices, num_tokens_per_expert = router(x)
# 2. 重排序 token 以匹配 experts 顺序
top_scores_sorted, token_indices_sorted, num_tokens = reorderer(...)
# 3. 路由输入
routed_input = x[token_indices_sorted // top_k]
if score_before_experts:
    routed_input = routed_input * top_scores_sorted  # 应用路由分数
# 4. Expert 计算
routed_output = experts(routed_input, num_tokens)
# 5. 恢复原始顺序并聚合
out_experts = unsort_and_sum(routed_output, token_indices_sorted, top_k)
# 6. 加上 shared experts
out = out_experts + shared_experts(x) if shared_experts else out_experts
```

### 6. 分布式并行化实现

#### 6.1 FSDP（`distributed/` 下隐式使用 PyTorch FSDP2 API）

FSDP 通过 `fully_shard` / `replicate` API 应用，结合 `MixedPrecisionPolicy` 支持混合精度。关键配置：
- `fsdp_reshard_after_forward` policy：`"always"` / `"never"` / `"default"`（PP 时默认不 reshard）
- CPU Offload：参数/梯度/优化器状态卸载到 CPU

#### 6.2 Tensor Parallel（`distributed/tensor_parallel.py`）

基于 PyTorch `distributed.tensor.parallel`：
- `ColwiseParallel`：列切分线性层（w1, w3）
- `RowwiseParallel`：行切分线性层（w2）
- `SequenceParallel`：序列维度并行
- `NoParallel`：确保参数在 TP mesh 上转换为 DTensor（用于 MoE router gate 等）

**自定义扩展：**
- `ColwiseParallelWithGradPlacement`：显式控制反向梯度 placement
- `maybe_enable_async_tp`：异步 TP，通过 `torch._inductor.config._micro_pipeline_tp = True`

#### 6.3 Pipeline Parallel（`distributed/pipeline_parallel.py`）

**模块分割（`pipeline_module_split`）：**
- 基于 `torch.distributed.pipelining.PipelineStage`
- 通过 `copy.deepcopy(whole_model)` + 删除不需要的层来创建各 stage
- 支持 V-style schedule（ZeroBubble、DualPipeV）

**调度器（`build_pipeline_schedule`）：**
- 支持 schedule：`1F1B`、`Interleaved1F1B`、`ZeroBubble`、`DualPipeV`
- 支持 CSV 自定义 schedule（`_PipelineScheduleRuntime`）
- 验证：`local_batch_size % microbatch_size == 0`

#### 6.4 Context Parallel（`distributed/context_parallel.py`）

基于 PyTorch DTensor experimental API：
- `_ContextParallel`：封装 SDPA/FlexAttention 的 CP 逻辑
- 支持两种 Load Balancer：`HeadTailLoadBalancer`（SDPA）、`PTRRLoadBalancer`（FlexAttention）
- `prepare_context_parallel_input`：准备 CP 输入，包括 position tensor 分片

#### 6.5 Expert Parallel（`distributed/expert_parallel.py`）

**三种 EP 实现：**
1. `ExpertParallel`：标准 all-to-all dispatch/combine
2. `ExpertTensorParallel`：支持 EP + ETP 组合
3. `DeepEPExpertParallel`：基于 DeepEP 自定义 kernel 的高效实现

**Token Dispatch 流程：**
```python
# 1. all-to-all 获取全局 token 分布
num_tokens_per_expert_group = all_to_all_single(num_tokens_per_expert)
# 2. _permute 填充和重排
routed_input, permuted_indices = _permute(routed_input, num_tokens_per_expert_group)
# 3. all-to-all dispatch
routed_input = all_to_all_single_autograd(routed_input, output_splits, input_splits)
```

### 7. RoPE 实现（`models/common/rope.py`）

支持三种后端和三种 scaling 策略：

**后端：**
- `complex`：复数指数形式（Llama3/4、DeepSeek V3）
- `cos_sin`：cos/sin 拼接形式（Qwen3、GPT-OSS）

**Scaling：**
- `none`：无 scaling
- `llama`：Llama3 风格的低/高频 scaling
- `yarn`：YaRN scaling 扩展上下文（DeepSeek V3、GPT-OSS）

### 8. 分布式检查点（`components/checkpoint.py`）

基于 PyTorch DCP（Distributed Checkpoint）：

**核心设计：**
- `ModelWrapper`：封装 model_parts，支持跨 PP rank 的 state_dict 展平
- `CheckpointManager`：管理保存/加载逻辑，支持异步保存
- `AsyncMode`：`"disabled"` / `"async"` / `"async_with_pinned_mem"`

**特殊处理：**
- Pipeline Parallel + Virtual Stages：需要 optimizer flattening 避免 rank 间 param_group 冲突
- 多 model chunk：每个 chunk 的 state_dict 独立展平保存

**与 HuggingFace 互操作：**
- `initial_load_in_hf=True`：从 safetensors 加载预训练模型
- `last_save_in_hf=True`：保存为 HuggingFace 格式

### 9. 模型注册与配置（`models/*/config_registry.py`）

每个模型通过 `config_registry.py` 注册配置和工厂函数：

```python
def llama3_8b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        model_spec=model_registry("8B"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        training=TrainingConfig(local_batch_size=1, seq_len=8192, steps=1000),
        parallelism=ParallelismConfig(...),
        ...
    )
```

通过 `--config` CLI 参数选择预设配置。

### 10. 协议层（`protocols/`）

- `Module`：所有 nn.Module 的基类，要求实现 `init_weights()`
- `BaseModel`：Decoder 基类，提供共享的 `forward()` 和 `get_attention_masks()`
- `ModelSpec`：封装模型的架构配置 + 工厂函数（build_loss_fn、parallelize_fn、pipelining_fn）
- `BaseStateDictAdapter`：native 格式与 HuggingFace 格式之间的 state_dict 转换

---

## 核心功能执行流程分析

### 训练启动流程

```
train.py::main()
  → ConfigManager.parse_args()  # 解析 CLI 参数
  → config.build()  # 构建 Trainer.Config
  → Trainer(config)
      → init_distributed()  # 初始化分布式环境 + 构建 DeviceMesh
      → build tokenizer & dataloader
      → build model (meta device)
          → model_config.build()  # 创建模型架构
          → model_converters.convert(model)  # 应用量化等转换
      → parallelize_fn(model)  # 应用 FSDP/TP/CP/EP
          → init_weights()  # 权重初始化
      → build optimizer & lr_scheduler
      → build checkpointer
      → metrics_processor
  → trainer.train()
```

### 单步训练流程

```
train_step(data_iterator)
  1. optimizer.zero_grad()
  2. 收集 microbatches，计算 global_valid_tokens
  3. for each microbatch:
       a. post_dataloading_process():
            - prepare_context_parallel_input() (if CP enabled)
            - get_attention_masks() (if flex/varlen)
       b. forward_backward_step():
            if PP:
              pp_schedule.step(inputs, target=labels, losses=[...])
            else:
              pred = model(inputs, **extra_kwargs)
              loss_sum = loss_fn(pred, labels)
              loss = loss_sum / global_valid_tokens
              loss.backward()
  4. clip_grad_norm_()
  5. checkpointer.maybe_wait_for_staging()
  6. optimizers.step()
  7. lr_schedulers.step()
  8. log metrics (loss, grad_norm, lr, tokens_seen)
```

### 模型并行化流程（非 PP）

```
parallelize_fn(model, parallel_dims, ...)
  1. TP 应用（如果启用）
       parallelize_module(attention, tp_mesh, ColwiseParallel + RowwiseParallel)
       parallelize_module(mlp, tp_mesh, ColwiseParallel + RowwiseParallel)
  2. CP 应用（如果启用）
       apply_cp_to_attention_module(attention_modules, cp_mesh, attn_backend)
  3. EP 应用（如果启用，MoE 模型）
       distribute_module(moe_experts, ep_mesh, ExpertParallel())
  4. AC 应用
       apply_ac(model, ac_config)
  5. torch.compile（如果启用）
       model = torch.compile(model, ...)
  6. FSDP 应用
       for param in model.parameters():
           fully_shard(param, mesh=fsdp_mesh)
  7. DDP replicate（如果启用）
       replicate(model, mesh=dp_replicate_mesh)
```

---

## 质量与性能评估

### 量化精度支持

| 精度 | 训练 dtype | 说明 |
|------|-----------|------|
| BF16 | bfloat16 | 完整 BF16 训练，RoPE/logits 仍用 FP32 |
| FP32 | float32 | 全精度训练 |
| FP8 | float8_e4m3fn | Float8 Linear，MXFP8 支持 |
| 混合精度 | param=bf16 + reduce=fp32 | 通过 FSDP MixedPrecisionPolicy |

### MFU（Model FLOPs Utilization）计算

基于 GPU 峰值 FLOPS 和实际 token 吞吐量计算：
```python
gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
mfu = (num_flops_per_token * tokens_per_sec) / gpu_peak_flops
```

### 内存占用估算

支持通过脚本 `scripts/estimate_memory.py` 在不实例化模型的情况下估算 FSDP/HSDP 内存占用。

### 注意力后端对比

| 后端 | 适用场景 | 特点 |
|------|---------|------|
| SDPA | 通用 | PyTorch 原生，支持 FlashAttention/CUDNN |
| FlexAttention | 需要自定义 mask | 可编译优化，支持复杂 mask 逻辑 |
| VarlenAttention | 变长序列/文档级 mask | 专为 block_causal mask 设计 |

---

## 项目构建与部署

### 开发环境

```bash
pip install -r requirements.txt -r requirements-dev.txt
pre-commit run --all-files  # lint + format
pytest tests/ -x             # 单元测试
```

### CI/CD 流程

**工作流类型：**
- `unit_test_cpu.yaml`：CPU 单元测试
- `integration_test_8gpu_features.yaml`：8 GPU 特性测试（并行组合）
- `integration_test_8gpu_models.yaml`：8 GPU 模型测试
- `integration_test_8gpu_torchcomms.yaml`：通信后端测试
- `integration_test_8gpu_graph_trainer*.yaml`：Graph Trainer 测试
- `docker-builds.yml`：Docker 镜像构建

**测试矩阵：** 通过 `set-matrix.yaml` 动态生成

### 分布式启动

通过 `torchrun` 或 Slurm 脚本启动：
```bash
# 单节点多卡
torchrun --nproc_per_node=8 train.py --config llama3_8b

# Slurm (multinode_trainer.slurm)
sbatch multinode_trainer.slurm
```

### 关键环境变量

- `WORLD_SIZE`：全局进程数
- `LOCAL_RANK`：本地 GPU 编号
- `CUDA_VISIBLE_DEVICES`：可见 GPU

---

## 二次开发指南

### 添加新模型

1. 在 `torchtitan/models/<model_name>/` 创建模型文件夹
2. 实现 `config_registry.py`：
   - 定义 `ModelSpec`（model config + callables）
   - 注册到 `model_registry()`
3. 实现 `model.py`：继承 `Decoder`，实现架构
4. 实现 `parallelize.py`：定义并行化策略
5. 如需 PP 支持，实现 `pipelining_fn`

参考文档：`torchtitan/models/README.md`

### 添加新的并行策略

1. 在 `torchtitan/distributed/` 下实现并行逻辑
2. 在 `parallelize.py` 的 `parallelize_fn` 中集成
3. 在 `ParallelDims` 中添加新的 mesh 维度（如需要）

### 添加新的量化方法

1. 在 `components/quantization/` 下实现 `ModelConverter`
2. 实现 `pre_optimizer_hook` / `post_optimizer_hook`（如需要）
3. 在 `ModelConvertersContainer.Config.converters` 中注册

### 实验代码管理

实验代码应放在 `torchtitan/experiments/`，遵循：
- 使用 torchtitan 的 config 系统，不引入自定义解析
- 不要修改 core 代码以适应实验需求
- 保持功能独立，不捆绑无关功能

---

## 附录：关键文件索引

| 模块 | 关键文件 |
|------|---------|
| 训练入口 | `train.py` |
| 训练器核心 | `trainer.py` |
| 配置系统 | `config/configs.py`、`config/manager.py` |
| 并行维度 | `distributed/parallel_dims.py` |
| FSDP | `distributed/`（PyTorch API） |
| Tensor Parallel | `distributed/tensor_parallel.py` |
| Pipeline Parallel | `distributed/pipeline_parallel.py` |
| Context Parallel | `distributed/context_parallel.py` |
| Expert Parallel | `distributed/expert_parallel.py` |
| DeepEP | `distributed/deepep/deepep.py` |
| 注意力 | `models/common/attention.py` |
| 前馈网络 | `models/common/feed_forward.py` |
| MoE | `models/common/moe/moe.py` |
| RoPE | `models/common/rope.py` |
| Decoder | `models/common/decoder.py` |
| 检查点 | `components/checkpoint.py` |
| 优化器 | `components/optimizer.py` |
| 学习率 | `components/lr_scheduler.py` |
| 数据加载 | `components/dataloader.py` |
| 指标 | `components/metrics.py` |
| 模型注册 | `models/*/config_registry.py` |
| 协议接口 | `protocols/module.py`、`protocols/model.py` |
