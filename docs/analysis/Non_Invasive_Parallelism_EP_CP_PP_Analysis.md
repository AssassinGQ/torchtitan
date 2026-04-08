# 非侵入式并行推理调研：EP、CP、PP 实现分析与约束

## 1. 背景：TP 非侵入式并行的实现原理

TorchTitan 的 TP（Tensor Parallelism）是非侵入式并行的基准实现。核心机制是 `torch.distributed.tensor.parallel.parallelize_module()`，它使用 `distribute_module()` 将 `nn.Module` 包装上 DTensor 分片。模块参数被替换为 DTensor replica/shard，在模块边界插入 hooks 进行 `torch.Tensor <-> DTensor` 转换。**模型代码本身从不直接引用分布式张量**——它只是正常执行前向传播。

## 2. Expert Parallelism (EP) 分析

### 2.1 架构实现

**核心组件位置**：`torchtitan/distributed/expert_parallel.py` + `torchtitan/models/common/moe.py`

**MoE 模块** (`models/common/moe.py`):
- `GroupedExperts`：包含形状为 `(num_experts, hidden_dim, dim)` 的 `w1/w2/w3` 权重张量。前向接收 `x` 和 `num_tokens_per_expert`，**不感知** EP rank。
- `TokenChoiceTopKRouter`：产生路由决策（每个 token 的 top-k experts）。
- `TokenReorderer`：按 expert 索引重排 tokens。
- `MoE`：编排上述组件。关键设计决策：`MoE.forward()` 调用 `self.experts(routed_input, num_tokens_per_expert)` 使用普通张量。EP 感知行为完全来自并行化包装器。

**并行化方式** (`ParallelStyle`):
- **`ExpertParallel`**：扩展 `parallelize_module`，带有自定义 `input_fn`（token 分发）和 `output_fn`（token 合并）：
  - `_token_dispatch`：使用 `all_to_all_single_autograd` 将 tokens 洗牌到正确的 EP rank。
  - `_token_combine`：反转置换并 all-to-all 恢复原始布局。
  - `_partition_fn`：通过 `distribute_tensor(..., [Shard(0)])` 在 expert 维度分片每个 expert 权重。
- **`TensorParallel`**：experts 内部的 TP（w1/w3 在 dim=1 分片，w2 在 dim=2）。
- **`ExpertTensorParallel`**：EP + TP 组合。
- **`DeepEPExpertParallel`**：使用 DeepEP 自定义内核（来自 `deep_ep` 库）而非 PyTorch all-to-all 进行分发/合并。支持 "deepep"（H100/NVLink Switch）和 "hybridep"（GB200/NVLink72）后端。
- **`TorchAOExpertParallel`**：包装标准 EP 与 torchao 的 `permute_and_pad`，用于 FP8/MXFP8 量化分组 GEMM。

### 2.2 非侵入性分析

**部分非侵入式**：MoE 模块使用普通张量编写，但 EP 需要 MoE 的 `GroupedExperts.forward()` 处理每个 expert 可变长度的 token 分块（通过 `num_tokens_per_expert`）。这是 MoE 特定的问题，不是模型通用的。然而路由逻辑和 expert 分发/合并**完全外部化**到并行化层——模型只调用 `experts(input, num_tokens_per_expert)`。

**对 HuggingFace 模型的约束**：HF MoE 模型通常将 experts 实现为 `nn.ModuleList` 并使用循环。非侵入式 EP 需要：
- (a) 将 expert 列表替换为 `GroupedExperts`（使用分组 GEMM）
- (b) 用 EP 并行风格包装 expert 列表

方案 (b) 更 HuggingFace 兼容但效率较低（无分组 GEMM）。

## 3. Context Parallelism (CP) 分析

### 3.1 架构实现

**核心机制** (`torchtitan/distributed/context_parallel.py`)

CP 通过 `torch.distributed.tensor.experimental._attention._ContextParallel` 实现，这是 PyTorch DTensor attention API 的一部分：

- **`apply_cp_to_attention_module()`**：使用 `_ContextParallel` plan 对 attention 模块应用 `parallelize_module()`（支持 `FlexAttention` 和 `ScaledDotProductAttention`）。
- **`prepare_context_parallel_input()`**：使用 `_context_parallel_shard()` 沿序列维度分片输入张量和 attention mask。支持负载均衡器：
  - `_HeadTailLoadBalancer`：用于 CP 的 SDPA。
  - `_PTRRLoadBalancer`：用于 FlexAttention。
- **`cp_shard()`**：使用 `_context_parallel_shard()` 将序列分片到 CP ranks。

### 3.2 非侵入性分析

**接近完全非侵入式**：CP 需要：
1. 在模型前向之前对输入序列进行分片。
2. 通过 `parallelize_module` with `_ContextParallel` 对 attention 模块进行并行化。
3. 序列长度必须能被 `2 * cp_degree` 整除（用于负载均衡）。
4. `torch.distributed.device_mesh` 必须有 CP 维度。

**对 HuggingFace 模型的约束**：HF 模型使用 `torch.nn.functional.scaled_dot_product_attention`（SDPA）或自定义 attention 实现。`apply_cp_to_attention_module()` 函数检查 `isinstance(first, FlexAttention | ScaledDotProductAttention)` 并为 SDPA 启用 CP dispatcher。对于使用原始 SDPA 调用的 HuggingFace 模型，需要用 CP dispatcher 包装 SDPA 调用或替换为 torchtitan 的 attention 模块。

## 4. Pipeline Parallelism (PP) 分析

### 4.1 架构实现

**核心机制** (`torchtitan/distributed/pipeline_parallel.py`)

PP 是**侵入式**的——它在本质上与 TP/EP/CP 不同：

- **`pipeline_module_split()`**：对每个 stage 执行 `copy.deepcopy(whole_model)`，然后**删除**不属于该 stage 的层。使用 `torch.distributed.pipelining.PipelineStage` 包装每个 stage。
- **`pipeline_llm()`**：通过以下步骤编排 PP：
  1. 通过 `generate_llm_fqn_per_model_part()` 创建每个 stage 的模块 FQN 列表。
  2. 将模型分割成 stage 块。
  3. 通过 `parallelize_fn()` 独立地将 SPMD 并行化（TP、FSDP 等）应用到每个块。
  4. 构建 pipeline 调度。
- **`build_pipeline_schedule()`**：创建调度（1F1B、GPipe、Interleaved1F1B、ZBVZeroBubble、DualPipeV 等）。
- **关键约束**：模型必须可以通过 `model.layers` 访问层列表，并使用数字索引（`model.layers.0`、`model.layers.1` 等），模块命名为 `tok_embeddings`、`norm`、`output`。

### 4.2 非侵入性分析

**侵入式**：PP 无法非侵入式，因为：
1. 它**物理分割模型**到各 rank——不同 rank 持有不同的子图。
2. 它使用 `torch.distributed.pipelining.PipelineStage`，需要通过调度进行 per-rank 前向/反向协调。
3. Microbatch 协调需要完整的训练循环集成（`_PipelineSchedule` 对象驱动迭代）。
4. 模型要求：`forward()` 必须容忍被删除的层；权重初始化必须容忍缺失的层；无嵌套 ModuleDict/ModuleList 结构。

**对 HuggingFace 模型的约束**：HF 模型有命名层（如 `model.layers` 作为 `ModuleList`）。torchtitan 的 `generate_llm_fqn_per_model_part()` 函数通过接受 `f"layers.{i}"` FQN 来处理这个问题。符合标准 `embed + layer_stack + norm + lm_head` 结构的 HuggingFace PP 兼容模型可以被分割。

## 5. Mesh 架构（所有并行如何组合）

从 `torchtitan/distributed/parallel_dims.py`，world mesh 被展平为：

```
["pp", "dp_replicate", "fsdp", "tp"]      -- dense_mesh（非 MoE 参数）
["pp", "dp_replicate", "efsdp", "ep", "etp"] -- sparse_mesh（MoE 参数）
["pp", "batch", "cp", "tp"]              -- dataloading_mesh
```

约束公式：`dp_replicate * dp_shard * cp * tp * pp = world_size`。

## 6. 关键发现总结

| 并行方式 | 非侵入式？ | 机制 | HuggingFace 约束 |
|---------|----------|------|-----------------|
| **TP** | 是（完全） | `parallelize_module` with ColwiseParallel/RowwiseParallel；模型从不接触 DTensors | 模型需要 `ModuleList` 层；模块名称必须 FQN 兼容 |
| **EP** | 部分 | `BaseExpertParallel` 作为 `ParallelStyle`，带有 all-to-all 的 input_fn/output_fn hooks；MoE 模块用普通张量编写 | MoE 必须将 `experts` 暴露为分片模块；EP 通信需要自定义内核或 A2A 集合操作 |
| **CP** | 接近完全 | 通过 DTensor attention API 的 `_ContextParallel`；只需输入分片和 attention 模块包装 | 序列长度必须能被 `2 * cp_degree` 整除；SDPA dispatcher 必须启用 |
| **PP** | 否（侵入式） | 模型被深拷贝并物理分割到各 stage；pipeline 调度驱动执行 | 模型必须有数字索引的 `ModuleList` 层；无嵌套 ModuleDict/ModuleList；forward 必须容忍被删除的层 |

## 7. 附加观察

1. **DeepEP/HybridEP**：`torchtitan/distributed/deepep/` 文件夹提供了专门的通信内核，通过 `torch.library` 与 PyTorch autograd 集成。这是**最非侵入式**的 EP 方法，因为 MoE 模型的前向完全不改变——分发/合并 hooks 在 `parallelize_module` 边界拦截。

2. **ETP (Expert Tensor Parallelism)**：支持两种模式：`ETP=TP`（EP 从 TP 借用）和 `ETP=1`（所有 TP 转向 EP）。代码通过 `ReordererSequenceParallel` 处理这两种情况。

3. **EP + PP 组合**：在集成测试中验证（`deepseek_v3_pp+fsdp+tp+ep`）。每个 PP stage 独立地将其 MoE 层应用 EP。sparse mesh `["pp", "dp_replicate", "efsdp", "ep", "etp"]` 保持 PP 作为最外层维度。

4. **权重绑定**：在 embeddings 和 output projection 之间有权重绑定的模型在 PP 中需要特殊处理（参见 `llama4/parallelize.py` 中的 `apply_fsdp`）。
