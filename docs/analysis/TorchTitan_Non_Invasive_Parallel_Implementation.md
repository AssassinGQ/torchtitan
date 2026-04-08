# TorchTitan 非侵入式并行策略实现详解

## 一、概述

TorchTitan 采用**非侵入式**的并行策略实现，模型本身是标准的 PyTorch `nn.Module`，无需感知分布式策略。所有并行能力通过外部 `parallelize_module()` API 组合实现。

---

## 二、并行策略定义：ParallelDims

### 2.1 核心数据结构

```python
@dataclass
class ParallelDims:
    dp_replicate: int   # 数据并行副本数 (HSDP外层)
    dp_shard: int      # 数据并行分片数 (FSDP维度)
    cp: int           # 上下文并行度
    tp: int           # 张量并行度
    pp: int           # 流水线并行度
    ep: int           # 专家并行度
    etp: int          # 专家张量并行度
    world_size: int   # 总GPU数
```

### 2.2 并行维度约束

```python
# 必须满足：dp_replicate * dp_shard * cp * tp * pp == world_size
dp_replicate * dp_shard * cp * tp * pp == world_size

# EP约束：ETP = TP 或 ETP = 1
if ep > 1:
    assert etp == tp or etp == 1
```

### 2.3 DeviceMesh 构建

```
World Mesh: [world_size]
     │
     ▼ unflatten
┌─────────────────────────────────────────────────────────────┐
│  dataloading_mesh: ["pp", "batch", "cp", "tp"]            │
│  dense_mesh:      ["pp", "dp_replicate", "fsdp", "tp"]    │
│  sparse_mesh:     ["pp", "dp_replicate", "efsdp", "ep", "etp"] │
└─────────────────────────────────────────────────────────────┘
```

**关键维度计算**：

| Mesh名称 | 维度 | 计算公式 | 说明 |
|---------|------|---------|------|
| batch | 数据加载 | `dp_replicate * dp_shard` | 包含replicate和shard |
| fsdp | FSDP分片 | `dp_shard * cp` | 标准FSDP维度 |
| efsdp | MoE的FSDP | `dp_shard * cp * tp // (etp * ep)` | EP区域专用 |
| loss | 损失计算 | `dp_replicate * dp_shard * cp` | AllReduce维度 |

### 2.4 Mesh获取API

```python
# 获取单个mesh
tp_mesh = parallel_dims.get_mesh("tp")
ep_mesh = parallel_dims.get_mesh("ep")
cp_mesh = parallel_dims.get_mesh("cp")

# 获取组合mesh
ep_etp_mesh = parallel_dims.get_mesh(["ep", "etp"])

# 可选mesh（可能为None）
edp_mesh = parallel_dims.get_optional_mesh(["dp_replicate", "efsdp"])
```

---

## 三、Tensor Parallel (TP) 非侵入式实现

### 3.1 核心API：`parallelize_module()`

TorchTitan 使用 PyTorch DTensor 的 `parallelize_module()` 函数实现非侵入式TP：

```python
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)

parallelize_module(
    module=model,
    device_mesh=tp_mesh,
    parallelize_plan={
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    },
)
```

### 3.2 ParallelStyle 体系

```
ParallelStyle (抽象基类)
 │
 ├── ColwiseParallel       # 列并行：Linear(in, out) → out维度分片
 │      │
 │      └── ColwiseParallelWithGradPlacement  # 带梯度控制
 │
 ├── RowwiseParallel       # 行并行：Linear(in, out) → in维度分片
 │
 ├── SequenceParallel      # 序列并行：沿序列维度分片
 │
 ├── NoParallel            # 复制计算（无分片）
 │
 ├── BaseExpertParallel    # 专家并行基类
 │      │
 │      ├── ExpertParallel       # 标准EP
 │      └── ExpertTensorParallel # EP + TP组合
 │
 └── TensorParallel        # MoE的TP
```

### 3.3 TP应用流程

```python
def apply_non_moe_tp(model, tp_mesh, loss_parallel, enable_float8_tensorwise_tp):
    # 1. Embedding层：行并行，输出沿hidden维度分片
    parallelize_module(model, tp_mesh, {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),  # hidden dim分片
        ),
    })

    # 2. Transformer Block
    for transformer_block in model.layers.values():
        layer_plan = {
            # 输入准备：序列维分片 → 复制
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            # QKV：列并行
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            # O：行并行
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        }

        # FFN（非MoE）
        layer_plan.update({
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        })

        parallelize_module(transformer_block, tp_mesh, layer_plan)
```

### 3.4 TP切分示意

```
原始权重: [hidden_dim, hidden_dim]  (TP=2)
         ┌─────────┬─────────┐
         │  W_col1 │  W_col2 │  → ColwiseParallel
         └─────────┴─────────┘

Attention输出: [seq, hidden] → [seq, hidden/2] (TP=2)
              ┌─────────┬─────────┐
              │  out_1  │  out_2  │  → RowwiseParallel
              └─────────┴─────────┘
```

---

## 四、Expert Parallel (EP) 非侵入式实现

### 4.1 EP策略类

```python
class BaseExpertParallel(ParallelStyle, ABC):
    @abstractmethod
    def _partition_fn(self, name, mod, device_mesh):
        """参数分区：专家权重如何分片"""

    @abstractmethod
    def _token_dispatch(self, mod, inputs, device_mesh):
        """Token分发：哪些token发送到哪个expert"""

    @abstractmethod
    def _token_combine(self, mod, routed_output, device_mesh):
        """Token合并：专家输出如何汇聚"""
```

### 4.2 ExpertParallel（标准EP）

```python
class ExpertParallel(BaseExpertParallel):
    def _partition_fn(self, name, mod, device_mesh):
        # 专家参数沿expert维度分片
        for param in mod.parameters():
            dist_param = distribute_tensor(param, device_mesh, [Shard(0)])

    def _token_dispatch(self, mod, inputs, device_mesh):
        # 1. 统计每个expert的token数
        num_tokens_per_expert = count_tokens_per_expert(routed_input)

        # 2. All-to-All：跨EP rank分发tokens
        num_tokens_grouped = all_to_all_single(num_tokens_per_expert, ...)
        routed_input = all_to_all_single_autograd(routed_input, ...)

        # 3. 重排：确保每个local expert收到正确数量的tokens
        routed_input = _permute(routed_input, ...)

    def _token_combine(self, mod, routed_output, device_mesh):
        # 1. 反重排
        routed_output = _unpermute(routed_output, ...)

        # 2. All-to-All：汇聚输出
        routed_output = all_to_all_single_autograd(routed_output, ...)
```

### 4.3 ExpertTensorParallel（EP+TP组合）

```python
class ExpertTensorParallel(ExpertParallel):
    def _partition_fn(self, name, mod, device_mesh):
        # 专家参数同时在EP和TP维度分片
        # w1: [experts, out_dim, in_dim] → [Shard(0), Shard(1)]
        mod.register_parameter("w1", distribute_tensor(mod.w1, device_mesh, [Shard(0), Shard(1)]))
        # w2: [experts, in_dim, out_dim] → [Shard(0), Shard(2)]
        mod.register_parameter("w2", distribute_tensor(mod.w2, device_mesh, [Shard(0), Shard(2)]))
```

### 4.4 EP应用流程

```python
def apply_moe_ep_tp(model, tp_mesh, ep_mesh, etp_mesh, ep_etp_mesh):
    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        # 1. MoE层应用TP（如果启用）
        if tp_mesh is not None:
            moe_layer_plan = {
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "moe.router.gate": NoParallel(),  # 路由器复制计算
            }
            parallelize_module(transformer_block, tp_mesh, moe_layer_plan)

        # 2. 专家应用EP
        if ep_mesh is not None and etp_mesh is None:
            experts_plan = ExpertParallel()
        elif ep_mesh is not None and etp_mesh is not None:
            experts_plan = ExpertTensorParallel()
        else:
            experts_plan = TensorParallel()  # 仅TP

        # 3. 分发到对应mesh
        if ep_mesh is not None:
            parallelize_module(transformer_block.moe.experts, ep_mesh, experts_plan)
```

### 4.5 EP通信模式

```
Token Dispatch ( All-to-All):
 EP Rank 0: [t0, t1, t2, t3] → [t0, t2] → EP Rank 0
  EP Rank 1: [t4, t5, t6, t7] → [t1, t3] → EP Rank 1

Token Combine (All-to-All反向):
  EP Rank 0: [out0, out2] → [out0, out1, out2, out3] → EP Rank 0
  EP Rank 1: [out1, out3] → [out4, out5, out6, out7] → EP Rank 1
```

---

## 五、Context Parallel (CP) 非侵入式实现

### 5.1 CP核心API

```python
from torchtitan.distributed.context_parallel import apply_cp_to_attention_module

# 应用CP到attention模块
apply_cp_to_attention_module(
    attention_modules=[block.attention for block in model.layers.values()],
    cp_mesh=cp_mesh,
    attention_type="sdpa",  # 或 "flex"
)
```

### 5.2 CP实现原理

CP使用 PyTorch DTensor 的 `_ContextParallel` 实现：

```python
from torch.distributed.tensor.experimental._attention import _ContextParallel

cp_plan = _ContextParallel(
    seq_dim=2,  # 序列维度
    attention_type=_ContextParallel.AttentionType.SDPA,
)

# 对每个attention模块应用CP
for attention_module in attention_modules:
    parallelize_module(attention_module, cp_mesh, cp_plan)
```

### 5.3 输入预处理

```python
def prepare_context_parallel_input(inputs, labels, cp_mesh, device):
    # 1. 创建位置编码
    positions = torch.arange(0, seq_len, device=device)

    # 2. 沿序列维度分片
    inputs_sharded = inputs.split(seq_len // cp_size, dim=1)[cp_rank]
    labels_sharded = labels.split(seq_len // cp_size, dim=1)[cp_rank]
    positions_sharded = positions.split(seq_len // cp_size, dim=1)[cp_rank]

    # 3. 可选：负载均衡
    if load_balancer_type == "headtail":
        # 头尾负载均衡
        pass
    elif load_balancer_type == "ptrr":
        # PTRR负载均衡
        pass

    return inputs_sharded, labels_sharded, positions_sharded
```

### 5.4 CP切分示意

```
原始序列: [s0, s1, s2, s3, s4, s5, s6, s7] (seq_len=8, CP=2)
          ┌──────────────────┬──────────────────┐
          │ [s0,s1,s2,s3]     │ [s4,s5,s6,s7]    │  → 沿seq维分片
          └──────────────────┴──────────────────┘
               CP Rank 0            CP Rank 1

Attention计算:
  - 每个CP rank计算局部attention
  - 通过Ring AllReduce交换信息
  - 最终得到完整attention结果
```

---

## 六、非侵入式并行的关键技术

### 6.1 DTensor 分布式张量

```
torch.Tensor          # 局部张量
    │
    ├── local_tensor  # 本地数据
    ├── device_mesh   # 设备网格
    └── placements    # 分片策略 [Shard(0), Replicate(), ...]
```

### 6.2 distribute_module 机制

```python
def distribute_module(
    module: nn.Module,
    device_mesh: DeviceMesh,
    partition_fn: Callable,      # 参数分区函数
    input_fn: Callable,           # 输入预处理
    output_fn: Callable,          # 输出后处理
):
    # 1. 调用partition_fn对每个参数创建DTensor
    for name, param in module.named_parameters():
        param = distribute_tensor(param, device_mesh, placements)

    # 2. 注册forward hooks处理输入输出
    module.register_forward_pre_hook(input_fn)
    module.register_forward_hook(output_fn)

    return module
```

### 6.3 ParallelStyle 工作流程

```
parallelize_module() 调用流程:

1. _apply(module, device_mesh)
      │
      ├── distribute_module()
      │     │
      │     ├── partition_fn(param) → DTensor
      │     │     例: Shard(0) 沿dim=0分片
      │     │
      │     └── 注册 input_fn / output_fn hooks
      │
      └── 返回分布式化后的module

2. Forward时:
      inputs → input_fn (DTensor转换) 
             → original_forward()
             → output_fn (DTensor转换/聚合)
             → outputs
```

### 6.4 layer_plan 配置原理与层间布局

`layer_plan` 是 `parallelize_module` 的核心参数，用于描述**每一层的输入输出期望布局**。系统根据用户的配置自动处理层间的数据分布转换。

#### 6.4.1 ParallelStyle 的默认布局约定

每种 ParallelStyle 都有**固定的默认输入输出布局**：

```python
# ColwiseParallel 默认布局
ColwiseParallel():
    input_layouts = Replicate()    # 默认输入复制
    output_layouts = Shard(1)       # 默认输出沿 hidden 分片

# RowwiseParallel 默认布局
RowwiseParallel():
    input_layouts = Shard(1)       # 默认输入沿 hidden 分片
    output_layouts = Replicate()    # 默认输出复制
```

#### 6.4.2 连续使用：自动匹配

当 ColwiseParallel 和 RowwiseParallel **连续使用**时，由于它们的输入输出布局是**互补**的，系统自动匹配，无需额外配置：

```python
layer_plan = {
    "attention.wq": ColwiseParallel(),  # 输出: Shard(1)
    "attention.wo": RowwiseParallel(),  # 默认输入: Shard(1) ← 自动匹配！
}
```

```
数据流:
attention.wq 输出: Shard(1) ─────────────────┐
                                               ├── 无需 redistribute！
attention.wo 期望: Shard(1) (默认) ←─────────┘
```

#### 6.4.3 单独使用：必须显式声明

当 ParallelStyle **单独使用**（如 tok_embeddings）时，由于没有前一层提供布局，必须显式声明输入输出布局：

```python
parallelize_module(model, tp_mesh, {
    "tok_embeddings": RowwiseParallel(
        input_layouts=Replicate(),   # 显式声明：输入是完整的
        output_layouts=Shard(1),     # 显式声明：输出沿 hidden 分片
    ),
})
```

**完整 Transformer Block 配置示例**：

```python
layer_plan = {
    # 输入准备：告诉系统 attention 模块期望的输入布局
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1),),           # 期望接收 Shard(1)
        desired_input_layouts=(Replicate(),), # 转换为 Replicate 供 QKV 使用
    ),
    
    # QKV：列并行，权重沿 hidden 分片
    "attention.wq": ColwiseParallel(),   # 输出: Shard(1)
    "attention.wk": ColwiseParallel(),   # 输出: Shard(1)
    "attention.wv": ColwiseParallel(),   # 输出: Shard(1)
    
    # O：行并行，输入分片，输出聚合
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    
    # FFN 输入准备
    "feed_forward": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    
    # FFN 权重
    "feed_forward.w1": ColwiseParallel(),  # 输出: Shard(1)
    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    "feed_forward.w3": ColwiseParallel(), # 输出: Shard(1)
}
```

#### 6.4.4 PrepareModuleInput / PrepareModuleInputOutput

当层之间布局不匹配时，使用 `PrepareModuleInput` 声明需要进行的转换：

```python
PrepareModuleInput(
    input_layouts=(Shard(1),),           # 实际接收到的布局
    desired_input_layouts=(Replicate(),), # 期望转换成的布局
)
```

| 场景 | 前层输出 | 后层期望 | 需要重分布？ |
|-----|---------|---------|------------|
| Col → Row | Shard(1) | Shard(1) | ❌ 不需要 |
| Col → Row | Shard(1) | Replicate | ✅ 需要 |
| Row → Col | Replicate | Replicate | ❌ 不需要 |

#### 6.4.5 Partial 布局：延迟聚合

对于 MoE 层，可以使用 `Partial` 布局避免不必要的 all-gather：

```python
PrepareModuleInputOutput(
    input_layouts=(Shard(1),),
    desired_input_layouts=(Replicate(),),
    output_layouts=(Partial(),),           # 输出 Partial，无需立即聚合
    desired_output_layouts=(Shard(1),),   # 下一层期望 Shard(1)
)
```

```
MoE 层数据流（零额外通信）：

输入:  Shard(1) → redistribute → Replicate
                │
                ▼
            MoE Forward
                │
                ▼
输出:  Partial (类似 all-reduce 的 partial 结果)
                │
                ▼ (下一层直接接收 Shard(1)，无需通信！)
```

#### 6.4.6 配置原则总结

| 情况 | 配置方式 | 说明 |
|-----|---------|------|
| Col → Row 连续使用 | `RowwiseParallel()` | 默认自动匹配 |
| Row → Col 连续使用 | `ColwiseParallel()` | 默认自动匹配 |
| 单独使用 | `RowwiseParallel(input_layouts=..., output_layouts=...)` | 必须显式声明 |
| 布局不匹配 | `PrepareModuleInput(input_layouts=..., desired_input_layouts=...)` | 显式声明转换 |
| 延迟聚合 | `output_layouts=Partial()` | 避免不必要的通信 |

---

## 七、完整并行应用示例

```python
# 假设: TP=2, EP=4, CP=2, DP_SHARD=2, world_size=16

# 1. 定义并行维度
parallel_dims = ParallelDims(
    dp_replicate=1,
    dp_shard=2,
    cp=2,
    tp=2,
    pp=1,
    ep=4,
    etp=1,
    world_size=16,
)
parallel_dims.build_mesh()

# 2. 应用TP（非MoE部分）
tp_mesh = parallel_dims.get_mesh("tp")
apply_non_moe_tp(model, tp_mesh, loss_parallel=True, cp_enabled=True)

# 3. 应用MoE + EP + TP
ep_mesh = parallel_dims.get_optional_mesh("ep")
etp_mesh = parallel_dims.get_optional_mesh("etp")
apply_moe_ep_tp(
    model,
    tp_mesh=tp_mesh,
    ep_mesh=ep_mesh,
    etp_mesh=etp_mesh,
    ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
)

# 4. 应用CP
if parallel_dims.cp_enabled:
    cp_mesh = parallel_dims.get_mesh("cp")
    apply_cp_to_attention_module(
        [block.attention for block in model.layers.values()],
        cp_mesh,
        "sdpa",
    )

# 5. 应用FSDP/HSDP
dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
model = data_parallel(model, dp_mesh, dp_mode="hybrid_shard")
```

---

## 八、Mesh维度映射表

| 并行度 | 公式 | 示例(TP=2, EP=4, CP=2, DP=2, PP=1) |
|-------|------|----------------------------------|
| batch | `dp_replicate * dp_shard` | 2 |
| fsdp | `dp_shard * cp` | 4 |
| efsdp | `dp_shard * cp * tp // (etp * ep)` | 1 |
| loss | `dp_replicate * dp_shard * cp` | 4 |
| ep | ep | 4 |
| etp | etp | 1 |

---

## 九、总结

TorchTitan 的非侵入式并行实现核心在于：

1. **ParallelDims**: 统一管理所有并行维度，自动构建DeviceMesh层次结构
2. **parallelize_module()**: 基于PyTorch DTensor的模块分布式化API
3. **ParallelStyle体系**: ColwiseParallel、RowwiseParallel、ExpertParallel等策略类
4. **DTensor**: 参数和激活的分布式表示，支持灵活的切分和通信
5. **Hooks机制**: 通过forward pre/hook处理输入输出的分布式转换

这种设计使得模型代码保持纯净，无需感知分布式细节，真正实现了渐进式引入并行能力。

---

## 十、多模态模型支持

TorchTitan 支持多模态理解（VLM）和多模态生成（Flux）模型的并行训练。

### 10.1 Vision Language Model (VLM) - 多模态理解

**项目路径**: `torchtitan/experiments/vlm/`

**核心特性**:
- 原生宽高比：不限于正方形裁剪
- 原生分辨率：batch中图像可以不同尺寸，无需图像切片和缩略图
- 原生交错数据：训练样本可包含不同数量的图像，与文本交织在不同位置

**模型架构**:

```
Llama3Siglip2Transformer (Llama3 + Siglip2)
  ├── VisionTransformer (encoder)     # Siglip2视觉编码器
  │     └── 支持原生分辨率和宽高比
  ├── Projector                       # 投影层 (Linear + SiLU + Linear)
  │     └── in_dim: encoder.dim → out_dim: llm.dim
  └── Llama3 (decoder)                # LLM主干
```

**并行支持状态**:

| 并行方式 | 支持状态 | 说明 |
|---------|---------|------|
| FSDP/HSDP | ✅ | Encoder和Decoder均支持 |
| CP | ✅ | 仅支持LLM部分 |
| TP | ❌ | 正在开发中 |
| PP | ❌ | 尚未支持 |

**实现文件**:
- [model.py](file:///home/hgq/workspace/aicoder/ws3/torchtitan/torchtitan/experiments/vlm/model/model.py): VLM模型定义
- [siglip2.py](file:///home/hgq/workspace/aicoder/ws3/torchtitan/torchtitan/experiments/vlm/model/siglip2.py): Siglip2 encoder
- [parallelize.py](file:///home/hgq/workspace/aicoder/ws3/torchtitan/torchtitan/experiments/vlm/infra/parallelize.py): 并行化应用

### 10.2 Flux - 多模态图像生成

**项目路径**: `torchtitan/models/flux/`

**核心特性**:
- 文本到图像生成 (Text-to-Image)
- 支持 FLUX.1-dev 和 FLUX.1-schnell 模型

**模型架构**:

```
FluxModel
  ├── img_in      # 图像输入层
  ├── time_in     # 时间步输入层
  ├── vector_in   # 条件向量输入层
  ├── txt_in      # 文本输入层
  ├── double_blocks   # 双块注意力 (self-attn + cross-attn)
  ├── single_blocks   # 单块注意力 (self-attn)
  └── final_layer    # 输出层
```

**并行支持状态**:

| 并行方式 | 支持状态 | 说明 |
|---------|---------|------|
| FSDP/HSDP | ✅ | 完全支持 |
| CP | ✅ | 支持 |
| TP | ❌ | 正在开发 |
| PP | ❌ | 尚未支持 |
| Activation Checkpointing | ✅ | 支持 |
| torch.compile | ❌ | 尚未支持 |

**实现文件**:
- [flux/model/](file:///home/hgq/workspace/aicoder/ws3/torchtitan/torchtitan/models/flux/model/): Flux模型定义
- [parallelize.py](file:///home/hgq/workspace/aicoder/ws3/torchtitan/torchtitan/models/flux/parallelize.py): 并行化应用

### 10.3 多模态并行策略应用示例

**VLM并行化流程**:

```python
# torchtitan/experiments/vlm/infra/parallelize.py

def parallelize_vlm(model, parallel_dims, training, ...):
    # 1. 应用Activation Checkpointing
    apply_ac(model, ac_config)
    apply_ac(model.encoder, ac_config)

    # 2. 应用torch.compile（可选）
    if compile_config.enable:
        apply_compile(model, compile_config)
        apply_compile(model.encoder, compile_config)

    # 3. 应用FSDP/HSDP到Encoder和Decoder
    if parallel_dims.fsdp_enabled:
        # FSDP for Encoder
        for layer in model.encoder.layers.values():
            fully_shard(transformer_block, **fsdp_config)

        # FSDP for Decoder (LLM)
        for layer in model.layers.values():
            fully_shard(transformer_block, **fsdp_config)

        # Context Parallel for LLM (not Encoder)
        if parallel_dims.cp_enabled:
            apply_cp_to_attention_module(
                [block.attention for block in model.layers.values()],
                cp_mesh,
                "sdpa",
            )

    # TP尚未支持（实现中）
    if parallel_dims.tp_enabled:
        raise NotImplementedError("TP support for VLM training is still in progress.")
```

### 10.4 多模态支持总结

| 模型类型 | 模型 | FSDP | HSDP | CP | TP | PP |
|---------|------|------|------|-----|-----|-----|
| 多模态理解 | Llama3+Siglip2 | ✅ | ✅ | ✅ | 🔄 | ❌ |
| 图像生成 | Flux | ✅ | ✅ | ✅ | 🔄 | ❌ |

✅: 已支持 | 🔄: 开发中 | ❌: 尚未支持
