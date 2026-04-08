# 训练反向传播对非侵入式并行的新要求

## 1. 背景：推理 vs 训练的反向传播挑战

**推理**只需要前向传播——激活值被计算后丢弃。**训练**需要反向传播，这意味着：

- 梯度必须流过所有并行边界
- 集合操作（all-reduce、all-to-all）必须是 autograd 可知的
- 梯度同步必须在正确的时机发生
- 激活和梯度的内存必须在分片 rank 之间管理

TorchTitan 的"非侵入式"理念意味着这些挑战通过 DTensor hooks 和并行风格包装器**在模型代码外部处理**，而不是修改模型的前向方法。

## 2. DTensor 自动微分 for TP

### 如何分片权重在反向传播中工作

当 `ColwiseParallel` 或 `RowwiseParallel` 通过 `distribute_module()` 并行化模块时，它用 DTensor 分片的参数替换模块的参数，并用 input/output hooks 包装前向函数。反向传播由 PyTorch 的 DTensor autograd 系统自动处理：

- **前向**：`distribute_module` 包装输入准备（`_prepare_input_fn`）和输出提取（`_prepare_output_fn`）hooks。Hooks 调用 `DTensor.from_local()` 将本地张量转换为 DTensor，调用 `to_local(grad_placements=...)` 转换回来。

- **反向**：`DTensor.from_local()` 的 backward（当设置了 `grad_placements` 时）产生梯度的 all-reduce 以匹配输入的布局。`to_local(grad_placements=...)` 的 backward 产生梯度重新分配。

在 torchtitan 的 `tensor_parallel.py` 中，**`ColwiseParallelWithGradPlacement`**（第109-188行）是一个关键子类，通过 `local_input_grad_placements` 对反向梯度布局提供**显式控制**。基础 `ColwiseParallel` 默认将 `d_x` all-reduce 为 Replicate，但这个子类让用户指定本地输入梯度应该具有的确切梯度布局——这很重要，因为 TP 分片的激活需要它们的梯度也为下一层分片。

```python
# 关键代码示例 - 显式梯度布局控制
input_tensor = DTensor.from_local(
    input_tensor,
    device_mesh,
    input_layouts,
    run_check=False,
    grad_placements=local_input_grad_placements,  # 控制反向行为
)
```

**`NoParallel`** 风格用于 TP mesh 上的复制计算（如 MoE router/gate）。它的 `_prepare_output_fn` 显示 `to_local(grad_placements=local_output_grad_placements)` —— `grad_placements` 参数是控制反向梯度流的关键 DTensor API。

## 3. FSDP + Tensor Parallelism 集成

### 核心挑战

当 TP 和 FSDP 同时启用时，它们在**正交 mesh 维度**上操作：

- **TP** 沿 dense mesh 的 `tp` 维度分片：`[pp, dp_replicate, fsdp, tp]`
- **FSDP** 沿 `fsdp` 维度分片：`[pp, dp_replicate, fsdp, tp]`，其中 `fsdp = dp_shard * cp`

关键洞察是：**FSDP 在 TP 外部/之后应用**（torchtitan 的 `apply_fsdp()`）：

```python
# 1. 先应用 TP（沿 tp dim 分片权重）
apply_tp(model, tp_mesh, ...)
# 2. 然后应用 FSDP（沿 fsdp dim 分片梯度）
apply_fsdp(model, dp_mesh, ...)
```

### FSDP 如何与 TP 分片参数交互

1. **前向**：FSDP 的 `fully_shard` 在前向之前 all-gather TP 分片参数。每个 FSDP rank 持有 TP 分片参数的**完整副本**（因为 FSDP 沿 `fsdp` 分片，每个 FSDP rank 持有所有 TP 分片）。

2. **反向**：FSDP 的反向：
   - 为完整参数计算本地梯度
   - **reduce-scatter** 沿 FSDP mesh 的梯度（所以每个 FSDP rank 最终持有一个分片）
   - 对于下一个 microbatch，再次 all-gather

`disable_fsdp_gradient_division()` 函数禁用 FSDP 的自动梯度除法（`gradient_divide_factor=1.0`），因为 torchtitan 使用 `global_valid_tokens` **手动处理**梯度缩放：

```python
# 手动缩放
loss = loss_sum / global_valid_tokens
```

这种手动缩放在 CP 或 FSDP 改变每个 rank 处理的 token 数量时至关重要。

## 4. 各并行方式的反向传播挑战

### TP (Tensor Parallelism)

**Optimizer state 的梯度 all-reduce**：反向后，每个 TP rank 持有其权重分片的梯度。在 DTensor 中，分片参数的梯度本身也是分片的。

**ColwiseParallel 反向**：对于列分片的 linear（`w = Shard(0)` 在输出维度），`y = x @ W` 的反向计算 `dW = x^T @ dy` 是本地的，然后 **all-reduces** `dW` 以便每个 rank 获得完整梯度。

**RowwiseParallel 反向**：对于行分片的 linear（`w = Shard(1)` 在输入维度），`y = x @ W` 的反向计算 `dx = dy @ W^T` 是本地的。但 `dx` 需要 all-reduce 如果输入是复制的。

**梯度范数裁剪**：`clip_grad_norm_()` 在 `utils.py` 中显式处理 `_NormPartial` 布局的 DTensor 梯度。当 PP 启用时，它额外跨 PP stage all-reduce 总范数。

### EP (Expert Parallelism)

**梯度路由是核心挑战**：前向传播使用 `all_to_all_single_autograd` 进行 token 分发（tokens 路由到 expert ranks）和合并（tokens 路由回来）。反向传播必须**反转这个路由**：

- **分发的反向**：流入 expert 计算的梯度必须路由回原始 rank —— 由 `all_to_all_single_autograd` 的反向自动处理
- **Expert 权重梯度**：每个 EP rank 只持有其本地 experts 的梯度。这些梯度**不在 EP ranks 之间 reduce**（与 FSDP 不同），因为每个 rank 拥有不同的 experts

**EP 梯度裁剪**：`_clip_grad_norm_with_ep()` 处理 EP params 和 non-EP params 位于不同 mesh 维度（"ep" vs 标准 mesh）的情况。它分别计算每组的范数然后合并。

**EP + OptimizersInBackward 不兼容**：代码显式拒绝此组合，因为通过 EP 的 all-to-all  collectives 路由梯度与 backbone 内优化器实现不兼容。

### CP (Context Parallelism)

**跨 CP ranks 的注意力梯度累积**：CP 沿序列维度分片到各 rank。反向时：
- 每个 CP rank 为其本地序列块计算注意力梯度
- 梯度必须正确跨 CP ranks 累积/分散

**CP + FSDP 总是同时启用**：当使用 CP 时，FSDP 总是应用以"利用其权重 all-gather 和梯度 reduce-scatter，即使可能没有数据并行（global batch size 为 1）"。

### PP (Pipeline Parallelism)

**反向传播调度**：PyTorch 的 `PipelineSchedule`（来自 `torch.distributed.pipelining`）管理反向传播调度。关键选项：
- `PipelineScheduleSingle`（1F1B）
- `PipelineScheduleMulti`（交错调度）
- `ScheduleDualPipeV`（V 型）
- `ScheduleZBVZeroBubble`（零气泡）

**PP + 激活重计算**：PyTorch 的 `CheckpointWrapper` 按 transformer 块应用。在 PP 中，检查点边界与 stage 边界对齐。`_apply_op_sac()` 实现 per-op 选择性激活检查点：
- **MUST_SAVE** 昂贵的计算 ops（matmuls、attention）以避免重计算
- **MUST_SAVE** 通信 ops（reduce_scatter、all_to_all）以避免重新通信
- 使用跟踪 mm 计数并选择性地每第二个 matmul 重计算的自定义策略

**`scale_grads=False` 的含义**：当 `False` 时，来自 microbatches 的梯度被**求和**而非平均，调用者负责正确的缩放。TorchTitan 通过除以 `global_valid_tokens` 来做到这一点。

## 5. 激活检查点（AC）与并行的交互

**SAC（选择性 AC）+ TP**：`activation_checkpoint.py` 中的 `_get_save_ops()` 显式列出输出重计算成本高昂的 ops：
- 通信 ops：`reduce_scatter_tensor`、`all_to_all_single`、DeepEP/HybridEP 分发/合并
- 计算 ops：SDPA、FlexAttention、linear

关键原则：**通信 ops 总是被保存**（MUST_SAVE），因为重新通信比保存更昂贵。这对 EP 的 all-to-all 和 TP 的隐式集合操作至关重要。

**AC + PP**：存在已知的 `SAC + PP + FlexAttention` 可能触发重编译问题的交互。解决方法（lines 221-232）禁用 Dynamo LRU 缓存以避免断言失败。

**AC 和 `torch.compile` 顺序**：
```python
# 1. 先应用 AC
if ac_config.mode != "none":
    apply_ac(model, ac_config, ...)
# 2. AC 之后应用 compile，FS之前
if model_compile_enabled:
    apply_compile_dense(model, compile_config)
# 3. 最后应用 FSDP
if parallel_dims.fsdp_enabled:
    apply_fsdp(model, dp_mesh, ...)
```

## 6. loss_parallel 机制

这是 TP 训练的关键基础设施。`torch.distributed.tensor.parallel.loss_parallel()` 上下文管理器：

1. TP 输出层使用 `Shard(-1)` 在 logits 维度（每个 TP rank 持有 `vocab_size // tp_degree` 的一个切片）
2. 在 `loss_parallel()` 上下文中，`cross_entropy` **并行计算**——每个 TP rank 为其 vocab 切片计算损失贡献
3. 上下文返回一个**复制的** DTensor 损失（所以所有 ranks 认同该值）
4. `loss_parallel` 的反向正确地将梯度分布回各 TP ranks（每个 rank 获得其 vocab 切片的梯度）

**为什么重要**：没有 `loss_parallel`，完整的 logits 需要在计算损失之前从所有 TP ranks 聚集，然后在反向时分散。有了 `loss_parallel`，损失计算本身是并行的，减少了通信。

## 7. Mesh 布局及其对反向传播的影响

TorchTitan 的 mesh 布局决定梯度流：

```
dense_mesh: ["pp", "dp_replicate", "fsdp", "tp"]
sparse_mesh: ["pp", "dp_replicate", "efsdp", "ep", "etp"]
```

关键含义：

- **TP 梯度保持在 TP mesh 内**——TP-only 参数不需要跨 TP rank 的梯度同步
- **FSDP 梯度在 fsdp mesh 内 reduce**（= dp_shard * cp）——处理数据并行和 CP 分片
- **PP 梯度流经 PP ranks**——每个 PP rank 拥有一个模型 stage 并计算本地梯度
- **EP 参数位于 sparse mesh**——expert 权重的梯度沿 EP ranks 分片，FSDP 沿 efsdp 分片

`fsdp_gradient_divide_factor` 属性（`dp_replicate * dp_shard * cp`）确保即使 expert 参数分片方式不同，FSDP 的梯度除法也是一致的。

## 8. 总结：非侵入式并行在训练中的新要求

| 并行方式 | 训练反向要求 | 非侵入式实现难度 |
|---------|------------|----------------|
| **TP** | DTensor autograd 处理梯度 all-reduce；loss_parallel 机制使损失计算并行化 | 低——hooks 自动处理 |
| **EP** | all-to-all 反向需要 autograd 可知；梯度不跨 EP ranks 同步 | 中——需要 all_to_all_autograd |
| **CP** | 注意力梯度跨序列分片累积；与 FSDP 耦合 | 中——SDPA dispatcher 处理 |
| **PP** | Pipeline 调度管理反向；激活重计算必须保存通信 ops | 高——需要 PipelineStage 协调 |

**训练对非侵入式的核心新要求**：

1. **通信 ops 必须是 autograd 可知的**：所有集合操作（all-reduce、all-to-all、reduce-scatter）必须参与反向传播计算
2. **梯度同步时机**：FSDP 的 all-gather（在反向前）和 reduce-scatter（在反向后）必须在正确的时机发生
3. **手动梯度缩放**：当并行方式改变有效 token 数量时（如 CP、FSDP），必须手动缩放损失
4. **激活内存管理**：AC 必须与并行方式协调——通信 ops 不能被重计算
5. **优化器状态分片**：FSDP 与 TP/EP 的组合需要仔细管理 optimizer state 的分片
