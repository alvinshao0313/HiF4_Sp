# MLP 128×128 Block Gradient Pruning 设计文档

## 1. 目标

实现一个仅针对 Transformer MLP 模块的结构化块剪枝流程：

1. 使用校准数据执行完整模型前向与 loss 反向传播；
2. 根据最终语言模型 loss 对每个 `128×128` 权重块的梯度敏感度估计块重要性；
3. 按重要性从低到高选择权重块；
4. 将选中的块整体置零，形成严格的二维块稀疏权重；
5. 保存块级 score、mask、剪枝后模型及统计报告。

本版本只负责：

- MLP 权重块重要性统计；
- 块级 mask 分配；
- 块剪枝和评估。

本版本不包含：

- SparseGPT/OBS 权重补偿；
- 剪枝后微调或蒸馏；
- Attention 模块剪枝；
- 运行时 block-sparse kernel；
- 通道剪枝或 residual width 压缩。

---

## 2. 支持范围

首版只支持 LLaMA/Qwen 类 SwiGLU MLP：

$$ U = XW_{\mathrm{up}}^\top, $$

$$ G = XW_{\mathrm{gate}}^\top, $$

$$ H = \operatorname{SiLU}(G)\odot U, $$

$$ Y = HW_{\mathrm{down}}^\top. $$

需要识别的 Linear：

- `mlp.up_proj`
- `mlp.gate_proj`
- `mlp.down_proj`

PyTorch 中权重形状为：

$$ W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}. $$

每个权重矩阵沿输出维和输入维划分为：

$$ W_{a,b} = W[128a:128(a+1),\,128b:128(b+1)]. $$

首版要求：

```text
d_out % 128 == 0
d_in  % 128 == 0
```

若不满足，直接报错，不对边界块做隐式 padding。这样可以保证生成的 mask 与后续 128×128 block-sparse kernel 的物理 tile 完全一致。

---

## 3. 核心思想：把整个权重块视为一个标量门控变量

对于第 $b$ 个权重块，引入标量 mask：

$$ \widetilde W_b=m_bW_b,\qquad m_b=1. $$

删除该块对应：

$$ m_b:1\rightarrow0. $$

在当前 dense 模型处，loss 对块 mask 的梯度为：

$$ \frac{\partial\mathcal L}{\partial m_b} = \sum_{(i,j)\in b} \frac{\partial\mathcal L}{\partial W_{ij}}W_{ij} = \left\langle \nabla_{W_b}\mathcal L,\, W_b \right\rangle. $$

因此不需要真的给每个块创建一个可训练标量 mask。只需在 loss 反向传播后，从权重和权重梯度直接计算：

$$ g_b = \left\langle \nabla_{W_b}\mathcal L,\, W_b \right\rangle. $$

其中 $g_b$ 是“缩放或删除整个块”时的一阶 loss 敏感度。

---

## 4. 推荐的重要性指标

### 4.1 不使用全校准集平均梯度后再取绝对值

禁止使用：

$$ S_b = \left| \left\langle \frac{1}{N}\sum_n\nabla_{W_b}\ell_n,\, W_b \right\rangle \right|. $$

原因是不同样本、不同 token 的梯度符号可能相反，先平均会产生严重抵消。

### 4.2 主指标：Block Empirical Fisher

对每个校准 batch 独立计算：

$$ g_b^{(t)} = \left\langle \nabla_{W_b}\mathcal L_t,\, W_b \right\rangle. $$

然后先平方，再跨 batch 累积：

$$ \boxed{ S_b^{\mathrm{Fisher}} = \frac{1}{T} \sum_{t=1}^{T} \left(g_b^{(t)}\right)^2 } $$

最终 score 必须满足：

```text
score >= 0
```

score 越大，表示该块对最终 loss 越敏感，越应该保留。

score 越小，表示该块可以优先剪除。

### 4.3 可选诊断指标

同时记录但不用于主排序：

$$ S_b^{\mathrm{AbsTaylor}} = \frac{1}{T} \sum_t \left|g_b^{(t)}\right|, $$

$$ S_b^{\mathrm{SignedMean}} = \frac{1}{T} \sum_t g_b^{(t)}. $$

用途：

- Fisher：主剪枝分数；
- AbsTaylor：检查 Fisher 是否被极少数异常 batch 主导；
- SignedMean：检查梯度抵消程度。

首版默认：

```yaml
score_type: fisher
```

---

## 5. Loss 定义

首版直接使用完整 causal language modeling loss：

$$ \mathcal L_{\mathrm{LM}} = -\frac{1}{|\Omega|} \sum_{t\in\Omega} \log p_\theta(x_t\mid x_{<t}), $$

其中 $\Omega$ 是有效 label token 集合。

实现要求：

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    use_cache=False,
)
loss = outputs.loss
```

注意：

1. padding token 对应 label 必须设置为 `-100`；
2. 模型必须处于 `eval()` 模式，关闭 dropout；
3. 不使用 teacher-student MSE 作为初始 score loss，因为 dense student 与 teacher 完全相同时，重建损失和梯度均可能为零；
4. 首版 calibration batch size 建议为 `1`，以减少样本间梯度抵消；
5. 若 batch size 大于 1，应固定 batch size 和 sequence length，避免不同 batch 的 score 尺度不可比。

推荐默认配置：

```yaml
calibration_samples: 128
sequence_length: 2048
score_batch_size: 1
loss_type: causal_lm
```

---

## 6. 需要参与梯度统计的参数

只对 MLP Linear 权重开启梯度：

```python
for param in model.parameters():
    param.requires_grad_(False)

for module_name, module in target_mlp_linears:
    module.weight.requires_grad_(True)
```

其他参数不需要保存 `.grad`，但完整计算图仍必须保留，以便最终 LM loss 的梯度能够穿过后续层回传到目标 MLP 权重。

目标参数：

```text
*.mlp.up_proj.weight
*.mlp.gate_proj.weight
*.mlp.down_proj.weight
```

首版不统计 bias。LLaMA/Qwen 通常也没有这些 bias。

---

## 7. Block score 的计算

给定：

```python
weight.shape == [d_out, d_in]
grad.shape   == [d_out, d_in]
block_size   == 128
```

先计算 element-wise Taylor signal：

```python
element_signal = weight.float() * grad.float()
```

再按 `128×128` 分块求和：

```python
block_signal = element_signal.reshape(
    d_out // 128,
    128,
    d_in // 128,
    128,
).sum(dim=(1, 3))
```

得到：

```python
block_signal.shape == [
    num_output_blocks,
    num_input_blocks,
]
```

对当前 batch：

```python
score_sq += block_signal.square()
score_abs += block_signal.abs()
score_signed += block_signal
num_batches += 1
```

最终：

```python
fisher_score = score_sq / num_batches
abs_taylor_score = score_abs / num_batches
signed_mean = score_signed / num_batches
```

必须在每个 batch 结束后执行：

```python
model.zero_grad(set_to_none=True)
```

禁止把多个 batch 的 weight gradient 累加后再计算 block score。

---

## 8. Score 数据结构

为每个 MLP Linear 创建独立记录：

```python
@dataclass
class BlockScoreRecord:
    module_name: str
    layer_index: int
    projection_type: str   # up_proj / gate_proj / down_proj
    weight_shape: tuple[int, int]
    block_size: int
    fisher: torch.Tensor
    abs_taylor: torch.Tensor
    signed_mean: torch.Tensor
    current_mask: torch.Tensor
```

其中：

```text
fisher.shape
abs_taylor.shape
signed_mean.shape
current_mask.shape
=
[num_output_blocks, num_input_blocks]
```

`current_mask` 语义：

```text
True  = 当前保留
False = 已剪除
```

所有 score 建议累计到 CPU FP64 或 CPU FP32：

```python
record.fisher += block_signal.detach().double().cpu().square()
```

权重和梯度可保持 BF16/FP16，但 block reduce 前需要转 FP32。

---

## 9. Mask 分配策略

### 9.1 不建议完全无约束地全模型 Bottom-K

直接把所有 MLP block 放在一起排序，可能导致少数层或少数 projection 被全部剪空。

首版采用：

$$ \boxed{ \text{Global ranking} + \text{per-matrix pruning cap} } $$

即：

1. score 使用最终 LM loss 的原始 Fisher score，不做层内归一化；
2. 所有 MLP block 全局排序；
3. 每个 Linear 设置最大可剪比例；
4. 按 score 从小到大遍历，满足约束时剪除；
5. 达到目标全局块数后停止。

推荐默认值：

```yaml
target_block_sparsity: 0.30
max_prune_ratio_per_matrix: 0.60
min_keep_blocks_per_matrix: 1
```

### 9.2 目标剪枝数

设全部候选块数为：

$$ B_{\mathrm{total}}. $$

目标剪枝数：

$$ B_{\mathrm{target}} = \left\lfloor \rho_{\mathrm{target}} B_{\mathrm{total}} \right\rfloor. $$

候选块按：

```text
(fisher_score, module_name, output_block_idx, input_block_idx)
```

升序排序。

选择时必须跳过：

- 已剪块；
- 达到 `max_prune_ratio_per_matrix` 的矩阵；
- 会违反 `min_keep_blocks_per_matrix` 的候选。

若约束导致无法达到目标稀疏率，应报错并输出最大可实现稀疏率，不允许静默降低目标。

---

## 10. Up/Gate mask 对齐模式

up_proj 和 gate_proj 形状相同、输入相同。为支持后续 fused SwiGLU block-sparse kernel，提供可选模式：

```yaml
share_up_gate_mask: false
```

### 独立模式

```text
up_proj 和 gate_proj 分别排序、分别生成 mask。
```

优势：

- 算法自由度高；
- 通常更容易获得较好精度。

### 共享模式

对于同一层、同一 block 坐标 $(a,b)$，定义联合分数：

$$ S_{a,b}^{ug} = S_{a,b}^{up} + S_{a,b}^{gate}. $$

候选单位变成一个 up/gate block pair：

```text
(layer_idx, output_block_idx, input_block_idx)
```

一旦剪除，同时执行：

```text
up_proj block   -> 0
gate_proj block -> 0
```

共享 pair 的成本按两个物理 block 计入全局稀疏率。

首版建议先实现独立模式，随后增加共享模式作为硬件约束消融。

down_proj 不与 up/gate 强制共享二维 tile mask。因为：

- up/gate 的中间维位于权重输出轴；
- down 的中间维位于权重输入轴；
- 二维 tile 并不存在简单的一对一映射。

---

## 11. 一次性剪枝与迭代剪枝

### 11.1 首版 baseline：一次性剪枝

流程：

```text
dense model
  -> score all MLP blocks
  -> allocate target mask
  -> prune once
  -> evaluate
```

适合：

- 验证 gradient score 是否有效；
- 与 block magnitude、random pruning 做公平比较；
- 目标稀疏率不高，例如 10%–30%。

### 11.2 推荐扩展：迭代剪枝

由于 128×128 block 较大，多个块同时删除会产生明显交互。高稀疏率建议采用：

```yaml
pruning_rounds: 4
```

若最终目标为 $\rho$，第 $r$ 轮累计目标可以设置为：

$$ \rho_r = \rho\frac{r}{R}, \qquad r=1,\dots,R. $$

每轮执行：

```text
1. 在当前稀疏模型上重新统计剩余 block 的 Fisher score；
2. 排除 current_mask=False 的 block；
3. 增量剪枝至当前累计目标；
4. 物理置零新剪 block；
5. 进入下一轮。
```

注意：已剪 block 的权重为零，其 `weight * grad` 也为零，因此不能依靠 score 自动排除，必须显式使用 `current_mask` 排除。

---

## 12. 应用剪枝

对某个 Linear 的 block mask：

```python
block_mask.shape == [
    d_out // 128,
    d_in // 128,
]
```

展开为 element mask：

```python
element_mask = block_mask.repeat_interleave(128, dim=0)
element_mask = element_mask.repeat_interleave(128, dim=1)
```

物理置零：

```python
with torch.no_grad():
    module.weight.mul_(
        element_mask.to(
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
    )
```

首版不替换 `nn.Linear`，仍使用 dense Linear 做正确性评估。

需要额外保存 block mask，供后续：

- block-sparse kernel 转换；
- 稀疏 checkpoint 加载；
- 剪枝后恢复训练；
- 稀疏率统计。

---

## 13. 推荐代码结构

```text
block_pruning/
├── config.py
├── mlp_registry.py
├── block_utils.py
├── gradient_scorer.py
├── mask_allocator.py
├── mask_apply.py
├── serialization.py
└── evaluator.py

scripts/
└── score_and_prune_mlp.py

tests/
├── test_block_reduce.py
├── test_mask_gradient_equivalence.py
├── test_global_allocator.py
├── test_apply_mask.py
└── test_mlp_only_changed.py
```

### 13.1 `config.py`

定义：

```python
@dataclass
class GradientBlockPruningConfig:
    model_path: str
    calibration_dataset: str
    output_dir: str

    block_size: int = 128
    target_block_sparsity: float = 0.30

    calibration_samples: int = 128
    sequence_length: int = 2048
    score_batch_size: int = 1

    score_type: str = "fisher"
    selection_mode: str = "global_constrained"

    max_prune_ratio_per_matrix: float = 0.60
    min_keep_blocks_per_matrix: int = 1

    share_up_gate_mask: bool = False
    pruning_rounds: int = 1

    seed: int = 42
    score_accumulation_dtype: str = "float64"
```

### 13.2 `mlp_registry.py`

职责：

- 遍历 `model.named_modules()`；
- 识别 `up_proj/gate_proj/down_proj`；
- 验证模块是 `nn.Linear`；
- 验证权重维度可被 128 整除；
- 解析 layer index 和 projection type；
- 返回稳定排序后的 target module 列表。

接口：

```python
def collect_mlp_linears(model, block_size: int) -> list[MLPLinearTarget]:
    ...
```

### 13.3 `block_utils.py`

提供：

```python
def reduce_weight_gradient_to_blocks(
    weight: torch.Tensor,
    grad: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    ...
```

```python
def expand_block_mask(
    block_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    ...
```

### 13.4 `gradient_scorer.py`

职责：

- 冻结非目标参数；
- 遍历校准 batch；
- 计算 causal LM loss；
- 反向传播；
- 逐模块计算 block signal；
- 先平方后累计；
- 返回 `dict[module_name, BlockScoreRecord]`。

主接口：

```python
def collect_mlp_block_scores(
    model,
    dataloader,
    targets,
    config,
    current_masks=None,
) -> dict[str, BlockScoreRecord]:
    ...
```

### 13.5 `mask_allocator.py`

职责：

- 生成全局候选列表；
- 实现 constrained global Bottom-K；
- 支持 up/gate 独立或共享 mask；
- 检查目标块数和约束；
- 输出新 mask 和统计报告。

主接口：

```python
def allocate_block_masks(
    score_records,
    config,
    current_masks=None,
    cumulative_target_sparsity=None,
) -> MaskAllocationResult:
    ...
```

### 13.6 `mask_apply.py`

主接口：

```python
def apply_mlp_block_masks(
    model,
    masks: dict[str, torch.Tensor],
    block_size: int,
) -> None:
    ...
```

### 13.7 `serialization.py`

保存：

```text
block_scores.pt
block_masks.pt
pruning_summary.json
per_matrix_report.csv
pruned_model/
```

---

## 14. 端到端主流程伪代码

```python
def main(config):
    set_seed(config.seed)

    model, tokenizer = load_model_and_tokenizer(config.model_path)
    model.eval()
    model.config.use_cache = False

    dataloader = build_calibration_dataloader(
        tokenizer=tokenizer,
        dataset_name=config.calibration_dataset,
        num_samples=config.calibration_samples,
        sequence_length=config.sequence_length,
        batch_size=config.score_batch_size,
    )

    targets = collect_mlp_linears(
        model=model,
        block_size=config.block_size,
    )

    current_masks = initialize_all_one_masks(targets)

    for round_idx in range(config.pruning_rounds):
        cumulative_target = (
            config.target_block_sparsity
            * (round_idx + 1)
            / config.pruning_rounds
        )

        score_records = collect_mlp_block_scores(
            model=model,
            dataloader=dataloader,
            targets=targets,
            config=config,
            current_masks=current_masks,
        )

        allocation = allocate_block_masks(
            score_records=score_records,
            config=config,
            current_masks=current_masks,
            cumulative_target_sparsity=cumulative_target,
        )

        current_masks = allocation.masks

        apply_mlp_block_masks(
            model=model,
            masks=current_masks,
            block_size=config.block_size,
        )

        save_round_artifacts(
            round_idx=round_idx,
            score_records=score_records,
            allocation=allocation,
            output_dir=config.output_dir,
        )

    verify_masks_and_weights(
        model=model,
        masks=current_masks,
        targets=targets,
        block_size=config.block_size,
    )

    evaluate_model(model, tokenizer, config)

    save_final_model_and_masks(
        model=model,
        tokenizer=tokenizer,
        masks=current_masks,
        output_dir=config.output_dir,
    )
```

---

## 15. 梯度统计伪代码

```python
def collect_mlp_block_scores(
    model,
    dataloader,
    targets,
    config,
    current_masks,
):
    freeze_all_parameters(model)

    for target in targets:
        target.module.weight.requires_grad_(True)

    accumulators = create_score_accumulators(
        targets=targets,
        block_size=config.block_size,
        dtype=torch.float64,
        device="cpu",
    )

    num_batches = 0

    for batch in dataloader:
        model.zero_grad(set_to_none=True)

        batch = move_batch_to_model_device(batch)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )

        loss = outputs.loss
        loss.backward()

        for target in targets:
            weight = target.module.weight
            grad = weight.grad

            if grad is None:
                raise RuntimeError(
                    f"No gradient for target module: {target.module_name}"
                )

            block_signal = reduce_weight_gradient_to_blocks(
                weight=weight.detach(),
                grad=grad.detach(),
                block_size=config.block_size,
            )

            # 已剪 block 不再参与后续排序，但统计文件中可以保留 0。
            active_mask = current_masks[target.module_name].to(
                block_signal.device
            )

            block_signal = block_signal * active_mask

            accumulators[target.module_name].score_sq += (
                block_signal.double().cpu().square()
            )
            accumulators[target.module_name].score_abs += (
                block_signal.double().cpu().abs()
            )
            accumulators[target.module_name].score_signed += (
                block_signal.double().cpu()
            )

        num_batches += 1

    records = finalize_score_records(
        accumulators=accumulators,
        num_batches=num_batches,
        current_masks=current_masks,
    )

    model.zero_grad(set_to_none=True)
    return records
```

---

## 16. Global constrained mask 分配伪代码

```python
def allocate_global_constrained(
    score_records,
    current_masks,
    target_block_sparsity,
    max_prune_ratio_per_matrix,
    min_keep_blocks_per_matrix,
):
    total_blocks = sum(mask.numel() for mask in current_masks.values())
    target_pruned = int(total_blocks * target_block_sparsity)

    already_pruned = sum(
        (~mask).sum().item()
        for mask in current_masks.values()
    )

    additional_needed = target_pruned - already_pruned
    if additional_needed <= 0:
        return current_masks

    candidates = []

    for module_name, record in score_records.items():
        mask = current_masks[module_name]
        score = record.fisher

        for out_block, in_block in active_block_indices(mask):
            candidates.append(
                (
                    float(score[out_block, in_block]),
                    module_name,
                    out_block,
                    in_block,
                )
            )

    candidates.sort(key=lambda x: x[0])

    new_masks = clone_masks(current_masks)
    pruned_per_matrix = get_pruned_count_per_matrix(new_masks)
    max_prunable_per_matrix = {}

    for module_name, mask in new_masks.items():
        max_by_ratio = int(
            mask.numel() * max_prune_ratio_per_matrix
        )
        max_by_keep_floor = (
            mask.numel() - min_keep_blocks_per_matrix
        )
        max_prunable_per_matrix[module_name] = min(
            max_by_ratio,
            max_by_keep_floor,
        )

    selected = 0

    for score, module_name, out_block, in_block in candidates:
        if selected >= additional_needed:
            break

        if (
            pruned_per_matrix[module_name]
            >= max_prunable_per_matrix[module_name]
        ):
            continue

        new_masks[module_name][out_block, in_block] = False
        pruned_per_matrix[module_name] += 1
        selected += 1

    if selected != additional_needed:
        raise RuntimeError(
            "Cannot reach target sparsity under current matrix constraints."
        )

    return new_masks
```

---

## 17. 输出文件格式

### 17.1 `block_scores.pt`

```python
{
    module_name: {
        "layer_index": int,
        "projection_type": str,
        "weight_shape": tuple,
        "block_size": 128,
        "fisher": Tensor[num_out_blocks, num_in_blocks],
        "abs_taylor": Tensor[num_out_blocks, num_in_blocks],
        "signed_mean": Tensor[num_out_blocks, num_in_blocks],
    }
}
```

### 17.2 `block_masks.pt`

```python
{
    module_name: BoolTensor[num_out_blocks, num_in_blocks]
}
```

其中：

```text
True  = keep
False = prune
```

### 17.3 `pruning_summary.json`

至少包含：

```json
{
  "block_size": 128,
  "target_block_sparsity": 0.3,
  "actual_block_sparsity": 0.3,
  "num_total_blocks": 0,
  "num_pruned_blocks": 0,
  "num_pruning_rounds": 1,
  "score_type": "fisher",
  "selection_mode": "global_constrained",
  "share_up_gate_mask": false,
  "calibration_samples": 128,
  "sequence_length": 2048,
  "seed": 42
}
```

### 17.4 `per_matrix_report.csv`

列：

```text
module_name
layer_index
projection_type
weight_shape
num_blocks
num_pruned_blocks
block_sparsity
score_min
score_median
score_mean
score_max
```

---

## 18. 必须实现的测试

### 18.1 Block reduce 正确性

对随机矩阵手工循环求：

$$ \sum_{(i,j)\in b}W_{ij}G_{ij} $$

与 reshape-vectorized 结果逐块比较。

要求：

```text
max_abs_error < 1e-5
```

### 18.2 显式 mask 梯度等价性

构造一个小 Linear 和一个标量 block mask $m_b$：

```python
masked_weight[block] = m_b * weight[block]
```

反向传播后验证：

$$ \frac{\partial L}{\partial m_b} \approx \sum_{(i,j)\in b} W_{ij} \frac{\partial L}{\partial W_{ij}}. $$

这是整个方法最关键的单元测试。

### 18.3 Fisher 必须先平方后累计

构造两个 batch，使同一 block 的 signal 分别为 $+a$ 和 $-a$。

应得到：

$$ S_b^{F}=a^2, $$

而不是 0。

### 18.4 剪枝数量准确

给定：

```text
100 blocks
target sparsity = 0.30
```

必须准确剪除 30 个 block，除非约束明确导致不可达并报错。

### 18.5 只修改 MLP

剪枝前后比较参数：

- Attention 权重必须完全一致；
- embedding、norm、lm_head 必须完全一致；
- 只有目标 MLP Linear 的被剪块发生变化。

### 18.6 零块严格性

对每个 `mask=False` 的 block：

```python
assert torch.count_nonzero(weight_block) == 0
```

对每个 `mask=True` 的 block，除非原始权重本身包含零，否则不应被误置零。

### 18.7 可复现性

相同：

- seed；
- calibration samples；
- batch order；
- model checkpoint；

应生成完全一致的 `block_masks.pt`。

---

## 19. 基线与必要实验

为了验证 loss-gradient block score 是否有效，至少比较：

### Baseline A：Random Block

随机选择相同数量的 128×128 block。

### Baseline B：Block Magnitude

$$ S_b^{\mathrm{Mag}} = \|W_b\|_F^2. $$

剪除 score 最小的块。

### Baseline C：Block Abs-Taylor

$$ S_b^{\mathrm{AbsTaylor}} = \mathbb E_t \left| \left\langle\nabla_{W_b}\mathcal L_t,W_b\right\rangle \right|. $$

### Main：Block Fisher

$$ S_b^{\mathrm{Fisher}} = \mathbb E_t \left[ \left( \left\langle\nabla_{W_b}\mathcal L_t,W_b\right\rangle \right)^2 \right]. $$

建议稀疏率：

```text
10%
20%
30%
40%
50%
```

至少汇报：

- WikiText-2 perplexity；
- 下游任务平均准确率；
- 每层 MLP block sparsity；
- up/gate/down 各自 block sparsity；
- score 分布；
- 校准数据规模消融。

---

## 20. 关键实现注意事项

### 20.1 不使用 `torch.no_grad()`

score 阶段必须构建完整反向图。

### 20.2 关闭 KV cache

```python
model.config.use_cache = False
```

否则训练式反向可能占用不必要显存或出现不兼容。

### 20.3 不使用 GradScaler

建议直接使用 BF16 autocast，不使用动态 loss scaling。

若使用 FP16 + GradScaler，必须在读取 gradient 前执行 unscale，否则 score 会被 scale 因子平方放大。

### 20.4 梯度累积顺序

正确：

```text
forward
backward
calculate block signal
square and accumulate
zero_grad
```

错误：

```text
多个 batch backward
gradient 先累加
最后计算一次 block signal
```

### 20.5 权重更新时机

score 统计期间禁止修改权重。

只有完成当前 round 的全部校准数据统计后，才能分配 mask 和置零权重。

### 20.6 OOM 策略

按优先级：

1. `score_batch_size=1`；
2. 缩短 sequence length；
3. BF16；
4. gradient checkpointing；
5. 分布式模型切分。

首版不建议按层逐个 backward，因为目标是完整 LM loss 的全模型梯度传播。

### 20.7 分布式统计

若使用 data parallel，每个 rank 独立累计：

```text
score_sq
score_abs
score_signed
num_batches
```

最后执行 all-reduce sum，再除以全局 batch 数。

首版可以只支持单进程，分布式作为后续扩展。

---

## 21. 验收标准

实现完成后必须满足：

1. 能从 Hugging Face LLaMA/Qwen checkpoint 自动识别全部 MLP Linear；
2. Attention 和其他模块不参与剪枝；
3. 每个目标权重严格按 128×128 分块；
4. block score 等价于显式 block mask 的 loss 梯度；
5. Fisher score 按 batch 先平方后平均；
6. 能生成精确目标数量的 block mask；
7. 每个被剪块严格全零；
8. 能保存和重新加载 mask；
9. 能输出逐层、逐 projection 的稀疏率报告；
10. 相同配置和随机种子可复现；
11. 能在同一评估脚本中比较 dense、random、magnitude 和 gradient-Fisher pruning。

---

## 22. 首版推荐配置

```yaml
model_path: Qwen/Qwen3-8B
block_size: 128

target_modules:
  - up_proj
  - gate_proj
  - down_proj

calibration_dataset: wikitext
calibration_samples: 128
sequence_length: 2048
score_batch_size: 1

loss_type: causal_lm
score_type: fisher

selection_mode: global_constrained
target_block_sparsity: 0.30
max_prune_ratio_per_matrix: 0.60
min_keep_blocks_per_matrix: 1

share_up_gate_mask: false
pruning_rounds: 1

score_accumulation_dtype: float64
seed: 42
```

首个验证完成后，再增加：

```yaml
pruning_rounds: 4
share_up_gate_mask: true
```

作为渐进剪枝和硬件对齐 mask 的扩展实验。

---

## 23. 后续扩展边界

以下功能不要混入首版实现：

1. SparseGPT/OBS 补偿；
2. 剪枝后 LoRA 或全参数恢复；
3. teacher KL loss；
4. MLP block 输出重建 loss；
5. Attention block pruning；
6. up/gate/down 联合中间组约束；
7. HiF4 fake-quant 下重新评分；
8. block-sparse CUDA kernel。

首版必须先回答一个单独问题：

> 在完全相同的 128×128 MLP 块稀疏率下，基于最终 LM loss 回传的 block empirical Fisher，是否比随机和块幅值指标更能保持模型性能？

只有这个结论成立后，再加入二阶补偿、量化联合和 kernel 约束。
