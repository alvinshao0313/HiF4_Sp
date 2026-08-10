# MLP Block-Masked OBS Compensation Initialization

独立二阶段流程：先用现有 `score_and_prune_mlp.py` 生成最终块 mask（和可选 MLP 排序），再用本目录在**原始稠密模型**上做 SparseGPT/OBS 风格补偿，导出可评测、可继续恢复训练的块稀疏初始化。

## 为什么必须重载稠密模型

现有剪枝入口在选完 mask 后直接 `weight.mul_(element_mask)` 置零，没有二阶补偿。OBS 必须在未置零的稠密权重上求解；**不要**把 Stage A 导出的直接置零 checkpoint 当作 OBS 输入。

## 源产物限制（第一版）

- `num_pruning_rounds == 1`
- `residual_permutation == none`
- `mlp_permutation` 仅支持 `none` 或 `wanda_shared`
- 需要文件：
  - `pruning_summary.json`
  - `block_masks.pt`（最终无后缀 mask）
  - `mlp_permutations.pt`（仅 `wanda_shared`；`none` 时必须不存在）

### 全剪 block row 的语义

固定 mask 允许某个 Linear 的一个或多个 block row 全部为 `False`。这些 block row 对应的输出行最终严格为零，OBS 不会跨输出行补偿它们；同一矩阵中其他仍有保留块的输出行继续正常补偿。整个 Linear 的 mask 全为 `False` 仍然是非法输入。

## Stage A：生成 mask / 排序

```bash
conda run -n hif4 --no-capture-output \
  python Block_Sparse/tools/score_and_prune_mlp.py \
  --model_path Qwen/Qwen3.5-4B \
  --output_dir Block_Sparse/outputs/qwen35_4b_mask_source \
  --score_type fisher_budget_wanda \
  --target_block_sparsity 0.20 \
  --block_size 64x32 \
  --calibration_dataset s1k \
  --calibration_samples 128 \
  --sequence_length 1024 \
  --mlp_permutation wanda_shared \
  --residual_permutation none \
  --pruning_rounds 1
```

`none` 与 `wanda_shared` 源产物必须分目录保存，不能互相覆盖。

## Stage B：OBS 补偿

```bash
conda run -n hif4 --no-capture-output \
  python Block_Sparse/obs_compensation/run_obs_pruning.py \
  --source_artifacts_dir Block_Sparse/outputs/qwen35_4b_mask_source/pruning_artifacts \
  --output_dir Block_Sparse/outputs/qwen35_4b_obs_init \
  --calibration_dataset s1k \
  --calibration_samples 128 \
  --sequence_length 1024 \
  --obs_percdamp 0.01 \
  --solver_block_size 128 \
  --obs_order_policy auto \
  --dtype bfloat16 \
  --device cuda \
  --seed 42
```

也可用 `run_obs_pruning.sh`（顶部变量可改）。`--model_path` 可选；传入时必须与 summary 中字符串完全一致。

支持校准集：`s1k`、`wikitext2`。

## 求解方向

OBS 只沿权重**输入列**传播误差，不跨输出行补偿。

- `gate/up` 的 `wanda_shared` 排序在行方向，因此始终左到右。
- `down` 排序在列方向：重要通道在左、不重要在右；`permutation_aware` 对 down 右到左处理。
- 列顺序只影响离线求解，不改模型已保存坐标，也不按 mask 剪枝数量动态重排。

| Source `mlp_permutation` | Requested | Resolved | gate/up | down |
|---|---|---|---|---|
| `none` | `auto` | `standard` | L→R | L→R |
| `none` | `standard` | `standard` | L→R | L→R |
| `none` | `permutation_aware` | invalid | error | error |
| `wanda_shared` | `auto` | `permutation_aware` | L→R | R→L |
| `wanda_shared` | `standard` | `standard` | L→R | L→R |
| `wanda_shared` | `permutation_aware` | `permutation_aware` | L→R | R→L |

## 内存

峰值来自 `down_proj` 的 float32 Hessian，形状约 `d_ff × d_ff`。逐层释放；不要同时缓存多层 Hessian。

校准捕获和逐层 OBS 全程运行在 `torch.no_grad()` 下。对于 mask 全为 `True` 的投影，不构建 Hessian、不做 Cholesky、不调用 solver；报告中记录 `solver_applied=false` 和 `skip_reason=mask_all_kept`。若整层 gate/up/down 都未剪枝，该层对每个样本只执行一次前向并直接传播到下一层。

## 输出

正式输出通过同父目录 staging 路径保存。例如目标为 `outputs/model_obs` 时，写入路径是 `outputs/.model_obs.incomplete`。模型与全部 OBS artifacts 成功后才 rename 为正式目录。保存失败时正式目录不会出现，staging 会保留用于检查；再次运行前需要人工检查并移除该 staging 目录。

```text
output_dir/
├── (HF model + tokenizer)
└── obs_artifacts/
    ├── source_pruning_summary.json
    ├── block_masks.pt
    ├── mlp_permutations.pt          # only wanda_shared
    ├── obs_config.json
    ├── obs_summary.json
    ├── per_module_obs.csv
    └── per_layer_reconstruction.csv
```

报告字段补充：

```text
solver_applied
skip_reason
num_fully_pruned_block_rows
num_fully_pruned_output_rows
num_solver_applied_modules
num_solver_skipped_modules
total_fully_pruned_block_rows
total_fully_pruned_output_rows
```

## 下游评测

本包不调用评测。完成后复用：

- `Block_Sparse/scripts/eval_ppl.sh`
- `Block_Sparse/scripts/eval_mmlu_no_thinking.sh`

普通下游用 lm_eval；reasoning（含 MMLU-Pro）用仓库根 `main.py` + lighteval。

## 五模型消融矩阵

同一稀疏度、块大小、校准与评测设置。同排序分支内，direct-zero 与 OBS 必须共用同一份最终 mask。

| ID | MLP sort | Initialization | `obs_order_policy` | Resolved down | Purpose |
|---|---|---|---|---|---|
| A | none | direct zero | N/A | N/A | unsorted zero baseline |
| B | none | OBS | `auto` | L→R | standard SparseGPT |
| C | wanda_shared | direct zero | N/A | N/A | sorted zero baseline |
| D | wanda_shared | OBS | `standard` | L→R | sorted + non-aware OBS |
| E | wanda_shared | OBS | `auto` | R→L | sorting-aware method |

对比：`B-A` 普通 OBS；`C-A` 排序增益；`D-C` 排序后普通 OBS；`E-D` 仅方向增益；`E-C` 排序下完整 OBS；`E-B` 完整方法 vs 无排序 OBS。不要做 unsorted + `permutation_aware`。

每份报告建议包含：WikiText-2 PPL、现有普通下游、恢复前后准确率、`per_layer_reconstruction.csv`、requested/resolved 方向、down 左右半区剪枝块数、保留权重最大幅值变化。
