# Qwen3.5-4B Residual π₀ 通道聚合实验总结

日期：2026-07-29

## 公共设定

| 项 | 值 |
|---|---|
| 模型 | Qwen3.5-4B |
| 剪枝 | `fisher_budget_wanda`，全局块稀疏率 0.20，块 `64×32`，`max_prune_ratio_per_matrix=0.80` |
| 校准 | s1k，128 samples |
| MLP 中间维置换 | `wanda_shared` |
| Residual 置换 | `block_loss`（除对照 `rpermnone`） |
| 搜索步数 | 穿刺主表为 `search_steps=0`（只用 π₀）；另有旧版 `equal` + 2000 步对照 |

π₀ 聚合公式类：

- `equal`：每矩阵 L1 后等权求和
- `layer_fisher` / `matrix_fisher`：L1 后按层/矩阵 Fisher 总量加权
- `raw_wanda`：原始 Wanda 直接求和（无 L1）
- `sparsity_raw_wanda`：原始 Wanda × ρ_m（ρ_m = K_m/N_m，Fisher 分配剪枝稀疏率）
- `density_raw_wanda`：原始 Wanda × (1−ρ_m)

## 1. lm_eval 0-shot（loglikelihood，`acc`）

协议：`arc_easy` / `arc_challenge` / `mmlu`，0-shot，与既有报告一致。

| 设定 | ARC-E ↑ | ARC-C ↑ | MMLU ↑ | 相对 rpermnone ΔMMLU |
|---|---:|---:|---:|---:|
| **rpermnone（无 residual perm）** | **63.51** | 35.84 | 71.17 | — |
| equal + search_steps=2000（旧） | 62.29 | 35.67 | 68.59 | −2.58 |
| equal s0 | 61.36 | 34.98 | 68.61 | −2.56 |
| layer_fisher s0 | 62.08 | 35.32 | 67.00 | −4.17 |
| matrix_fisher s0 | 62.29 | 34.47 | 64.59 | −6.58 |
| **raw_wanda s0** | 62.21 | 34.73 | **71.93** | **+0.76** |
| sparsity_raw_wanda s0 | 62.08 | 35.84 | 70.82 | −0.35 |
| density_raw_wanda s0 | 62.88 | **36.26** | 70.77 | −0.40 |

原始 JSON：`Block_Sparse/results/lm_eval_4b/*_arc_mmlu.json`

### 简要结论（lm_eval）

1. 全局 residual 置换整体难打过 `rpermnone`；多数聚合伤 MMLU。
2. 唯有 `raw_wanda`（无 L1）在 MMLU 上超过对照（71.93 vs 71.17），但 ARC 仍略低。
3. Fisher 加权（layer/matrix）明显更差；L1 等权也不好。
4. 用稀疏率 / 保留率加权（`sparsity_` / `density_`）介于 `equal` 与 `raw_wanda` 之间：ARC 可接近或略超对照，MMLU 仍略低于 `rpermnone`，且不如 `raw_wanda`。
5. `search_steps=2000` 相对 `steps=0` 的 `equal` 几乎不改善下游（MMLU 持平、ARC 略好一点但仍低于 none）。

## 2. lighteval MMLU-Pro（300 题）

协议对齐 `report.html` §12：

- 后端：vLLM + `main.py` / lighteval
- 任务：`mmlu_pro|0`（0-shot）
- `max_samples=300`
- `disable_thinking`
- `max_new_tokens=32768`，temperature=0.7 / top_p=0.8 / top_k=20
- 指标：`extractive_match`
- 4B 使用 TP=1（报告 27B 为 TP=2）

| 设定 | extractive_match | stderr | 相对 dense |
|---|---:|---:|---:|
| **dense（未剪枝 Qwen3.5-4B）** | **71.00%** | ±2.62% | — |
| rpermnone | 16.00% | ±2.12% | −55.00 |
| raw_wanda s0 | 18.67% | ±2.25% | −52.33 |
| density_raw_wanda s0 | 18.67% | ±2.25% | −52.33 |
| **sparsity_raw_wanda s0**（剪枝组内最高） | **20.33%** | ±2.33% | −50.67 |

Dense 来源：`Block_Sparse/experiments/qwen35_4b_dense/`（`dense_baseline.json`；结果 `results/mmlu_pro/.../results_2026-07-28T07-23-11.214996.json`）。协议同表：`mmlu_pro|0`，`max_samples=300`，`disable_thinking`，生成参数同上。

剪枝组原始结果目录：

- `Block_Sparse/results/<ckpt名>/results/results_*.json`
- 运行日志：`Block_Sparse/results/lm_eval_4b/lighteval_mmlu_pro_300/`

### 简要结论（MMLU-Pro 300）

1. 稠密基线 **71.00%**；所有 20% 块剪枝设定掉到约 16–20%，相对 dense 约 −50～−55 pt。
2. 剪枝组内：三条 residual-perm 设定均高于 `rpermnone`（16.00%）。
3. `sparsity_raw_wanda` 剪枝组最高（20.33%）；`raw_wanda` 与 `density_raw_wanda` 并列 18.67%。
4. 这是生成式抽取匹配，与 lm_eval 0-shot loglikelihood MMLU **不可直接横比**（同报告说明）。
5. 长生成 / 循环输出仍可能拉低分数；本表未做循环子集二次分析。

## 3. 对照解读

| 目标 | 更优选择（本轮 4B） |
|---|---|
| lm_eval MMLU | `raw_wanda`（唯一高于 none） |
| lm_eval ARC-C | `density_raw_wanda` |
| lighteval MMLU-Pro 300 | dense 71%；剪枝组内 `sparsity_raw_wanda` |
| 整体是否值得开 residual perm | 未形成一致收益；指标间排序冲突；相对 dense 的 MMLU-Pro 掉点远大于 residual 组内差异 |

## 4. 相关 ckpt 路径

前缀：`Block_Sparse/outputs/`

- `..._rpermnone`
- `..._rpermblock_loss`（equal + 2000，旧）
- `..._rpermblock_loss_s0_aggequal`
- `..._rpermblock_loss_s0_agglayer_fisher`
- `..._rpermblock_loss_s0_aggmatrix_fisher`
- `..._rpermblock_loss_s0_aggraw_wanda`
- `..._rpermblock_loss_s0_aggsparsity_raw_wanda`
- `..._rpermblock_loss_s0_aggdensity_raw_wanda`

完整前缀名：`qwen35_4b_fisher_budget_wanda_s0.20_b64x32_s1k_permwanda_shared_`
