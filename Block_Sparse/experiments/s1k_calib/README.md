# s1K-1.1 校准实验归档

本目录存放 **s1K-1.1 校准** 的评测日志与指标（ckpt 仍在 `Block_Sparse/outputs/`，约 51G，不重复拷贝）。

汇报对照页见 [`../wikitext2_calib/report.html`](../wikitext2_calib/report.html) 第 7 节。

## 内容

| 路径 | 说明 |
|------|------|
| `results/ppl/` | WikiText-2 PPL（seq=2048）json + log |
| `results/lm_eval/` | lm_eval 0-shot ARC-E / ARC-C / MMLU json + log |
| `results/pruning_artifacts/` | `pruning_summary.json`、报告 CSV / permutation 产物 |
| `metrics_summary.json` | 机器可读指标汇总（ARC 用 `acc`） |
| `mmlu_pro_300/{results,details}/` | lighteval MMLU-Pro 300（`extractive_match`；含 details，与 results 已去重合并） |

## 主结果（block=64 · sparsity=20%，ARC / MMLU 均为 `acc`）

| 方案 | perm | max_prune | PPL ↓ | ARC-E ↑ | ARC-C ↑ | MMLU ↑ |
|------|------|----------:|------:|--------:|--------:|-------:|
| fisher（wiki 对照） | none | 0.30 | **13.37** | 76.98 | 45.82 | 75.96 |
| fisher | none | 0.30 | 16.63 | **75.13** | 45.99 | 76.04 |
| fisher_budget_wanda | none | 0.30 | 19.79 | 74.41 | 46.59 | 76.14 |
| fisher_budget_wanda | wanda_shared | **0.50** | 38.52 | 71.46 | 44.88 | **81.96** |

注意：最后一行同时改了 `mlp_permutation` 与 `max_prune`，不是单因素对照。稠密基线：ARC-E 84.85 / ARC-C 59.90 / MMLU 84.43。

ckpt（若仍在）：

- `Block_Sparse/outputs/qwen35_27b_fisher_s0.20_b64_s1k`
- `Block_Sparse/outputs/qwen35_27b_fisher_budget_wanda_s0.20_b64_s1k_permnone`
- `Block_Sparse/outputs/qwen35_27b_fisher_budget_wanda_s0.20_b64_s1k_permwanda_shared`
