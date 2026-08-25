# Qwen3.5-4B · Residual π₀ 通道聚合

试验记录目录（唯一副本）。

## 设定摘要

- 剪枝：`fisher_budget_wanda`，sparsity 0.20，block `64×32`，calib s1k，`wanda_shared`
- Residual：`block_loss`，主表 `search_steps=0`；π₀ 聚合消融见报告
- 对照：同配方 `rpermnone`；稠密基线见 [`../qwen35_4b_dense/`](../qwen35_4b_dense/)

## 目录

| 路径 | 说明 |
|------|------|
| `results/lm_eval_0shot/` | ARC-E / ARC-C / MMLU（lm_eval 0-shot `acc`） |
| `results/lighteval_mmlu_pro_300/` | MMLU-Pro 300（lighteval `extractive_match` + details） |
| `results/ppl/` | 仅 residual 置换、未剪枝时的 WikiText-2 PPL |
| `results/run_logs/` | 编排 / 中止任务日志 |
| `reports/paper_report_residual_perm_channel_agg.md` | 论文式汇报 |
| `reports/metrics_tables.json` | 机器可读指标表 |
| `reports/notes_residual_channel_agg_draft.md` | 早期草稿 |

## Checkpoint

`Block_Sparse/outputs/qwen35_4b_fisher_budget_wanda_s0.20_b64x32_s1k_permwanda_shared_*`
