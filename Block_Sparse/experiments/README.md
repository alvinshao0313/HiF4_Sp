# Block_Sparse 实验结果目录

**唯一归档根目录：`Block_Sparse/experiments/`**  
`Block_Sparse/results/` 仅作新评测临时输出；稳定后迁入本目录对应实验下。

更新：2026-08-07

---

## 怎么查

1. 先看下表「做了什么」定位实验目录  
2. 摘要 / 指针看该目录 `README.md`  
3. 详细数字与分析看「详细结果」列指向的文件  

统一评测协议（4B 下游系列）：WikiText-2 PPL seq=2048；lm_eval 0-shot ARC/MMLU（`acc`）；lighteval MMLU-Pro-300（`extractive_match`，`disable_thinking`）。

---

## 4B 块稀疏主线

| 目录 | 做了什么 | 摘要入口 | 详细结果 |
|------|----------|----------|----------|
| [`qwen35_4b_dense/`](qwen35_4b_dense/) | 未剪枝全精度稠密基线（PPL / ARC / MMLU / MMLU-Pro） | [`README.md`](qwen35_4b_dense/README.md) | [`dense_baseline.json`](qwen35_4b_dense/dense_baseline.json)；原始跑分在 `results/{ppl,lm_eval,mmlu_pro}/` |
| [`qwen35_4b_residual_perm_channel_agg/`](qwen35_4b_residual_perm_channel_agg/) | Residual π₀ 通道聚合 vs `rpermnone`；同配方 `fisher_budget_wanda` s0.20 b64×32 s1k `wanda_shared` | [`README.md`](qwen35_4b_residual_perm_channel_agg/README.md) | 正式报告 [`reports/paper_report_residual_perm_channel_agg.md`](qwen35_4b_residual_perm_channel_agg/reports/paper_report_residual_perm_channel_agg.md)；机器表 [`reports/metrics_tables.json`](qwen35_4b_residual_perm_channel_agg/reports/metrics_tables.json)；跑分在 `results/` |
| [`qwen35_4b_obs_compensation/`](qwen35_4b_obs_compensation/) | 同一 Stage A mask 上 **OBS 补偿 vs Direct-Zero** 下游对比（PPL / ARC / MMLU / MMLU-Pro） | [`README.md`](qwen35_4b_obs_compensation/README.md) | 汇总 [`comparison.json`](qwen35_4b_obs_compensation/comparison.json)；跑分在 `results/{ppl,lm_eval_0shot,lighteval_mmlu_pro_300}/` |
| [`qwen35_4b_lora_distill_recovery/`](qwen35_4b_lora_distill_recovery/) | 剪枝模型 Masked-LoRA 恢复：M0 Dense / M1 剪枝 / M2 纯 CE / M3 QAD 蒸馏 | [`README.md`](qwen35_4b_lora_distill_recovery/README.md) | 正式报告 [`reports/paper_report_lora_distill_recovery.md`](qwen35_4b_lora_distill_recovery/reports/paper_report_lora_distill_recovery.md)；机器表 [`reports/metrics_tables.json`](qwen35_4b_lora_distill_recovery/reports/metrics_tables.json)；跑分在 `results/`，训练日志在 `run_logs/` |

### 4B 主线 lineage（便于对照）

```text
M0 Dense  ──►  Stage A 剪枝 (rpermnone) = M1 / Direct-Zero
                    │
                    ├─► OBS 补偿          → qwen35_4b_obs_compensation
                    ├─► Residual π₀ 穿刺  → qwen35_4b_residual_perm_channel_agg
                    ├─► Masked-LoRA CE    → M2 (lora_distill_recovery)
                    └─► Masked-LoRA QAD   → M3 (lora_distill_recovery)
```

ckpt 在 `Block_Sparse/outputs/`，不进本目录。

---

## 4B 独立系列（不与主线混并）

| 目录 | 做了什么 | 摘要入口 | 详细结果 |
|------|----------|----------|----------|
| [`qwen35_4b_w4a4_proj_ablation/`](qwen35_4b_w4a4_proj_ablation/) | HiF4 W4A4 RTN 下跳过部分投影（gate/up、down、o、整 MLP）的量化消融 | [`README.md`](qwen35_4b_w4a4_proj_ablation/README.md) | [`results/summary.json`](qwen35_4b_w4a4_proj_ablation/results/summary.json)；各变体跑分在 `results/` |
| [`qwen35_4b_input_mask_proxy_ablation/`](qwen35_4b_input_mask_proxy_ablation/) | 单 `up_proj` 上 8 种「输出块 mask → 输入 K 块 mask」代理反推对比（含 M7=`mean(S0)`、M8=无 MY 条件化；精度 / 条件 oracle / 原型延迟） | [`README.md`](qwen35_4b_input_mask_proxy_ablation/README.md) | 正式报告 [`reports/paper_report_input_mask_proxy_ablation.md`](qwen35_4b_input_mask_proxy_ablation/reports/paper_report_input_mask_proxy_ablation.md)；八方法结果 [`results/20260807T070223Z/`](qwen35_4b_input_mask_proxy_ablation/results/20260807T070223Z/)；七/六方法历史 [`results/20260807T025747Z/`](qwen35_4b_input_mask_proxy_ablation/results/20260807T025747Z/)、[`results/20260805T114616Z/`](qwen35_4b_input_mask_proxy_ablation/results/20260805T114616Z/) |

---

## 27B 校准剪枝系列

| 目录 | 做了什么 | 摘要入口 | 详细结果 |
|------|----------|----------|----------|
| [`wikitext2_calib/`](wikitext2_calib/) | WikiText-2 校准：magnitude / fisher / random 等块剪枝对照 | [`README.md`](wikitext2_calib/README.md) | 汇报页 [`report.html`](wikitext2_calib/report.html)；[`metrics_summary.json`](wikitext2_calib/metrics_summary.json)；[`dense_baseline.json`](wikitext2_calib/dense_baseline.json)；跑分在 `results/` |
| [`s1k_calib/`](s1k_calib/) | s1K-1.1 校准：fisher / fisher_budget_wanda ± permutation 等 | [`README.md`](s1k_calib/README.md) | [`metrics_summary.json`](s1k_calib/metrics_summary.json)；MMLU-Pro 见 `mmlu_pro_300/`；对照汇报仍看 [`wikitext2_calib/report.html`](wikitext2_calib/report.html) 第 7 节 |

---

## 杂项

| 目录 | 做了什么 | 详细结果 |
|------|----------|----------|
| [`_archive_orphan/`](_archive_orphan/) | 无法归属的 hash 结果堆 | `results_misc_hash_dumps/`；**不入正表** |

---

## 写新结果时

1. 评测可先写到 `Block_Sparse/results/<run>/`  
2. 稳定后迁入上表对应实验目录（或新建独立系列目录）  
3. 更新该实验 `README.md`，并在本文件补一行目录指针  
4. 有正式结论时：写 `reports/paper_report_*.md` + `reports/metrics_tables.json`（或 `comparison.json` / `metrics_summary.json`）
