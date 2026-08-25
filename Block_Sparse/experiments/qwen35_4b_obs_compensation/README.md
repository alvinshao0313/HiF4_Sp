# Qwen3.5-4B OBS 补偿 vs Direct-Zero 下游对比

同一 Stage A mask（`fisher_budget_wanda` s0.20 b64x32 s1k `wanda_shared` `rpermnone`），对比做 / 不做 OBS 补偿。

| 臂 | 说明 | 路径 |
|---|---|---|
| Dense | 未剪枝对照 | `Qwen/Qwen3.5-4B`（数字来自 `../qwen35_4b_dense`） |
| Direct-Zero | Stage A 直接置零 | `Block_Sparse/outputs/qwen35_4b_fisher_budget_wanda_s0.20_b64x32_s1k_permwanda_shared_rpermnone` |
| OBS | 固定 mask SparseGPT/OBS 补偿 | `Block_Sparse/outputs/qwen35_4b_obs_s0.20_b64x32_s1k_permwanda_shared_rpermnone` |

## 评测协议（对齐 dense 基线）

| 协议 | 工具 | 参数 |
|---|---|---|
| WikiText-2 PPL | `eval_ppl.py` | seq=2048, bf16 |
| ARC-E / ARC-C / MMLU | `eval_lm_eval.py` | 0-shot, batch=16, 报 `acc` |
| MMLU-Pro-300 | `main.py` + lighteval | `mmlu_pro\|0`, max_samples=300, TP=1, `DISABLE_THINKING=1` |

## 结果总表

| 指标 | Dense | Direct-Zero | OBS | OBS−DZ |
|---|---:|---:|---:|---:|
| WikiText-2 PPL (2048) | 9.58 | 33.13 | 36.58 | **+3.45** |
| ARC-Easy acc (%) | 81.40 | 63.51 | 63.55 | +0.04 |
| ARC-Challenge acc (%) | 51.54 | 35.84 | 35.84 | 0.00 |
| MMLU acc (%) | 74.37 | 71.17 | 72.06 | **+0.89** |
| MMLU-Pro-300 extractive_match (%) | 71.00 ±2.62 | 18.00 ±2.22 | 18.67 ±2.25 | +0.67 |

## 结论

- **PPL 变差**：OBS 比直接置零高约 3.5，说明当前固定 mask / `permutation_aware` OBS 并未改善语言建模损失。
- **ARC 基本持平**：ARC-E/C 差异在噪声内。
- **MMLU 小幅提升**：约 +0.9pp。
- **MMLU-Pro 几乎无增益**：+0.67pp，小于 stderr（约 ±2.2），不能当作可靠收益。

机器可读汇总见 `comparison.json`。

## 目录

| 路径 | 说明 |
|------|------|
| `comparison.json` | 三臂数字 + OBS−DZ 差分 |
| `results/ppl/` | PPL json + log |
| `results/lm_eval_0shot/` | ARC/MMLU json |
| `results/lighteval_mmlu_pro_300/` | lighteval MMLU-Pro 300 |
| `run_logs/` | lm_eval / MMLU-Pro 运行日志 |
