# Qwen3.5-4B 稠密基线

未剪枝全精度对照。

## 指标（摘要）

见 `dense_baseline.json`：

| 指标 | 值 |
|------|---:|
| WikiText-2 PPL (2048) | 9.58 |
| ARC-E / ARC-C / MMLU (`acc`) | 81.40% / 51.54% / 74.37% |
| MMLU-Pro 300 (`extractive_match`) | 71.00% ± 2.62% |

## 目录

| 路径 | 说明 |
|------|------|
| `dense_baseline.json` | 汇总卡 |
| `results/ppl/` | PPL json + log |
| `results/lm_eval/` | ARC/MMLU json + log |
| `results/mmlu_pro/` | lighteval MMLU-Pro 300（results + details） |
