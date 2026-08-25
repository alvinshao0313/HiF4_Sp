# WikiText-2 校准实验归档

本目录存放 **Fisher/Magnitude/Random 使用 WikiText-2 做校准（或脚本默认标定）** 的剪枝 ckpt 与评测结果。

后续 s1K-1.1 校准评测已归档到 [`../s1k_calib/`](../s1k_calib/)（ckpt 仍在 `Block_Sparse/outputs/`）。

## 内容

| 路径 | 说明 |
|------|------|
| `outputs/` | 剪枝后 HF ckpt（约 51G/个） |
| `results/` | MMLU / 其它 lighteval 结果 |
| `results/ppl/` | WikiText-2 PPL（seq=2048） |
| `results/lm_eval/` | lm_eval 0-shot ARC-E / ARC-C / MMLU |
| `dense_baseline.json` | 未剪枝全精度 lm_eval 基线 |
| `metrics_summary.json` | 机器可读指标汇总（ARC 用 `acc`） |
| `report.html` | 汇报用结果页（浏览器打开） |

## 稠密基线（对照用 `acc`）

| 任务 | 指标 | 分数 |
|------|------|-----:|
| MMLU | acc | **84.43%** |
| ARC Easy | acc | **84.85%** |
| ARC Challenge | acc | **59.90%** |

## 主结果（ARC / MMLU 均为 `acc`）

WikiText-2 PPL（seq=2048）+ lm_eval 0-shot：

| 方案 | Block | max_prune | PPL ↓ | ARC-E ↑ | ARC-C ↑ | MMLU ↑ |
|------|-------|----------:|------:|--------:|--------:|-------:|
| magnitude | 64 | 0.30 | **9.80** | **81.94** | **54.52** | 68.74 |
| magnitude | 128 | 0.60 | 10.55 | 79.92 | 50.26 | 69.06 |
| fisher | 64 | 0.30 | 13.37 | 76.98 | 45.82 | **75.96** |
| random | 64 | 0.30 | 13.87 | 78.49 | 46.84 | 73.37 |
| random | 128 | 0.60 | 15.52 | 78.87 | 51.37 | 73.70 |
| fisher | 64×32 | 0.60 | 15.95 | 72.77 | 41.55 | 70.40 |
| fisher | 128 | 0.60 | 17.30 | 74.03 | 42.75 | 68.07 |

PPL/ARC 最优：`magnitude` · `block=64`。MMLU 最优：`fisher` · `block=64`。详见 [`report.html`](report.html)。
