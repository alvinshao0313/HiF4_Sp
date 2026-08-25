# Qwen3.5-4B HiF4 W4A4 RTN 投影消融

比较在 HiF4 W4A4 RTN 下，跳过部分投影（权重 + 激活都不量化）对精度的影响。

## 变体

| 变体 | 不量化（W+A） |
|------|----------------|
| `full` | 仅 `lm_head` |
| `skip_gate_up` | `gate_proj` + `up_proj`（对应 vLLM `gate_up_proj`） |
| `skip_down` | `down_proj` |
| `skip_o_proj` | `o_proj` |
| `skip_mlp` | `gate_proj` + `up_proj` + `down_proj` |

## 评测协议（对齐 `Block_Sparse/experiments/wikitext2_calib/report.html`）

| 任务 | 后端 | 设置 |
|------|------|------|
| ARC-E / ARC-C / MMLU | **lm_eval** 0-shot | 指标优先 `acc`；权重用 RTN ckpt，激活用 HiF4 QLinear2 |
| MMLU-Pro | **lighteval** / `main.py` | `mmlu_pro\|0`，`max_samples=300`，`disable_thinking`，`max_new_tokens=32768`，temp=0.7 / top_p=0.8 / top_k=20 |

量化 ckpt 仅临时使用，评完删除；保留 `results/` 与脚本。

## 运行

```bash
cd /home/shaoyuantian/program/HiF4_Sp
GPUS=7 bash HiF4_exp/qwen35_4b_w4a4_proj_ablation/run_ablation.sh
```

常用环境变量：`GPUS`、`TP`、`VARIANTS`、`MMLU_PRO_MAX_SAMPLES`（默认 300）。

汇总：

```bash
/home/shaoyuantian/anaconda3/envs/hif4/bin/python \
  HiF4_exp/qwen35_4b_w4a4_proj_ablation/summarize_results.py
```
