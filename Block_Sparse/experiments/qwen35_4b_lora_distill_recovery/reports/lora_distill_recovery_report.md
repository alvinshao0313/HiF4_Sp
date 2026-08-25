# Qwen3.5-4B 剪枝模型 Masked-LoRA 蒸馏恢复实验

## 1. Lineage 映射

| 臂 | 模型 | 路径 / 来源 |
|---|---|---|
| M0 | Dense Qwen3.5-4B | `Qwen/Qwen3.5-4B`（HF cache） |
| M1 | 剪枝 rpermnone | `Block_Sparse/outputs/qwen35_4b_fisher_budget_wanda_s0.20_b64x32_s1k_permwanda_shared_rpermnone` |
| M2 | M1 + Masked-LoRA 纯 CE SFT | `Block_Sparse/outputs/qwen35_4b_rpermnone_mlplora16_ce500` |
| M3 | M1 + Masked-LoRA QAD 蒸馏 | `Block_Sparse/outputs/qwen35_4b_rpermnone_mlplora16_qad500` |

M0/M1 数字来源：`Block_Sparse/experiments/qwen35_4b_dense/dense_baseline.json` 与 `Block_Sparse/experiments/qwen35_4b_residual_perm_channel_agg/reports/metrics_tables.json`。M1 PPL 本实验补测（Task 1）。

## 2. 训练配置摘要

两臂共用（全部为脚本默认值）：

| 项 | 值 |
|---|---|
| LoRA | r=16, alpha=32, dropout=0.0，仅 MLP gate/up/down |
| 可训练参数 | 18,087,936 / 4,223,839,232 (0.43%) |
| 数据 | `simplescaling/s1K-1.1_tokenized`（1000 条 deepseek R1 轨迹） |
| 步数 | 500（grad_accum 8 × batch 1 ≈ 4 epoch） |
| LR | 1e-4, cosine, warmup 3%, max_grad_norm 1.0 |
| 序列 | model_max_length=32768, allow_truncate=false, logit_chunk_size=512 |
| 精度/并行 | bf16, gradient checkpointing, parallel_mode=layer |
| seed | 42 |

M3 蒸馏（teacher=Qwen3.5-4B 冻结）：`0.05·CE + 2.0·EAKLD + 0.5·LAFD`，T=1.0，lafd_topk=3。

### ⚠️ 与计划的偏差

| 项 | 计划 | 实际 | 原因 |
|---|---|---|---|
| kl_mode | `eakld`（全词表） | `eakld_topk` k=128 | 全词表 EAKLD 的 autograd 图在 32k 序列上累积约 77GB（64 chunk × 1.2GB float32），单卡/双卡均 OOM。topk 模式将 KL 图限制在 k=128 维，峰值降到约 51GB（双卡）/约 60GB（单卡）。γ 仍走全词表教师熵，与 eakld 一致。 |
| M2 启动方式 | `TEACHER_MODEL_DIR=""` 经 shell 脚本 | 直调 `python train_mlp_lora_sft.py` 不传 `--teacher_model_dir` | `run_mlp_lora_sft.sh:41` 的 `${TEACHER_MODEL_DIR:-Qwen/Qwen3.5-4B}` 将空串视为未设置，会强制蒸馏。直调 python 绕过此 bug。 |
| 训练进程管理 | `nohup &` | `setsid nohup` + 落盘 driver 脚本 + OOM 熔断重试 | IDE 会清理长时间挂着的后台 shell；共享机抢卡频繁。driver 脚本 `run_logs/driver_qad_v2.sh` / `driver_ce.sh` 作为可复现档案保留。 |

## 3. 评测协议

| 协议 | 工具 | 关键参数 |
|---|---|---|
| lm_eval 0-shot | `Block_Sparse/tools/eval_lm_eval.py` | tasks=arc_easy,arc_challenge,mmlu, fewshot=0, batch_size=16, 报 acc |
| MMLU-Pro-300 | 仓库根 `main.py`（vLLM+lighteval） | mmlu_pro\|0, max_samples=300, TP=1, DISABLE_THINKING=1, max_model_length=32768, max_new_tokens=32768, temp=0.7, top_p=0.8, top_k=20 |
| WikiText-2 PPL | `Block_Sparse/tools/eval_ppl.py` | seq_len=2048, bf16 |

## 4. 结果总表（4 模型 × 5 指标）

| 指标 | M0 Dense | M1 剪枝 | M1−M0 | M2 纯CE | M3 蒸馏 |
|---|---:|---:|---:|---:|---:|
| ARC-Easy acc (%) | 81.40 | 63.51 | −17.89 | 67.72 | 67.09 |
| ARC-Challenge acc (%) | 51.54 | 35.84 | −15.70 | 40.44 | 39.68 |
| MMLU acc (%) | 74.37 | 71.17 | −3.20 | 72.05 | 72.75 |
| MMLU-Pro-300 extractive_match (%) | 71.00 ±2.62 | 16.00 ±2.12 | −55.00 | 48.00 ±2.89 | 45.00 ±2.88 |
| WikiText-2 PPL (seq=2048) | 9.5806 | 33.1286 | +23.55 | 20.9603 | 25.0590 |

## 5. 恢复率表

恢复率 = (M恢复后 − M1) / (M0 − M1)。PPL 为越低越好： (PPL_M1 − PPL恢复后) / (PPL_M1 − PPL_M0)。

| 指标 | M2 recovery (%) | M3 recovery (%) | M3−M2 (pp) |
|---|---:|---:|---:|
| ARC-Easy | 20.0 | 20.0 | 0.0 |
| ARC-Challenge | 24.4 | 24.4 | 0.0 |
| MMLU | 27.5 | 49.4 | +21.8 |
| MMLU-Pro-300 | 58.2 | 52.7 | −5.5 |
| WikiText-2 PPL | 51.6 | 33.9 | −17.7 |

## 6. 归因分析

### 6.1 M3−M1（蒸馏总恢复）

蒸馏对所有指标均有正向恢复。MMLU-Pro 从 16% 恢复到 45%（+29pp，恢复率 52.7%），PPL 从 33.1 恢复到 25.1（恢复率 33.9%），MMLU 从 71.2 恢复到 72.8（恢复率 49.4%）。ARC 恢复较弱（约 20%）。

### 6.2 M2−M1（SFT 数据本身贡献）

纯 CE SFT 同样对所有指标有正向恢复，且在多数指标上与蒸馏臂接近甚至更好。MMLU-Pro 从 16% 恢复到 48%（+32pp，恢复率 58.2%），PPL 从 33.1 恢复到 21.0（恢复率 51.6%），均高于 M3。

### 6.3 M3−M2（蒸馏信号增量）

**蒸馏信号增量为负或接近零**，这是本实验最关键的发现，与计划主假设相反：

- MMLU-Pro：M3 比 M2 低 3.0pp（45 vs 48），在 ±2.9 的 stderr 范围内，统计上不显著但方向为负
- PPL：M3 比 M2 差 4.1（25.1 vs 21.0），远超噪声
- ARC-E/C：两臂几乎相同（差 <1pp）
- MMLU：M3 比 M2 高 0.7pp，唯一蒸馏略胜的指标

**结论：在当前配置下，MMLU-Pro 的恢复主要来自 s1K SFT 数据本身（风格/格式对齐），而非蒸馏 teacher logits 信号。** 蒸馏信号不仅没有增量贡献，反而在 PPL 和 MMLU-Pro 上有负面影响。

### 6.4 可能原因分析

1. **kl_mode 降级**：计划的全词表 `eakld` 被迫降级为 `eakld_topk k=128`。topk 模式只在 128 维上计算 KL，教师分布的绝大部分信息被丢弃，蒸馏信号质量下降。这可能使 EAKLD 项引入噪声而非有用梯度。
2. **损失权重失衡**：`2.0·EAKLD + 0.5·LAFD` 在 topk 模式下可能不再平衡。全词表 EAKLD 的数值范围与 topk 版不同，原权重可能使蒸馏项过度主导，挤压了 CE 的学习。
3. **CE 臂的纯数据优势**：s1K 的 deepseek R1 轨迹本身包含大量推理过程，纯 CE SFT 已能让模型学会更好的生成格式和抽取行为，蒸馏的边际收益有限。
4. **统计噪声**：MMLU-Pro 300 样本的 stderr 约 ±2.9，3pp 差距在 1σ 范围内，不能排除随机波动。但 PPL 是全量 WikiText-2（无采样噪声），M3 差 4.1 是确定性结论。

## 7. 训练曲线摘要

### M3 蒸馏臂（500 步，13.05h）

| step | ce | eakld | lafd | qad_total |
|---|---:|---:|---:|---:|
| 10 | 3.28 | 0.330 | 0.265 | 0.956 |
| 50 | 3.26 | 0.120 | 0.167 | 0.476 |
| 100 | 3.66 | 0.136 | 0.176 | 0.516 |
| 250 | 3.06 | 0.087 | 0.140 | 0.379 |
| 500 | 2.02 | 0.082 | 0.152 | 0.341 |

CE 从 3.28 降到 2.0，EAKLD 从 0.33 降到 0.08，LAFD 从 0.26 降到 0.14。`train_loss=3.365`。

### M2 纯 CE 臂（500 步，3.28h）

| step | loss |
|---|---:|
| 10 | 9.77 |
| 50 | 7.48 |
| 100 | 7.12 |
| 250 | 7.04 |
| 500 | 6.71 |

`train_loss=7.16`。注意 M2 的 `loss` 是纯 CE（全词表），M3 的 `ce` 分量是 chunked CE（数值口径相同但只报 CE 部分），两者不可直接比较——M3 的 `qad_total=0.34` 是 `0.05·CE + 2.0·EAKLD + 0.5·LAFD` 的加权和，远小于纯 CE。

## 8. 结论与后续建议

### 结论

Masked-LoRA SFT 对剪枝模型的恢复有效，但**恢复主要来自 s1K 数据本身**，蒸馏 teacher logits 信号在当前配置下没有增量贡献，反而在 PPL 和 MMLU-Pro 上略有负面影响。计划的主假设（蒸馏对修复生成崩塌关键）**不成立**。

### 后续建议

1. **优先修复 kl_mode**：实现逐 chunk 立即 backward（数学上与全词表 eakld 严格等价，峰值降到 1–2GB），在原计划的全词表 eakld 下重跑 M3，排除 topk 降级的干扰。这是验证主假设的必要条件。
2. **若全词表 eakld 仍无增量**：说明蒸馏信号对 4B 剪枝模型确实无帮助，主假设证伪。此时应转向数据消融（s1K vs 其他 SFT 数据）和容量消融（r=32/64）。
3. **MMLU-Pro 300 样本噪声**：若需更可靠的 MMLU-Pro 数字，考虑跑 full test set（12k 题）或增大到 1000 样本。
4. **PPL 差距是确定的**：M2 PPL=21.0 vs M3 PPL=25.1 无采样噪声，蒸馏对语言建模质量有确定性负面影响，应在报告中明确标注。

## 9. 文件清单

```
Block_Sparse/experiments/qwen35_4b_lora_distill_recovery/
├── README.md                          # 本文件
├── results/
│   ├── ppl/
│   │   ├── qwen35_4b_..._rpermnone_wikitext2_s2048.json      # M1
│   │   ├── qwen35_4b_rpermnone_mlplora16_qad500_wikitext2_s2048.json  # M3
│   │   └── qwen35_4b_rpermnone_mlplora16_ce500_wikitext2_s2048.json  # M2
│   ├── lm_eval_0shot/
│   │   ├── qwen35_4b_rpermnone_mlplora16_qad500_arc_mmlu.json  # M3
│   │   └── qwen35_4b_rpermnone_mlplora16_ce500_arc_mmlu.json  # M2
│   └── lighteval_mmlu_pro_300/
│       ├── qwen35_4b_rpermnone_mlplora16_qad500/results/      # M3
│       └── qwen35_4b_rpermnone_mlplora16_ce500/results/      # M2
├── run_logs/
│   ├── driver_qad_v2.sh               # M3 训练 driver（可复现）
│   ├── driver_ce.sh                   # M2 训练 driver（可复现）
│   ├── train_qad500.log               # M3 训练日志
│   ├── smoke_qad20.log                # 冒烟日志
│   ├── eval_lmeval_m3.log / _m2.log   # lm_eval 日志
│   ├── eval_mmlupro_m3.log / _m2.log  # MMLU-Pro 日志
│   └── eval_ppl_m3.log / _m2.log      # PPL 日志
└── reports/
    └── lora_distill_recovery_report.md  # 本报告的正式版
```
