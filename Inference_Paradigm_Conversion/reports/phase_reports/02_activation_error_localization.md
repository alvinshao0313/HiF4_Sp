# 02 激活量化误差定位（增量 AX1–AX5）

## 1. 研究问题

在主计划已确认 HiF4 激活转换误差较大的前提下，误差主要来自 S0 位置、64-group 粒度、层级指数、payload 网格，还是 NVFP4/HiF4 尺度系统差异？各机制在真实 Linear 输出上最多可恢复多少？

## 2. 已有三格式基线

复用主计划 A1 / repr-al 结果，不重跑。

## 3. 已有 HiF4 内部步骤消融

复用 A2：`continuous_s0` / `oracle_e8_e4_joint` / `continuous_payload_clipped` 等。

## 4. AX1：S0 位置是否合理

平均 Oracle 输出恢复率 R_Y=0.0197。详见 `experiment_logs/AX1_s0_divisor_oracle.md`。

## 5. AX2：64-group 是否过大

见 `ax2_group_size_ablation.csv` 与 AX2 日志。

## 6. AX3：网格与真实占用

见 `ax3_grid_occupancy.csv`、`ax3_theoretical_grid.json`。

## 7. AX4：Scale 与 Payload 谁主导

见 `ax4_cross_format_factorization.csv`。

## 8. NVFP4 Source 自身误差

引用主计划 A2 NVFP4 内部消融，不重跑。

## 9. 所有机制 Linear 输出可恢复误差排名

- #1: **NVFP4 payload 连续化** (来源 A2, R=0.9442915833304937, n=672, note=A2_R_cf_vs_X)
- #2: **Payload/Clipping** (来源 A2, R=0.8413027320916704, n=672, note=A2_R_cf_vs_X)
- #3: **S0 表示精度** (来源 A2, R=0.13390777117496613, n=672, note=A2_R_cf_vs_X)
- #4: **S0 E6M2 表示** (来源 A2, R=0.13299094238507458, n=672, note=A2_R_cf_vs_X)
- #5: **64-group 共享粒度(G16)** (来源 AX2, R=0.12679782557528582, n=672, note=)
- #6: **HiF4 Scale + NVFP4 Payload (HN/range_matched)** (来源 AX4, R=0.09080701546248303, n=672, note=)
- #7: **64-group 共享粒度(G32)** (来源 AX2, R=0.07895281770526177, n=672, note=)
- #8: **NVFP4 local-scale 连续化** (来源 A2, R=0.05488908087656939, n=672, note=A2_R_cf_vs_X)
- #9: **NVFP4 global-scale Oracle** (来源 A2, R=0.02238317098301605, n=615, note=A2_R_cf_vs_X)
- #10: **S0 位置** (来源 AX1, R=0.01966297673000657, n=672, note=)


## 10. Prefill/Decode、Layer、Projection 差异

本 run phases 以结果 CSV 中 `phase/projection/layer_idx` 字段为准；若仅含 prefill，需补跑 decode 后再更新。

## 11. 低开销规则可行性

状态：`skipped_due_to_low_s0_recovery`；candidate_for_e2e=`False`。

## 12. 验证集结论

若本 run 仅为 discovery，请在 validation run 上复验前三根因是否反转。

## 13. 激活量化前三根因

### 根因 #1：NVFP4 payload 连续化

- 机制：NVFP4 payload 连续化
- 观察/反事实证据来源：A2
- Linear 输出可恢复误差（聚合）：0.9442915833304937
- 备注：A2_R_cf_vs_X

### 根因 #2：Payload/Clipping

- 机制：Payload/Clipping
- 观察/反事实证据来源：A2
- Linear 输出可恢复误差（聚合）：0.8413027320916704
- 备注：A2_R_cf_vs_X

### 根因 #3：S0 表示精度

- 机制：S0 表示精度
- 观察/反事实证据来源：A2
- Linear 输出可恢复误差（聚合）：0.13390777117496613
- 备注：A2_R_cf_vs_X


## 图表

共生成 23 张图，目录：`Inference_Paradigm_Conversion/results/20260811T_ax_final_consolidated/figures`。

## 实验日志

- `/home/shaoyuantian/program/HiF4_Sp/Inference_Paradigm_Conversion/reports/experiment_logs/AX1_s0_divisor_oracle.md`
- `/home/shaoyuantian/program/HiF4_Sp/Inference_Paradigm_Conversion/reports/experiment_logs/AX2_group_size_ablation.md`
- `/home/shaoyuantian/program/HiF4_Sp/Inference_Paradigm_Conversion/reports/experiment_logs/AX3_grid_occupancy.md`
- `/home/shaoyuantian/program/HiF4_Sp/Inference_Paradigm_Conversion/reports/experiment_logs/AX4_scale_payload_factorization.md`
- `/home/shaoyuantian/program/HiF4_Sp/Inference_Paradigm_Conversion/reports/experiment_logs/AX5_rule_validation.md`
