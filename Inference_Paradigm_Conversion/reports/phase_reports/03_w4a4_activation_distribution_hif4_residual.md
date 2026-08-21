# 03 W4A4 激活分布与 NVFP4→HiF4 残差可视化

## 1. 实验目的

在 **NVFP4 W4A4 semantic inference** 下，刻画真正进入 Linear GEMM 的激活 `A_N` 分布，并量化同一 `X_in` 上 counterfactual `A_H=Q_HiF4(X_in)` 相对 `A_N` 的转换残差 `ΔA=A_H−A_N` 的结构，指导后续激活优化优先级。

## 2. W4A4 semantic inference 定义

本实验是 NVFP4 W4A4 **semantic inference**：weight 使用 NVFP4-QAT fake-dequant BF16 source value（不重新做 packed W4 fake quant），activation 在每个目标 Linear 前执行 NVFP4 A4 QDQ；`A_N` 回传进入 GEMM，`A_H` 仅旁路统计。不是 packed W4A4 kernel 性能实验。

主残差固定：`ΔA = A_H − A_N`（禁止改成相对 `X_in` 或绝对值差）。

## 3. 数据来源与采样方式

- run_dir: `Inference_Paradigm_Conversion/results/20260811T074651Z_activation_viz_validation`
- 分布形状图 = **stratified deterministic sample estimate**（`activation_viz_points.pt`）
- 每个 capture 的数值指标 = **full-tensor exact statistic**（`activation_capture_summary.csv` / `activation_group_residual.csv`）
- 抽样策略：每个代表层 capture 最多 `max_point_samples_per_capture=1024` 个元素级点；prefill 统计 token 上限 128（均匀子采样）；seed=`20260810`
- 实际绘图点数：`3,096,576`；capture 数：`3024`；group 行数：`248832`

## 4. 理论 NVFP4 / HiF4 完整内部网格

- NVFP4：**E4M3FN × signed E2M1**（排除最外层 FP32 per-tensor scale）。标题与正文一律写 **E4M3FN**，不是 E8M0。
- HiF4：S0 × 2^(e8+e4) × signed S1P2
- nv_unique=475, hf_unique=1341; source=full_internal_grid = unique representable values after removing outer FP32 per-tensor scale; NVFP4 uses E4M3FN×signed E2M1; HiF4 uses S0×2^(e8+e4)×signed S1P2.

## 5. 真实 NVFP4 W4A4 激活分布（sampled points）

| 指标 | 值 |
|---|---|
| mean(A_N) | 0.000791579 |
| std(A_N) | 1.49201 |
| RMS(A_N) | 1.49201 |
| zero rate | 0.12987 |
| q90/q99/q99.9 \|A_N\| | 1.15625 / 4.40625 / 15.0625 |
| projection RMS max/min | q_proj / down_proj |
| prefill vs decode mean RMS(A_N) | 0.922466 vs 0.910958 |

## 6. NVFP4→HiF4 元素级残差

| 指标 | 值 |
|---|---|
| mean bias(ΔA) | -0.000218734 |
| median(ΔA) | 0 |
| RMS(ΔA) sampled | 0.188025 |
| mean capture NMSE | 0.0149953 |
| q90/q99/q99.9 \|ΔA\| | 0.175781 / 0.796875 / 1.875 |
| sign flip (sampled, both nonzero) | 0 |
| NV≠0→HF=0 mean | 0.0561637 |
| NV=0→HF≠0 mean | 0.0151623 |
| HF−NV zero rate mean | 0.0410014 |
| sampled energy top0.1/1/5/10% | 0.2755 / 0.5985 / 0.8665 / 0.9481 |
| capture median energy top0.1/1/5/10% | 0.1166 / 0.3224 / 0.5817 / 0.7182 |

## 7. 零点迁移与 payload 迁移

见 R7/R8。payload 热图只描述 codebook 使用迁移，因 scale 系统不同，不能解释为最终数值一一映射。

## 8. layer / projection / phase 热点

最高 NMSE capture（top5）：

- NMSE=0.0210328 | decode L18 v_proj
- NMSE=0.0210328 | decode L18 q_proj
- NMSE=0.0210328 | decode L18 k_proj
- NMSE=0.0204286 | decode L34 o_proj
- NMSE=0.0202455 | decode L34 o_proj

## 9. token×group 空间结构

见 R11（discovery/prefill 各代表层 NMSE 最大 capture 的 token×64-group RMS(ΔA) 表面与 2D heatmap）。

## 10. group 动态范围/离散度与残差

- Spearman(log2(amax64_x_mean), log10(rms_delta)) = **0.973748**
- Spearman(mean_sub16_dispersion, log10(rms_delta)) = **0.165049**
- mean / q90 sub16 dispersion = 1.62577 / 2.92759

## 11. discovery vs validation 稳定性

stable=True；max_global_js=9.299752242085792e-05；projections_exceeding_0p05=[]

## 12. 对激活优化的直接启示

A. residual energy 高度集中于少量元素（sampled top1% =0.599）：优先 outlier-aware / selective scale / protected group。

## 13. 限制与下一步

- 元素级图依赖确定性分层抽样，不能代替 full-tensor 指标。
- D4 三联图比较的是「内部可表示点密度」与「真实概率质量」，不是直接覆盖率。
- 下一步按第 12 节方向做针对性消融，并在 validation 上复验热点是否反转。

---

# §0 核心问题逐条回答

### Q1. 真正进入 GEMM 的 `A_N` 是什么分布？是否重尾、强零点、正负不对称？

`A_N` 抽样点：mean=0.0007916，std=1.492，RMS=1.492，zero_rate=0.1299；\|A_N\| 的 q99/q99.9=4.406/15.06。见 D1–D3：若 q99.9 ≫ RMS 则重尾明显；zero_rate 反映零点集中；mean 相对 0 的偏离反映正负不对称。本报告以抽样密度估计形状，full-tensor 矩见 capture CSV。

### Q2. 各 projection / 早中晚层 / prefill·decode 是否不同？

projection 平均 RMS(A_N) 最大=q_proj、最小=down_proj；prefill vs decode mean RMS=0.9225 vs 0.911。见 D5（projection×phase log2\|A_N\|）、D6（layer×projection log10 RMS）。

### Q3. 理论可表示点与真实 `A_N` 概率质量分别集中在哪些数量级？

理论网格为 **E4M3FN×E2M1**（NV）与 HiF4 层级网格（均去掉最外层 FP32 scale）；真实 `A_N` 为 dequantized 概率密度。见 D4：A/B 纵轴为 unique count，C 为 density；x 不归一化。三者 y 含义不同，不可直接当覆盖率。

### Q4. `ΔA` 是否近似零均值？是否有系统偏置？

sampled mean bias=-0.000218734，median=0，RMS=0.188025。若 \|mean\|/RMS 不可忽略，则存在系统偏置。见 R1。

### Q5. 大量小误差还是少量极大 outlier 主导？

sampled top0.1/1/5/10% 能量份额=0.2755/0.5985/0.8665/0.9481；per-capture exact median 同序=0.1166/0.3224/0.5817/0.7182。见 R5。

### Q6. 大残差主要在小/中/大激活区间？

见 R4（A_N vs ΔA）与 R6（\|A_N\| 分位 bin 的 RMS/mean\|ΔA\|）。不得用接近 0 时爆炸的相对误差替代。

### Q7. HiF4 是否更容易把非零压成 0？

capture 均值：NV≠0→HF=0 = 0.0561637；反方向 NV=0→HF≠0 = 0.0151623；HF−NV zero_rate = 0.0410014。见 R7。

### Q8. `ΔA` 在 token×64-group 空间均匀还是集中？

见 R11 各代表层最差 NMSE capture 的 3D/2D 图：若表面出现少数 token/group 尖峰则为空间集中，否则较弥散。

### Q9. amax64、sub16 离散度与 group residual RMS 的关系？AX2 能否视觉复现？

Spearman(amax, rms_delta)=0.9737；Spearman(dispersion, rms_delta)=0.1650。见 R12。若 dispersion 相关强，则与 AX2「64-group 过大/子块不均」方向一致。

### Q10. 哪些 layer/projection/phase 残差最高，下一步优先打哪里？

热点见第 8 节与 R9/R10/R13。算法方向：A. residual energy 高度集中于少量元素（sampled top1% =0.599）：优先 outlier-aware / selective scale / protected group。

---

## 生成的 figures

- `fig_d1_w4a4_activation_hist_full.png`
- `fig_d2_w4a4_activation_hist_central.png`
- `fig_d3_w4a4_activation_log2_abs.png`
- `fig_d4_theory_vs_real_activation_triptych.png`
- `fig_d5_activation_distribution_by_projection_phase.png`
- `fig_d6_activation_rms_layer_projection_heatmap.png`
- `fig_r1_delta_hist_full.png`
- `fig_r2_delta_log10_abs_hist.png`
- `fig_r3_an_vs_ah_hexbin_full.png`
- `fig_r3_an_vs_ah_hexbin_central.png`
- `fig_r4_an_vs_delta_hexbin.png`
- `fig_r5_residual_energy_concentration.png`
- `fig_r6_residual_vs_activation_quantile.png`
- `fig_r7_zero_transition.png`
- `fig_r8_payload_transition_heatmap_count.png`
- `fig_r8_payload_transition_heatmap_row_normalized.png`
- `fig_r9_residual_nmse_layer_projection_heatmap.png`
- `fig_r9_residual_rms_layer_projection_heatmap.png`
- `fig_r10_residual_by_projection_boxplot.png`
- `fig_r11_3d_token_group_residual_surface_layer4.png`
- `fig_r11_token_group_residual_heatmap_layer4.png`
- `fig_r11_3d_token_group_residual_surface_layer18.png`
- `fig_r11_token_group_residual_heatmap_layer18.png`
- `fig_r11_3d_token_group_residual_surface_layer34.png`
- `fig_r11_token_group_residual_heatmap_layer34.png`
- `fig_r12_3d_group_mechanism_scatter.png`
- `fig_r12_amax64_vs_rms_delta_2d.png`
- `fig_r12_dispersion_vs_rms_delta_2d.png`
- `fig_r13_3d_layer_projection_residual_landscape.png`
