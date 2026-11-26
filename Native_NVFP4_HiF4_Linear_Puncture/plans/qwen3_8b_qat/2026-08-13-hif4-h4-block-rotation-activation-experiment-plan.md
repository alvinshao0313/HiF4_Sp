# HiF4 4维 Hadamard 块旋转实验计划

日期：2026-08-13

## 1. 研究目标

验证用户给出的 4×4 正交 Hadamard 旋转，是否能够在不改变 Linear 浮点数学结果的前提下，降低真实保存激活从现有源格式转换到 HiF4 后的量化误差。

本实验只回答三个问题：

1. 对保存激活本身，4维块旋转是否降低 HiF4 量化误差？
2. 对对应 Linear 权重同步做等价旋转后，是否降低最终 Linear 输出误差？
3. 如果有效，收益来自哪里：是否主要来自 G4 内异常值被摊平、64-group 的 amax/S0 下降、以及 e8/e4/local-scale 使用更合理？

本实验不做端到端下游任务，不重新训练，不搜索旋转矩阵，不修改现有 DIAG 算法。

---

## 2. 新实验目录

新建：

```text
Native_NVFP4_HiF4_Linear_Puncture/
└── experiments/
    └── h4_block_rotation/
        ├── PLAN.md
        ├── h4_transform.py
        ├── run_h4_activation_experiment.py
        ├── analyze_h4_results.py
        ├── run_smoke.sh
        ├── run_full.sh
        └── README.md
```

结果统一写到：

```text
Native_NVFP4_HiF4_Linear_Puncture/
└── results/
    └── h4_block_rotation/
        └── <run_id>/
            ├── config.json
            ├── resolved_inputs.json
            ├── group_metrics.csv
            ├── layer_metrics.csv
            ├── summary.json
            ├── figures/
            └── report.md
```

禁止修改现有 `src/diagonal_search.py`、`src/rotation.py`、`src/linear_cases.py` 的算法语义。现有 DIAG 只能作为对照组复用。

---

## 3. 固定旋转矩阵

用户给出的未归一化矩阵：

\[
H_4=
\begin{bmatrix}
1&1&1&-1\\
1&1&-1&1\\
1&-1&1&1\\
-1&1&1&1
\end{bmatrix}
\]

真正使用：

\[
R_4=\frac{1}{2}H_4.
\]

必须在测试中验证：

\[
R_4R_4^\top=I,\qquad R_4^\top R_4=I.
\]

此矩阵对称，因此：

\[
R_4^{-1}=R_4^\top=R_4.
\]

禁止把未归一化的 `H4` 直接用于实验。

---

## 4. 旋转布局

HiF4 的最小结构是连续 4 元素，因此旋转必须严格对齐 HiF4 的 G4 边界。

对于最后一维：

```text
[..., 64]
→ [..., 16, 4]
→ 每个连续 4 元素右乘 R4
→ [..., 64]
```

对于更长维度：

```text
[..., N64, 64]
→ [..., N64, 16, 4]
→ 每个 G4 独立乘 R4
```

严禁：

- 跨越 64-group 混合；
- 跨越两个 G4 混合；
- 对整个 64 维做一次 64×64 Hadamard；
- 改变元素顺序；
- 在旋转后额外做归一化。

输入最后一维必须能被 64 整除，否则直接报错。

定义整体变换：

\[
R=\operatorname{blockdiag}(R_4,R_4,\ldots,R_4).
\]

---

## 5. Linear 等价变换

当前 Linear 定义：

\[
Y=XW^\top.
\]

激活做：

\[
X'=XR,
\]

权重必须同步做：

\[
W'=WR.
\]

于是：

\[
X'(W')^\top
=XR(WR)^\top
=XRR^\top W^\top
=XW^\top.
\]

所以浮点域中该变换必须严格等价。

必须加入 pre-quant equivalence check：

```text
Y_ref = linear(X, W)
Y_rot = linear(X @ R, W @ R)
```

FP32 相对 Frobenius 误差必须 `< 1e-6`。

如果该检查失败，本层实验直接失败，不允许继续把误差归因于 HiF4。

---

## 6. 数据来源：只能复用已经保存的激活

不得重新 capture 激活。

Cursor 首先检查当前 `Native_NVFP4_HiF4_Linear_Puncture` 的 capture/config/manifest，找到现有 DIAG/Linear puncture 实验实际读取的保存激活。

实验必须复用**与现有 DIAG 实验完全相同的 activation tensor、layer 列表、样本和对应权重**。

特别注意：

- 如果保存激活已经处于模型原生 online rotation 之后，不允许撤销该旋转；
- 如果现有 pipeline 使用 NVFP4 QDQ 后的 activation 作为转换源，本实验也必须使用同一个 tensor；
- 如果现有 pipeline 使用 BF16 pre-quant activation 作为源，本实验同样保持该语义；
- 不允许为了 H4 实验重新定义“reference activation”。

运行时把最终解析到的每个输入文件路径、tensor shape、dtype、layer 名称写入：

```text
resolved_inputs.json
```

若找不到保存激活，直接报错退出；禁止静默重新采集。

---

## 7. 对照组

至少固定以下三组：

### Case A：Identity

不做新变换：

\[
X_I=X,\qquad W_I=W.
\]

执行现有 HiF4 转换，作为最重要基线。

### Case B：现有 DIAG

复用当前已有 DIAG 最优结果/现有实现。

目的只是回答：

> H4 相比现在已经做过的 DIAG 大概处于什么水平？

不允许为了本实验重新扩大 DIAG 搜索空间。

### Case C：H4

\[
X_H=XR,\qquad W_H=WR.
\]

对旋转后的张量执行与 Identity 完全相同的 HiF4 quant/dequant。

H4 不允许搜索、不允许调参。

---

## 8. 第一阶段：只看保存激活的量化误差

这是主机制实验，必须先做。

对每个保存 activation：

### 8.1 Identity

\[
\hat X_I=Q_{\mathrm{HiF4}}(X)
\]

计算：

\[
NMSE_X^{I}
=
\frac{\|\hat X_I-X\|_F^2}{\|X\|_F^2}.
\]

### 8.2 H4

先：

\[
X_H=XR
\]

再：

\[
\hat X_H=Q_{\mathrm{HiF4}}(X_H).
\]

计算旋转域误差：

\[
NMSE_X^{H}
=
\frac{\|\hat X_H-X_H\|_F^2}{\|X_H\|_F^2}.
\]

因为 R 正交，也可以旋转回来：

\[
\tilde X_H=\hat X_HR^\top
\]

并验证：

\[
\frac{\|\tilde X_H-X\|_F^2}{\|X\|_F^2}
\]

应与旋转域 NMSE 数值一致（允许浮点误差）。

### 8.3 核心效果量

记录：

\[
ratio_X=
\frac{NMSE_X^H}{NMSE_X^I}
\]

和

\[
gain_X=
1-ratio_X.
\]

`ratio_X < 1` 才表示 H4 对 activation HiF4 量化有效。

---

## 9. 第二阶段：权重量化误差

对每个对应 Linear 权重：

Identity：

\[
\hat W_I=Q_{\mathrm{HiF4}}(W)
\]

H4：

\[
W_H=WR
\]

\[
\hat W_H=Q_{\mathrm{HiF4}}(W_H)
\]

记录：

\[
NMSE_W^I,\quad NMSE_W^H,\quad
ratio_W=\frac{NMSE_W^H}{NMSE_W^I}.
\]

这一阶段用于判断 H4 是否只改善激活，却破坏权重。

---

## 10. 第三阶段：Linear 输出误差

这是最终是否值得继续研究的关键指标。

### 10.1 Reference

必须沿用当前 Linear puncture 实验已有的 source/reference 语义。

若当前实验以 native NVFP4 QDQ 的 `X_src` 和 `W_src` 为转换源，则：

\[
Y_{\mathrm{ref}}=X_{\mathrm{src}}W_{\mathrm{src}}^\top.
\]

不要擅自改成另外一个 BF16 reference。

### 10.2 Identity HiF4

\[
Y_I
=
Q_H(X)Q_H(W)^\top.
\]

### 10.3 H4 HiF4

\[
Y_H
=
Q_H(XR)Q_H(WR)^\top.
\]

记录：

\[
NMSE_Y^I
=
\frac{\|Y_I-Y_{\mathrm{ref}}\|_F^2}
{\|Y_{\mathrm{ref}}\|_F^2},
\]

\[
NMSE_Y^H
=
\frac{\|Y_H-Y_{\mathrm{ref}}\|_F^2}
{\|Y_{\mathrm{ref}}\|_F^2}.
\]

以及：

\[
ratio_Y=\frac{NMSE_Y^H}{NMSE_Y^I}.
\]

除 NMSE 外同时记录：

- relative Frobenius error；
- cosine similarity；
- max absolute error。

不能只凭 activation NMSE 宣布 H4 有效，必须同时报告 Linear output。

---

## 11. 必须做的 G4/G8/G64 机制统计

目的不是只看“误差降了多少”，而是解释为什么。

### 11.1 G4 crest factor

对每个四元组：

\[
CF_4
=
\frac{\max_i |x_i|}
{\sqrt{\frac14\sum_i x_i^2}}.
\]

4维情况下：

\[
1\le CF_4\le2.
\]

H4 若能摊平单点异常值，应降低 `CF4`。

记录旋转前后：

- mean CF4；
- median CF4；
- p90 CF4；
- p99 CF4。

### 11.2 64-group crest factor

\[
CF_{64}
=
\frac{\max_i |x_i|}
{\sqrt{\frac1{64}\sum_i x_i^2}}.
\]

记录旋转前后变化。

### 11.3 amax

记录：

- `amax4`；
- `amax8`；
- `amax64`；

以及：

```text
amax64_ratio = amax64_after / amax64_before
```

### 11.4 HiF4 内部状态

必须复用现有 reference quantizer 能拿到的 metadata，统计：

- S0；
- e8=1 比例；
- e4=1 比例；
- local scale 分布；
- payload=0 比例；
- payload=1.75 的 clipping 比例。

不要根据近似公式重新模拟另一套 HiF4。

### 11.5 group-level 误差

每个 64-group 都记录：

```text
layer
sample/capture id
group_id
nmse_identity
nmse_h4
nmse_ratio
cf4_mean_before
cf4_mean_after
cf64_before
cf64_after
amax64_before
amax64_after
s0_before
s0_after
e8_rate_before
e8_rate_after
e4_rate_before
e4_rate_after
zero_rate_before
zero_rate_after
clip_rate_before
clip_rate_after
```

写入：

```text
group_metrics.csv
```

---

## 12. 机制相关性分析

分析 H4 的 group-level 收益：

\[
\Delta \log NMSE
=
\log NMSE_H-\log NMSE_I.
\]

分别与以下量做 Spearman：

\[
\Delta CF_4,
\quad
\Delta CF_{64},
\quad
\Delta \log amax_{64},
\quad
\Delta S_0,
\quad
\Delta e8\_rate,
\quad
\Delta e4\_rate.
\]

主要判断：

> H4 的收益是不是来自“4维内部能量摊平 → G4 peak 下降 → 64-group amax/S0 更合理 → HiF4 payload 网格更细”。

不要把相关性写成因果结论；这里只用于机制解释。

---

## 13. FP32 与 BF16 旋转

主实验使用 FP32 完成 H4 线性变换，以隔离“旋转几何本身是否改善 HiF4”。

然后额外增加一个固定的 deployment sanity case：

```text
H4_BF16
```

即保存激活/权重先转 BF16，在 BF16 语义下执行同一个 4维变换，再做 HiF4。

只用于回答：

> 真正低精度在线实现以后，收益是否仍然存在？

不允许把 BF16 case 和 FP32 case 混在同一主结论中。

---

## 14. Smoke 测试

正式全量前先跑一个最小 smoke。

从现有保存 activation 中固定选择：

- 按 manifest 顺序第一个可完整匹配 activation + weight 的 Linear；
- 只取该 capture 的前 256 行/token；
- 不随机抽样。

检查：

1. R4 正交性；
2. reshape 后 shape 完全恢复；
3. FP32 pre-quant Linear equivalence `<1e-6`；
4. Identity HiF4 与现有 baseline 在同一 tensor 上结果一致；
5. H4 旋转前后 L2 norm 守恒；
6. 输出 CSV/JSON 字段完整；
7. 无 NaN/Inf。

Smoke 通过后才能 full run。

---

## 15. Full run

Full run 使用：

- 当前保存的全部目标 Linear；
- 当前保存的全部 activation 样本；
- 不重新随机抽样；
- Identity / DIAG / H4_FP32 / H4_BF16 使用完全相同的数据。

所有 Python 命令必须在：

```bash
conda activate hif4
```

环境运行。

---

## 16. 汇总方式

### 16.1 Layer-level

每层记录：

- activation NMSE；
- weight NMSE；
- Linear output NMSE；
- Identity→H4 relative gain；
- Identity→DIAG relative gain；
- H4 vs DIAG；
- 64-group 改善比例；
- group NMSE ratio 的 median/p90/p99。

写入：

```text
layer_metrics.csv
```

### 16.2 总体

主报告至少给出：

```text
median_layer_activation_ratio
median_layer_weight_ratio
median_layer_output_ratio
fraction_layers_activation_improved
fraction_layers_output_improved
fraction_groups_improved
worst_layer_output_ratio
best_layer_output_ratio
```

避免只把所有元素拼起来算一个 global NMSE，因为大层/大样本会淹没层间差异。

---

## 17. 必须输出的图

至少生成 6 张图：

1. `fig01_activation_nmse_by_layer.png`
   - Identity / DIAG / H4_FP32 / H4_BF16

2. `fig02_output_nmse_by_layer.png`
   - Identity / DIAG / H4_FP32 / H4_BF16

3. `fig03_group_nmse_ratio_hist.png`
   - 横轴 `NMSE_H4 / NMSE_identity`

4. `fig04_cf4_before_after.png`
   - G4 crest factor 前后分布

5. `fig05_amax64_before_after.png`
   - 每组 amax64 before vs after

6. `fig06_nmse_gain_vs_cf4_change.png`
   - group-level 机制散点图

所有图必须有中文或清晰英文轴标签，不能只输出图而没有对应 CSV。

---

## 18. 结论判据

不能因为某一层改善就宣布成功。

按 `median layer output ratio` 给主判定：

### 明显有效

\[
ratio_Y^{median}\le0.90
\]

且至少 70% 的层：

\[
ratio_Y<1.
\]

### 小幅有效

\[
0.90<ratio_Y^{median}<0.98
\]

且超过一半层改善。

### 基本中性

\[
0.98\le ratio_Y^{median}\le1.02.
\]

### 负收益

\[
ratio_Y^{median}>1.02.
\]

同时必须单独报告 activation ratio 与 weight ratio。

若：

```text
activation 明显改善
weight 明显恶化
最终 output 无改善
```

结论必须写：

> H4 能改善激活的 HiF4-friendly 分布，但激活收益被权重量化损失抵消，因此当前等价 Linear 方案无净收益。

不能写“H4 有效”。

---

## 19. 额外需要特别检查的现象

### 情况 A：activation 改善，weight 恶化

说明同一个 R4 对激活和权重的最佳方向不一致。

### 情况 B：activation NMSE 几乎不变，但 output 改善

检查量化误差方向与权重/激活敏感方向的改变，不能只看 raw NMSE。

### 情况 C：G4 crest 明显下降但 HiF4 NMSE 不降

检查：

- S0 E6M2 rounding；
- e8/e4 阈值跳变；
- payload grid occupancy；
- rotation 后是否把元素推到不利的 0.25 half-step 附近。

### 情况 D：少数层收益很大，大部分层中性

后续才考虑做 layer-selective H4；本实验阶段不要直接实现选择策略。

---

## 20. 实现约束

Cursor 必须遵守：

1. 不修改现有 DIAG 算法。
2. 不修改现有 HiF4 quantizer 数学语义。
3. 不重新 capture 激活。
4. 不做端到端任务。
5. 不搜索新的 Hadamard 矩阵。
6. 不搜索 sign/permutation。
7. 不增加 learnable 参数。
8. H4 固定为用户给出的矩阵除以 2。
9. 所有 baseline 使用同一批保存数据。
10. 每一步保存原始指标，不能只保留最终 report。
11. 发现 reference 语义不明确时，必须以现有 Linear puncture baseline 的代码路径为准，不得自行创造新 reference。
12. 如果现有量化器可返回真实 S0/e8/e4/payload metadata，必须直接用真实 metadata，禁止写一套近似统计替代。

---

## 21. Cursor 执行顺序

严格按以下顺序：

### Step 1：代码审计

只读检查：

- 保存 activation 的 manifest/config；
- 当前 DIAG 实验入口；
- 当前 HiF4 quant/dequant API；
- 当前 native/source reference 的定义；
- weight 与 activation 的匹配方式。

输出一份 `README.md` 中的“现有数据语义”小节。

### Step 2：实现 H4 transform

只实现：

```python
apply_h4_g4(x, dim=-1)
```

并写最小单测：

- orthogonal；
- norm preservation；
- involution；
- group boundary；
- shape；
- Linear equivalence。

### Step 3：实现 Identity/H4 paired activation test

先不碰 DIAG，不碰 weight。

Smoke 跑通并确认 H4 activation NMSE 可计算。

### Step 4：加入 weight paired test

验证同一 R4 应用于 weight in_features 轴。

### Step 5：加入 Linear output test

先验证 pre-quant exact equivalence，再量化。

### Step 6：接入 DIAG baseline

只复用已有实现/已有最优设置。

### Step 7：输出 group diagnostics

S0/e8/e4/payload + CF4/CF64/amax。

### Step 8：Full run

全部保存 activation。

### Step 9：分析与报告

生成：

```text
summary.json
report.md
figures/*
```

报告必须明确区分：

- activation 自身；
- weight 自身；
- Linear 最终输出；
- 机制统计。

---

## 22. 最终报告必须回答的五句话

最终 `report.md` 开头必须直接回答：

1. 用户给出的 H4 是否降低保存 activation 的 HiF4 NMSE？
2. H4 是否降低对应 weight 的 HiF4 NMSE？
3. 激活和权重同步旋转后，Linear output error 是否下降？
4. 相比现有 DIAG，H4 是更好、相近还是更差？
5. 收益/退化主要与哪些 G4/G8/G64 结构变化相关？

若这五句话回答不出来，实验不算闭环完成。
