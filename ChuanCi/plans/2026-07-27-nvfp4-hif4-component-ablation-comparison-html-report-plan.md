# NVFP4 与 HiF4 综合对比、组件消融及 HTML 报告实施计划

> **生成日期：2026-07-27**
>
> **执行要求：**后续实现时使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans` 逐任务执行；本文件只定义实验、接口、结果口径和报告规划，本轮不修改量化实现、不运行正式实验。

## 1. 目标

在 `ChuanCi` 现有纯 PyTorch 实验基础上，建立一套可复现的 NVFP4–HiF4 对比与 HiF4 组件归因实验，最终自动生成一个离线可打开的综合分析 HTML。报告必须清楚回答以下问题：

1. 对同一份 BF16 原始权重，直接量化为 NVFP4 和直接量化为 HiF4，二者的权重误差、线性层输出误差和模型性能差多少。
2. HiF4 相比 NVFP4 的优势和劣势分别是什么，结论在哪些分布、层、模块和任务上成立。
3. 从真实 packed NVFP4 权重出发，分别经 FP32 解码载体和 BF16 解码载体转换为 HiF4，会额外引入多少误差；BF16 载体投影在总转换损失中占多少。
4. HiF4 的顶层 scale、两级 micro-exponent 和 payload 格式分别贡献了多少损失或收益。
5. 将 HiF4 的 S1P2 payload 替换为 E2M1 后损失是否下降；若 payload 不做 4-bit 量化而保留为 BF16，精度上限是多少。
6. 现有 PTS/direct 转换路径中，哪些结论可靠，哪些实验需要修正或扩展。
7. 所有图表和结论能否由结果 JSON 自动重建，而不是把数值手工写进 HTML。

## 2. 当前实现审计结论

### 2.1 已经具备的能力

`ChuanCi/nvfp4_hif4_torch.py` 已经提供：

- NVFP4 合成伪量化；
- packed NVFP4 解码，包括 E2M1 payload、E4M3 block scale 和 tensor-level global scale；
- HiF4 标准 `group_size=64` 量化；
- 顶层 scale 的四种模式：`continuous`、`bf16_math`、`e6m2_only`、`hardware`；
- 两级 micro-exponent：每 8 元素一级、每 4 元素一级；
- S1P2 payload 量化；
- NVFP4 direct（完整权重以 FP32 解码载体进入 HiF4）、PTS-FP32、PTS-BF16；当前尚无完整权重 BF16 carrier direct 路径；
- BF16-native → HiF4 路径；
- tensor、category、global 的能量加权误差聚合；
- packed checkpoint 与普通 checkpoint 读取；
- 可选的线性层输出误差；
- E1–E7 合成实验与代表层真实权重实验。

因此本计划不重写数值核心，而是在现有逻辑上增加“可控组件开关、同源配对比较、严格误差分解和自动报告”。

### 2.2 当前缺口

当前实验还不能严谨回答用户关心的问题，原因如下：

1. **BF16→NVFP4 与 BF16→HiF4没有同一个 reference 下的直接比较。**现有 E1 分别以 `W_NV` 和 `W_BF16` 为 reference，两个 NMSE 不能直接解释为格式优劣。
2. **HiF4组件消融不完整。**现有 E5 只看顶层 scale，E6 只看 group size；没有单独关闭或启用每 8 / 每 4 元素 micro-exponent，也没有 S1P2、E2M1 与 BF16 payload 上限的直接比较。
3. **S1P2损失占比尚未定义。**HiF4组件之间存在耦合，不能把若干独立 NMSE 简单相减后称为可加损失占比；需要以冻结层级决策后的 BF16 payload 上限作为条件参照。
4. **没有原生 NVFP4 载体路径分解。**真实输入是 packed NVFP4，只存在 `NVFP4→FP32→HiF4` 与 `NVFP4→BF16→HiF4` 两条解码载体路径；当前实验没有把 BF16 载体投影损失、后续 HiF4 量化损失及二者交互项分开。
5. **现有 E5/E6 的多 repeat 结果在循环中被覆盖。**当前结构只保留最后一次 repeat 的结果，不能作为正式统计结论；E7 的部分结果也只保存最后一个样本。
6. **HTML是手工汇总。**当前 `NVFP4_HiF4_experiment_report.html` 内嵌固定数值，不能保证结果更新后同步变化，也依赖 CDN。
7. **真实 BF16 权重来源需要严格确认。**`Qmodel/Qwen3.5-27B-NVFP4-BF16` 从命名上更像 NVFP4 解码后的 BF16 模型，不能未经验证就当作原始 BF16 reference。

### 2.3 必须先修正的统计问题

正式新增实验前，先修正以下问题：

- E5、E6 每个 `distribution × mode × repeat` 都保存 `ErrorSums`，结束后统一能量聚合；
- E7 对四种分布和全部 repeats 统一聚合，不使用 `[-1]` 取最后一次结果；
- 所有 global/category/layer 结果均累计原始 numerator/denominator，不平均逐 tensor NMSE；
- paired gap 必须基于同一 tensor、同一 BF16 reference 和同一元素集合；
- 任意缺失 tensor、shape 不匹配或 provenance 不明确都直接报错，不静默跳过。

## 3. 总体实验架构

采用三层证据链，三者必须在报告中分开命名。

### 3.1 同源格式对比：回答 NVFP4 和 HiF4 谁更准确

对同一个 BF16 reference `W_BF`：

\[
W_{NV}=Q_{NVFP4}(W_{BF}),
\qquad
W_H=Q_{HiF4}(W_{BF}).
\]

分别计算：

\[
L_{NV}=\frac{\|W_{NV}-W_{BF}\|_F^2}{\|W_{BF}\|_F^2},
\qquad
L_H=\frac{\|W_H-W_{BF}\|_F^2}{\|W_{BF}\|_F^2}.
\]

主对比：

\[
\Delta_{H-NV}=L_H-L_{NV},
\qquad
R_{H/NV}=\frac{L_H}{L_{NV}}.
\]

其中：

- `Δ < 0` 表示 HiF4 权重 NMSE 更低；
- `R < 1` 表示 HiF4 更优；
- 这组实验是“格式能力”的主要证据；
- NVFP4 和 HiF4 必须使用相同 BF16 权重、相同 tensor 覆盖范围、相同 RTN 原则和相同分组维度。

### 3.2 原生转换分析：回答 NVFP4 经 FP32/BF16 载体转 HiF4 的差距

真实输入是 packed NVFP4。先定义其数学解码值：

\[
W_{NV}^{32}=D_{FP32}(P_{NV}),
\]

其中 `P_NV` 包含 E2M1 payload、E4M3 block scale 和 tensor-level global scale。`W_NV^32` 是本实验唯一的 native reference，不需要也不假设存在对应的原始 BF16 权重。

两条主路径固定为：

\[
\widehat W_{H,32}=Q_{HiF4}(W_{NV}^{32}),
\]

\[
W_{NV}^{16}=BF16(W_{NV}^{32}),
\qquad
\widehat W_{H,16}=Q_{HiF4}(W_{NV}^{16}).
\]

主指标统一相对 `W_NV^32`：

\[
L_{FP32\ carrier}
=
\frac{\|\widehat W_{H,32}-W_{NV}^{32}\|_F^2}
{\|W_{NV}^{32}\|_F^2},
\]

\[
L_{BF16\ carrier,total}
=
\frac{\|\widehat W_{H,16}-W_{NV}^{32}\|_F^2}
{\|W_{NV}^{32}\|_F^2}.
\]

BF16 路径还要单独报告相对 BF16 载体自身的 HiF4 量化损失：

\[
L_{HiF4\mid BF16}
=
\frac{\|\widehat W_{H,16}-W_{NV}^{16}\|_F^2}
{\|W_{NV}^{16}\|_F^2}.
\]

为解释 BF16 路径总损失，定义：

\[
e_{carrier}=W_{NV}^{16}-W_{NV}^{32},
\qquad
e_H=\widehat W_{H,16}-W_{NV}^{16}.
\]

验证精确恒等式：

\[
\|\widehat W_{H,16}-W_{NV}^{32}\|^2
=
\|e_{carrier}\|^2+\|e_H\|^2+2\langle e_{carrier},e_H\rangle.
\]

分解中的三项统一除以 `||W_NV^32||²` 后输出：

- `carrier_projection_term`；
- `hif4_after_bf16_term`；
- `carrier_hif4_cross_term`；
- `bf16_carrier_total`；
- `fp32_carrier_total`；
- `bf16_minus_fp32_delta`；
- `identity_residual`。

这组原生转换实验禁止出现 `BF16→NVFP4→HiF4` 的命名。原始 BF16 权重只用于 3.1 的独立格式公平比较。

### 3.3 HiF4内部归因：回答各组件的增益与损失

HiF4拆为四类决策：

1. 顶层 64-group scale；
2. 每 8 元素 micro-exponent；
3. 每 4 元素 micro-exponent；
4. S1P2 payload 舍入与饱和。

组件实验既要提供直观的顺序 waterfall，也要提供不依赖启用顺序的 factorial effect。

## 4. 新增公共接口设计

### 4.1 HiF4消融配置

在 `ChuanCi/nvfp4_hif4_torch.py` 中扩展配置，但默认行为必须与当前 `hardware / group64 / full hierarchy / S1P2` 完全一致。

建议接口：

```python
@dataclass(frozen=True)
class HiF4AblationConfig:
    group_size: int = 64
    group_dim: int = -1
    scale_mode: str = "hardware"
    enable_exp8: bool = True
    enable_exp4: bool = True
    payload_format: Literal["s1p2", "e2m1", "bf16", "fp32"] = "s1p2"
    payload_clip_max: float | None = 1.75
```

语义：

- `payload_format="s1p2", payload_clip_max=1.75`：标准 HiF4；
- `payload_format="e2m1"`：使用 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` 非负码本；
- `payload_format="bf16"`：normalized payload 保持 BF16，用于精度上限；
- `payload_format="fp32"`：仅作数学 oracle 和实现校验；
- `payload_clip_max=1.75`：保持标准 S1P2 动态范围；
- `payload_clip_max=None`：不裁剪，用于 BF16/FP32 payload 的绝对上限；
- `enable_exp4=True, enable_exp8=False` 时，exp4 直接相对 `S0` 判定，不借用不存在的 exp8；
- E2M1 的“冻结标准层级”和“E2M1-aware 重调层级”由实验编排器显式区分；
- 任意非标准配置都在结果中标记 `is_standard_hif4=false`。

### 4.2 同源格式对比与原生转换结果

两个问题使用两个独立接口，禁止把原始 BF16 格式对比与 native NVFP4 转换混成一个所谓 cascade。

同源格式对比：

```python
def evaluate_paired_formats(
    bf16_weight: torch.Tensor,
    *,
    hif4_config: HiF4AblationConfig = HiF4AblationConfig(),
    return_reconstructions: bool = False,
) -> dict[str, object]:
    ...
```

只返回：

- `bf16_reference`；
- `nvfp4_from_bf16`；
- `hif4_from_bf16`；
- `hif4_minus_nvfp4`；
- `hif4_over_nvfp4`。

原生 packed NVFP4 转换：

```python
def evaluate_native_nvfp4_conversion(
    nvfp4_fp32: torch.Tensor,
    *,
    pts_scale: torch.Tensor | float | None,
    hif4_config: HiF4AblationConfig = HiF4AblationConfig(),
    actual_bf16_decoded_weight: torch.Tensor | None = None,
    return_reconstructions: bool = False,
) -> dict[str, object]:
    ...
```

必须返回：

- `nvfp4_reference_fp32`；
- `nvfp4_carrier_bf16 = BF16(nvfp4_reference_fp32)`；
- `fp32_carrier_to_hif4`；
- `bf16_carrier_to_hif4`；
- `bf16_carrier_projection`；
- `bf16_carrier_decomposition`；
- `bf16_minus_fp32_delta`；
- PTS 路径作为次级变体单独保存，不能替代两条主载体路径。

若提供真实解码后的 BF16 checkpoint，还要比较它与 `BF16(nvfp4_reference_fp32)` 的逐元素一致性；不一致时分别标记“数学 BF16 投影”和“实际 BF16 解码实现”。

### 4.3 组件消融接口

```python
def evaluate_hif4_component_ablation(
    reference: torch.Tensor,
    *,
    source_kind: Literal["bf16", "nvfp4_fake"],
    base_config: HiF4AblationConfig,
) -> dict[str, object]:
    ...
```

返回至少包含：

- micro-exponent 2×2 四个组合；
- top-scale 2×2 四个组合；
- S1P2-native、E2M1-native、E2M1-fixed、S1P2-search-oracle、E2M1-search-oracle、BF16-range-matched、BF16-unclipped payload；
- sequential gains；
- factorial main effects；
- interaction；
- saturation rate；
- payload bin occupancy；
- FP64 原始 `ErrorSums`。

### 4.4 实验编排与报告文件

推荐最短路径文件结构：

```text
ChuanCi/
├── nvfp4_hif4_torch.py                 # 数值核心和底层评测接口
├── nvfp4_hif4_study.py                 # 新增：实验编排、配对 checkpoint、schema v2
├── render_nvfp4_hif4_report.py         # 新增：JSON → 离线 HTML
├── run_comprehensive_study.sh          # 新增：预检、合成、真实权重、渲染入口
├── tests/
│   ├── test_nvfp4_hif4_torch_core.py
│   ├── test_nvfp4_hif4_torch_eval.py
│   ├── test_nvfp4_hif4_ablation.py     # 新增
│   └── test_nvfp4_hif4_report.py       # 新增
└── results/comprehensive_nvfp4_hif4/
    ├── synthetic/
    ├── real_controlled/
    ├── real_packed/
    ├── model_eval/
    └── report/
```

不建议继续把所有实验编排和 HTML 拼装塞进现有 2200 行主脚本；也不做与本任务无关的框架重构。

## 5. 实验 A：合成数据上的同源 NVFP4–HiF4 公平对比

### 5.1 基础随机分布

保留现有四种分布：

| 分布 | 定义 | 作用 |
|---|---|---|
| Gaussian | `N(0,1)` | 常规近似高斯权重 |
| Laplace | 方差归一化 Laplace | 更高峰、更重尾 |
| Student-t3 | 方差归一化 t3 | 强重尾 |
| Outlier0.1pct20x | 0.1% 元素乘 20 | 稀疏离群值 |

设置：

- seed：`20260723`；
- 每个分布每 repeat：`320,000` 元素，即 `5,000 × 64` groups；
- repeats：`10`；
- 样本在 CPU 生成后再搬至目标 device；
- BF16 reference：`FP32 → BF16 → FP32`；
- NVFP4 block size：16；
- HiF4 group size：64，group_dim=-1；
- 主 HiF4：hardware scale、完整 exp8+exp4、S1P2；
- 量化计算 FP32；误差累计 FP64；
- 主统计：energy-weighted NMSE；附带 NRMSE、SQNR、Cosine、MAE、最大绝对误差。

每个 base tensor 同时产生：

```text
W_BF16
W_NVFP4_FP32 = Q_NVFP4(W_BF16)                 # 数学解码值，FP32 载体
W_NVFP4_BF16 = BF16(W_NVFP4_FP32)              # BF16 解码载体
W_HiF4       = Q_HiF4(W_BF16)                  # 仅用于同源格式比较
W_NV_H_FP32  = Q_HiF4(W_NVFP4_FP32)            # 原生 FP32 载体转换
W_NV_H_BF16  = Q_HiF4(W_NVFP4_BF16)            # 原生 BF16 载体转换
```

前两条转换路径都以 `W_NVFP4_FP32` 为 native reference；`W_BF16` 只用于 NVFP4 与 HiF4 的独立同源格式比较。

### 5.2 结构化 64-group 分布

随机分布不足以精准激活层级 scale，增加以下 group 模板。每种模板每 repeat 仍生成 5,000 个 group，并在组内叠加小幅随机扰动，以避免只得到离散特例。

| 名称 | 64-group 内部结构 | 主要压力点 |
|---|---|---|
| homogeneous | 四个 16-block 都为相同方差 | 基础 payload 误差 |
| monotonic_1_2_4_8 | 四个 16-block scale 为 1/2/4/8 | 64-group 跨 block 动态范围 |
| single_hot_8x | 三块 scale=1，一块 scale=8 | 单个高能 block 对其他块的压制 |
| alternating_per4 | 每 4 元素在 scale 1 与 4 间交替 | exp4 有效性 |
| alternating_per8 | 每 8 元素在 scale 1 与 4 间交替 | exp8 有效性 |
| mixed_tail | 普通、重尾、离群 16-block 混在同一 group | 真实异质组 |

### 5.3 决策边界扫描

增加确定性边界实验，不做随机 CI：

- exp4 阈值附近：`2 ± {2^-12, 2^-10, 2^-8, 2^-6}`；
- exp8 阈值附近：`4 ± {2^-12, 2^-10, 2^-8, 2^-6}`；
- S1P2 中点：`(k+0.5)/4` 附近，`k=0..6`；
- payload 饱和边界 1.75 附近；
- E6M2 相邻码点中点附近；
- BF16 乘法/倒数可能翻转判定的边界。

输出：

- decision flip rate；
- 逐边界 NMSE；
- exp8/exp4 bit map 差异率；
- payload code 变化率；
- 最大局部误差案例。

## 6. 实验 B：HiF4 两级 micro-exponent 的严格消融

### 6.1 2×2组合

固定：

- 同一 reference；
- group=64；
- hardware top scale；
- S1P2 payload；
- payload max=1.75。

比较：

| 编号 | exp8 | exp4 | 含义 |
|---|---:|---:|---|
| H00 | 关闭 | 关闭 | 仅一个 S0 + S1P2 |
| H10 | 开启 | 关闭 | 每 8 元素一级调整 |
| H01 | 关闭 | 开启 | 每 4 元素相对 S0 独立调整 |
| H11 | 开启 | 开启 | 完整标准 HiF4 |

### 6.2 直观顺序增益

报告两条顺序链：

```text
H00 → H10 → H11
H00 → H01 → H11
```

每一步显示：

\[
Gain_{A\to B}=NMSE_A-NMSE_B.
\]

同时报告相对降幅：

\[
RelativeGain_{A\to B}=\frac{NMSE_A-NMSE_B}{NMSE_A}.
\]

### 6.3 不依赖顺序的 factorial effect

定义 exp8 主效应：

\[
G_{exp8}=\frac{(L_{00}-L_{10})+(L_{01}-L_{11})}{2}.
\]

定义 exp4 主效应：

\[
G_{exp4}=\frac{(L_{00}-L_{01})+(L_{10}-L_{11})}{2}.
\]

定义交互项：

\[
I=L_{00}-L_{10}-L_{01}+L_{11}.
\]

报告中明确：

- 主效应是跨另一组件状态的平均收益；
- `I>0` 表示两级联合收益低于简单相加或存在冗余；
- `I<0` 表示联合启用出现协同增益；
- 不把主效应和交互项强行解释为物理上唯一的损失分配。

### 6.4 数据范围

该实验必须覆盖：

- 四种随机合成分布；
- 六种结构化 group；
- BF16-native source；
- NVFP4-native source；
- 真实 BF16 权重；
- 真实 packed NVFP4 权重。

## 7. 实验 C：顶层 scale 的 2×2 因子消融

现有四种 scale mode 正好形成两个因素：

1. BF16 scale math 是否启用；
2. E6M2 scale codebook 是否启用。

| 模式 | BF16 math | E6M2 | 含义 |
|---|---:|---:|---|
| continuous | 否 | 否 | 理想连续 scale |
| bf16_math | 是 | 否 | 只看 BF16 计算 |
| e6m2_only | 否 | 是 | 只看 E6M2 离散化 |
| hardware | 是 | 是 | 标准实现路径 |

除现有四列 NMSE 外，新增：

\[
Penalty_{BF16}
=
\frac{(L_{bf16}-L_{cont})+(L_{hw}-L_{e6m2})}{2},
\]

\[
Penalty_{E6M2}
=
\frac{(L_{e6m2}-L_{cont})+(L_{hw}-L_{bf16})}{2},
\]

\[
Interaction_{scale}
=L_{hw}-L_{bf16}-L_{e6m2}+L_{cont}.
\]

注意：

- penalty 允许为负，表示离散化反而选到了更优的重建 scale；
- 合成实验按 10 repeats 聚合；
- 真实权重按 energy-weighted global/category/layer 聚合；
- 不再只展示 `hardware-continuous` 一个差值。

## 8. 实验 D：S1P2、E2M1 与 BF16 payload 精度上限

### 8.1 要回答的问题

本实验直接回答：

1. 将 S1P2 完整替换为 E2M1，并按 E2M1 重新计算 S0 和两级 exponent 后，误差会变小还是变大；
2. 如果只换 E2M1 码点、却错误地沿用 S1P2 的 S0 和 exponent，结果会偏离多少；
3. 对 S1P2 和 E2M1 使用同样的局部搜索后，两种格式各自还能提升多少；
4. 如果 payload 不再压缩成 4-bit，而是保留为 BF16，精度上限是多少。

S1P2 与 E2M1 都有 8 个非负 magnitude code，但分布完全不同：

```text
S1P2: {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75}
E2M1: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
```

S1P2 在小幅值区间更密，E2M1 动态范围更大。因此不能只做一个未重调 scale 的 E2M1 实验后就判断码本优劣。

### 8.2 主实验与辅助实验

主实验不是“只把最后一步的 S1P2 码点换成 E2M1”，而是把这一层的数据格式完整替换掉。E2M1 必须使用自己的最大值重新计算 S0，并重新决定每 8 个元素和每 4 个元素的 exponent。

| 变体 | S0 与 exponent 如何确定 | Payload | 作用 |
|---|---|---|---|
| `s1p2_native` | 按标准 HiF4 规则计算 | S1P2 | 标准基线 |
| `e2m1_native` | 按 E2M1 的数值范围重新计算 S0，并重新计算两级 exponent | E2M1 | 主实验，回答完整换成 E2M1 后是否更准 |
| `e2m1_fixed_hierarchy` | 沿用 S1P2 的 S0 和 exponent | E2M1 | 辅助实验，只说明“只换码点、不改 scale”会发生什么 |
| `s1p2_search_oracle` | 在附近搜索更优的 S0 和 exponent 组合 | S1P2 | 分析标准 S1P2 规则距离局部最优还有多远 |
| `e2m1_search_oracle` | 使用完全相同的搜索方法 | E2M1 | 分析 E2M1 原生规则距离局部最优还有多远 |
| `bf16_range_matched` | 固定 `s1p2_native` 的 S0 和 exponent | BF16，范围仍限制在 0–1.75 | 只去掉 S1P2 码点舍入后的精度上限 |
| `bf16_unclipped` | 固定 `s1p2_native` 的 S0 和 exponent | BF16，不限制到 1.75 | Payload 保持 BF16 时的绝对精度上限 |

再增加 `fp32_unclipped` 作为数值正确性校验，不放入主性能表。它应近似精确重建输入，用来检查整个计算链路是否实现正确。

所有变体必须相对同一个输入计算误差；BF16 原始权重和 NVFP4 原生权重分开报告。

### 8.3 S0 必须随数据格式一起改变

可以把 S0 理解成一把“总尺子”。后面的两级 exponent 最多能把这把尺子再放大 4 倍，所以 S0 要根据 payload 能表示的最大值来计算。

S1P2 的最大非负值是 1.75：

\[
S_{0,S1P2}\approx\frac{amax_{64}}{1.75\times2\times2}=\frac{amax_{64}}{7}.
\]

E2M1 的最大非负值是 6：

\[
S_{0,E2M1}\approx\frac{amax_{64}}{6\times2\times2}=\frac{amax_{64}}{24}.
\]

因此，主实验中的 E2M1 必须从 `amax64/24` 附近选择 E6M2 scale，不能继续沿用 S1P2 的 `amax64/7`。得到新的 S0 后，还要重新判断每 8 个元素和每 4 个元素是否需要乘 2。这样测到的才是完整替换为 E2M1 后的结果。

实现时优先使用简单而可靠的规则：

1. 先按对应格式计算 S0；
2. 对每 8 个元素，分别尝试 exponent 为 0 和 1；
3. 对其中每 4 个元素，再分别尝试 exponent 为 0 和 1；
4. 选择重建误差更小的合法组合。

这里每一级都只有两个合法选择，因此计算逻辑清楚，也不会引入额外的可学习参数。`e2m1_fixed_hierarchy` 仍会保留，但只用于展示错误地沿用 S1P2 尺子会带来多大影响，不作为主结论。

### 8.4 使用相同搜索方法得到参考上限

为了避免把“搜索方法更好”误认为“E2M1 格式更好”，再为 S1P2 和 E2M1 各运行一次完全相同的局部搜索：

1. 从各自原生计算得到的 S0 开始；
2. 枚举附近的 E6M2 scale；
3. 对每个候选 scale，尝试所有合法的 exp8 和 exp4 组合；
4. 选择 64 个元素总平方误差最小的组合。

主结论来自 `s1p2_native` 与 `e2m1_native`。`s1p2_search_oracle` 和 `e2m1_search_oracle` 只用于回答“原生计算规则距离更优结果还有多少空间”。

为确认局部搜索窗口没有漏掉更优结果：

- 合成实验随机抽取 10,000 个 group；
- 代表层真实权重每个 tensor 抽取 2,048 个 group；
- 对这些 group 穷举全部 255 个 E6M2 top-scale code；
- 报告局部搜索找到全局最优结果的比例，以及与全局最优结果之间的误差差距。

### 8.5 BF16 payload 精度上限

固定标准 HiF4 的 `S_i`，构造：

\[
p_{BF16,matched}=BF16\left(clip\left(|x|/S_i,0,1.75\right)\right),
\]

\[
p_{BF16,unclipped}=BF16\left(|x|/S_i\right).
\]

对应重建：

\[
\widehat x_{BF16,*}=sign(x)S_ip_{BF16,*}.
\]

两种上限含义不同：

- `bf16_range_matched`：保持标准 HiF4 的动态范围，只去除 S1P2 的离散码点误差；
- `bf16_unclipped`：连 1.75 范围约束也去掉，表示“payload 真正保持 BF16”的绝对上限，但存储不再是4-bit。

主比例定义：

\[
Recoverable_{S1P2\to BF16,matched}
=
\frac{L_{S1P2}-L_{BF16,matched}}{L_{S1P2}},
\]

\[
Recoverable_{S1P2\to BF16,unclipped}
=
\frac{L_{S1P2}-L_{BF16,unclipped}}{L_{S1P2}}.
\]

第一项是最适合作为“S1P2码点量化损失占比”的条件口径；第二项是包括 payload 动态范围限制在内的绝对可恢复上限。

### 8.6 S1P2相对BF16上限的精确分解

令：

\[
e_{upper}=\widehat x_{BF16,matched}-x,
\qquad
\delta_{S1P2}=\widehat x_{S1P2}-\widehat x_{BF16,matched}.
\]

则：

\[
\|\widehat x_{S1P2}-x\|^2
=
\|e_{upper}\|^2
+
\|\delta_{S1P2}\|^2
+
2\langle e_{upper},\delta_{S1P2}\rangle.
\]

输出：

- `bf16_range_matched_floor`；
- `s1p2_increment_energy`；
- `s1p2_upper_interaction`；
- `conditional_s1p2_recoverable_fraction`；
- `identity_residual`。

报告必须表述为“在冻结标准层级 scale/exponent 和动态范围后的条件贡献”，不能声称这是与 top scale、micro-exponent 完全独立的物理损失。

### 8.7 E2M1与S1P2配对指标

主格式比较：

\[
\Delta_{native}=L_{E2M1,native}-L_{S1P2,native}.
\]

- `Δ_native < 0`：完整 E2M1 格式误差更小；
- `Δ_native > 0`：标准 S1P2 格式误差更小。

另外报告三组辅助结果：

1. `e2m1_fixed_hierarchy - s1p2_native`：说明只换码点、不重新计算 S0 会造成什么影响；
2. `e2m1_search_oracle - s1p2_search_oracle`：在两种格式使用相同搜索方法时，哪一种潜在上限更高；
3. native 与 search-oracle 之间的差距：说明当前 S0 和 exponent 计算规则还有多少优化空间。

所有结果还要统计：

- E2M1 在不同 repeat、tensor 和 group 中胜出的比例；
- E2M1 与 `bf16_range_matched` 之间还剩多少误差；
- 两种格式在小数值、大数值和接近最大码点区域的误差；
- 在不同组内动态范围下，哪种格式更占优势。

### 8.8 Payload码点诊断

按 source、tensor 和 group 统计：

- S1P2 与 E2M1 各 magnitude code 的 occupancy；
- zero code 占比；
- S1P2 1.75 与 E2M1 6.0 最大码点占比；
- 各码点的误差能量贡献；
- S1P2/E2M1 决策不一致率；
- BF16 upper bound 与两种 4-bit payload 的逐元素 gap；
- top 1%、5%、10% 高误差 group 中的 payload 分布。

## 9. 实验 E：BF16→NVFP4 与 BF16→HiF4 性能差距

### 9.1 权重域主实验

必须使用同一 BF16 reference，报告：

- NMSE；
- NRMSE；
- SQNR；
- Cosine；
- MAE；
- approximation/reference energy ratio；
- 每 tensor 胜负；
- 全局 energy-weighted gap；
- category 和 layer gap；
- group NMSE 分位数。

主表列：

```text
BF16 reference
NVFP4 RTN
HiF4 RTN
HiF4 − NVFP4 absolute NMSE
HiF4 / NVFP4 NMSE ratio
winner
```

### 9.2 存储成本对照

按格式定义计算有效权重位数：

- NVFP4：4-bit payload + `8/16` bit block scale = 4.5 bit/weight，另有可忽略的 tensor global scale；
- HiF4：4-bit payload + `1/8` + `1/4` micro-exponent + `8/64` top scale = 4.5 bit/weight。

报告必须说明：

- 二者主存储开销近似相同，因此精度比较有意义；
- tensor/global metadata 单独列出，不用四舍五入掩盖；
- 若实际实现的 scale 位宽与上述定义不一致，以 checkpoint metadata 为准并标注。

### 9.3 线性层输出误差

用同一批校准激活 `X`：

\[
Y_{BF}=XW_{BF}^T,
\quad
Y_{NV}=XW_{NV}^T,
\quad
Y_H=XW_H^T.
\]

报告：

- `output_nmse_nvfp4`；
- `output_nmse_hif4`；
- `output_gap`；
- weight NMSE 与 output NMSE 的 Spearman/Pearson 相关性；
- 权重域 winner 与输出域 winner 不一致的 tensor 列表。

校准激活设置：

- 数据：`allenai/c4` English train split；若服务器使用本地镜像，记录实际数据路径和数据集指纹；
- 用模型自身 tokenizer 随机抽取并拼接为 128 条、每条 2048 tokens 的校准序列；
- seed：31；样本索引保存到结果目录，所有格式路径复用同一索引；
- 只缓存进入目标 Linear 的 FP16/BF16 activation；
- 每层最多采样 4096 tokens，固定采样位置；
- 同一 activation 文件供 NVFP4、HiF4、转换路径复用。

如果现有工程尚无统一 activation dump 脚本，新增一个最小脚本，只负责保存 `dict[weight_name, activation_tensor]`，不把模型评测逻辑混入量化核心。

## 10. 实验 F：真实权重设置

### 10.1 权重来源与 provenance

正式实验分两类输入：

1. **同源格式比较**需要原始 BF16 checkpoint、由其生成或与其严格同源的 packed NVFP4，以及从同一个 BF16 checkpoint直接 RTN 得到的 HiF4；
2. **原生 NVFP4转换**只需要真实 packed NVFP4 checkpoint，由脚本分别解码为 FP32 carrier 和 BF16 carrier；
3. 必要时，由原始 BF16 本地模拟得到 NVFP4，用于验证本地 NVFP4 生成器是否复现实际 packed checkpoint；
4. 可选提供一个实际 NVFP4 解码后的 BF16 checkpoint，用于验证真实 BF16 解码实现是否等于 `BF16(FP32_decode)`。

每份 checkpoint 保存：

- 路径；
- config 中模型类型和层数；
- 文件列表；
- index/config/recipe SHA-256；
- tensor 名称和 shape 摘要；
- 来源说明。

`Qmodel/Qwen3.5-27B-NVFP4-BF16` 应优先按“实际 NVFP4 解码后的 BF16 carrier checkpoint”核验：若其逐元素等于或接近 `BF16(FP32_decode(packed NVFP4))`，则用于 `NVFP4→BF16→HiF4` 的真实载体路径；它不能作为原始 BF16 reference，也不能用于 BF16→NVFP4 与 BF16→HiF4 的格式公平比较。

### 10.2 tensor覆盖范围

tensor覆盖按实验类型分别确定：

- **同源格式比较**：只取原始 BF16 与 packed NVFP4 的名称/shape 交集；
- **原生 NVFP4转换**：覆盖 packed NVFP4 recipe 中的全部目标 tensor，不要求原始 BF16 存在；
- **实际 BF16 carrier验证**：只取 packed NVFP4 FP32 decode 与 NVFP4-BF16 checkpoint 的名称/shape 交集。

统一遵循 NVFP4 recipe：

- MLP：`gate_proj/up_proj/down_proj`；
- full-attention：`q_proj/k_proj/v_proj/o_proj`；
- recipe 明确忽略的 `lm_head` 和 `linear_attn` 不混入主表；
- embedding、lm_head 和其他 tensor 可作为附录；
- shape 不匹配立即报错；
- 名称交集、遗漏原因和参数覆盖率写入结果。

### 10.3 两阶段运行

#### 代表层预检

沿用现有 Qwen3.5-27B 的 0-based 第 3、31、63 层：

- early / middle / late；
- 目标是验证接口、显存、结果 schema 和图表；
- 预检通过后才运行全层。

#### 全层正式实验

覆盖所有 recipe 实际量化的 Linear：

- global；
- category；
- layer；
- tensor；
- group quantile；
- 高误差 group 特征。

全局指标使用 FP64 能量累计，不平均层 NMSE。

### 10.4 simulated NVFP4 与 actual packed NVFP4 交叉验证

对同一 BF16 权重同时得到：

- `Q_NVFP4_local(W_BF16)`；
- `decode(actual_packed_NVFP4)`。

比较：

- exact element fraction；
- NMSE；
- max absolute error；
- block scale exact fraction；
- payload exact fraction；
- global scale；
- 每 tensor mismatch 原因。

若不能逐元素复现：

- **受控格式比较**使用本地 simulated NVFP4；
- **真实部署转换**使用 actual packed NVFP4；
- 两组结果在 HTML 中分栏，不得混合。

## 11. 实验 G：NVFP4→HiF4转换路径完善

### 11.1 两条原生载体路径

以 packed NVFP4 的 FP32 数学解码值 `W_NV^32` 为唯一 reference，主实验只比较：

- `fp32_carrier_direct`：`Q_HiF4(W_NV^32)`；
- `bf16_carrier_direct`：`Q_HiF4(BF16(W_NV^32))`。

每条路径报告：

- 相对 `W_NV^32` 的 native conversion NMSE；
- BF16 路径相对 `BF16(W_NV^32)` 的条件 HiF4 NMSE；
- BF16 carrier projection NMSE；
- BF16 carrier 的 projection / HiF4 / cross-term 精确分解；
- 重建能量比；
- tensor/category/layer/global；
- FP32 carrier 与 BF16 carrier 的配对胜率和 delta。

### 11.2 PTS路径作为次级变体

保留现有：

- PTS-FP32：`s_T * Q_HiF4(W_NV^32/s_T)`；
- PTS-BF16：`s_T * Q_HiF4(BF16(W_NV^32/s_T))`。

但必须明确：

- PTS-FP32/PTS-BF16 是“是否提出 tensor global scale 以及 normalized carrier dtype”的消融；
- 它们不等同于完整权重的 `NVFP4→FP32/BF16→HiF4` 两条主载体路径；
- PTS 路径同样只相对 `W_NV^32` 计算误差；
- 主报告先展示两条 direct carrier 路径，再单列 PTS 分析。

### 11.3 FP32与BF16载体的决策分歧

比较 `Q_HiF4(W_NV^32)` 与 `Q_HiF4(BF16(W_NV^32))` 的内部决策：

- top-scale code exact fraction；
- exp8 bit exact fraction；
- exp4 bit exact fraction；
- payload code exact fraction；
- 最终重建 exact fraction；
- 仅 top-scale 改变、仅 exponent 改变、仅 payload 改变和多项同时改变的 group 比例；
- 各类 decision-change group 对 BF16-vs-FP32 carrier gap 的误差能量贡献。

额外构造冻结决策对照：

1. 使用 FP32 carrier 路径的 scale/exponent map 量化 BF16 carrier；
2. 使用 BF16 carrier 路径的 scale/exponent map 量化 FP32 carrier；
3. 分别固定 top scale、固定 exponent、固定 payload code，定位载体投影导致的差距主要由哪一级决策翻转产生。

这些冻结决策实验只用于归因，必须标记为非部署路径。

### 11.4 64-group内四个NVFP4 block冲突分析

每个 HiF4 group 包含四个 NVFP4 16-block。记录：

- 四个 NV block scale；
- `max/min`；
- `log2(max/min)`；
- scale 均值、标准差、变异系数；
- 四块 reference energy；
- HiF4 group NMSE；
- FP32 carrier 与 BF16 carrier gap；
- direct 与 PTS gap；
- exp8/exp4 使用率；
- S1P2、E2M1 与 BF16 payload gap。

按 `log2 scale range` 分桶：

```text
[0,0.5), [0.5,1), [1,2), [2,3), [3,+∞)
```

报告每桶：

- group 数量；
- reference energy 占比；
- conversion error 占比；
- direct/PTS winner；
- 各组件收益。

这一步用于验证：NVFP4 的 16-block 与 HiF4 的 64-group 粒度冲突是否是转换损失的主要来源。

## 12. 实验 H：端到端模型性能

### 12.1 主公平实验

主模型性能比较必须满足：

- 同一原始 BF16 checkpoint；
- 相同 Linear 覆盖范围；
- 相同 weight-only 设定；
- 相同 RTN，不混入 GPTQ；
- 相同 activation/KV 精度；
- 相同 tokenizer、prompt、evaluation harness；
- NVFP4 和 HiF4 均在同一 forward graph 中做 fake quant 或等价加载。

主模型：Qwen3.5-27B；资源允许时增加一个不同架构的 7B/8B 模型，用来验证结论不是 Qwen3.5 特例。

### 12.2 模型版本

至少比较：

1. BF16；
2. BF16→NVFP4 RTN；
3. BF16→HiF4 RTN；
4. 原生 packed NVFP4→FP32 carrier→HiF4；
5. 原生 packed NVFP4→BF16 carrier→HiF4；
6. 原生 packed NVFP4→PTS-FP32/PTS-BF16→HiF4，作为次级转换变体；
7. 可选：现有 HiF4 GPTQ，仅作为“算法补偿后结果”，不与 RTN 格式能力混为一谈。

### 12.3 评测集合

第一轮最小稳定集合：

- WikiText-2 PPL；
- C4 PPL 子集；
- MMLU；
- ARC-Easy / ARC-Challenge；
- HellaSwag；
- WinoGrande；
- GSM8K 或项目当前固定 reasoning 任务。

设置：

- lighteval/vLLM 使用项目现有 `main.py`；
- 所有模型固定同一 `max_model_length`；
- 选择题任务采用确定性 log-likelihood；
- 生成任务固定 temperature=0、top_p=1、同一 max_new_tokens；
- 数据版本和缓存路径写入结果；
- 同一任务的样本集合与顺序一致；
- 首先小样本预检，再运行完整评测。

### 12.4 部署模型作为次要结果

现有真实 packed NVFP4 与已生成 HiF4 RTN/GPTQ checkpoint 可以另设“当前部署结果”章节，但必须标明：

- quantization algorithm 是否相同；
- activation quantization 是否相同；
- KV cache 是否相同；
- 是否经过 GPTQ；
- tensor 覆盖是否相同。

不满足同源同配置时，不得据此宣称格式本身优劣。

## 13. 统一结果 schema v2

建议 schema：

```json
{
  "schema_version": 2,
  "study": "nvfp4_hif4_comprehensive",
  "provenance": {},
  "config": {},
  "synthetic": {
    "paired_format": {},
    "micro_exponent_ablation": {},
    "top_scale_ablation": {},
    "payload_format_ablation": {},
    "native_carrier_conversion": {},
    "threshold_sweeps": {}
  },
  "real_controlled": {
    "coverage": {},
    "global": {},
    "categories": {},
    "layers": {},
    "tensors": {},
    "groups": {}
  },
  "real_packed": {},
  "model_eval": {},
  "validation": {}
}
```

每个误差对象至少保留：

```text
numel
reference_energy
approximation_energy
error_energy
dot
absolute_error_sum
max_absolute_error
nmse
nrmse
cosine
sqnr_db
mae
```

组件分解额外保存：

```text
component_energy
interaction_energy
normalized_component_share
identity_residual
```

报告渲染器只能读取 schema v2，不从 HTML 中读取旧结果，也不允许缺失字段时填 0。

## 14. HTML报告内容规划

### 14.1 页面原则

- 单文件离线 HTML；
- 不依赖 CDN；
- 图表由结果 JSON 自动生成；
- HTML 中不手写任何正式实验数值；
- 预检报告允许可选章节显示“未运行”，不伪造 0；最终报告若缺少主实验章节则渲染失败；
- 每张图都注明比较基准和统计方式；
- same-source 格式损失、FP32 carrier 原生转换、BF16 carrier 原生转换使用不同视觉标签；
- 适合浏览器阅读和科研汇报截图；
- 颜色采用白底、钴蓝主色，橙色强调差距，灰色表示分析性非标准格式；
- 以高中生能够顺畅阅读为目标，不要求读者预先理解量化格式；
- 每个实验章节先用三句话说明：为什么做、具体改了什么、结果应该怎么看；
- 专业术语第一次出现时必须立刻解释，例如把 S0 解释为“一组数据共用的总尺子”，把 payload 解释为“每个权重最终保存的4-bit数值”；
- 每个公式后紧跟一段通俗解释，说明分子、分母和正负号分别代表什么；
- 正文优先展示结论和关键数字，完整公式、实现参数和逐层结果放在可展开的“实验细节”区域；
- 不使用没有解释的缩写和英文术语，不把“相关”写成“导致”，不根据少量代表层直接推广到整个模型。

### 14.2 页面结构

#### 0. Executive Summary

四个核心结论卡片：

- BF16→NVFP4 与 BF16→HiF4 主差距；
- NVFP4→FP32/BF16 carrier→HiF4 的差距；
- HiF4最大损失组件；
- S1P2换成E2M1是否获益，以及BF16 payload精度上限。

每个结论卡片显示：数值、数据范围、reference、是否来自真实权重。

#### 1. 实验公平性与 provenance

表格：

- 模型来源；
- checkpoint hash；
- tensor 覆盖；
- quantization method；
- activation/KV 精度；
- 是否可用于格式公平比较。

#### 2. NVFP4 与 HiF4 格式结构

- 两种格式的层级结构示意；
- 元数据粒度；
- effective bits/weight 表；
- NVFP4 16-block 与 HiF4 64-group 的对应图。

#### 3. 同源 BF16→NVFP4 与 BF16→HiF4

图表：

1. 按合成分布的 grouped bar：NVFP4 NMSE vs HiF4 NMSE；
2. 真实 tensor paired scatter：x=NVFP4 NMSE，y=HiF4 NMSE，带 `y=x`；
3. category grouped bar；
4. layer heatmap：`log10(NMSE_H/NMSE_NV)`；
5. winner 统计和相对 gap 分布。

表格：global/category/layer 主结果。

#### 4. NVFP4→FP32/BF16载体→HiF4原生转换

图表：

1. FP32 carrier 与 BF16 carrier 的 native conversion NMSE grouped bar；
2. BF16 carrier 路径 waterfall：carrier projection、HiF4 after BF16、cross term、总损失；
3. FP32/BF16 carrier 的内部 scale/exponent/payload decision flip 统计；
4. direct carrier 与 PTS 路径 category bar；
5. 各层 BF16-minus-FP32 carrier overhead 热图。

#### 5. HiF4 micro-exponent消融

图表：

1. `H00→H10→H11` waterfall；
2. `H00→H01→H11` waterfall；
3. exp8/exp4 factorial main effect + interaction；
4. 结构化 group 上的收益热图。

#### 6. 顶层scale消融

图表：

- continuous/bf16/e6m2/hardware grouped bar；
- BF16 penalty、E6M2 penalty、interaction 的 signed bar；
- 重尾程度与 E6M2 penalty 的关系。

#### 7. S1P2、E2M1与BF16 payload上限

图表：

1. 主图只比较 S1P2-native、E2M1-native、BF16-range-matched 和 BF16-unclipped，让读者先看清完整格式替换后的结果；
2. 辅助图再展示 E2M1-fixed、S1P2-search-oracle 和 E2M1-search-oracle，说明不改 S0 与额外搜索分别会带来什么影响；
3. 展示 E2M1-native 相对 S1P2-native 的误差变化和胜出比例；
4. 展示 S1P2→BF16-range-matched 和 S1P2→BF16-unclipped 的可恢复损失比例；
5. 展示 BF16 floor、S1P2 increment 和 interaction 的误差分解；
6. 展示 S1P2/E2M1 码点使用频率，以及小数值、大数值和接近最大码点区域的误差。

必须在图下写清：BF16-range-matched 是冻结标准 scale/exponent 和动态范围后的条件上限；BF16-unclipped 是非4-bit绝对上限。

#### 8. 真实权重深入分析

图表：

- category bar；
- layer heatmap；
- group NMSE ECDF；
- NV block-scale conflict 与 conversion loss scatter；
- conflict 分桶的误差贡献；
- 高误差 tensor 表。

#### 9. 线性层输出与任务性能

- weight NMSE vs output NMSE scatter；
- 输出域 winner 翻转表；
- PPL/accuracy grouped bar；
- 相对 BF16 degradation 表。

#### 10. 结论、边界与建议

分三类：

- 被合成+真实权重+任务共同支持的结论；
- 只在权重域支持的结论；
- 尚需验证的假设。

### 14.3 图表实现

推荐使用 Python 生成内联 SVG，HTML 只负责布局：

- 无 CDN；
- 结果可完全离线；
- SVG 文本和数值可复制；
- 报告可稳定复现；
- 不需要引入前端构建链。

必要的交互只使用少量原生 JavaScript：

- 切换 synthetic/real；
- 展开 tensor 表；
- 过滤 category/layer；
- tooltip 读取嵌入 JSON。

## 15. 实施任务

### Task 1：修正现有聚合与建立 schema v2

**修改：**

- `ChuanCi/nvfp4_hif4_torch.py`
- `ChuanCi/tests/test_nvfp4_hif4_torch_eval.py`

**完成条件：**

- E5/E6 不再覆盖 repeat；
- E7 不再只取最后结果；
- 所有正式结果保留 `ErrorSums`；
- schema v2 validator 对缺失字段直接失败；
- 旧默认量化数值不变。

### Task 2：增加可控 HiF4 组件开关

**修改：**

- `ChuanCi/nvfp4_hif4_torch.py`
- `ChuanCi/tests/test_nvfp4_hif4_torch_core.py`

**新增测试：**

- H11 与当前标准实现逐元素一致；
- H00/H10/H01/H11 的手算 case；
- S1P2 与 E2M1 码本边界舍入；
- BF16 range-matched / unclipped payload 行为；
- 全零、饱和、阈值附近行为；
- CPU/CUDA 输入一致性。

### Task 3：实现同源格式配对与原生载体误差分解

**创建：**

- `ChuanCi/tests/test_nvfp4_hif4_ablation.py`

**修改：**

- `ChuanCi/nvfp4_hif4_torch.py`

**测试恒等式：**

- 同源格式比较使用同一 BF16 reference；
- native 转换使用同一 `W_NV^32` reference；
- BF16 carrier projection + HiF4 + cross-term energy identity；
- cross term 符号可正可负；
- paired tensor name/shape 严格一致；
- global 结果等于 atomic sums 合并。

### Task 4：实现 micro-exponent、scale 和 payload 格式消融

**修改：**

- `ChuanCi/nvfp4_hif4_torch.py`

**测试：**

- factorial effect 公式；
- 顺序 gain 与原始四组合一致；
- S1P2 相对 BF16-range-matched 的 energy identity；
- S1P2-native 的连续 S0 基准为 `amax64/7`；
- E2M1-native 的连续 S0 基准为 `amax64/24`，并重新计算 exp8/exp4；
- E2M1-fixed 完全复用 S1P2-native 的 S0/exp8/exp4，只作为辅助消融；
- S1P2-search-oracle 的结果不劣于 S1P2-native；
- E2M1-search-oracle 的结果不劣于 E2M1-native；
- S1P2-search-oracle 与 E2M1-search-oracle 使用相同的候选规模和误差选择逻辑；
- BF16-range-matched 只改变 payload，不改变 S1P2-native 的 scale/exponent map；
- FP32-unclipped oracle 近似精确重建；
- identity residual 在 FP64 容差内。

### Task 5：建立综合实验编排器

**创建：**

- `ChuanCi/nvfp4_hif4_study.py`

**子命令：**

```text
synthetic
real-preflight
real-full
activation-output
merge-results
```

**要求：**

- 每个子命令独立生成 JSON/CSV；
- 支持断点按 tensor 重跑，但不得静默使用缺失结果；
- 写出完整 config/provenance；
- 结果采用原子写入；
- 不在内存同时保留全模型全部 reconstruction。

### Task 6：原始BF16、packed NVFP4与BF16 carrier checkpoint配对

**修改或新增：**

- `ChuanCi/nvfp4_hif4_study.py`
- 必要时新增 `ChuanCi/dump_linear_activations.py`

**要求：**

- 原始 BF16 路径仅用于同源格式比较，可在只运行 native conversion 时省略；
- 实际 packed NVFP4 同时生成 FP32 carrier 与数学 BF16 carrier；
- 可选传入 `Qmodel/Qwen3.5-27B-NVFP4-BF16`，验证实际 BF16 carrier checkpoint；
- 分别输出格式比较、native conversion、实际 BF16 carrier验证的 tensor intersection 和 coverage；
- 代表层预检后再全层；
- simulated NVFP4、actual packed NVFP4、实际 BF16 carrier 分开保存。

### Task 7：端到端评测入口

**复用：**

- 根目录 `main.py`；
- 项目已有 lighteval/vLLM 环境。

**新增：**

- 一份固定模型列表和任务列表的 shell 配置；
- 结果归一化脚本，将 lighteval 输出转换进 schema v2。

### Task 8：自动 HTML 渲染器

**创建：**

- `ChuanCi/render_nvfp4_hif4_report.py`
- `ChuanCi/tests/test_nvfp4_hif4_report.py`

**要求：**

- 只读 schema v2；
- 不硬编码实验数值；
- 不访问网络；
- 生成单文件 HTML；
- 所有图为内联 SVG；
- 缺关键结果时报错；
- 支持只生成预检版，并显著标注“代表层预检”。

### Task 9：统一运行脚本

**创建：**

- `ChuanCi/run_comprehensive_study.sh`

**执行顺序：**

1. 单元测试；
2. quick synthetic；
3. representative-layer real preflight；
4. 预检 HTML；
5. full synthetic；
6. full real weights；
7. activation/output；
8. model evaluation；
9. final merge；
10. final HTML。

脚本必须在非 `hif4` 环境时直接退出。

## 16. 推荐命令

### 16.1 测试

```bash
conda run -n hif4 python -m unittest \
  ChuanCi.tests.test_nvfp4_hif4_torch_core \
  ChuanCi.tests.test_nvfp4_hif4_torch_eval \
  ChuanCi.tests.test_nvfp4_hif4_ablation \
  ChuanCi.tests.test_nvfp4_hif4_report -v
```

### 16.2 快速合成预检

```bash
conda run -n hif4 python ChuanCi/nvfp4_hif4_study.py synthetic \
  --quick \
  --device cuda \
  --output-dir ChuanCi/results/comprehensive_nvfp4_hif4/synthetic_quick
```

### 16.3 真实代表层

```bash
conda run -n hif4 python ChuanCi/nvfp4_hif4_study.py real-preflight \
  --bf16-checkpoint "$ORIGINAL_BF16_CHECKPOINT" \
  --nvfp4-checkpoint Qmodel/Qwen3.5-27B-NVFP4 \
  --nvfp4-bf16-checkpoint Qmodel/Qwen3.5-27B-NVFP4-BF16 \
  --layers 3,31,63 \
  --device cuda \
  --output-dir ChuanCi/results/comprehensive_nvfp4_hif4/real_preflight
```

### 16.4 全层真实权重

```bash
conda run -n hif4 python ChuanCi/nvfp4_hif4_study.py real-full \
  --bf16-checkpoint "$ORIGINAL_BF16_CHECKPOINT" \
  --nvfp4-checkpoint Qmodel/Qwen3.5-27B-NVFP4 \
  --nvfp4-bf16-checkpoint Qmodel/Qwen3.5-27B-NVFP4-BF16 \
  --device cuda \
  --output-dir ChuanCi/results/comprehensive_nvfp4_hif4/real_full
```

### 16.5 生成报告

```bash
conda run -n hif4 python ChuanCi/render_nvfp4_hif4_report.py \
  --input-root ChuanCi/results/comprehensive_nvfp4_hif4 \
  --output ChuanCi/results/comprehensive_nvfp4_hif4/report/NVFP4_HiF4_comprehensive_report.html
```

## 17. 运行顺序与停止条件

### 第一阶段：数值和统计正确性

完成 Task 1–4。

停止条件：

- 标准 H11 与旧实现逐元素一致；
- BF16 carrier projection / HiF4 / cross-term 分解恒等式通过；
- S1P2 相对 BF16-range-matched 的分解恒等式通过；
- E2M1-native 的 S0 与两级 exponent 重新计算通过手算测试，E2M1-fixed、两种 search-oracle 和 BF16 upper-bound 路径通过对应校验；
- E5/E6 重复实验正确聚合；
- quick synthetic 输出 schema v2。

### 第二阶段：代表层真实权重

完成 Qwen3.5-27B 第 3、31、63 层。

停止条件：

- 原始 BF16 provenance 明确，且与 NVFP4-BF16 carrier checkpoint 严格区分；
- 格式比较、native conversion、实际 BF16 carrier验证三套 tensor coverage 无歧义；
- 21 个代表 tensor 的 packed NVFP4 FP32/BF16 carrier 路径全部完成；
- 预检 HTML 能离线打开；
- 所有主图的数值能从 JSON 反算一致。

### 第三阶段：全层和端到端

只有代表层通过后才运行。

停止条件：

- 全层 tensor/category/layer/global 完整；
- activation/output 实验覆盖主要 Linear；
- 至少 WikiText-2 PPL 和一组下游任务完成；
- final HTML 无“未运行”的主章节。

## 18. 验收标准

### 18.1 数值正确性

- 标准 HiF4 结果与当前实现完全一致；
- packed NVFP4 解码继续通过现有交叉验证；
- 所有能量累计使用 FP64；
- BF16 carrier 分解与 S1P2-vs-BF16 分解的 identity residual 小于 `1e-10 × reference_energy` 或更严格的数值容差；
- FP32-unclipped payload oracle 的残余误差只来自规定的浮点计算路径；
- 全局指标由原始能量合并得到。

### 18.2 实验完整性

- 同源 BF16→NVFP4 和 BF16→HiF4 主对比完成；
- micro-exponent 2×2 完成；
- scale 2×2 完成；
- S1P2-native 与 E2M1-native 主对比完成，E2M1-fixed、两种 search-oracle、BF16-range-matched 和 BF16-unclipped 辅助对比完成；
- NVFP4→FP32 carrier→HiF4 与 NVFP4→BF16 carrier→HiF4 对比及 BF16 carrier 分解完成；
- synthetic 与 real 都有结果；
- 代表层与全层状态分开标注。

### 18.3 报告正确性

- HTML 中无手工填写的正式数值；
- 无 CDN 和网络依赖；
- 页面离线打开；
- 每张图注明 reference、聚合方式和数据范围；
- 非标准分析变体明确标注；
- 同一个输入 JSON 重复渲染得到稳定 HTML；
- 缺失关键字段时渲染失败而不是填零。

### 18.4 研究结论边界

最终结论必须分别注明属于：

- 格式本身的同源 RTN 对比；
- 实际 packed checkpoint 转换；
- HiF4 分析性非标准消融；
- 端到端模型表现。

禁止用以下方式下结论：

- 比较不同 reference 的 NMSE；
- 把原生 NVFP4 转换写成不存在的 `BF16→NVFP4→HiF4` 链路；
- 把 E2M1-search-oracle 当作标准 HiF4；
- 把 BF16 payload 上限当作4-bit可部署结果；
- 用 GPTQ HiF4 对比 RTN NVFP4 后宣称格式优势；
- 用 NVFP4 解码后的 BF16 模型当原始 BF16；
- 用 3 个代表层外推全部 64 层。

## 19. 预期最终交付物

```text
ChuanCi/results/comprehensive_nvfp4_hif4/
├── synthetic/results.json
├── synthetic/results.csv
├── real_preflight/results.json
├── real_full/results.json
├── activation_output/results.json
├── model_eval/results.json
├── merged/results.json
└── report/NVFP4_HiF4_comprehensive_report.html
```

最终 HTML 应能直接回答：

1. 同等约 4.5 bit/weight 下，NVFP4 与 HiF4 在同源 BF16 权重上的真实差距；
2. HiF4 的优势来自层级 micro-exponent 还是其他部分；
3. S1P2 换成 E2M1 后是否更准，固定层级与 E2M1-aware 重调的结论是否一致；
4. payload 保持 BF16 时，同范围条件上限和无裁剪绝对上限分别是多少；
5. 原生 NVFP4 经 FP32 或 BF16 载体转 HiF4 的损失差多少，BF16 carrier 投影是否会改变层级决策；
6. 这些权重域结论是否转化为输出误差、PPL 和任务准确率差异。

## 20. 2026-07-27 本轮实际完成状态

本轮完成的是权重域分析报告主线，不包含端到端 PPL、下游任务或全 64 层外推。

### 20.1 新增正式实验

- 合成数据和真实 packed NVFP4 权重均加入 `group_size={16,32,64}` 消融；
- 三种 group 均保留完整三级量化：S1P2 payload、每 8 元素指数、每 4 元素指数；
- 真实结果按 global、category、layer、tensor 四个层级聚合；
- 所有总体 NMSE 由 FP64 误差能量和参考能量合并后计算，不平均张量 NMSE。

### 20.2 正式运行配置

- Conda：`hif4`；
- PyTorch：`2.10.0+cu128`；
- device：CUDA；
- 合成数据：4 种分布，每种每次 320,000 元素，10 repeats；
- 真实 checkpoint：`Qmodel/Qwen3.5-27B-NVFP4`；
- 层：3、31、63；
- 张量：21 个；
- 权重元素：1,116,733,440。

### 20.3 group size 主要结果

真实 packed 权重：

| group | NMSE |
|---:|---:|
| 64 | 0.0070349210 |
| 32 | 0.0067972221 |
| 16 | 0.0066025754 |

- 64→32：相对降低 3.3788%；
- 32→16：相对降低 2.8636%；
- 64→16：恢复标准 g64 误差的 6.1457%；
- 边际收益递减。

合成 NVFP4 输入的 64→16 恢复比例：

- Gaussian：4.12%；
- Laplace：8.09%；
- Student-t3：15.16%；
- 0.1%×20 离群值：18.84%。

这说明顶层 64 元素共享范围确实造成损失，但其重要性高度依赖重尾和离群结构。真实权重上的总体影响明显小于 S1P2 payload 码点误差。

### 20.4 当前误差研发优先级

真实权重上的可恢复 NMSE 机会：

1. S1P2 范围内码点离散：0.0063855241；
2. group 64→16：0.0004323457；
3. hardware S0 相对连续 S0：0.0001051811；
4. BF16 中转载体：0.0000026324。

这些数值不是正交可加的误差分解，而是各控制变量实验提供的可恢复机会。现有证据支持优先研究码点感知量化、HiF4-aware 局部平滑和最终三级误差搜索，不支持把单独提高 BF16/S0 计算精度作为首要方向。

### 20.5 输出

- JSON：`ChuanCi/results/comprehensive_nvfp4_hif4/final/NVFP4_HiF4_comprehensive_results.json`；
- HTML：`ChuanCi/results/comprehensive_nvfp4_hif4/final/NVFP4_HiF4_comprehensive_report.html`。

HTML 已重构为“研究问题—实验设计—结果—机理解释—结论—算法含义”的分析报告，并加入汇报摘要、论文式实验设置、误差来源综合排序、算法指导和有效性边界。
