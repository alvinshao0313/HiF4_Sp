# 从零编写 PyTorch NVFP4 → HiF4 原生转换与真实权重评测脚本实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从空文件开始编写一个纯 PyTorch 的 NVFP4/HiF4 数值模拟与真实权重评测脚本，分别测量 NVFP4 伪量化值原生转换到 HiF4、BF16 原生转换到 HiF4 的损失，并系统完成 PTS、误差来源、分组粒度、真实模型和激活输出误差实验。

**Architecture:** 新建单一可导入、可执行脚本 `nvfp4_hif4_torch.py`，内部按“格式与舍入 → 分组与量化 → 指标 → tensor 评测 → checkpoint 评测 → 合成实验 → 报告与 CLI”分区。所有数值核心、真实权重接口和实验编排均从零实现；旧脚本既不修改、不导入，也不作为输出对齐基线。单元测试只使用格式定义、手算样例、穷举 oracle 和数学不变量判断正确性。

**Tech Stack:** Python 3.10+、PyTorch 2.2+、标准库 `argparse/csv/dataclasses/json/math/pathlib/re/statistics/typing`；`safetensors>=0.4` 仅为可选读取依赖；测试使用 `unittest`。

## Global Constraints

- 这是绿地实现计划，不是 NumPy→PyTorch 迁移计划。
- 新建两份测试。
- 数值定义只以本计划明确给出的格式、公式和不变量为准。
- 真实 NVFP4 输入是已经伪量化并解码、以 `torch.float32` 或 `torch.bfloat16` 保存的浮点 tensor；不解析 packed payload。
- `input_kind` 必须由调用者显式指定；不能依据 tensor dtype 猜测其语义。
- NVFP4 原生转换的参考始终是同一个已存在的伪量化值 \(W_{\mathrm{NV}}\)。
- BF16 原生转换的参考始终是输入显式投影到 BF16 后的值 \(W_{\mathrm{BF16}}\)。
- 不把 BF16→NVFP4 的历史误差计入任何 NVFP4→HiF4 或 BF16→HiF4 主指标。
- PTS 路径只在调用者提供真实 `pts_scale` 时计算；不得从 fake weight 的 `amax` 反推 PTS。
- direct 与 PTS 路径必须使用同一个 \(W_{\mathrm{NV}}\) 及同一个 reference energy。
- 标准 HiF4 主实验固定 `group_size=64`，沿 Linear 权重的输入/K 维，即 `group_dim=-1`。
- `group_size=16/32` 仅是分析性消融，输出中必须标注 `is_standard_hif4=false`。
- 所有量化运算使用 FP32 tensor；需要模拟 BF16 时显式执行 `.to(torch.bfloat16).to(torch.float32)`。
- 所有误差能量和点积使用 FP64 累计，再转 Python `float`。
- 真实 checkpoint 的全局指标通过累计 numerator/denominator 得到；不得平均逐层 NMSE。
- 第一个可用版本不实现 CUDA kernel、不实现 packed NVFP4、不实现训练/QAT、不依赖 Transformers。
- 合成实验的随机样本先在 CPU 生成，再搬到目标 device，保证 CPU/CUDA 使用相同输入。
- 输出必须同时保存原始能量、比例型指标、配置、seed、device、dtype 和跳过原因。

---

## 1. 数学口径与待回答问题

### 1.1 NVFP4 伪量化值的三条 HiF4 评测路径

已存在的 fake-quant 数值写成：

\[
W_{\mathrm{NV}}
=
s_T\bar W_{\mathrm{NV}},
\qquad
\bar W_{\mathrm{NV}}
=
s_{\mathrm{E4M3}}q_{\mathrm{E2M1}}.
\]

`W_NV` 可以保存为 FP32 或 BF16；容器 dtype 不改变它的
`input_kind="nvfp4_fake"` 语义。

必须评测：

1. 完整解码值直接转换：

   \[
   \widehat W_{\mathrm{direct}}
   =
   Q_{\mathrm{HiF4}}(W_{\mathrm{NV}}).
   \]

2. 提出真实 PTS，内部保持 FP32：

   \[
   \widehat W_{\mathrm{PTS\text{-}FP32}}
   =
   s_TQ_{\mathrm{HiF4}}(W_{\mathrm{NV}}/s_T).
   \]

3. 提出真实 PTS，内部先投影到 BF16：

   \[
   \boxed{
   \widehat W_{\mathrm{PTS\text{-}BF16}}
   =
   s_TQ_{\mathrm{HiF4}}
   \left(
   \operatorname{BF16}(W_{\mathrm{NV}}/s_T)
   \right)
   }.
   \]

路径 2 用来隔离 BF16 carrier 的贡献；路径 3 是用户重点关注的实际方案。
三条路径都相对原来的 \(W_{\mathrm{NV}}\) 计算误差。

### 1.2 BF16 原生转换

对于独立高精度输入 \(W\)：

\[
W_{\mathrm{BF16}}
=
\operatorname{BF16}(W),
\qquad
\widehat W_{\mathrm{BF16}}
=
Q_{\mathrm{HiF4}}(W_{\mathrm{BF16}}).
\]

若调用者传入 FP32，函数先将其投影为 BF16，并把投影后的数值作为
reference。因此指标只包含 BF16→HiF4 的转换损失，不包含 FP32→BF16
的 storage loss。

### 1.3 指标

对 reference \(R\) 和 reconstruction \(\widehat R\)：

\[
\mathrm{NMSE}
=
\frac{\|\widehat R-R\|_F^2}{\|R\|_F^2},
\qquad
\mathrm{NRMSE}
=
\sqrt{\mathrm{NMSE}},
\]

\[
\mathrm{CosSim}
=
\frac{\langle R,\widehat R\rangle}
{\|R\|_F\|\widehat R\|_F},
\qquad
\mathrm{SQNR}
=
10\log_{10}
\frac{\|R\|_F^2}{\|\widehat R-R\|_F^2}.
\]

每次评测必须先保存：

```text
numel
reference_energy
approximation_energy
error_energy
dot
absolute_error_sum
max_absolute_error
```

再派生：

```text
nmse
nrmse
cosine
sqnr_db
mae
max_absolute_error
```

全零约定：

- reference 和 reconstruction 都全零：`nmse=0`、`nrmse=0`、
  `cosine=1`、`sqnr_db="inf"`；
- reference 全零但 reconstruction 非零：`nmse="inf"`、
  `cosine=0`；
- JSON 不允许写非标准的 `Infinity/NaN`，使用字符串 `"inf"` 或 `null`
  加状态字段。

### 1.4 PTS 配对指标

\[
\Delta_{\mathrm{PTS}}
=
\mathrm{NMSE}_{\mathrm{PTS\text{-}BF16}}
-
\mathrm{NMSE}_{\mathrm{direct}},
\]

\[
\mathrm{RelativeChange}_{\mathrm{PTS}}
=
\frac{
\mathrm{NMSE}_{\mathrm{PTS\text{-}BF16}}
-
\mathrm{NMSE}_{\mathrm{direct}}
}{
\mathrm{NMSE}_{\mathrm{direct}}
}.
\]

`delta < 0` 表示提出 PTS 更好，`delta > 0` 表示 direct 更好。计划不得
预设哪条路径一定获胜。

---

## 2. 格式定义与从零实现算法

### 2.1 NVFP4 合成参考生成器

真实权重接口不会重新量化输入；本生成器只用于合成实验构造合法
NVFP4 fake-quant reference。

- 每个 tensor 一个 FP32 PTS；
- 每 16 个值共享一个正 E4M3FN block scale；
- payload 为带符号 E2M1；
- E2M1 正幅值码本：

  \[
  \{0,0.5,1,1.5,2,3,4,6\}.
  \]

给定高精度 \(x\)，合成路径固定为：

```text
tensor_amax = max(abs(x))
s_T = float32(tensor_amax / (448 * 6))  # 非零张量
block_amax = max(abs(block_16))
raw_block_scale = block_amax / (6 * s_T)
block_scale = RNE_E4M3FN(raw_block_scale)
effective_scale = s_T * block_scale
payload = sign(x) * RNE_E2M1(abs(x) / effective_scale)
W_NV = effective_scale * payload
```

全零 tensor 规定 `s_T=1`、block scale/payload/reference 全零。

### 2.2 E4M3FN 正 scale 码本

枚举非负 7-bit 编码 `0..126`，排除 `127`：

```text
exponent_code == 0:
    value = mantissa * 2^-9
otherwise:
    value = (1 + mantissa / 8) * 2^(exponent_code - 7)
```

应得到 127 个升序值：

```text
min = 0
smallest_positive = 2^-9
max = 448
```

### 2.3 HiF4 顶层 E6M2 scale 码本

枚举无符号编码 `0..254`，排除保留编码 `255`：

```text
value = (1 + mantissa / 4) * 2^(exponent_code - 48)
```

应得到 255 个升序正值：

```text
min = 2^-48
max = 1.5 * 2^15
```

### 2.4 通用 RNE 码本舍入

对非负输入：

1. 使用 `torch.searchsorted` 找到相邻低/高码点；
2. 越界时饱和；
3. 距离更近者获胜；
4. 完全等距时选择原始整数 code 最低位为 0 的码点；
5. 不依赖 `torch.float8_e4m3fn`，避免设备支持差异。

### 2.5 HiF4 分层量化

对每个 64 元素标准 group：

1. 计算每 4、每 8、每 64 元素的局部绝对最大值；
2. 顶层 scale：

   ```text
   continuous:
       S0 = amax64 / 7
       reciprocal = 1 / S0

   bf16_math:
       S0 = BF16(amax64 * BF16(1/7))
       reciprocal = BF16(1 / S0)

   e6m2_only:
       S0 = E6M2(amax64 / 7)
       reciprocal = 1 / S0

   hardware:
       S0 = E6M2(BF16(amax64 * BF16(1/7)))
       reciprocal = BF16(1 / S0)
   ```

3. 第一层 micro-exponent：

   \[
   e^{(8)} =
   \mathbf 1[\mathrm{amax}_8/S_0\ge4].
   \]

4. 第二层 micro-exponent：

   \[
   e^{(4)} =
   \mathbf 1[
   \mathrm{amax}_4/(S_0 2^{e^{(8)}})\ge2
   ].
   \]

5. 每个值的 local scale：

   \[
   S_i=S_0\,2^{e_i^{(8)}+e_i^{(4)}}.
   \]

6. S1P2 payload 正幅值：

   \[
   p_i
   =
   \min
   \left(
   1.75,
   \frac{\lfloor 4|x_i|/S_i+0.5\rfloor}{4}
   \right).
   \]

7. 重建：

   \[
   \widehat x_i=\operatorname{sign}(x_i)S_ip_i.
   \]

全零 group 必须显式返回零 reconstruction，不允许因为 `S0=0` 产生
NaN/Inf。

---

## 3. 新建文件与职责

### 必须新建

- `nvfp4_hif4_torch.py`
  - 定义所有 dataclass、码本、舍入、NVFP4 合成器、HiF4 量化器；
  - 提供真实 tensor/checkpoint API；
  - 实现 E0–E8 合成与权重实验；
  - 提供可选 E9 激活输出误差；
  - 提供 CLI、JSON/CSV/Markdown 输出。

- `tests/test_nvfp4_hif4_torch_core.py`
  - 独立验证码本、RNE、BF16、分组、NVFP4 合成器和 HiF4 核心；
  - 使用手算/穷举 oracle，不读取旧脚本。

- `tests/test_nvfp4_hif4_torch_eval.py`
  - 验证指标、NV/BF16 tensor API、PTS、checkpoint loader、聚合、
    activation output error、CLI 与输出 schema。

### 只在仓库没有依赖文件时新建

- `requirements-hif4-torch.txt`

  ```text
  torch>=2.2
  safetensors>=0.4
  ```

  `safetensors` 在代码中仍须延迟导入；只评测 `.pt/.pth` 时不能因为未安装
  `safetensors` 而失败。

### 明确不触碰

```text
reproduce_nvfp4_to_hif4.py
tests/test_reproduce_nvfp4_to_hif4.py
docs/superpowers/plans/2026-07-23-nvfp4-hif4-reproduction.md
docs/superpowers/plans/2026-07-23-pytorch-nvfp4-hif4-native-conversion-experiments.md
```

---

## 4. 最终公共 API

脚本必须暴露以下数据结构：

```python
@dataclass(frozen=True)
class HiF4Config:
    group_size: int = 64
    group_dim: int = -1
    scale_mode: str = "hardware"
    compute_dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260723
    samples_per_repeat: int = 320_000
    repeats: int = 10
    phase_points: int = 257
    phase_seed: int = 7


@dataclass
class NVFP4SimulationResult:
    values: torch.Tensor
    global_scale: torch.Tensor
    block_scales: torch.Tensor
    payload: torch.Tensor


@dataclass
class HiF4Result:
    values: torch.Tensor
    top_scale: torch.Tensor
    e1_per_8: torch.Tensor
    e1_per_4: torch.Tensor
    payload_magnitude: torch.Tensor
    local_scale: torch.Tensor


@dataclass
class ErrorSums:
    numel: int = 0
    reference_energy: float = 0.0
    approximation_energy: float = 0.0
    error_energy: float = 0.0
    dot: float = 0.0
    absolute_error_sum: float = 0.0
    max_absolute_error: float = 0.0
```

公共函数签名固定为：

```text
build_e4m3fn_codebook()
    -> tuple[torch.Tensor, torch.Tensor]

build_e6m2_codebook()
    -> tuple[torch.Tensor, torch.Tensor]

round_positive_to_codebook(
    values: torch.Tensor,
    codebook_values: torch.Tensor,
    codebook_codes: torch.Tensor,
) -> torch.Tensor

round_bfloat16(values: torch.Tensor) -> torch.Tensor

simulate_nvfp4(
    values: torch.Tensor,
    *,
    block_dim: int = -1,
) -> NVFP4SimulationResult

quantize_hif4(
    values: torch.Tensor,
    *,
    config: HiF4Config = HiF4Config(),
) -> HiF4Result

compute_error_sums(
    reference: torch.Tensor,
    approximation: torch.Tensor,
) -> ErrorSums

merge_error_sums(
    destination: ErrorSums,
    source: ErrorSums,
) -> ErrorSums

finalize_error_metrics(
    sums: ErrorSums,
) -> dict[str, float | int | str | None]

evaluate_nvfp4_fake_weight(
    weight: torch.Tensor,
    *,
    pts_scale: torch.Tensor | float | None = None,
    hif4_config: HiF4Config = HiF4Config(),
    return_reconstructions: bool = False,
) -> dict[str, Any]

evaluate_bf16_weight(
    weight: torch.Tensor,
    *,
    hif4_config: HiF4Config = HiF4Config(),
    return_reconstruction: bool = False,
) -> dict[str, Any]

evaluate_output_error(
    activations: torch.Tensor,
    reference_weight: torch.Tensor,
    approximation_weight: torch.Tensor,
    *,
    token_batch_size: int = 256,
) -> dict[str, float | int | str | None]

iter_checkpoint_tensors(
    checkpoint_path: Path,
) -> Iterator[tuple[str, torch.Tensor]]

load_pts_scales(
    path: Path | None,
) -> dict[str, torch.Tensor | float]

evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    input_kind: Literal["nvfp4_fake", "bf16"],
    pts_scales_path: Path | None,
    include_regex: str | None,
    exclude_regex: str | None,
    device: torch.device,
    hif4_config: HiF4Config,
    chunk_groups: int = 16_384,
    require_pts: bool = False,
    tensor_names: tuple[str, ...] = (),
    max_tensors: int | None = None,
    activations_path: Path | None = None,
    token_batch_size: int = 256,
) -> dict[str, Any]

run_simulation(
    config: ExperimentConfig,
    *,
    device: torch.device,
    quick: bool = False,
) -> dict[str, Any]

main(argv: list[str] | None = None) -> int
```

不得因为输入 dtype 是 BF16 就自动调用 `evaluate_bf16_weight`。语义由
调用者选择的 API 或 CLI `--input-kind` 决定。

---

## 5. 实验矩阵

| 编号 | 要回答的问题 | 数据 | 主对照 | 输出 |
|---|---|---|---|---|
| E0 | 从零实现的格式、舍入、分组和设备行为是否正确 | 手算、穷举、随机小 tensor | 独立 oracle/不变量 | 单测与数值验证报告 |
| E1 | NVFP4-native 与 BF16-native 各自转 HiF4 的损失是多少 | 四种合成分布 | 各自 reference | NMSE/NRMSE/Cos/SQNR |
| E2 | 同一个 \(W_{\mathrm{NV}}\) 上提出 PTS 是否优于 direct | E1 的 NV reference | direct、PTS-FP32、PTS-BF16 | 配对差值与 CI |
| E3 | BF16 carrier 是否精确，若不精确贡献多少 | 码本穷举、E1、真实权重 | PTS-FP32 vs PTS-BF16 | carrier NMSE/code equal |
| E4 | PTS 尾数相位如何改变 E6M2 网格 | 固定归一化 NV code × 257 phases | 三条 NV 路径 | phase-NMSE 曲线 |
| E5 | 误差主要来自 BF16 scale math、E6M2 还是层级 payload | E1 sources | 4 个 `scale_mode` | 分解表 |
| E6 | 16/32/64 group 带来多少误差变化 | E1 sources | group size × scale mode | 分组消融表 |
| E7 | 同一 NV fake 值以 FP32/BF16 保存有何影响 | 同一 NV reference 两种容器 | storage projection + conversion | 两阶段损失表 |
| E8 | 合成结论能否在真实模型权重重现 | BF16/NV fake checkpoints | 逐层/类别/全局 | 主模型表与曲线 |
| E9 | weight NMSE 是否对应线性层输出误差 | 真实激活与 E8 权重 | weight vs output NMSE | 相关性与反例 |
| E10 | 转换损失是否影响 PPL/任务准确率 | 完整模型 | 各自 source baseline | 可选端到端结果 |

---

## 6. 统一实验设置

### 6.1 合成实验

| 配置 | 完整运行 | 快速冒烟 |
|---|---:|---:|
| 主 seed | `20260723` | `20260723` |
| 每个 distribution/repeat 的 base 样本数 | `320_000` | `6_400` |
| repeats | `10` | `1` |
| NVFP4 block size | `16` | `16` |
| 标准 HiF4 group size | `64` | `64` |
| phase points | `257` | `17` |
| phase seed | `7` | `7` |
| 量化 dtype | FP32 | FP32 |
| 累计 dtype | FP64 | FP64 |
| 样本生成设备 | CPU | CPU |

每次 repeat 的 base tensor 一次生成，然后由它独立得到：

```text
W_NV = simulate_nvfp4(W_base).values
W_BF16 = BF16(W_base)
```

E1 的两项指标分别相对 `W_NV` 与 `W_BF16`；这只是共享随机底样本以降低
方差，不代表计算 BF16→NVFP4→HiF4 级联。

seed 固定为：

```text
distribution_seed = 20260723 + repeat_index
outlier_index_seed = 20260723 + 1000 + repeat_index
phase_value_seed   = 7
```

### 6.2 四种输入分布

1. Gaussian：

   \[
   x\sim\mathcal N(0,1).
   \]

2. Laplace：

   \[
   x\sim\operatorname{Laplace}(0,1/\sqrt2).
   \]

3. Student-t3：

   \[
   x=t_3/\sqrt3.
   \]

4. `Outlier0.1pct20x`：

   - 先生成标准 Gaussian；
   - 用独立固定 permutation 精确选择
     `round(0.001 * samples_per_repeat)` 个位置；
   - 这些位置乘 20；
   - 不用 Bernoulli mask，保证每次 outlier 数一致。

样本数、block size、group size 均可整除；不要为凑整而静默截断。

### 6.3 repeat 汇总

每个 repeat 保存每条路径的 `ErrorSums`。最终同时报告：

- 所有 repeat 合并后的 energy-weighted NMSE：主结果；
- repeat NMSE 的算术均值；
- repeat 标准差；
- 95% t 置信区间：

  \[
  \bar x\pm2.262\,s/\sqrt{10}.
  \]

E2 的差值必须是同一 repeat 内的配对差：

```text
paired_delta_r = nmse_pts_bf16_r - nmse_direct_r
```

再对 10 个 `paired_delta_r` 计算均值、标准差和 95% CI。

### 6.4 真实权重

| 项目 | 默认值 |
|---|---|
| `group_size` | `64` |
| `group_dim` | `-1` |
| `chunk_groups` | `16_384` |
| 设备 | CUDA 可用时 `cuda`，否则 `cpu` |
| tensor 范围 | 浮点且 `ndim>=2` |
| 主类别 | `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` |
| 单独报告 | `embed_tokens/lm_head/other` |
| 默认跳过 | 1D norm/bias、非浮点、分组维度不能整除 64 |
| 随机性 | 无 |

最低完整实验：

- 同一模型架构的一份 BF16 checkpoint；
- 一份对应的 NVFP4 fake-quant checkpoint；
- 至少覆盖七类 Linear；
- 有真实 PTS mapping 时评测 PTS，否则仅 direct；
- 双方对照表只取参数名与 shape 均匹配的交集；
- 每条指标仍以各自 native reference 为准。

论文级推荐：

- 主模型：Qwen3-8B；
- 再增加一个不同架构的 7B/8B 模型；
- 主表不混入 embedding/lm_head；
- embedding/lm_head 作为附加实验单独报告。

---

## 7. 结果 schema

合成结果：

```json
{
  "schema_version": 1,
  "implementation": "greenfield_torch",
  "run_kind": "simulation",
  "config": {},
  "conventions": {
    "nvfp4_reference": "decoded_fake_quantized_value",
    "bf16_reference": "value_after_bfloat16_projection",
    "direct_path": "Q_hif4(W_nv)",
    "pts_fp32_path": "s_T * Q_hif4(W_nv / s_T)",
    "pts_bf16_path": "s_T * Q_hif4(BF16(W_nv / s_T))",
    "hif4_standard_group_size": 64,
    "group_dim": -1
  },
  "experiments": {
    "e0_correctness": {},
    "e1_native_source": {},
    "e2_pts_paths": {},
    "e3_bf16_carrier": {},
    "e4_phase_sweep": {},
    "e5_scale_decomposition": {},
    "e6_group_size": {},
    "e7_storage_dtype": {}
  }
}
```

checkpoint 结果：

```json
{
  "schema_version": 1,
  "implementation": "greenfield_torch",
  "run_kind": "checkpoint",
  "checkpoint": "/path",
  "input_kind": "nvfp4_fake",
  "config": {},
  "global": {},
  "categories": {},
  "tensors": {
    "model.layers.0.self_attn.q_proj.weight": {
      "shape": [4096, 4096],
      "storage_dtype": "torch.float32",
      "category": "q_proj",
      "direct": {},
      "pts_fp32": {},
      "pts_bf16": {},
      "pts_delta": {},
      "status": "evaluated"
    }
  },
  "skipped": {}
}
```

未提供 PTS 时：

```text
direct = 正常指标
pts_fp32 = null
pts_bf16 = null
pts_status = "not_provided"
```

不得用 0 填充缺失路径。

---

## Task 1: 建立完全独立的 Torch 脚本骨架与语义红灯测试

**Files:**
- Create: `nvfp4_hif4_torch.py`
- Create: `tests/test_nvfp4_hif4_torch_core.py`
- Create: `tests/test_nvfp4_hif4_torch_eval.py`

**Interfaces:**
- Consumes: 本计划第 1–7 节，不依赖任何现有实现。
- Produces: 可导入的新模块、精确失败的新 API 测试、CLI 骨架。

- [ ] **Step 0: 固定两份测试文件的 class 结构**

`tests/test_nvfp4_hif4_torch_core.py`：

```text
GreenfieldModuleTests
CodebookAndRoundingTests
HiF4CoreTests
NVFP4SimulationTests
```

`tests/test_nvfp4_hif4_torch_eval.py`：

```text
NativeReferenceTests
MetricTests
TensorEvaluationTests
SimulationExperimentTests
CheckpointLoadingTests
CheckpointEvaluationTests
OutputErrorTests
CLITests
```

后续命令中的 unittest dotted path 必须使用以上名称，不临时改名。

两份文件的固定 imports：

```python
# core
import importlib
import unittest

import torch

# eval
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
```

- [ ] **Step 1: 创建核心测试文件并验证导入目标是新模块**

加入：

```python
import importlib
import unittest

import torch

module = importlib.import_module("nvfp4_hif4_torch")


class GreenfieldModuleTests(unittest.TestCase):
    def test_imports_greenfield_module(self) -> None:
        self.assertEqual(module.__name__, "nvfp4_hif4_torch")
        self.assertTrue(hasattr(module, "HiF4Config"))
        self.assertTrue(hasattr(module, "quantize_hif4"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 创建评测语义红灯测试**

测试必须明确 reference：

```python
class NativeReferenceTests(unittest.TestCase):
    def test_nvfp4_paths_share_same_reference(self) -> None:
        reference = torch.tensor(
            [[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0] * 8],
            dtype=torch.float32,
        )
        result = module.evaluate_nvfp4_fake_weight(
            reference,
            pts_scale=0.25,
            return_reconstructions=True,
        )
        torch.testing.assert_close(
            result["reference"],
            reference,
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            set(result["paths"]),
            {"direct", "pts_fp32", "pts_bf16"},
        )

    def test_bf16_uses_post_cast_reference(self) -> None:
        fp32 = torch.tensor(
            [[1.001, -0.997, 0.333, -0.125] * 16],
            dtype=torch.float32,
        )
        expected = fp32.to(torch.bfloat16).to(torch.float32)
        result = module.evaluate_bf16_weight(
            fp32,
            return_reconstruction=True,
        )
        torch.testing.assert_close(
            result["reference"],
            expected,
            rtol=0,
            atol=0,
        )
```

- [ ] **Step 3: 创建缺失 PTS 红灯测试**

```python
def test_missing_pts_never_gets_inferred(self) -> None:
    weight = torch.linspace(-1, 1, 64).reshape(1, 64)
    result = module.evaluate_nvfp4_fake_weight(weight)
    self.assertIsNotNone(result["paths"]["direct"])
    self.assertIsNone(result["paths"]["pts_fp32"])
    self.assertIsNone(result["paths"]["pts_bf16"])
    self.assertEqual(result["pts_status"], "not_provided")
```

- [ ] **Step 4: 运行测试确认先失败**

Run:

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_core \
  tests.test_nvfp4_hif4_torch_eval -v
```

Expected: 因 `nvfp4_hif4_torch` 尚不存在而 FAIL；不得因测试语法错误失败。

- [ ] **Step 5: 从空文件创建最小模块骨架**

只加入 imports、dataclass 和准确签名；未实现函数统一抛出：

```python
raise NotImplementedError("greenfield API contract")
```

`main()` 暂时只创建空 parser 并返回 0。不要在此任务复制旧脚本内容。

- [ ] **Step 6: 运行编译与导入检查**

```bash
python -m py_compile \
  nvfp4_hif4_torch.py \
  tests/test_nvfp4_hif4_torch_core.py \
  tests/test_nvfp4_hif4_torch_eval.py
```

Expected: exit code 0。

- [ ] **Step 7: 提交**

```bash
git add nvfp4_hif4_torch.py \
        tests/test_nvfp4_hif4_torch_core.py \
        tests/test_nvfp4_hif4_torch_eval.py
git commit -m "test: define greenfield torch conversion contract"
```

---

## Task 2: 从格式定义实现码本、RNE 与 BF16 carrier

**Files:**
- Modify: `nvfp4_hif4_torch.py`
- Modify: `tests/test_nvfp4_hif4_torch_core.py`

**Interfaces:**
- Produces: `build_e4m3fn_codebook`、`build_e6m2_codebook`、
  `round_positive_to_codebook`、`round_bfloat16`、
  `quantize_e2m1_magnitude`。

- [ ] **Step 1: 写 E4M3FN 码本红灯测试**

```python
def test_e4m3fn_positive_codebook(self) -> None:
    values, codes = module.build_e4m3fn_codebook()
    self.assertEqual(values.dtype, torch.float32)
    self.assertEqual(codes.dtype, torch.int16)
    self.assertEqual(values.numel(), 127)
    self.assertEqual(values[0].item(), 0.0)
    self.assertEqual(values[1].item(), 2.0**-9)
    self.assertEqual(values[-1].item(), 448.0)
    self.assertTrue(torch.all(values[1:] > values[:-1]).item())
    torch.testing.assert_close(
        codes,
        torch.arange(127, dtype=torch.int16),
        rtol=0,
        atol=0,
    )
```

- [ ] **Step 2: 写 E6M2 码本红灯测试**

```python
def test_e6m2_unsigned_scale_codebook(self) -> None:
    values, codes = module.build_e6m2_codebook()
    self.assertEqual(values.numel(), 255)
    self.assertEqual(values[0].item(), 2.0**-48)
    self.assertEqual(values[-1].item(), 1.5 * 2.0**15)
    self.assertEqual(codes[0].item(), 0)
    self.assertEqual(codes[-1].item(), 254)
    self.assertTrue(torch.all(values[1:] > values[:-1]).item())
```

- [ ] **Step 3: 写通用 RNE 与饱和测试**

使用人工码本：

```python
def test_round_positive_to_codebook_uses_even_code_on_ties(self) -> None:
    values = torch.tensor([0.0, 1.0, 2.0, 4.0])
    codes = torch.tensor([0, 1, 2, 3], dtype=torch.int16)
    x = torch.tensor([-1.0, 0.5, 1.5, 3.0, 8.0])
    expected = torch.tensor([0.0, 0.0, 2.0, 2.0, 4.0])
    actual = module.round_positive_to_codebook(x, values, codes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

- [ ] **Step 4: 写 E2M1 合法值与 midpoint 测试**

```python
def test_e2m1_midpoints(self) -> None:
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    expected = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    actual = module.quantize_e2m1_magnitude(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

- [ ] **Step 5: 写 BF16 carrier 精确实现测试**

```python
def test_round_bfloat16_matches_native_cast(self) -> None:
    values = torch.tensor(
        [0.0, -0.0, 1.0, 1.001, -3.1415926, 2.0**-120, 2.0**120],
        dtype=torch.float32,
    )
    expected = values.to(torch.bfloat16).to(torch.float32)
    actual = module.round_bfloat16(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

- [ ] **Step 6: 运行红灯**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_core.CodebookAndRoundingTests -v
```

Expected: 新函数仍抛 `NotImplementedError`。

- [ ] **Step 7: 实现码本构造**

在新脚本内直接按第 2 节公式枚举；返回 CPU FP32 values 与 INT16 codes。
模块加载时构建一次：

```python
E4M3FN_VALUES, E4M3FN_CODES = build_e4m3fn_codebook()
E6M2_VALUES, E6M2_CODES = build_e6m2_codebook()
E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
E2M1_CODES = torch.arange(8, dtype=torch.int16)
```

- [ ] **Step 8: 实现 device-aware RNE**

算法固定为：

```python
x = values.to(torch.float32).clamp_min(0)
book = codebook_values.to(device=x.device, dtype=torch.float32)
codes = codebook_codes.to(device=x.device)
hi = torch.searchsorted(book, x)
hi = hi.clamp(0, book.numel() - 1)
lo = (hi - 1).clamp(0, book.numel() - 1)
d_lo = x - book[lo]
d_hi = book[hi] - x
choose_hi = d_hi < d_lo
tie = d_hi == d_lo
choose_hi |= tie & ((codes[hi] & 1) == 0)
return torch.where(choose_hi, book[hi], book[lo])
```

添加有限性检查；负数按 0 处理，因为该函数只处理 magnitude/scale。

- [ ] **Step 9: 实现 BF16 carrier**

```python
def round_bfloat16(values: torch.Tensor) -> torch.Tensor:
    if not values.is_floating_point():
        raise TypeError("values must be floating point")
    return values.to(torch.bfloat16).to(torch.float32)
```

- [ ] **Step 10: 跑测试并提交**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_core.CodebookAndRoundingTests -v
git add nvfp4_hif4_torch.py tests/test_nvfp4_hif4_torch_core.py
git commit -m "feat: implement torch format codebooks and rounding"
```

Expected: 全部 PASS。

---

## Task 3: 从零实现通用分组工具与 HiF4 核心

**Files:**
- Modify: `nvfp4_hif4_torch.py`
- Modify: `tests/test_nvfp4_hif4_torch_core.py`

**Interfaces:**
- Consumes: Task 2 的码本/RNE/BF16。
- Produces: `HiF4Config` 验证、内部 group reshape 工具、
  `quantize_hif4`。

- [ ] **Step 1: 写最后一维分组 shape 测试**

```python
def test_hif4_groups_last_dimension_without_crossing_rows(self) -> None:
    x = torch.randn(3, 128, generator=torch.Generator().manual_seed(1))
    result = module.quantize_hif4(
        x,
        config=module.HiF4Config(group_size=64, group_dim=-1),
    )
    self.assertEqual(result.values.shape, x.shape)
    self.assertEqual(result.top_scale.shape, (3, 2))
    self.assertEqual(result.e1_per_8.shape, (3, 2, 8))
    self.assertEqual(result.e1_per_4.shape, (3, 2, 16))
```

- [ ] **Step 2: 写任意 `group_dim` 测试**

```python
def test_hif4_groups_dimension_zero(self) -> None:
    x = torch.randn(128, 3, generator=torch.Generator().manual_seed(2))
    result = module.quantize_hif4(
        x,
        config=module.HiF4Config(group_size=64, group_dim=0),
    )
    self.assertEqual(result.values.shape, x.shape)
```

- [ ] **Step 3: 写非法输入测试**

逐项断言抛 `TypeError` 或 `ValueError`：

```text
integer tensor
NaN/Inf
group_size < 8
group_size % 8 != 0
grouped dimension length % group_size != 0
group_dim out of range
compute_dtype != torch.float32
scale_mode not in {continuous,bf16_math,e6m2_only,hardware}
```

- [ ] **Step 4: 写零值与 payload 合法性测试**

```python
def test_hif4_zero_and_payload_domain(self) -> None:
    zero = torch.zeros(2, 64)
    zero_result = module.quantize_hif4(zero)
    self.assertTrue(torch.equal(zero_result.values, zero))
    self.assertTrue(torch.isfinite(zero_result.local_scale).all().item())

    x = torch.linspace(-7, 7, 128).reshape(2, 64)
    result = module.quantize_hif4(x)
    payload = result.payload_magnitude
    self.assertTrue(torch.all(payload >= 0).item())
    self.assertTrue(torch.all(payload <= 1.75).item())
    self.assertTrue(torch.equal(payload * 4, torch.round(payload * 4)))
    self.assertTrue(torch.all((result.e1_per_8 == 0) | (result.e1_per_8 == 1)).item())
    self.assertTrue(torch.all((result.e1_per_4 == 0) | (result.e1_per_4 == 1)).item())
```

- [ ] **Step 5: 写手算 group oracle**

构造 64 个值全部为 1：

```python
def test_hif4_constant_one_group(self) -> None:
    x = torch.ones(64)
    result = module.quantize_hif4(
        x,
        config=module.HiF4Config(scale_mode="continuous"),
    )
    torch.testing.assert_close(
        result.top_scale,
        torch.tensor([1.0 / 7.0], dtype=torch.float32),
        rtol=1e-6,
        atol=0,
    )
    self.assertTrue(torch.equal(result.e1_per_8, torch.ones_like(result.e1_per_8)))
    self.assertTrue(torch.equal(result.e1_per_4, torch.ones_like(result.e1_per_4)))
    torch.testing.assert_close(
        result.values,
        torch.ones_like(x),
        rtol=1e-6,
        atol=1e-7,
    )
```

手算依据是 \(1/(1/7)=7\ge4\)，因此第一层 E1 为 1；再有
\(1/((1/7)\times2)=3.5\ge2\)，因此第二层 E1 也为 1。local scale
为 \(4/7\)，S1P2 payload 为 1.75，最终恰好重建 1。

- [ ] **Step 6: 写连续 scale 等变性测试**

```python
def test_continuous_mode_is_scale_equivariant(self) -> None:
    x = torch.randn(4, 64, generator=torch.Generator().manual_seed(3))
    factor = torch.tensor(1.371)
    base = module.quantize_hif4(
        x,
        config=module.HiF4Config(scale_mode="continuous"),
    )
    scaled = module.quantize_hif4(
        x * factor,
        config=module.HiF4Config(scale_mode="continuous"),
    )
    torch.testing.assert_close(
        scaled.values,
        base.values * factor,
        rtol=2e-6,
        atol=1e-7,
    )
```

- [ ] **Step 7: 运行红灯**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_core.HiF4CoreTests -v
```

- [ ] **Step 8: 实现无跨行分组工具**

固定数据流：

```python
normalized_dim = group_dim % values.ndim
moved = values.movedim(normalized_dim, -1).contiguous()
moved_shape = moved.shape
groups_per_row = moved_shape[-1] // group_size
groups = moved.reshape(-1, group_size)
# quantize groups
restored = reconstructed.reshape(moved_shape)
restored = restored.movedim(-1, normalized_dim)
```

metadata 重新 reshape 成：

```text
leading moved dimensions + groups_per_row + local metadata dimensions
```

不得对整个二维矩阵直接 flatten 后分组。

- [ ] **Step 9: 实现四种顶层 scale mode**

严格执行第 2.5 节。对全零 group：

```python
nonzero = peak_per_group > 0
safe_scale = torch.where(nonzero, computed_scale, torch.ones_like(computed_scale))
```

最终 reconstruction 对零 group 覆盖为 0。

- [ ] **Step 10: 实现两级 E1 与 S1P2**

以 `(num_groups, group_size)` 内部 shape 计算；使用 `repeat_interleave`
扩展 8→4→element，不使用 Python element 循环。

- [ ] **Step 11: 跑测试、确定性检查并提交**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_core.HiF4CoreTests -v
python - <<'PY'
import torch
import nvfp4_hif4_torch as m
x = torch.randn(8, 128, generator=torch.Generator().manual_seed(9))
a = m.quantize_hif4(x)
b = m.quantize_hif4(x)
assert torch.equal(a.values, b.values)
assert torch.equal(a.e1_per_8, b.e1_per_8)
assert torch.equal(a.e1_per_4, b.e1_per_4)
PY
git add nvfp4_hif4_torch.py tests/test_nvfp4_hif4_torch_core.py
git commit -m "feat: implement greenfield hif4 quantizer"
```

---

## Task 4: 从零实现仅供合成实验使用的 NVFP4 生成器

**Files:**
- Modify: `nvfp4_hif4_torch.py`
- Modify: `tests/test_nvfp4_hif4_torch_core.py`

**Interfaces:**
- Consumes: Task 2 的 E4M3/E2M1 舍入与 Task 3 的分组方式。
- Produces: `simulate_nvfp4`。

- [ ] **Step 1: 写 shape、scale 与合法 payload 测试**

```python
def test_simulate_nvfp4_returns_legal_fake_quant_values(self) -> None:
    x = torch.randn(3, 32, generator=torch.Generator().manual_seed(4))
    result = module.simulate_nvfp4(x)
    self.assertEqual(result.values.shape, x.shape)
    self.assertEqual(result.payload.shape, x.shape)
    self.assertEqual(result.block_scales.shape, (3, 2))
    self.assertEqual(result.global_scale.ndim, 0)
    legal = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    for value in result.payload.abs().unique():
        self.assertTrue(torch.any(legal == value).item())
```

- [ ] **Step 2: 写解码恒等测试**

```python
def test_nvfp4_values_equal_scale_times_payload(self) -> None:
    x = torch.randn(2, 32, generator=torch.Generator().manual_seed(5))
    result = module.simulate_nvfp4(x)
    effective = (
        result.block_scales.unsqueeze(-1)
        * result.global_scale
    ).expand(2, 2, 16).reshape_as(x)
    expected = effective * result.payload
    torch.testing.assert_close(result.values, expected, rtol=0, atol=0)
```

- [ ] **Step 3: 写全零和非法输入测试**

全零输入应得到：

```text
global_scale = 1
block_scales = 0
payload = 0
values = 0
```

拒绝非浮点、非有限、分组维度不能被 16 整除。

- [ ] **Step 4: 写非最后维度测试**

对 shape `[32, 3]`、`block_dim=0`，结果 shape 必须保持 `[32, 3]`，每列
独立分 block。

- [ ] **Step 5: 运行红灯**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_core.NVFP4SimulationTests -v
```

- [ ] **Step 6: 按第 2.1 节从零实现**

关键约束：

```python
tensor_amax = values.float().abs().amax()
if tensor_amax == 0:
    global_scale = torch.tensor(1.0, device=values.device)
else:
    global_scale = (tensor_amax / (448.0 * 6.0)).to(torch.float32)
    if global_scale == 0:
        global_scale = torch.nextafter(
            torch.tensor(0.0, device=values.device),
            torch.tensor(1.0, device=values.device),
        )
```

block scale 与 payload 全部 tensor 化，不对 block 写 Python 循环。

- [ ] **Step 7: 加入防误用文档**

`simulate_nvfp4` docstring 明确写：

```text
Only generates synthetic NVFP4 references for controlled experiments.
Never call this function inside evaluate_nvfp4_fake_weight or
evaluate_checkpoint(input_kind="nvfp4_fake").
```

- [ ] **Step 8: 跑测试并提交**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_core.NVFP4SimulationTests -v
git add nvfp4_hif4_torch.py tests/test_nvfp4_hif4_torch_core.py
git commit -m "feat: add independent nvfp4 simulation source"
```

---

## Task 5: 实现 FP64 指标与两个真实 tensor API

**Files:**
- Modify: `nvfp4_hif4_torch.py`
- Modify: `tests/test_nvfp4_hif4_torch_eval.py`

**Interfaces:**
- Consumes: Task 3 的 `quantize_hif4`。
- Produces: `ErrorSums`、`compute_error_sums`、`merge_error_sums`、
  `finalize_error_metrics`、`evaluate_nvfp4_fake_weight`、
  `evaluate_bf16_weight`。

- [ ] **Step 1: 写手算指标测试**

```python
def test_error_sums_and_metrics_match_manual_values(self) -> None:
    reference = torch.tensor([1.0, 2.0])
    approximation = torch.tensor([1.0, 1.0])
    sums = module.compute_error_sums(reference, approximation)
    metrics = module.finalize_error_metrics(sums)
    self.assertEqual(sums.numel, 2)
    self.assertEqual(sums.reference_energy, 5.0)
    self.assertEqual(sums.approximation_energy, 2.0)
    self.assertEqual(sums.error_energy, 1.0)
    self.assertEqual(sums.dot, 3.0)
    self.assertEqual(sums.absolute_error_sum, 1.0)
    self.assertEqual(sums.max_absolute_error, 1.0)
    self.assertAlmostEqual(metrics["nmse"], 0.2)
    self.assertAlmostEqual(metrics["nrmse"], math.sqrt(0.2))
    self.assertAlmostEqual(metrics["mae"], 0.5)
```

- [ ] **Step 2: 写 merge 等价测试**

将长度 128 的 tensor 分成两个 chunk，分别累计后 merge；与一次完整累计
的每个 `ErrorSums` 字段完全一致或误差低于 `1e-12`。

- [ ] **Step 3: 写全零 JSON-safe 测试**

先执行：

```python
metrics = module.finalize_error_metrics(sums)
json.dumps(metrics, allow_nan=False)
```

并验证不抛异常。

- [ ] **Step 4: 写 NV fake API 三路径测试**

```python
def test_nvfp4_api_builds_three_paths_without_requantizing_reference(self) -> None:
    reference = torch.linspace(-2, 2, 64).reshape(1, 64)
    result = module.evaluate_nvfp4_fake_weight(
        reference,
        pts_scale=torch.tensor(0.25),
        return_reconstructions=True,
    )
    torch.testing.assert_close(result["reference"], reference, rtol=0, atol=0)
    for name in ("direct", "pts_fp32", "pts_bf16"):
        self.assertIn("metrics", result["paths"][name])
        self.assertEqual(
            result["paths"][name]["sums"]["reference_energy"],
            result["paths"]["direct"]["sums"]["reference_energy"],
        )
```

- [ ] **Step 5: 写 PTS 计算顺序 oracle**

```python
def test_pts_bf16_path_matches_explicit_formula(self) -> None:
    reference = torch.randn(2, 64, generator=torch.Generator().manual_seed(6))
    scale = torch.tensor(0.137, dtype=torch.float32)
    result = module.evaluate_nvfp4_fake_weight(
        reference,
        pts_scale=scale,
        return_reconstructions=True,
    )
    normalized = (reference.float() / scale).to(torch.bfloat16).to(torch.float32)
    expected = module.quantize_hif4(normalized).values * scale
    torch.testing.assert_close(
        result["paths"]["pts_bf16"]["reconstruction"],
        expected,
        rtol=0,
        atol=0,
    )
```

- [ ] **Step 6: 写 PTS validation**

接受：

```text
Python float
0-D tensor
可广播到 weight 的 tensor
```

拒绝：

```text
scale <= 0
NaN/Inf
不能广播
complex/integer scale tensor
```

scale 安全搬到 weight device，以 FP32 执行乘除；结果记录原始 dtype/shape。

- [ ] **Step 7: 写 BF16-native API 测试**

```python
def test_bf16_api_does_not_include_fp32_to_bf16_loss(self) -> None:
    source = torch.tensor([[1.001, -0.997] * 32], dtype=torch.float32)
    result = module.evaluate_bf16_weight(
        source,
        return_reconstruction=True,
    )
    expected_reference = source.to(torch.bfloat16).to(torch.float32)
    torch.testing.assert_close(
        result["reference"],
        expected_reference,
        rtol=0,
        atol=0,
    )
    self.assertEqual(result["input_kind"], "bf16")
    self.assertNotIn("nvfp4", json.dumps(result).lower())
```

- [ ] **Step 8: 运行红灯**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_eval.MetricTests \
  tests.test_nvfp4_hif4_torch_eval.TensorEvaluationTests -v
```

- [ ] **Step 9: 实现 FP64 累计**

核心必须是：

```python
reference64 = reference.detach().to(torch.float64)
approximation64 = approximation.detach().to(torch.float64)
difference64 = approximation64 - reference64
```

所有 sum/dot 在 FP64 中完成，再 `.item()`。shape 必须完全一致；拒绝
NaN/Inf。

- [ ] **Step 10: 实现 NV 三路径**

固定顺序：

```python
reference_fp32 = weight.detach().to(torch.float32)
direct = quantize_hif4(reference_fp32, config=hif4_config).values

normalized_fp32 = reference_fp32 / pts_scale_fp32
pts_fp32 = quantize_hif4(normalized_fp32, config=hif4_config).values
pts_fp32 = pts_fp32 * pts_scale_fp32

normalized_bf16 = normalized_fp32.to(torch.bfloat16).to(torch.float32)
pts_bf16 = quantize_hif4(normalized_bf16, config=hif4_config).values
pts_bf16 = pts_bf16 * pts_scale_fp32
```

额外报告：

```text
inner_bf16_projection metrics
pts_fp32_vs_pts_bf16_value_equal_fraction
pts_delta_nmse
pts_relative_change
```

没有 `pts_scale` 时只计算 direct。

- [ ] **Step 11: 实现 BF16-native**

固定：

```python
reference = weight.detach().to(torch.bfloat16).to(torch.float32)
reconstruction = quantize_hif4(reference, config=hif4_config).values
```

不得调用 `simulate_nvfp4`。

- [ ] **Step 12: 运行测试并提交**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_eval.MetricTests \
  tests.test_nvfp4_hif4_torch_eval.TensorEvaluationTests -v
git add nvfp4_hif4_torch.py tests/test_nvfp4_hif4_torch_eval.py
git commit -m "feat: add native tensor conversion evaluators"
```

---

## Task 6: 从零实现 E1–E7 合成实验引擎

**Files:**
- Modify: `nvfp4_hif4_torch.py`
- Modify: `tests/test_nvfp4_hif4_torch_eval.py`

**Interfaces:**
- Consumes: `simulate_nvfp4`、`quantize_hif4`、两个 tensor evaluator。
- Produces: `make_distribution`、`run_simulation` 及 E1–E7 的结构化结果。

- [ ] **Step 1: 写四种分布的确定性测试**

测试：

```python
def test_distributions_are_deterministic_and_correct_size(self) -> None:
    for name in (
        "gaussian",
        "laplace",
        "student_t3",
        "outlier_0p1pct_20x",
    ):
        a = module.make_distribution(name, 6_400, seed=20260723)
        b = module.make_distribution(name, 6_400, seed=20260723)
        self.assertEqual(a.device.type, "cpu")
        self.assertEqual(a.dtype, torch.float32)
        self.assertEqual(a.numel(), 6_400)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.isfinite(a).all().item())
```

outlier 分布另验证恰好 6 个位置相对同 seed 的 Gaussian base 被乘 20。

- [ ] **Step 2: 从基础随机数明确实现分布**

只使用本地 `torch.Generator(device="cpu")`：

```python
generator = torch.Generator(device="cpu").manual_seed(seed)
```

实现：

```text
Gaussian:
    torch.randn(n, generator=generator)

Laplace:
    u = torch.rand(n, generator=generator) - 0.5
    x = -sign(u) * log1p(-2*abs(u)) / sqrt(2)

Student-t3 normalized to variance 1:
    z0,z1,z2,z3 = four independent standard normal arrays
    x = z0 / sqrt(z1^2 + z2^2 + z3^2)

Outlier:
    base = Gaussian
    index_generator uses seed + 1000
    indices = randperm(n)[:round(0.001*n)]
    base[indices] *= 20
```

Student-t3 公式来自 \(t_3/\sqrt3=z_0/\sqrt{\chi_3^2}\)。

- [ ] **Step 3: 写 E1 快速实验 schema 红灯测试**

```python
def test_quick_simulation_contains_e1_native_paths(self) -> None:
    result = module.run_simulation(
        module.ExperimentConfig(
            samples_per_repeat=6_400,
            repeats=1,
            phase_points=17,
        ),
        device=torch.device("cpu"),
        quick=True,
    )
    e1 = result["experiments"]["e1_native_source"]
    for distribution in (
        "gaussian",
        "laplace",
        "student_t3",
        "outlier_0p1pct_20x",
    ):
        self.assertIn("nv_direct", e1[distribution])
        self.assertIn("nv_pts_fp32", e1[distribution])
        self.assertIn("nv_pts_bf16", e1[distribution])
        self.assertIn("bf16_native", e1[distribution])
```

- [ ] **Step 4: 实现 E1 原生转换实验**

每个 distribution/repeat 固定流程：

```python
base = make_distribution(name, n, seed=seed)
nv = simulate_nvfp4(base)
bf16_reference = base.to(torch.bfloat16).to(torch.float32)

nv_result = evaluate_nvfp4_fake_weight(
    nv.values.to(device),
    pts_scale=nv.global_scale.to(device),
)
bf16_result = evaluate_bf16_weight(
    bf16_reference.to(device),
)
```

必须保存：

```text
distribution
repeat_index
seed
base_numel
nv_global_scale
nv storage dtype
每条路径 ErrorSums 和 metrics
```

绝不把 `base → NV` 的误差加进 NV conversion；绝不让 BF16-native
使用 `nv.values`。

- [ ] **Step 5: 写 E2 配对一致性测试**

对每个 repeat 断言：

```text
nv_direct.reference_energy
== nv_pts_fp32.reference_energy
== nv_pts_bf16.reference_energy
```

结果结构包含：

```text
delta_nmse
relative_change
paired_delta_mean
paired_delta_std
paired_delta_ci95_low
paired_delta_ci95_high
pts_bf16_win_count
```

- [ ] **Step 6: 实现 E2 配对统计**

只在同 distribution、同 repeat 内做差。`repeats=1` 时 CI 字段写 `null`，
不能除以零或伪造置信区间。

- [ ] **Step 7: 写 E3 码本穷举测试**

构造：

```python
e4, _ = module.build_e4m3fn_codebook()
e2 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
products = (e4[:, None] * e2[None, :]).reshape(-1)
carrier = products.to(torch.bfloat16).to(torch.float32)
```

保存并断言：

```text
total_pairs = 127 * 8
exact_count
exact_fraction
max_absolute_error
projection_nmse
```

期望 `exact_fraction=1.0`；如果失败，先检查格式公式和 overflow，不得删除
断言。

- [ ] **Step 8: 实现 E3 三层 carrier 分析**

E3 输出三部分：

1. `legal_codebook_products`：E4M3FN×E2M1 穷举；
2. `synthetic_normalized_values`：

   ```python
   normalized_fp32 = nv.values.float() / nv.global_scale.float()
   normalized_bf16 = normalized_fp32.to(torch.bfloat16).to(torch.float32)
   ```

   报告 projection NMSE 和 exact fraction；
3. `final_hif4_impact`：PTS-FP32 与 PTS-BF16 的 reconstruction 相等比例、
   NMSE 差值。

- [ ] **Step 9: 写 E4 phase sweep 测试**

phase 定义：

```python
k = torch.arange(phase_points, dtype=torch.float64)
phase = torch.pow(2.0, k / phase_points).to(torch.float32)
```

要求：

```text
phase[0] == 1
所有 phase < 2
严格递增
点数与 phase_points 完全一致
phase=1 时 direct reconstruction == PTS-FP32 reconstruction
```

- [ ] **Step 10: 实现 E4**

使用 `phase_seed=7` 生成固定 base，得到：

```python
nv = simulate_nvfp4(base)
expanded_block_scales = (
    nv.block_scales.unsqueeze(-1)
    .expand(*nv.block_scales.shape, 16)
    .reshape_as(nv.payload)
)
normalized_legal = expanded_block_scales * nv.payload
```

这里直接由 E4M3FN block scale 与 E2M1 payload 重建归一化合法值，避免
`(s_T * value) / s_T` 的浮点乘除噪声混入 phase 实验。

对每个 phase \(g\)：

```text
reference = g * normalized_legal
direct = Q_HiF4(reference)
pts_fp32 = g * Q_HiF4(normalized_legal)
pts_bf16 = g * Q_HiF4(BF16(normalized_legal))
```

每点保存三条路径 NMSE、E6M2 top-scale 概要和 direct-vs-PTS 差值。报告：

```text
argmin/argmax phase
PTS 有利 phase 占比
direct NMSE peak-to-peak
PTS-FP32 NMSE peak-to-peak
PTS-BF16 NMSE peak-to-peak
```

- [ ] **Step 11: 写 E5 scale mode 测试**

每个 source/distribution 必须有：

```text
continuous
bf16_math
e6m2_only
hardware
```

`hardware` 必须与 E1 对应路径完全一致。不得用
`hardware-continuous` 再强行拆成可加的两个误差。

- [ ] **Step 12: 实现 E5**

对 `nv_direct` 与 `bf16_native` 至少运行四种 mode；推荐再对
`nv_pts_bf16` 运行四种 mode。每条都保存完整 `ErrorSums`。

派生但不作可加性承诺：

```text
bf16_math_minus_continuous
e6m2_only_minus_continuous
hardware_minus_continuous
hardware_minus_e6m2_only
```

- [ ] **Step 13: 写 E6 group-size 测试**

组合固定为：

```text
source_kind in {nv_direct, bf16_native}
group_size in {16, 32, 64}
scale_mode in {continuous, hardware}
```

输出每项包含：

```text
group_size
is_standard_hif4 = (group_size == 64)
scale_mode
metrics
```

- [ ] **Step 14: 实现 E6**

同一个 repeat/source 上配对比较。16/32/64 均可被默认 320K 和 6.4K
整除。若调用者覆盖 sample count 导致不能整除，立即报错，不静默截断。

- [ ] **Step 15: 写 E7 storage dtype 测试**

固定步骤：

```python
nv = simulate_nvfp4(base)
nv_fp32 = nv.values
nv_bf16 = nv_fp32.to(torch.bfloat16)
storage_reference = nv_bf16.to(torch.float32)
fp32_result = evaluate_nvfp4_fake_weight(
    nv_fp32,
    pts_scale=nv.global_scale,
)
bf16_result = evaluate_nvfp4_fake_weight(
    nv_bf16,
    pts_scale=nv.global_scale,
)
```

结果必须分开：

```text
storage_projection: nv_fp32 -> nv_bf16 value
fp32_container_conversion: Q_HiF4(nv_fp32) relative to nv_fp32
bf16_container_conversion: Q_HiF4(storage_reference) relative to storage_reference
```

不得把前两段误差相加叫作 native conversion。

- [ ] **Step 16: 实现 E7**

除 NMSE 外报告：

```text
storage_exact_fraction
HiF4 reconstruction equal fraction
e1_per_8 equal fraction
e1_per_4 equal fraction
```

- [ ] **Step 17: 实现 repeat 汇总**

独立 helper：

```python
def summarize_repeats(
    repeat_sums: list[ErrorSums],
    repeat_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    merged = ErrorSums()
    for sums in repeat_sums:
        merged = merge_error_sums(merged, sums)
    nmse_values = [float(item["nmse"]) for item in repeat_metrics]
    mean_nmse = statistics.fmean(nmse_values)
    std_nmse = statistics.stdev(nmse_values) if len(nmse_values) > 1 else 0.0
    if len(nmse_values) == 10:
        half_width = 2.262 * std_nmse / math.sqrt(10)
        ci95 = [mean_nmse - half_width, mean_nmse + half_width]
    else:
        ci95 = None
    return {
        "energy_weighted": finalize_error_metrics(merged),
        "repeat_nmse": nmse_values,
        "repeat_count": len(nmse_values),
        "mean_nmse": mean_nmse,
        "std_nmse": std_nmse,
        "ci95": ci95,
    }
```

要求：

- merge 后得到 energy-weighted 主指标；
- 均值/std/CI 基于 repeat NMSE；
- 默认 `repeats=10` 时使用 \(t_{0.975,9}=2.262\)；
- `repeats=1` 或用户自定义为非 10 次时 `std` 仍照常给出、CI 为 `null`；
- 保存每个 repeat 的原始记录，不只留汇总。

- [ ] **Step 18: 运行快速 E1–E7 测试**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_eval.SimulationExperimentTests -v
```

验收：

```text
无 NaN
所有非零 reference 的 0 < NMSE < 0.05
cosine 位于 [-1, 1]
nrmse == sqrt(nmse)
direct/PTS reference energy 完全一致
E3 legal-code exact fraction == 1
phase=1 direct == PTS-FP32
E6 的 16/32 均标非标准
```

- [ ] **Step 19: 提交**

```bash
git add nvfp4_hif4_torch.py tests/test_nvfp4_hif4_torch_eval.py
git commit -m "feat: implement native conversion simulation experiments"
```

---

## Task 7: 从零实现真实 checkpoint 读取、筛选和 PTS 映射

**Files:**
- Modify: `nvfp4_hif4_torch.py`
- Modify: `tests/test_nvfp4_hif4_torch_eval.py`

**Interfaces:**
- Produces: `iter_checkpoint_tensors`、`load_pts_scales`、tensor 分类与过滤。

- [ ] **Step 1: 写单文件 state_dict 测试**

用 `TemporaryDirectory` 保存：

```python
state = {
    "model.layers.0.self_attn.q_proj.weight":
        torch.randn(64, 128, dtype=torch.bfloat16),
    "model.layers.0.input_layernorm.weight":
        torch.ones(128, dtype=torch.bfloat16),
}
torch.save(state, path / "model.pt")
```

`iter_checkpoint_tensors` 应按 key 排序返回两个 tensor，全部先位于 CPU。

- [ ] **Step 2: 写 wrapped state_dict 测试**

仅支持明确 wrapper：

```python
{"state_dict": state}
{"model_state_dict": state}
```

若文件是 `nn.Module`、optimizer state 或含多个无法判断的 tensor mapping，
抛出带文件名的 `ValueError`，不猜测。

- [ ] **Step 3: 写 `.safetensors` 延迟依赖测试**

当依赖可用时创建小文件并读回；依赖不可用时：

```text
.pt/.pth 仍正常
读取 .safetensors 时才抛 RuntimeError
错误信息包含 pip install safetensors
```

- [ ] **Step 4: 写 Hugging Face 分片目录测试**

支持以下优先级：

```text
model.safetensors.index.json
pytorch_model.bin.index.json
单个 model.safetensors
目录内唯一 .pt/.pth/.bin
目录内按文件名排序的多个 .safetensors
```

index 中同一 shard 只加载一次，按 `weight_map` 的参数名顺序产出；不得一次
把所有 shard 留在内存。

- [ ] **Step 5: 写 PTS JSON 测试**

合法文件：

```json
{
  "model.layers.0.self_attn.q_proj.weight": 0.001953125,
  "model.layers.0.mlp.up_proj.weight": 0.00390625
}
```

要求：

- key 为非空字符串；
- value 为正、有限 scalar；
- 重复 key 由 JSON parser 已覆盖时不能检测，因此 README/report 说明
  使用规范 JSON；
- 非数值、0、负数、NaN/Inf 均拒绝。

- [ ] **Step 6: 写 PTS `.pt/.pth` mapping 测试**

允许：

```python
dict[str, float | torch.Tensor]
```

tensor 必须是单元素；统一转成 CPU FP32 0-D tensor。真实 checkpoint 第一版
只支持 per-tensor scalar PTS。

- [ ] **Step 7: 实现 checkpoint iterator**

使用：

```python
torch.load(path, map_location="cpu", weights_only=True)
```

若当前 PyTorch 不支持 `weights_only` 参数，捕获 `TypeError` 并给出明确版本
要求；不要静默退回不安全 pickle 加载。

每个 yield 后调用方处理完即可释放 shard；不 clone 大 tensor，除非底层
reader 生命周期要求。

- [ ] **Step 8: 实现参数筛选**

固定顺序：

1. `tensor_names` 精确白名单；
2. `include_regex`；
3. `exclude_regex`；
4. dtype 是否 floating；
5. `ndim>=2`；
6. `group_dim` 合法；
7. 分组维度长度可被 `group_size` 整除；
8. `max_tensors`。

每个跳过项记录一个枚举原因：

```text
not_requested
include_regex_miss
exclude_regex_match
non_floating
ndim_lt_2
invalid_group_dim
not_group_divisible
max_tensors_reached
```

- [ ] **Step 9: 实现类别映射**

按参数名 suffix/substring 分类：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
embed_tokens
lm_head
other
```

匹配顺序从具体到一般；每个 tensor 只能属于一个类别。

- [ ] **Step 10: 运行 loader 测试**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_eval.CheckpointLoadingTests -v
```

Expected: `.pt/.pth` 全部 PASS；safetensors 测试按依赖条件 PASS/SKIP。

- [ ] **Step 11: 提交**

```bash
git add nvfp4_hif4_torch.py tests/test_nvfp4_hif4_torch_eval.py
git commit -m "feat: add safe checkpoint and pts loading"
```

---

## Task 8: 实现 group-aligned chunk 评测、聚合和激活输出误差

**Files:**
- Modify: `nvfp4_hif4_torch.py`
- Modify: `tests/test_nvfp4_hif4_torch_eval.py`

**Interfaces:**
- Consumes: Task 5 tensor API、Task 7 iterator/filter。
- Produces: `evaluate_checkpoint`、`evaluate_output_error`、逐 tensor/类别/全局
  汇总。

- [ ] **Step 1: 写 chunk 与整 tensor 等价测试**

使用 shape `[192, 128]` 的小权重，分别：

```text
不分 chunk
chunk_groups=1
chunk_groups=17
chunk_groups=16384
```

对 direct、PTS-FP32、PTS-BF16 的 `ErrorSums` 逐字段比较：

```text
relative/absolute difference <= 1e-12
```

重建 tensor 不要求保留。

- [ ] **Step 2: 实现 group-aligned chunk iterator**

内部先：

```python
moved = weight.movedim(group_dim, -1).contiguous()
rows = moved.reshape(-1, moved.shape[-1])
groups_per_row = rows.shape[-1] // group_size
grouped = rows.reshape(-1, group_size)
```

每次取最多 `chunk_groups` 个完整 group。chunk 的 `HiF4Config` 使用
`group_dim=-1`。不得切断 group。

- [ ] **Step 3: 写逐 tensor/category/global 聚合测试**

临时 checkpoint 放入两层 q_proj 和一层 up_proj。手动对所有 reference /
reconstruction 拼接计算 global NMSE，要求与 `evaluate_checkpoint` 的：

```text
global.direct
categories.q_proj.direct
categories.up_proj.direct
```

完全一致。逐 tensor NMSE 的平均值应被刻意构造成不同，以防实现误用平均。

- [ ] **Step 4: 实现聚合器**

维护：

```python
global_sums: dict[path_name, ErrorSums]
category_sums: dict[category, dict[path_name, ErrorSums]]
tensor_results: dict[tensor_name, dict[str, Any]]
```

路径：

```text
BF16 input: native
NV input: direct
NV + PTS: pts_fp32, pts_bf16
```

NV 全局 PTS 聚合只累计有 PTS 的 tensor，并额外记录：

```text
pts_tensor_coverage
pts_numel_coverage
pts_reference_energy_coverage
```

- [ ] **Step 5: 写 PTS 缺失/require 测试**

```text
require_pts=False:
    missing tensor 仍评测 direct，PTS 为 null

require_pts=True:
    任一已选 NV tensor 缺 PTS 时整次运行非零失败

BF16 input:
    pts_scales_path 非空时参数错误
```

- [ ] **Step 6: 实现逐 group 误差摘要**

每个 tensor 内按标准 group 计算：

```text
group_reference_energy
group_error_energy
group_nmse
```

reference energy 为 0 的 group 单独计数。保存：

```text
group_count
zero_reference_group_count
p50/p90/p95/p99/max group NMSE
top-20 group index and NMSE
```

每个 tensor 的 group NMSE 可暂存在 CPU 以算精确 quantile，写完摘要立即
释放；不得把全模型所有 group 原始值写入 JSON。

- [ ] **Step 7: 写 output-error 小矩阵 oracle**

```python
def test_output_error_matches_full_matmul(self) -> None:
    x = torch.randn(13, 8, generator=torch.Generator().manual_seed(7))
    w = torch.randn(6, 8, generator=torch.Generator().manual_seed(8))
    w_hat = w + 0.01
    result = module.evaluate_output_error(
        x,
        w,
        w_hat,
        token_batch_size=4,
    )
    y = x.float() @ w.float().T
    y_hat = x.float() @ w_hat.float().T
    expected = module.finalize_error_metrics(
        module.compute_error_sums(y, y_hat)
    )
    self.assertAlmostEqual(result["nmse"], expected["nmse"], places=12)
```

- [ ] **Step 8: 实现 output error**

约定：

```text
activations shape = [num_tokens, in_features]
weight shape = [out_features, in_features]
output = activations @ weight.T
```

按 token batch 分块：

```python
reference_output = x_batch.float() @ reference_weight.float().T
approx_output = x_batch.float() @ approximation_weight.float().T
```

每个 batch 立即累计 FP64 ErrorSums。拒绝 shape 不匹配、非有限 activation、
非二维 weight、`token_batch_size<=0`。

- [ ] **Step 9: 实现 activation 文件读取**

第一版只接受 `.pt/.pth`：

```python
dict[str, torch.Tensor]
```

key 可以是完整 weight 名或去掉 `.weight` 的 module 名。若两个 key 都存在
且数值不一致，报错，不猜测。

- [ ] **Step 10: 将 activation 路径接入 checkpoint 评测**

只对有对应 activation 的二维 Linear 计算。每条 HiF4 reconstruction 按
输出行 chunk 量化并立刻 matmul；每个 chunk 必须包含这些输出行的完整 K
维，不能把一行的 K 维拆开后独立计算输出。这样可避免永久保存整模型重建
权重。结果中记录：

```text
activation_status
token_count
weight_nmse
output_nmse
```

- [ ] **Step 11: 运行评测测试**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_eval.CheckpointEvaluationTests \
  tests.test_nvfp4_hif4_torch_eval.OutputErrorTests -v
```

- [ ] **Step 12: 提交**

```bash
git add nvfp4_hif4_torch.py tests/test_nvfp4_hif4_torch_eval.py
git commit -m "feat: add chunked checkpoint and output evaluation"
```

---

## Task 9: 从零实现 CLI、JSON/CSV/Markdown 报告

**Files:**
- Modify: `nvfp4_hif4_torch.py`
- Modify: `tests/test_nvfp4_hif4_torch_eval.py`

**Interfaces:**
- Produces: `simulate`、`evaluate-tensor-file`、`evaluate-checkpoint`
  三个子命令和三类输出文件。

- [ ] **Step 1: 定义 `simulate` CLI**

完整运行：

```bash
python nvfp4_hif4_torch.py simulate \
  --device cuda \
  --seed 20260723 \
  --samples-per-repeat 320000 \
  --repeats 10 \
  --phase-points 257 \
  --output-dir results/nvfp4_hif4_torch/simulation
```

快速运行：

```bash
python nvfp4_hif4_torch.py simulate \
  --quick \
  --device cpu \
  --output-dir results/nvfp4_hif4_torch/quick
```

`--quick` 仅在用户没有显式覆盖时设置：

```text
samples_per_repeat=6400
repeats=1
phase_points=17
```

- [ ] **Step 2: 定义真实单 tensor 文件 CLI**

支持 `.pt/.pth` 中直接保存的单 tensor：

```bash
python nvfp4_hif4_torch.py evaluate-tensor-file \
  --tensor /path/to/fake_weight.pt \
  --input-kind nvfp4_fake \
  --pts-scale 0.001953125 \
  --device cuda \
  --group-size 64 \
  --group-dim -1 \
  --output-dir results/nv_single_tensor
```

BF16：

```bash
python nvfp4_hif4_torch.py evaluate-tensor-file \
  --tensor /path/to/bf16_weight.pt \
  --input-kind bf16 \
  --device cuda \
  --output-dir results/bf16_single_tensor
```

NV 模式的 `--pts-scale` 可省略；BF16 模式传入必须报错。

- [ ] **Step 3: 定义 NV checkpoint CLI**

```bash
python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint /path/to/nvfp4_fake_model \
  --input-kind nvfp4_fake \
  --pts-scales /path/to/pts_scales.json \
  --include-regex '(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$' \
  --device cuda \
  --group-size 64 \
  --group-dim -1 \
  --chunk-groups 16384 \
  --output-dir results/qwen3_8b_nvfp4
```

- [ ] **Step 4: 定义 BF16 checkpoint CLI**

```bash
python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint /path/to/bf16_model \
  --input-kind bf16 \
  --include-regex '(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$' \
  --device cuda \
  --group-size 64 \
  --group-dim -1 \
  --chunk-groups 16384 \
  --output-dir results/qwen3_8b_bf16
```

- [ ] **Step 5: 定义通用参数**

必须支持：

```text
--include-regex REGEX
--exclude-regex REGEX
--tensor-name NAME             # 可重复
--max-tensors N                # 仅调试，结果标注 truncated=true
--require-pts
--device cpu|cuda|cuda:N
--group-size 64
--group-dim -1
--scale-mode hardware
--chunk-groups 16384
--activations PATH
--token-batch-size 256
--output-dir PATH
```

在参数解析后立即验证冲突：

```text
input_kind=bf16 与 pts 参数冲突
require_pts 只允许 nvfp4_fake
group_size 非 8 的正倍数
chunk_groups <= 0
token_batch_size <= 0
repeats <= 0
samples <= 0 或不能被 64 整除
```

- [ ] **Step 6: 实现原子输出**

每次成功运行写：

```text
results.json
results.csv
report.md
```

先写同目录临时文件，再用 `Path.replace` 原子替换，防止中断留下半文件。
JSON 使用：

```python
json.dump(result, file, indent=2, ensure_ascii=False, allow_nan=False)
```

- [ ] **Step 7: 定义 long-form CSV**

至少包含：

```text
schema_version
run_kind
scope
tensor_name
category
distribution
repeat
path
metric
value
unit
```

比例 `nmse` 写原始比例，`unit="ratio"`；Markdown 展示时才乘 100。

- [ ] **Step 8: 定义 simulation Markdown**

依次输出：

1. 配置与语义；
2. E1 native source 主表；
3. E2 PTS 配对差值；
4. E3 BF16 carrier；
5. E4 phase sweep 摘要；
6. E5 scale decomposition；
7. E6 group-size；
8. E7 storage dtype；
9. warnings 与验收结果。

- [ ] **Step 9: 定义 checkpoint Markdown**

依次输出：

1. checkpoint、input kind、device、筛选；
2. 评测/跳过 tensor 数；
3. global 指标；
4. category 指标；
5. PTS 覆盖率和获胜比例；
6. top-20 tensor；
7. 逐 group 尾部摘要；
8. activation output error（若提供）；
9. skipped reasons。

- [ ] **Step 10: 写 CLI 端到端测试**

使用临时小 checkpoint 调：

```python
exit_code = module.main([
    "evaluate-checkpoint",
    "--checkpoint", str(checkpoint_path),
    "--input-kind", "bf16",
    "--device", "cpu",
    "--output-dir", str(output_dir),
])
```

验证：

```text
exit_code == 0
三个文件存在
JSON allow_nan=False 可重读
CSV 有表头和数据
report.md 包含 input_kind=bf16
report.md 不含 no_pts/without_pts/legacy
错误参数返回非零
```

- [ ] **Step 11: 跑 CLI 测试并提交**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_eval.CLITests -v
git add nvfp4_hif4_torch.py tests/test_nvfp4_hif4_torch_eval.py
git commit -m "feat: add greenfield torch evaluation cli and reports"
```

---

## Task 10: 执行 E0——独立数值正确性与跨设备验证

**Files:**
- Modify only when E0 exposes a verified defect:
  `nvfp4_hif4_torch.py`
- Modify only when the independent oracle was incomplete:
  `tests/test_nvfp4_hif4_torch_core.py`
  `tests/test_nvfp4_hif4_torch_eval.py`

**Interfaces:**
- Consumes: Tasks 1–9 的完整绿地实现。
- Produces: 可接受合成/真实实验的数值正确性门槛。

- [ ] **Step 1: 从零运行新测试**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_core \
  tests.test_nvfp4_hif4_torch_eval -v
```

Expected: 全部 PASS。

- [ ] **Step 2: 确认新实现没有 NumPy**

```bash
rg -n 'import numpy|from numpy|np\.' \
  nvfp4_hif4_torch.py \
  tests/test_nvfp4_hif4_torch_core.py \
  tests/test_nvfp4_hif4_torch_eval.py
```

Expected: 0 匹配。

- [ ] **Step 3: 确认新实现没有旧脚本依赖**

```bash
rg -n \
  'reproduce_nvfp4_to_hif4|test_reproduce_nvfp4_to_hif4|legacy' \
  nvfp4_hif4_torch.py \
  tests/test_nvfp4_hif4_torch_core.py \
  tests/test_nvfp4_hif4_torch_eval.py
```

Expected: 0 匹配。测试说明文字也不要依赖旧文件。

- [ ] **Step 4: 做 E4M3FN/E2M1/E6M2 oracle 穷举**

验证：

```text
E4M3FN values 数量 127、严格升序、端点正确
E6M2 values 数量 255、严格升序、端点正确
所有码本 midpoint 的 RNE 与 code parity 一致
低于最小值/高于最大值正确饱和
E2M1 输出只属于 8 个合法正幅值
```

midpoint 测试不得只随机抽样；对相邻码点全部穷举。

- [ ] **Step 5: 做 E4M3FN×E2M1 BF16 carrier 穷举**

```bash
python - <<'PY'
import torch
import nvfp4_hif4_torch as m

e4, _ = m.build_e4m3fn_codebook()
e2 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
x = (e4[:, None] * e2[None, :]).reshape(-1)
y = x.to(torch.bfloat16).to(torch.float32)
assert torch.equal(x, y), (x != y).sum().item()
print({"pairs": x.numel(), "exact": int(torch.equal(x, y))})
PY
```

Expected:

```text
pairs = 1016
exact = 1
```

- [ ] **Step 6: 做 S1P2 手算边界测试**

对 normalized magnitude：

```text
0.000 -> 0.00
0.124 -> 0.00
0.125 -> 0.25
0.374 -> 0.25
0.375 -> 0.50
1.624 -> 1.50
1.625 -> 1.75
1.875 -> 1.75  # saturation
```

验证正数 half-away-from-zero 和 1.75 saturation；负数通过 sign 单独恢复。

- [ ] **Step 7: 做分组不跨行验证**

构造两行动态范围差异极大的 tensor：

```python
x = torch.cat([
    torch.ones(1, 64),
    torch.full((1, 64), 1024.0),
], dim=0)
```

逐行单独量化再拼接，必须与二维 tensor 一次量化完全一致。该测试可抓出错误
的全局 flatten 分组。

- [ ] **Step 8: 做 CPU/CUDA 对齐**

仅在 CUDA 可用时运行。输入先在 CPU 固定生成，再复制：

```python
x_cpu = torch.randn(
    31,
    128,
    generator=torch.Generator().manual_seed(20260723),
)
x_cuda = x_cpu.cuda()
```

对四种 `scale_mode`、NVFP4 simulation、三条 NV evaluator 路径检查：

```text
payload/e1 完全一致
top scale/reconstruction rtol<=1e-6, atol<=1e-8
NMSE 绝对差 <=1e-8
```

若 midpoint tie 出现设备差异，修正显式 tie-breaking；不能仅放宽到“统计
接近”。

- [ ] **Step 9: 做 chunk 一致性**

对 CPU（及可用时 CUDA）比较：

```text
完整 tensor
1 group/chunk
17 groups/chunk
16,384 groups/chunk
```

每个 ErrorSums 字段一致到 `1e-12`，最终比例指标一致到 `1e-12`。

- [ ] **Step 10: 跑 quick CLI**

```bash
python nvfp4_hif4_torch.py simulate \
  --quick \
  --device cpu \
  --output-dir /tmp/nvfp4_hif4_greenfield_e0
```

Expected:

```text
exit code 0
E1–E7 全部存在
results.json 可用 allow_nan=False 读取
无 NaN/Inf 数值
phase=1 direct 与 PTS-FP32 相同
```

- [ ] **Step 11: 只提交被 E0 证明必要的修复**

```bash
git add nvfp4_hif4_torch.py \
        tests/test_nvfp4_hif4_torch_core.py \
        tests/test_nvfp4_hif4_torch_eval.py
git commit -m "fix: satisfy independent torch format oracles"
```

如果没有修复，不创建空提交。

---

## Task 11: 执行完整合成实验 E1–E7

**Files:**
- Generated only:
  `results/nvfp4_hif4_torch/simulation/*`

**Interfaces:**
- Consumes: 已通过 E0 的脚本。
- Produces: 合成主表、PTS 配对结论、carrier/scale/group/storage 消融。

- [ ] **Step 1: 运行完整配置**

CUDA：

```bash
python nvfp4_hif4_torch.py simulate \
  --device cuda \
  --seed 20260723 \
  --samples-per-repeat 320000 \
  --repeats 10 \
  --phase-points 257 \
  --output-dir results/nvfp4_hif4_torch/simulation
```

无 CUDA 时把 `--device` 改为 `cpu`，其他设置不变，并在结果中保留实际
device。

- [ ] **Step 2: 审核 E1**

四种分布每种必须同时有：

```text
NV direct
NV PTS-FP32
NV PTS-BF16
BF16-native
```

逐项核对：

- NV 三路径共用 `W_NV` reference；
- BF16-native 使用 `BF16(W_base)` reference；
- 不存在名为 `total_error`、`cascade_error` 或
  `bf16_to_nvfp4_to_hif4` 的主指标；
- 每项既有合并能量指标，也有 repeat mean/std/CI。

- [ ] **Step 3: 审核 E2**

对每个 distribution 输出：

```text
direct NMSE
PTS-FP32 NMSE
PTS-BF16 NMSE
PTS-BF16 - direct
relative change
paired 95% CI
10 repeat 中 PTS-BF16 获胜次数
```

如果 CI 跨 0，结论写“未观察到稳定优势”，不能只依据均值方向宣称获胜。

- [ ] **Step 4: 审核 E3**

分开报告：

1. 合法 E4M3FN×E2M1 code product 的 BF16 exactness；
2. 合成 `W_NV/s_T` 的 BF16 projection；
3. PTS-FP32 与 PTS-BF16 的最终 HiF4 reconstruction 差异。

第一项必须 100%；后两项即使非零也不得被解释成 NVFP4 source 的历史
量化损失。

- [ ] **Step 5: 审核 E4**

JSON/CSV 必须保存全部 257 点。核对：

```text
phase 范围 [1, 2)
phase 单调
phase=1 invariant
direct、PTS-FP32、PTS-BF16 三条线
argmin/argmax 与原始点一致
```

科学结论只描述“PTS 改变 E6M2 网格相位后，在哪些 phase 有利或不利”，
不能推广成所有真实模型都相同。

- [ ] **Step 6: 审核 E5**

四种 mode：

```text
continuous
bf16_math
e6m2_only
hardware
```

分别针对 source/distribution 输出。差值只作 diagnostic，不把
`BF16 contribution + E6M2 contribution` 当作严格可加分解。

- [ ] **Step 7: 审核 E6**

主表按：

```text
source × distribution × scale_mode × group_size
```

确认：

- 64 标为标准；
- 16/32 标为分析性；
- continuous 与 hardware 分开；
- 不把 group size 与 E6M2 scale 舍入混为同一因素。

- [ ] **Step 8: 审核 E7**

必须有两段不同结果：

```text
storage projection: FP32 fake values -> BF16 container values
native conversion: each stored value -> HiF4 relative to itself
```

报告 storage exact fraction、两种 container conversion NMSE，以及 HiF4
code/reconstruction equal fraction。

- [ ] **Step 9: 执行不依赖结果方向的科学验收**

固定检查：

```text
所有非零 source: 0 < NMSE < 0.05
所有 cosine 在 [-1, 1]
NRMSE 与 sqrt(NMSE) 一致
同 repeat 的 direct/PTS reference energy 完全一致
BF16-native reference energy 可由 BF16 reference 重算
E3 legal-code exactness = 1
phase=1 invariant
continuous scale 等变误差 < 1e-6
所有 repeat 无 NaN/Inf
schema/config/seed 完整
```

首次 full run 不设置狭窄的“预期 NMSE 必须落在某个历史区间”。获得可信的
绿地基线并复核后，未来才可添加宽容的数值回归区间。

- [ ] **Step 10: 固化实验快照**

在 `report.md` 记录：

```text
git commit
Python version
torch version
CUDA version
GPU/CPU name
完整 CLI
耗时
输出目录
```

结果文件是否提交由项目策略决定；不得把大型中间 tensor 加入 git。

---

## Task 12: 执行 E8——真实 NVFP4 fake 与 BF16 模型权重实验

**Files:**
- Generated result directories only.

**Interfaces:**
- Consumes: 用户提供的真实 checkpoints，以及可选真实 PTS mapping。
- Produces: 逐 tensor、类别、层和全模型 native conversion 结果。

### 12.1 运行前元数据确认

- [ ] **Step 1: 记录 NV fake checkpoint 语义**

必须记录：

```text
模型名称/版本
checkpoint 路径
tensor container dtype（FP32/BF16/混合）
数值确实已经 fake-quant 为 NVFP4
数值是否已经乘入 PTS
PTS 是否可按参数名获得
权重布局是否 [out_features, in_features]
HiF4 应沿哪一维分组
```

若无法确认 fake-quant 语义，不运行或将结果明确标为
`unverified_input_semantics`。

- [ ] **Step 2: 记录 BF16 checkpoint 语义**

记录：

```text
模型名称/版本
是否与 NV checkpoint 同架构
参数名/shape 是否可对齐
权重保存 dtype
是否含其他量化 wrapper
```

BF16 evaluator 会再次显式投影 BF16，所以 FP32 容器中的 BF16-exact
数值也可使用。

### 12.2 单 tensor 预检

- [ ] **Step 3: 先跑一层 q_proj NV fake**

```bash
python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint /path/to/nvfp4_fake_model \
  --input-kind nvfp4_fake \
  --tensor-name model.layers.0.self_attn.q_proj.weight \
  --pts-scales /path/to/pts_scales.json \
  --device cuda \
  --output-dir results/nv_single_q_proj
```

没有 PTS 文件时删去该参数。检查：

```text
shape/dtype 正确
group_dim=-1
direct 存在
PTS 是否与 mapping 一致
reference energy > 0
NMSE 合理且无 NaN
chunk_groups 改变不影响结果
```

- [ ] **Step 4: 跑对应 BF16 q_proj**

```bash
python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint /path/to/bf16_model \
  --input-kind bf16 \
  --tensor-name model.layers.0.self_attn.q_proj.weight \
  --device cuda \
  --output-dir results/bf16_single_q_proj
```

确认参数名和 shape 与 NV 侧一致；两个 NMSE 各自使用自己的 reference。

### 12.3 七类 Linear 主实验

- [ ] **Step 5: 跑 NV fake 七类权重**

```bash
python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint /path/to/nvfp4_fake_model \
  --input-kind nvfp4_fake \
  --pts-scales /path/to/pts_scales.json \
  --include-regex '(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$' \
  --device cuda \
  --group-size 64 \
  --group-dim -1 \
  --chunk-groups 16384 \
  --output-dir results/model_nvfp4_linear
```

没有 PTS 时删去 `--pts-scales`，并确认 PTS 字段为 null，而非估计值。

- [ ] **Step 6: 跑 BF16 七类权重**

```bash
python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint /path/to/bf16_model \
  --input-kind bf16 \
  --include-regex '(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$' \
  --device cuda \
  --group-size 64 \
  --group-dim -1 \
  --chunk-groups 16384 \
  --output-dir results/model_bf16_linear
```

- [ ] **Step 7: 只取参数名与 shape 交集比较**

生成比较表：

| Tensor/Category | NV direct | NV PTS-FP32 | NV PTS-BF16 | BF16-native | NV storage dtype | PTS available |
|---|---:|---:|---:|---:|---|:---:|

说明：

- 同一行只是比较两种 native source 到 HiF4 的转换难度；
- 不能计算 `BF16-native NMSE - NV-direct NMSE` 后称为历史量化误差；
- 若同名 shape 不一致，排除并记录。

- [ ] **Step 8: 做逐类别聚合**

对 q/k/v/o/gate/up/down 分别报告：

```text
tensor count
numel
reference energy
energy-weighted NMSE
NRMSE
cosine
SQNR
PTS coverage
PTS win tensor ratio
PTS win numel ratio
PTS win reference-energy ratio
group NMSE p50/p90/p95/p99
```

- [ ] **Step 9: 做逐层分析**

从参数名解析 layer index；无法解析者归为 `layer_unknown`。输出每层：

```text
q/k/v/o/gate/up/down direct NMSE
PTS-BF16 NMSE（若有）
BF16-native NMSE
delta
```

列出：

- top-20 direct 高误差 tensor；
- top-20 PTS 改善 tensor；
- top-20 PTS 恶化 tensor；
- group-tail 最重的 tensor。

- [ ] **Step 10: 单独跑 embedding/lm_head**

```bash
python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint /path/to/model \
  --input-kind INPUT_KIND \
  --include-regex '(embed_tokens|lm_head)\.weight$' \
  --device cuda \
  --output-dir results/model_embedding_head
```

不把这些大矩阵加入七类 Linear 主全局指标，避免能量/numel 主导。

### 12.4 模型数量

- [ ] **Step 11: 完成最低可用配置**

至少：

```text
1 个 7B/8B 模型架构
1 份 BF16 checkpoint
1 份对应 NVFP4 fake checkpoint
七类 Linear
NV direct
BF16-native
有真实 PTS 时加 NV PTS-FP32/BF16
```

- [ ] **Step 12: 完成论文级推荐配置**

建议：

```text
Qwen3-8B
另一个不同架构的 7B/8B 模型
每个架构 BF16/NV fake 各一份
统一 group/rounding/filter
逐模型、类别、层、全局报告
```

更大模型为扩展项，不阻塞第一轮真实权重结论。

---

## Task 13: 执行 E9 激活输出误差，并规划可选 E10

**Files:**
- Generated activations and results only.
- 本任务不要求把模型 forward/hook 逻辑加入
  `nvfp4_hif4_torch.py`。

**Interfaces:**
- Consumes: E8 权重及外部采集的 calibration activations。
- Produces: weight NMSE 与线性层 output NMSE 的关系。

### 13.1 激活采集

- [ ] **Step 1: 使用模型侧独立脚本/现有框架采集**

推荐设置：

| 项目 | 设置 |
|---|---|
| 数据 | 与模型量化一致的 calibration corpus；无统一口径时用 C4 |
| sequences | `128` |
| sequence length | `2048` |
| seed | `20260723` |
| 每个 Linear 保存 token vectors | `4096` |
| reservoir 采样 | 固定 seed |
| 保存 dtype | BF16 |
| 保存 device | CPU |
| 模块 | q/k/v/o/gate/up/down |

保存：

```python
dict[str, torch.Tensor]
```

其中 key 是 weight 参数名或去掉 `.weight` 的 module 名，value shape 是
`[num_tokens, in_features]`。

- [ ] **Step 2: 决定 activation reference**

推荐各自 source 使用各自激活：

```text
BF16-native -> BF16 模型 forward 的输入激活
NV-native -> NV fake 模型 forward 的输入激活
```

如果只能使用一套共同激活，结果 metadata 必须标：

```text
activation_semantics = "common_activation_proxy"
```

不能把 proxy 结果表述为完整端到端行为。

### 13.2 输出误差

- [ ] **Step 3: 跑 NV output error**

```bash
python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint /path/to/nvfp4_fake_model \
  --input-kind nvfp4_fake \
  --pts-scales /path/to/pts_scales.json \
  --activations /path/to/nv_activations.pt \
  --token-batch-size 256 \
  --include-regex '(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$' \
  --device cuda \
  --output-dir results/model_nvfp4_output_error
```

- [ ] **Step 4: 跑 BF16 output error**

```bash
python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint /path/to/bf16_model \
  --input-kind bf16 \
  --activations /path/to/bf16_activations.pt \
  --token-batch-size 256 \
  --include-regex '(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$' \
  --device cuda \
  --output-dir results/model_bf16_output_error
```

- [ ] **Step 5: 分析 weight/output 关系**

逐 tensor 报告：

```text
weight NMSE
output NMSE
direct vs PTS-BF16 weight delta
direct vs PTS-BF16 output delta
```

计算：

```text
Spearman(weight NMSE, output NMSE)
Pearson(log10 weight NMSE, log10 output NMSE)
weight 改善且 output 改善的比例
weight 改善但 output 恶化的 tensor 列表
```

Spearman 可在脚本外用 pandas/scipy 生成；核心脚本结果需提供完整逐 tensor
数值，不强制增加 SciPy 依赖。

- [ ] **Step 6: 限制结论**

若 PTS 只降低 weight NMSE、不降低 output NMSE，则结论必须限制为：

```text
PTS reduces numeric weight re-encoding loss under this setting.
```

不能宣称模型精度必然提升。

### 13.3 可选 E10 端到端

- [ ] **Step 7: 建立各自 source baseline**

至少评测：

```text
BF16 source
BF16 -> HiF4
NVFP4 fake source
NVFP4 fake -> HiF4 direct
NVFP4 fake -> HiF4 PTS-BF16（有真实 PTS 时）
```

BF16 与 NV 各自相对自己的 source，不能合并成一条
`BF16 -> NVFP4 -> HiF4` 损失链。

- [ ] **Step 8: PPL 设置**

建议：

```text
WikiText-2
sequence length = 2048
固定 tokenizer/version
关闭 sampling
报告绝对 PPL 与相对各自 source 的 delta
```

- [ ] **Step 9: 下游任务设置**

复用项目现有评测套件，并覆盖至少：

```text
常识/阅读
数学或推理
MMLU 类知识
```

所有路径统一 prompt、shots、batch size、seed 和评测版本。

- [ ] **Step 10: 关联分析**

比较：

```text
global weight NMSE vs PPL delta
output NMSE vs PPL delta
高误差层/类别 vs 任务退化
PTS 数值差异是否达到模型指标可见量级
```

E10 不属于第一版 Torch 脚本完成门槛。

---

## Task 14: 最终验证与交付

**Files:**
- Modify only if verification finds a defect:
  `nvfp4_hif4_torch.py`
  `tests/test_nvfp4_hif4_torch_core.py`
  `tests/test_nvfp4_hif4_torch_eval.py`

**Interfaces:**
- Produces: 可交付的新脚本、测试、实验协议和至少一组真实权重结果。

- [ ] **Step 1: 完整测试**

```bash
python -m unittest \
  tests.test_nvfp4_hif4_torch_core \
  tests.test_nvfp4_hif4_torch_eval -v
```

Expected: 全部 PASS。

- [ ] **Step 2: 编译**

```bash
python -m py_compile \
  nvfp4_hif4_torch.py \
  tests/test_nvfp4_hif4_torch_core.py \
  tests/test_nvfp4_hif4_torch_eval.py
```

Expected: exit code 0。

- [ ] **Step 3: 绿地边界检查**

```bash
rg -n \
  'reproduce_nvfp4_to_hif4|test_reproduce_nvfp4_to_hif4|import numpy|from numpy|np\.' \
  nvfp4_hif4_torch.py \
  tests/test_nvfp4_hif4_torch_core.py \
  tests/test_nvfp4_hif4_torch_eval.py
```

Expected: 0 匹配。

- [ ] **Step 4: 旧语义残留检查**

```bash
rg -n \
  'without_pts|no_pts|fresh_nvfp4|cascade_error|double_quantization_loss|total_minus' \
  nvfp4_hif4_torch.py \
  tests/test_nvfp4_hif4_torch_core.py \
  tests/test_nvfp4_hif4_torch_eval.py
```

Expected: 0 匹配。

- [ ] **Step 5: quick simulation**

```bash
python nvfp4_hif4_torch.py simulate \
  --quick \
  --device cpu \
  --output-dir /tmp/nvfp4_hif4_greenfield_final
```

Expected: exit code 0，E1–E7 与三个输出文件齐全。

- [ ] **Step 6: 最小 tensor API smoke**

```bash
python - <<'PY'
import torch
import nvfp4_hif4_torch as m

x = torch.randn(64, 128)
nv = m.simulate_nvfp4(x)
nv_result = m.evaluate_nvfp4_fake_weight(
    nv.values,
    pts_scale=nv.global_scale,
)
bf_result = m.evaluate_bf16_weight(x)
assert nv_result["paths"]["direct"]["metrics"]["nmse"] >= 0
assert bf_result["metrics"]["nmse"] >= 0
PY
```

- [ ] **Step 7: 最小 checkpoint CLI smoke**

创建只含一个 `[64,128]` q_proj 的临时 state dict，分别以
`input_kind=nvfp4_fake` 和 `input_kind=bf16` 跑通。确认 NV 缺 PTS 时只
有 direct。

- [ ] **Step 8: 科学语义审查**

逐项签字确认：

```text
NV direct/PTS 使用同一个 W_NV
BF16-native reference 是 BF16 投影值
缺 PTS 不推断
container dtype 不决定 input_kind
真实权重沿 K 维分组
16/32 标为非标准
全局指标用能量累计
无 BF16->NVFP4 历史误差混入
无旧脚本调用或输出对齐
```

- [ ] **Step 9: 检查任务拥有的文件范围**

实现任务应只新增/修改：

```text
nvfp4_hif4_torch.py
tests/test_nvfp4_hif4_torch_core.py
tests/test_nvfp4_hif4_torch_eval.py
requirements-hif4-torch.txt  # 仅在需要时
实验输出目录
```

不得为了完成此计划改写旧 NumPy 脚本。

- [ ] **Step 10: 最终提交**

```bash
git add nvfp4_hif4_torch.py \
        tests/test_nvfp4_hif4_torch_core.py \
        tests/test_nvfp4_hif4_torch_eval.py
git commit -m "feat: complete greenfield torch nvfp4 hif4 evaluation"
```

如果创建了依赖文件，将其一并加入。

---

## 8. 最终必须生成的表与图

### 主表 1：合成 native conversion

| Distribution | NV direct | NV PTS-FP32 | NV PTS-BF16 | BF16-native | PTS-BF16 minus direct | 95% CI |
|---|---:|---:|---:|---:|---:|---:|

四种 distribution 全部填写；NMSE 同时提供 ratio 和百分比展示。

### 主表 2：真实模型 native conversion

| Model | Category | NV direct | NV PTS-BF16 | BF16-native | Tensors | Numel | PTS coverage |
|---|---|---:|---:|---:|---:|---:|---:|

主表只包含 q/k/v/o/gate/up/down。

### 消融表 1：BF16 carrier

| Scope | Projection NMSE | Exact fraction | Final HiF4 NMSE delta | Reconstruction equal fraction |
|---|---:|---:|---:|---:|

三行：

```text
legal E4M3FN×E2M1 products
synthetic W_NV/s_T
real model W_NV/s_T
```

### 消融表 2：scale path

| Source | Distribution | Continuous | BF16 math | E6M2 only | Hardware |
|---|---|---:|---:|---:|---:|

### 消融表 3：group size

| Source | Distribution | Scale mode | G16 | G32 | G64 |
|---|---|---|---:|---:|---:|

脚注注明 G16/G32 非标准。

### 消融表 4：storage dtype

| Distribution/Model | NV FP32→BF16 storage NMSE | FP32-container native NMSE | BF16-container native NMSE | HiF4 equal fraction |
|---|---:|---:|---:|---:|

### 图 1：PTS phase sweep

- x：\(s_T\) 尾数相位，范围 `[1,2)`，对数/线性横轴任选但需标注；
- y：conversion NMSE；
- 三条线：direct、PTS-FP32、PTS-BF16；
- 标出 PTS-BF16 优于 direct 的区间。

### 图 2：真实模型逐层 NMSE

- x：layer index；
- y：NMSE；
- q/k/v/o/gate/up/down 分面或分色；
- 对比 NV direct、NV PTS-BF16、BF16-native；
- source reference 不同，图注必须说明这是 native difficulty comparison。

### 图 3：逐 group NMSE CDF

- 按权重类别绘制；
- 重点显示 p90–p99 尾部；
- direct 与 PTS-BF16 对比；
- 不把零 reference group 作为普通有限 NMSE 混入。

### 图 4：weight NMSE 与 output NMSE

仅 E9 完成后生成：

- 每点一个 tensor；
- 颜色为 category；
- direct/PTS 使用不同 marker；
- 标注 Spearman；
- 单独列出方向不一致的 tensor。

---

## 9. 明确禁止的实现与实验错误

1. 修改或“顺手清理”旧的 `reproduce_nvfp4_to_hif4.py`；
2. 从旧脚本复制函数后只替换 `np` 为 `torch`；
3. import 旧模块并包一层 Torch tensor；
4. 用旧脚本的固定输出作为新实现正确性的唯一 oracle；
5. 用 BF16 权重先生成 NVFP4，再把级联误差称为 BF16-native；
6. 对真实 NV fake weight 再调用 `simulate_nvfp4`；
7. 没有真实 PTS 时用 `amax/(448*6)` 估计 PTS；
8. 把 `use_pts=False` 重新量化得到的另一个 NV tensor 当作“提出 PTS”；
9. 依据 storage dtype 自动判断 `input_kind`；
10. 对二维权重全局 flatten，导致跨行组成 HiF4 group；
11. 平均逐 tensor NMSE 作为全模型 NMSE；
12. 把 FP32→BF16 storage projection 加进 BF16-native conversion；
13. 把 group size 16/32 称为标准 HiF4；
14. 假设 PTS 必然优于 direct；
15. 把 `hardware-continuous` 拆成严格可加的 BF16/E6M2 误差；
16. 为首次 full run 根据观察结果即时造狭窄回归区间；
17. 默认返回或保存整模型所有 reconstruction，造成双倍内存；
18. 把所有 group 原始 NMSE 写入巨型 JSON；
19. 在同一表列混用比例和百分数而不标单位；
20. 用非法 JSON `NaN/Infinity`；
21. 为读取 checkpoint 静默启用不安全 pickle；
22. 因可选 `safetensors` 未安装而阻塞 `.pt/.pth`；
23. 没有确认 fake-quant 输入语义就下模型级结论；
24. 将 E9 proxy activation 结果声称为端到端精度；
25. 把 E10 当作第一版脚本完成的强制条件。

---

## 10. 完成定义

只有同时满足以下条件，才可声称计划实现完成：

- 新建 `nvfp4_hif4_torch.py`，且数值路径为纯 PyTorch；
- 旧 NumPy 脚本及其测试完全未修改、未导入；
- 新建的两份测试全部通过；
- 码本 midpoint、BF16 carrier、S1P2 边界和分组有独立 oracle；
- CPU/CUDA 一致性达到 E0 要求；
- NV direct、PTS-FP32、PTS-BF16 使用同一个 reference；
- BF16-native 使用独立 BF16 reference；
- 缺失 PTS 时只报告 direct；
- E1–E7 完整运行并生成 JSON/CSV/Markdown；
- `.pt/.pth` checkpoint 可评测；
- 安装可选依赖时 `.safetensors`/HF shard 可评测；
- 真实模型逐 tensor、逐类别和全局 energy aggregation 正确；
- 至少完成一组 7B/8B BF16 与 NV fake checkpoint 的独立损失实验；
- E9 API 可使用外部 activation 文件计算 output NMSE；
- 所有结果含配置、seed、dtype、device、原始能量和跳过原因；
- 任何结论都没有混入 BF16→NVFP4 历史误差；
- 没有把本工作描述成“移植旧脚本”。
