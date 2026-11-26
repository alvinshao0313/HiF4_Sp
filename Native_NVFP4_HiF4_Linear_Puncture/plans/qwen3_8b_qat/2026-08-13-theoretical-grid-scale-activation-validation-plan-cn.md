# NVFP4→HiF4 理论均权最优网格缩放：真实捕获激活验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 不重新跑 Qwen3-8B、不重新捕获激活，只复用 `Native_NVFP4_HiF4_Linear_Puncture` 已保存的 post-rotation / pre-quant `X_rot`，验证理论推导得到的均权绝对 MSE 最优网格缩放 `s*=1.299295470055` 在真实 NVFP4 QAT 激活上是否能降低 NVFP4→HiF4 的激活转换误差。

**Architecture:** 新增一个完全离线的 activation-only 验证入口。对同一份保存的 `X_rot`，先按 checkpoint 的原生 NVFP4 activation 语义重建 `A_N`，再分别计算标准 HiF4 `A_H,std` 与使用理论缩放后的 HiF4 `A_H,theory`；主指标固定为二者相对 `A_N` 的误差。理论缩放不从这些激活上重新搜索，不读取 validation 选择参数，也不加载模型/权重，从而严格回答“纯理论均权最优 scale 能否直接迁移到真实激活”。

**Tech Stack:** Python 3.11、PyTorch、pandas、matplotlib（仅用于一张必要对比图）；运行环境固定 `hif4` conda；复用当前目录的 `native_nvfp4.py`、`formats.py`、`metrics.py`、`io_utils.py` 与已有 capture 文件。

## Global Constraints

- 实验目录固定为 `Native_NVFP4_HiF4_Linear_Puncture/`，禁止写入 `Inference_Paradigm_Conversion/`。
- **只使用已经保存的 activation capture，不重新加载 Qwen3-8B，不重新 forward，不重新 capture。**
- 本次正式输入 capture 固定来自：`results/20260812T103800Z_native_nvfp4_hif4_linear_puncture/`。
- 正式 capture 必须满足：`capture_mode=formal`、`capture_coverage=35/35`、`capture_point=post_rotation_pre_activation_quant`、`source_semantic_version=native_nvfp4_rot_a4_v1`；任何一项不满足直接失败。
- 正式被测模块固定为已有 5 层 × 7 projection = 35 个 Linear；`cal` 与 `val` 两个 split 都评估，但**不使用任一 split 搜索或修改理论 scale**。
- 理论 scale 固定：`THEORY_GRID_SCALE = 1.299295470055`。这是前序数学推导中“NVFP4 / HiF4 theoretical unique grid，每个 NVFP4 点等权、绝对 squared error 最小”的结果，不允许 Cursor 再用真实 activation 搜索新的 scale。
- 标准 HiF4 配置继续固定：`group_size=64, group_dim=-1, s0_divisor=7.0, e8_threshold=4.0, e4_threshold=2.0, s0_mode="hardware"`。
- 本实验把理论 scale 落到当前真实 HiF4 quantizer 的唯一方式固定为修改 S0 divisor：

  ```text
  s0_raw,std    = amax64 / 7
  s0_raw,theory = s* × amax64 / 7
                 = amax64 / (7 / s*)
  ```

  因而：

  ```text
  THEORY_S0_DIVISOR = 7 / 1.299295470055
                    = 5.387535138334382
  ```

- 除 `s0_divisor` 外，HiF4 的 `group_size / group_dim / e8_threshold / e4_threshold / s0_mode / enable_exp8 / enable_exp4` 全部保持标准值，禁止额外调阈值、group size、对角阵 D 或 weight。
- 这里验证的是“理论 lattice phase 通过 S0 divisor 注入当前硬件语义 HiF4 后是否有实际收益”。由于 `S0` 还要做 BF16/E6M2 rounding、e8/e4 也会依据新的 S0 重新判定，因此它**不是**对理论连续网格 `s·H` 的逐点完全等价实现；报告必须明确这一边界。
- 主 reference 固定为原生 NVFP4 activation：

  ```text
  A_N = Q_NVFP4(X_rot; input_global_scale_fp32)
  ```

  主问题是 `A_H` 相对 `A_N` 的转换误差，而不是 HiF4 相对 BF16 的普通量化误差。
- `X_rot` 作为第二 reference 只做诊断：用于判断理论 scale 是否只是“更像 NVFP4”，同时却离原始 pre-quant activation 更远。
- 不做 Linear GEMM，不加载权重，不计算 output NMSE，不做 end-to-end benchmark。
- 不修改已有 `linear_cases.py`、`diagonal_search.py`、capture 数据和已有正式结果；本实验必须是独立增量入口。
- 全局 MSE/NMSE 必须通过总 `error_energy / num_elements` 或总 `error_energy / total_reference_energy` 计算，禁止对 35 个 module 的 MSE/NMSE 做简单算术平均作为主结论。
- 不引入 fallback、自动重新 capture、自动替换 capture run、自动搜索 scale。

---

# 1. 先锁死本实验的数学问题

对每个已保存的真实 post-rotation 激活张量 `X_rot`：

```text
A_N       = Q_NVFP4(X_rot; checkpoint input_global_scale)
A_H,std   = Q_HiF4(X_rot; s0_divisor=7.0)
A_H,theory= Q_HiF4(X_rot; s0_divisor=5.387535138334382)
```

其中理论 scale 的来源是：

```text
s* = 1.299295470055
```

在 theoretical unique lattice 上，它最小化：

```text
(1/M) * sum_i min_h (n_i - s h)^2
```

本实验**不再求 s**，只验证这个固定 `s*` 对真实 activation 是否有迁移收益。

## 1.1 主指标：NVFP4→HiF4 conversion absolute MSE

每个 capture tensor：

```text
E_std    = Σ (A_H,std    - A_N)^2
E_theory = Σ (A_H,theory - A_N)^2
N        = numel(A_N)

MSE_std    = E_std / N
MSE_theory = E_theory / N

Recovery_MSE = 1 - E_theory / E_std
```

由于两种方案的 `N` 完全相同，使用 error energy 或 MSE 算 recovery 等价。

全局 split 结果必须使用：

```text
Global_MSE_std    = Σ_module E_std    / Σ_module N
Global_MSE_theory = Σ_module E_theory / Σ_module N
Global_Recovery   = 1 - Σ_module E_theory / Σ_module E_std
```

**主结论只看 validation split 的 `Global_Recovery`。** `cal` 只用于看方向是否一致，不参与任何参数选择。

## 1.2 同时记录 conversion NMSE

```text
RefEnergy_AN = Σ A_N^2
NMSE_std     = E_std    / RefEnergy_AN
NMSE_theory  = E_theory / RefEnergy_AN
```

全局 NMSE 同样必须先累计 numerator / denominator，再相除。

## 1.3 第二 reference：相对 X_rot 的普通 HiF4 量化误差

同时记录：

```text
E_std_x    = Σ (A_H,std    - X_rot)^2
E_theory_x = Σ (A_H,theory - X_rot)^2
```

目的仅有一个：区分下面两种情况。

```text
情况 A：vs A_N 改善，同时 vs X_rot 也改善
=> 理论 scale 不仅更贴近 NVFP4 source，也提高了对原始 activation 的重构。

情况 B：vs A_N 改善，但 vs X_rot 恶化
=> 理论 scale 主要是在匹配 NVFP4 lattice；它对“转换”有利，但不是普通 HiF4 activation MSE 的统一最优解。
```

不要用 `vs X_rot` 覆盖主结论。

## 1.4 零点诊断

至少记录：

```text
zero_rate_an
zero_rate_hif4_std
zero_rate_hif4_theory
an_nonzero_to_hif4_std_zero_rate
an_nonzero_to_hif4_theory_zero_rate
```

其中：

```python
mask_an_nonzero = A_N != 0
an_nonzero_to_h_zero_rate = ((A_H == 0) & mask_an_nonzero).sum() / mask_an_nonzero.sum()
```

如果 `A_N` 全零则该项为 NaN，禁止除零。

---

# 2. 文件结构与最短实现路径

只新增以下文件，不修改已有正式 runner：

```text
Native_NVFP4_HiF4_Linear_Puncture/
├── src/
│   └── grid_scale_validation.py                 # 新增：全部实验逻辑
├── scripts/
│   └── run_grid_scale_validation.sh             # 新增：hif4 环境入口
├── tests/
│   └── test_grid_scale_validation.py            # 新增：最小必要单元测试
└── plans/
    └── 2026-08-13-theoretical-grid-scale-activation-validation-plan-cn.md
```

实验运行后新增结果目录：

```text
results/<NEW_RUN_ID>/
├── grid_scale_per_module.csv
├── grid_scale_by_projection.csv
├── grid_scale_summary.json
├── grid_scale_report_cn.md
└── figures/
    └── fig01_grid_scale_conversion_mse_recovery.png
```

禁止把结果写回 source capture run 目录，避免污染原始 capture。

---

# 3. Task 1：实现理论 scale 到 HiF4 S0 divisor 的唯一映射

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/src/grid_scale_validation.py`
- Test: `Native_NVFP4_HiF4_Linear_Puncture/tests/test_grid_scale_validation.py`

**Interfaces:**

```python
THEORY_GRID_SCALE: float = 1.299295470055


def theory_s0_divisor(base_divisor: float = 7.0) -> float:
    """Return base_divisor / THEORY_GRID_SCALE."""
    if base_divisor <= 0:
        raise ValueError("base_divisor must be positive")
    return float(base_divisor / THEORY_GRID_SCALE)


def build_hif4_config(
    *,
    group_size: int,
    s0_divisor: float,
    e8_threshold: float,
    e4_threshold: float,
    s0_mode: str,
) -> HiF4QuantConfig:
    return HiF4QuantConfig(
        group_size=group_size,
        group_dim=-1,
        s0_divisor=s0_divisor,
        e8_threshold=e8_threshold,
        e4_threshold=e4_threshold,
        s0_mode=s0_mode,
        enable_exp8=True,
        enable_exp4=True,
    )
```

要求：

```text
theory_s0_divisor(7.0) == 5.387535138334382（允许 1e-12 浮点误差）
```

- [x] **Step 1: 写失败测试，锁死 scale 数值和方向**

```python
def test_theory_s0_divisor_scales_grid_up():
    d = theory_s0_divisor(7.0)
    assert abs(d - 5.387535138334382) < 1e-12
    assert d < 7.0
```

这里 `d < 7` 是方向 gate：因为 `s*>1` 意味着 `S0_raw` 应该更大，所以 divisor 必须更小。禁止写反成 `7*s*`。

- [x] **Step 2: 写配置保持性测试**

构造 standard/theory 两个 `HiF4QuantConfig`，明确断言只有 `s0_divisor` 不同，其余字段完全一致。

- [x] **Step 3: 运行最小测试确认先失败**

```bash
conda run -n hif4 --no-capture-output \
  pytest Native_NVFP4_HiF4_Linear_Puncture/tests/test_grid_scale_validation.py -q
```

- [x] **Step 4: 实现最小函数使测试通过**

不要修改 `formats.py` 的 standard 常量；本实验在新模块内显式构造 config，并继续调用现有：

```python
qdq_hif4_direct(x, config=cfg, output_dtype=torch.float32)
```

- [x] **Step 5: 再运行测试确认通过**

---

# 4. Task 2：实现单个 capture 的三路离线量化与指标

**Files:**
- Modify: `src/grid_scale_validation.py`
- Modify: `tests/test_grid_scale_validation.py`

**Consumes:**

capture `.pt` 中已有字段：

```text
module_name
layer_idx
projection
split
x_rot_bf16
input_global_scale_fp32
```

**Produces:**

```python
def nonzero_to_zero_rate(source: torch.Tensor, target: torch.Tensor) -> float:
    source_flat = source.detach().reshape(-1)
    target_flat = target.detach().reshape(-1)
    mask = source_flat != 0
    denom = int(mask.sum().item())
    if denom == 0:
        return float("nan")
    numer = int(((target_flat == 0) & mask).sum().item())
    return float(numer / denom)


def evaluate_capture(
    capture: dict[str, Any],
    *,
    hif4_base_config: HiF4QuantConfig,
    hif4_theory_config: HiF4QuantConfig,
    nvfp4_group_size: int,
    device: torch.device,
) -> dict[str, Any]:
    x = capture["x_rot_bf16"].to(device=device, dtype=torch.float32)
    scale = capture["input_global_scale_fp32"].to(device=device, dtype=torch.float32)

    a_n = qdq_nvfp4_post_rotation(
        x, scale, group_size=nvfp4_group_size
    ).to(torch.float32)
    a_h_std = qdq_hif4_direct(
        x, config=hif4_base_config, output_dtype=torch.float32
    )
    a_h_theory = qdq_hif4_direct(
        x, config=hif4_theory_config, output_dtype=torch.float32
    )

    n = int(a_n.numel())
    e_std_an = error_energy(a_h_std, a_n)
    e_theory_an = error_energy(a_h_theory, a_n)
    e_std_x = error_energy(a_h_std, x)
    e_theory_x = error_energy(a_h_theory, x)
    ref_an = reference_energy(a_n)
    ref_x = reference_energy(x)

    return {
        "num_elements": n,
        "error_energy_std_vs_an": e_std_an,
        "error_energy_theory_vs_an": e_theory_an,
        "mse_std_vs_an": e_std_an / n,
        "mse_theory_vs_an": e_theory_an / n,
        "nmse_std_vs_an": e_std_an / max(ref_an, 1e-30),
        "nmse_theory_vs_an": e_theory_an / max(ref_an, 1e-30),
        "recovery_mse_vs_an": float("nan") if e_std_an == 0 else 1.0 - e_theory_an / e_std_an,
        "error_energy_std_vs_xrot": e_std_x,
        "error_energy_theory_vs_xrot": e_theory_x,
        "mse_std_vs_xrot": e_std_x / n,
        "mse_theory_vs_xrot": e_theory_x / n,
        "recovery_mse_vs_xrot": float("nan") if e_std_x == 0 else 1.0 - e_theory_x / e_std_x,
        "reference_energy_an": ref_an,
        "reference_energy_xrot": ref_x,
        "zero_rate_an": zero_rate(a_n),
        "zero_rate_hif4_std": zero_rate(a_h_std),
        "zero_rate_hif4_theory": zero_rate(a_h_theory),
        "an_nonzero_to_hif4_std_zero_rate": nonzero_to_zero_rate(a_n, a_h_std),
        "an_nonzero_to_hif4_theory_zero_rate": nonzero_to_zero_rate(a_n, a_h_theory),
    }
```

- [x] **Step 1: 先写 synthetic 64-d 测试**

使用固定 seed 构造 `[4, 64]` FP32/BF16 tensor，并用当前 `qdq_nvfp4_post_rotation` 生成 `A_N`。测试只检查以下语义，不预设理论 scale 一定改善 synthetic 随机数据：

```text
A_N 的 reference 由 qdq_nvfp4_post_rotation 生成
standard 与 theory 都直接量化同一个 X_rot
两个 variant 输出 shape/dtype 正确
num_elements 与输入一致
所有 error energy >= 0
```

- [x] **Step 2: 明确计算三组 tensor**

```python
x = capture["x_rot_bf16"].to(device=device, dtype=torch.float32)
scale = capture["input_global_scale_fp32"].to(device=device, dtype=torch.float32)

a_n = qdq_nvfp4_post_rotation(
    x,
    scale,
    group_size=nvfp4_group_size,
).to(torch.float32)

a_h_std = qdq_hif4_direct(
    x,
    config=hif4_base_config,
    output_dtype=torch.float32,
)

a_h_theory = qdq_hif4_direct(
    x,
    config=hif4_theory_config,
    output_dtype=torch.float32,
)
```

当前 `qdq_nvfp4_post_rotation` 已固定内部输出 BF16，因此正式实现必须按上述现有签名调用后 `.to(torch.float32)`；禁止为本实验修改 NVFP4 QDQ 数学语义或函数签名。

- [x] **Step 3: 指标全部用 FP64 accumulator**

复用 `metrics.py` 的 `error_energy / reference_energy / nmse / zero_rate`；`nonzero_to_zero_rate()` 使用本 Task 前面已经锁死的实现，不再创建第二套统计函数。不要为了这一个实验修改通用 `metrics.py`。

每 module/split 的一行结果至少包含：

```text
run_id
source_capture_run_id
split
layer_idx
projection
module_name
num_elements

theory_grid_scale
standard_s0_divisor
theory_s0_divisor

error_energy_std_vs_an
error_energy_theory_vs_an
mse_std_vs_an
mse_theory_vs_an
nmse_std_vs_an
nmse_theory_vs_an
recovery_mse_vs_an

error_energy_std_vs_xrot
error_energy_theory_vs_xrot
mse_std_vs_xrot
mse_theory_vs_xrot
recovery_mse_vs_xrot

reference_energy_an
reference_energy_xrot

zero_rate_an
zero_rate_hif4_std
zero_rate_hif4_theory
an_nonzero_to_hif4_std_zero_rate
an_nonzero_to_hif4_theory_zero_rate
```

- [x] **Step 4: recovery 的零分母规则锁死**

若 `E_std == 0`：

```text
recovery_mse_vs_an = NaN
```

禁止填 0 或 1。

- [x] **Step 5: 运行 unit tests**

---

# 5. Task 3：只读复用正式 capture run，增加严格 manifest gate

**Files:**
- Modify: `src/grid_scale_validation.py`
- Modify: `tests/test_grid_scale_validation.py`

**CLI 固定：**

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.src.grid_scale_validation \
  --config Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml \
  --capture-run-id 20260812T103800Z_native_nvfp4_hif4_linear_puncture \
  --run-id <NEW_RUN_ID> \
  --device cuda:0
```

不要提供 `--theory-scale` 参数。理论 scale 必须锁死在代码常量，防止运行时把这个实验偷换成 data-driven search。

- [x] **Step 1: manifest gate**

读取：

```text
results/<capture-run-id>/capture_manifest.json
```

严格检查：

```text
capture_mode == "formal"
capture_coverage == "35/35"
module_count == 35
capture_point == "post_rotation_pre_activation_quant"
source_semantic_version == "native_nvfp4_rot_a4_v1"
formal_layers == [2,10,18,26,34]
```

并检查 `config.formal_module_names` 的 35×2 个 `{cal,val}` capture 文件全部存在。

任何缺失立即 fail；禁止跳过缺失 module 后继续汇总。

- [x] **Step 2: 禁止 source/destination run 相同**

如果：

```text
capture_run_id == run_id
```

立即失败，防止覆盖原实验。

- [x] **Step 3: 逐文件串行处理，避免不必要的显存峰值**

35 module × 2 split 逐个 load → GPU 量化 → 立即转为标量指标 → 删除 tensor；不要一次性把 70 个 activation tensor 搬上 GPU。

- [x] **Step 4: 写 `grid_scale_per_module.csv`**

固定 70 行：

```text
35 modules × 2 splits = 70 rows
```

不是 variant-long 两倍行数；同一行直接并列 standard/theory，便于一眼看 recovery。

---

# 6. Task 4：实现全局与 projection 聚合，禁止平均 NMSE 误导

**Files:**
- Modify: `src/grid_scale_validation.py`
- Modify: `tests/test_grid_scale_validation.py`

## 6.1 split-global 聚合

对 `cal` 和 `val` 分别计算：

```text
N_total = Σ num_elements
E_std_total = Σ error_energy_std_vs_an
E_theory_total = Σ error_energy_theory_vs_an
R_AN_total = Σ reference_energy_an

Global_MSE_std = E_std_total / N_total
Global_MSE_theory = E_theory_total / N_total
Global_NMSE_std = E_std_total / R_AN_total
Global_NMSE_theory = E_theory_total / R_AN_total
Global_Recovery = 1 - E_theory_total / E_std_total
```

相对 `X_rot` 的聚合同理。

还要记录：

```text
num_modules_positive_recovery
fraction_modules_positive_recovery
median_module_recovery
min_module_recovery
max_module_recovery
```

这些只是稳定性诊断，不能替代 global energy 结果。

## 6.2 projection 聚合

按：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

分别跨 5 层累计 error energy / numel / reference energy，输出：

```text
grid_scale_by_projection.csv
```

每个 split × projection 一行，共：

```text
2 × 7 = 14 rows
```

至少包含：

```text
split
projection
num_modules
num_elements
global_mse_std_vs_an
global_mse_theory_vs_an
global_nmse_std_vs_an
global_nmse_theory_vs_an
recovery_vs_an
recovery_vs_xrot
zero_rate_delta_std_to_theory
an_nonzero_to_zero_delta_std_to_theory
```

## 6.3 单元测试锁死聚合方式

构造两个 module：一个小 tensor、一个大 tensor，使“module MSE 算术平均”和“总 error / 总 elements”明显不同。断言正式 aggregate 选择后者。

---

# 7. Task 5：输出最小但足够解释结论的 summary / report / figure

**Files:**
- Modify: `src/grid_scale_validation.py`

不改现有 `report.py`。本次实验小而独立，直接由新模块输出一个短报告。

## 7.1 `grid_scale_summary.json`

固定结构：

```json
{
  "experiment": "theoretical_uniform_abs_mse_grid_scale_activation_validation",
  "source_capture_run_id": "20260812T103800Z_native_nvfp4_hif4_linear_puncture",
  "theory": {
    "objective": "uniform unique NVFP4 grid points, absolute squared error",
    "grid_scale": 1.299295470055,
    "standard_s0_divisor": 7.0,
    "theory_s0_divisor": 5.387535138334382,
    "scale_selected_from_activation_data": false
  },
  "cal": {},
  "val": {},
  "conclusion": {}
}
```

`conclusion` 只能依据 val 结果机械生成：

```text
val Global_Recovery > 0
=> theory_scale_helpful_on_captured_activation = true

val Global_Recovery <= 0
=> false
```

同时记录真实数值，禁止只输出 boolean。

## 7.2 `grid_scale_report_cn.md`

报告保持 1~2 页，顺序固定：

1. 实验问题；
2. 理论 scale 的来源与实际使用方式；
3. 数据来源：已有 35 Linear capture；
4. validation 全局结果；
5. 7 类 projection 结果；
6. `vs A_N` 与 `vs X_rot` 是否方向一致；
7. zero / NV非零→HiF4零变化；
8. 明确结论；
9. 边界：没有重新搜索 scale、没有 Linear output、没有 E2E。

禁止在本报告中宣称：

```text
“1.2993 是真实 activation 最优 scale”
“1.2993 是部署最优 scale”
“已经证明 occupancy weighting 无需做”
```

只能说：

```text
固定的理论均权 scale 在真实 captured activation 上有/没有直接迁移收益。
```

## 7.3 只画一张必要图

`fig01_grid_scale_conversion_mse_recovery.png`：

- x 轴：7 projection；
- y 轴：validation `recovery_vs_an`（百分比）；
- 再画一条/一组 global validation recovery 参考；
- 图标题明确 `s=1.29929547, S0 divisor=5.38753514`。

不生成大批装饰性图。

---

# 8. Task 6：增加 shell 入口并执行闭环验证

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/scripts/run_grid_scale_validation.sh`

脚本风格跟随现有 `run_linear_cases.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CAPTURE_RUN_ID=${CAPTURE_RUN_ID:-20260812T103800Z_native_nvfp4_hif4_linear_puncture}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_theory_grid_scale_validation}
CONFIG=${CONFIG:-Native_NVFP4_HiF4_Linear_Puncture/configs/qwen3_8b_native_nvfp4_linear_puncture.yaml}
DEVICE=${DEVICE:-cuda:0}

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

conda run -n hif4 --no-capture-output \
  python -m Native_NVFP4_HiF4_Linear_Puncture.src.grid_scale_validation \
  --config "$CONFIG" \
  --capture-run-id "$CAPTURE_RUN_ID" \
  --run-id "$RUN_ID" \
  --device "$DEVICE"
```

- [x] **Step 1: 运行本实验新增 unit tests**

```bash
conda run -n hif4 --no-capture-output \
  pytest Native_NVFP4_HiF4_Linear_Puncture/tests/test_grid_scale_validation.py -q
```

必须 PASS。

- [x] **Step 2: 运行当前目录原有测试，确认没有破坏既有行为**

```bash
conda run -n hif4 --no-capture-output \
  pytest Native_NVFP4_HiF4_Linear_Puncture/tests -q
```

必须 PASS。

- [x] **Step 3: 正式运行离线 activation 实验**

```bash
CAPTURE_RUN_ID=20260812T103800Z_native_nvfp4_hif4_linear_puncture \
DEVICE=cuda:0 \
Native_NVFP4_HiF4_Linear_Puncture/scripts/run_grid_scale_validation.sh
```

这个步骤不应该加载 8B 模型；若日志出现 checkpoint shard/model loading，说明实现路径错误，立即停止检查。

- [x] **Step 4: 输出完整性 gate**

正式 run 结束后必须检查：

```text
grid_scale_per_module.csv        70 rows
grid_scale_by_projection.csv     14 rows
grid_scale_summary.json          存在且含 cal/val
grid_scale_report_cn.md          存在
figures/fig01_*.png               存在
```

同时检查：

```text
所有 error energy finite 且 >=0
所有 MSE/NMSE finite（零 reference 的特殊情况除外）
theory_grid_scale == 1.299295470055
theory_s0_divisor == 5.387535138334382
scale_selected_from_activation_data == false
35/35 module 在 cal/val 均出现
```

- [x] **Step 5: 人工复核 val 主结论**

最终只回答四个数字：

```text
1. standard HiF4 vs A_N validation global MSE
2. theory-scale HiF4 vs A_N validation global MSE
3. validation global recovery
4. 35 个 module 中有多少个 recovery > 0
```

然后再补 7 projection 的 recovery 表。

---

# 9. 实验结果应该怎样解释

## 9.1 如果 `Global_Recovery > 0`

可以得出：

```text
理论上由“均权绝对 MSE 网格对齐”得到的 s*=1.29929547，
在没有读取真实 activation occupancy 重新搜索的情况下，
直接迁移到真实 Qwen3-8B NVFP4 QAT captured activation 后，
仍能降低 NVFP4→HiF4 activation conversion MSE。
```

这说明 theoretical lattice mismatch 与真实数据之间存在可迁移联系，后续继续做 occupancy-weighted 最优 scale 有充分价值。

## 9.2 如果 `Global_Recovery <= 0`

不能推翻前面的数学推导。正确结论是：

```text
均权 unique-grid 的 theoretical optimum 没有直接转化成真实 activation optimum。
```

最可能需要下一步回答的是：真实 grid occupancy 权重改变后，最优 scale 是否显著偏离 1.2993。

不要在本实验里临时搜索新的 scale 来“救结果”；那属于下一阶段 occupancy-weighted 实验，必须保持分开。

## 9.3 如果 cal 与 val 方向不同

如实报告，不做任何调参。理论 scale 本身没有拟合 cal，因此不存在“根据 cal 调到 val”的步骤。

---

# 10. 明确禁止 Cursor 自由发挥的事项

- 不重新跑 capture。
- 不加载完整模型。
- 不加载或量化权重。
- 不做 Linear output。
- 不把 `s=1.2993` 和已有 channel-wise diagonal `D` 混在一起。
- 不搜索 `s`。
- 不搜索 `s0_divisor`。
- 不调 e8/e4 threshold。
- 不改 group size。
- 不把 validation 用于参数选择。
- 不引入 occupancy weighting；那是下一份实验计划。
- 不用 raw per-module average 作为全局主指标。
- 不修改已有正式结果文件。
- 不在失败时 fallback 到其他 capture run。

---

# 11. 最终验收标准

计划执行完成后，必须能够用一句话回答：

> 固定理论均权绝对 MSE 最优 scale `s*=1.299295470055`，通过 `s0_divisor: 7.0 → 5.387535138334382` 注入标准 HiF4 activation quantizer 后，在已有 35 个真实 NVFP4 QAT post-rotation activation captures 的 validation split 上，相对原生 `A_N` 的全局 conversion MSE 是改善还是恶化，幅度是多少？

验收所需最小证据：

```text
70-row per-module CSV
14-row projection CSV
1 个 summary JSON
1 个短中文报告
1 张 projection recovery 图
新增 tests + 原目录 tests 全通过
```

本计划到此结束。下一阶段如果用户确认继续，再单独设计“统计每个 NVFP4 internal lattice point occupancy，并按 occupancy 推导/求解新的最优 scale”的实验；不得提前混入本次验证。