"""Offline NVFP4→HiF4 theoretical grid-scale activation validation.

Reuses saved post-rotation captures. Does not load the 8B model, weights,
or search a new scale.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from Native_NVFP4_HiF4_Linear_Puncture.src.config import (
    TARGET_PROJECTIONS,
    AppConfig,
    Hif4Config,
    load_config,
    results_dir,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import HiF4QuantConfig, qdq_hif4_direct
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import (
    ensure_dir,
    load_pt,
    module_capture_stem,
    read_json,
    write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.metrics import (
    compute_recovery,
    error_energy,
    reference_energy,
    zero_rate,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.native_nvfp4 import qdq_nvfp4_post_rotation

THEORY_GRID_SCALE: float = 1.299295470055
STANDARD_S0_DIVISOR: float = 7.0
THEORY_S0_DIVISOR: float = 5.387535138334382
REQUIRED_FORMAL_LAYERS: tuple[int, ...] = (2, 10, 18, 26, 34)
REQUIRED_MODULE_COUNT = 35
SPLITS: tuple[str, ...] = ("cal", "val")
NMSE_EPS = 1e-30

PER_MODULE_COLUMNS = [
    "run_id",
    "source_capture_run_id",
    "split",
    "layer_idx",
    "projection",
    "module_name",
    "num_elements",
    "theory_grid_scale",
    "standard_s0_divisor",
    "theory_s0_divisor",
    "error_energy_std_vs_an",
    "error_energy_theory_vs_an",
    "mse_std_vs_an",
    "mse_theory_vs_an",
    "nmse_std_vs_an",
    "nmse_theory_vs_an",
    "recovery_mse_vs_an",
    "error_energy_std_vs_xrot",
    "error_energy_theory_vs_xrot",
    "mse_std_vs_xrot",
    "mse_theory_vs_xrot",
    "recovery_mse_vs_xrot",
    "reference_energy_an",
    "reference_energy_xrot",
    "zero_rate_an",
    "zero_rate_hif4_std",
    "zero_rate_hif4_theory",
    "an_nonzero_to_hif4_std_zero_rate",
    "an_nonzero_to_hif4_theory_zero_rate",
]

PROJECTION_COLUMNS = [
    "split",
    "projection",
    "num_modules",
    "num_elements",
    "global_mse_std_vs_an",
    "global_mse_theory_vs_an",
    "global_nmse_std_vs_an",
    "global_nmse_theory_vs_an",
    "recovery_vs_an",
    "recovery_vs_xrot",
    "zero_rate_delta_std_to_theory",
    "an_nonzero_to_zero_delta_std_to_theory",
]


def theory_s0_divisor(base_divisor: float = STANDARD_S0_DIVISOR) -> float:
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


def configs_differ_only_in_s0_divisor(
    standard: HiF4QuantConfig, theory: HiF4QuantConfig
) -> bool:
    d_std = asdict(standard)
    d_theory = asdict(theory)
    if d_std.keys() != d_theory.keys():
        return False
    for key in d_std:
        if key == "s0_divisor":
            if d_std[key] == d_theory[key]:
                return False
        elif d_std[key] != d_theory[key]:
            return False
    return True


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

    if a_n.shape != x.shape or a_h_std.shape != x.shape or a_h_theory.shape != x.shape:
        raise RuntimeError(
            f"QDQ shape mismatch: x={tuple(x.shape)} a_n={tuple(a_n.shape)} "
            f"a_h_std={tuple(a_h_std.shape)} a_h_theory={tuple(a_h_theory.shape)}"
        )
    if a_n.dtype != torch.float32 or a_h_std.dtype != torch.float32 or a_h_theory.dtype != torch.float32:
        raise RuntimeError("QDQ outputs must be float32")

    n = int(a_n.numel())
    if n != int(x.numel()):
        raise RuntimeError(f"num_elements mismatch: a_n={n} x={int(x.numel())}")

    e_std_an = error_energy(a_h_std, a_n)
    e_theory_an = error_energy(a_h_theory, a_n)
    e_std_x = error_energy(a_h_std, x)
    e_theory_x = error_energy(a_h_theory, x)
    ref_an = reference_energy(a_n)
    ref_x = reference_energy(x)

    for name, value in (
        ("error_energy_std_vs_an", e_std_an),
        ("error_energy_theory_vs_an", e_theory_an),
        ("error_energy_std_vs_xrot", e_std_x),
        ("error_energy_theory_vs_xrot", e_theory_x),
        ("reference_energy_an", ref_an),
        ("reference_energy_xrot", ref_x),
    ):
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f"{name} must be finite and >= 0, got {value}")

    return {
        "num_elements": n,
        "error_energy_std_vs_an": e_std_an,
        "error_energy_theory_vs_an": e_theory_an,
        "mse_std_vs_an": e_std_an / n,
        "mse_theory_vs_an": e_theory_an / n,
        "nmse_std_vs_an": e_std_an / max(ref_an, NMSE_EPS),
        "nmse_theory_vs_an": e_theory_an / max(ref_an, NMSE_EPS),
        "recovery_mse_vs_an": compute_recovery(e_std_an, e_theory_an),
        "error_energy_std_vs_xrot": e_std_x,
        "error_energy_theory_vs_xrot": e_theory_x,
        "mse_std_vs_xrot": e_std_x / n,
        "mse_theory_vs_xrot": e_theory_x / n,
        "recovery_mse_vs_xrot": compute_recovery(e_std_x, e_theory_x),
        "reference_energy_an": ref_an,
        "reference_energy_xrot": ref_x,
        "zero_rate_an": zero_rate(a_n),
        "zero_rate_hif4_std": zero_rate(a_h_std),
        "zero_rate_hif4_theory": zero_rate(a_h_theory),
        "an_nonzero_to_hif4_std_zero_rate": nonzero_to_zero_rate(a_n, a_h_std),
        "an_nonzero_to_hif4_theory_zero_rate": nonzero_to_zero_rate(a_n, a_h_theory),
    }


def capture_file_path(capture_dir: Path, module_name: str, split: str) -> Path:
    return capture_dir / "captures" / f"{module_capture_stem(module_name)}_{split}.pt"


def expected_capture_paths(config: AppConfig, capture_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for module_name in config.formal_module_names:
        for split in SPLITS:
            paths.append(capture_file_path(capture_dir, module_name, split))
    return paths


def assert_distinct_run_ids(capture_run_id: str, run_id: str) -> None:
    if capture_run_id == run_id:
        raise ValueError(
            "capture-run-id and run-id must differ; refusing to write into the source capture run"
        )


def assert_standard_hif4(hif4: Hif4Config) -> None:
    if hif4.group_size != 64:
        raise ValueError(f"HiF4 group_size must be 64, got {hif4.group_size}")
    if hif4.s0_divisor != STANDARD_S0_DIVISOR:
        raise ValueError(
            f"HiF4 s0_divisor must be {STANDARD_S0_DIVISOR}, got {hif4.s0_divisor}"
        )
    if hif4.e8_threshold != 4.0:
        raise ValueError(f"HiF4 e8_threshold must be 4.0, got {hif4.e8_threshold}")
    if hif4.e4_threshold != 2.0:
        raise ValueError(f"HiF4 e4_threshold must be 2.0, got {hif4.e4_threshold}")
    if hif4.s0_mode != "hardware":
        raise ValueError(f"HiF4 s0_mode must be 'hardware', got {hif4.s0_mode!r}")


def validate_capture_manifest(
    config: AppConfig, capture_dir: Path, manifest: dict[str, Any]
) -> None:
    checks = {
        "capture_mode": "formal",
        "capture_coverage": "35/35",
        "module_count": REQUIRED_MODULE_COUNT,
        "capture_point": "post_rotation_pre_activation_quant",
        "source_semantic_version": "native_nvfp4_rot_a4_v1",
    }
    for key, expected in checks.items():
        got = manifest.get(key)
        if got != expected:
            raise ValueError(f"capture manifest {key} must be {expected!r}, got {got!r}")
    layers = tuple(int(x) for x in manifest.get("formal_layers", ()))
    if layers != REQUIRED_FORMAL_LAYERS:
        raise ValueError(
            f"capture manifest formal_layers must be {list(REQUIRED_FORMAL_LAYERS)}, got {list(layers)}"
        )
    if tuple(int(x) for x in config.experiment.formal_layers) != REQUIRED_FORMAL_LAYERS:
        raise ValueError(
            f"config formal_layers must be {list(REQUIRED_FORMAL_LAYERS)}, "
            f"got {list(config.experiment.formal_layers)}"
        )
    if len(config.formal_module_names) != REQUIRED_MODULE_COUNT:
        raise ValueError(
            f"config.formal_module_names must have {REQUIRED_MODULE_COUNT} entries, "
            f"got {len(config.formal_module_names)}"
        )
    missing = [p for p in expected_capture_paths(config, capture_dir) if not p.is_file()]
    if missing:
        preview = ", ".join(str(p) for p in missing[:5])
        raise FileNotFoundError(
            f"missing {len(missing)} capture files; first missing: {preview}"
        )


def _safe_div(numer: float, denom: float) -> float:
    if denom == 0.0:
        return float("nan")
    return float(numer / denom)


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(r[key]) for r in rows))


def _weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    numer = 0.0
    denom = 0.0
    for row in rows:
        n = float(row["num_elements"])
        numer += float(row[key]) * n
        denom += n
    return _safe_div(numer, denom)


def _an_nonzero_to_zero_aggregate(rows: list[dict[str, Any]], key: str) -> float:
    numer = 0.0
    denom = 0.0
    for row in rows:
        n_nz = (1.0 - float(row["zero_rate_an"])) * float(row["num_elements"])
        value = float(row[key])
        if n_nz <= 0.0 or math.isnan(value):
            continue
        numer += value * n_nz
        denom += n_nz
    return _safe_div(numer, denom)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty row list")
    n_total = int(_sum(rows, "num_elements"))
    e_std_an = _sum(rows, "error_energy_std_vs_an")
    e_theory_an = _sum(rows, "error_energy_theory_vs_an")
    e_std_x = _sum(rows, "error_energy_std_vs_xrot")
    e_theory_x = _sum(rows, "error_energy_theory_vs_xrot")
    ref_an = _sum(rows, "reference_energy_an")
    ref_x = _sum(rows, "reference_energy_xrot")
    recoveries = np.array([float(r["recovery_mse_vs_an"]) for r in rows], dtype=np.float64)
    finite = recoveries[np.isfinite(recoveries)]
    num_positive = int(np.sum(finite > 0.0))
    return {
        "num_modules": len(rows),
        "num_elements": n_total,
        "error_energy_std_vs_an": e_std_an,
        "error_energy_theory_vs_an": e_theory_an,
        "global_mse_std_vs_an": _safe_div(e_std_an, n_total),
        "global_mse_theory_vs_an": _safe_div(e_theory_an, n_total),
        "global_nmse_std_vs_an": _safe_div(e_std_an, ref_an),
        "global_nmse_theory_vs_an": _safe_div(e_theory_an, ref_an),
        "global_recovery_vs_an": compute_recovery(e_std_an, e_theory_an),
        "error_energy_std_vs_xrot": e_std_x,
        "error_energy_theory_vs_xrot": e_theory_x,
        "global_mse_std_vs_xrot": _safe_div(e_std_x, n_total),
        "global_mse_theory_vs_xrot": _safe_div(e_theory_x, n_total),
        "global_nmse_std_vs_xrot": _safe_div(e_std_x, ref_x),
        "global_nmse_theory_vs_xrot": _safe_div(e_theory_x, ref_x),
        "global_recovery_vs_xrot": compute_recovery(e_std_x, e_theory_x),
        "reference_energy_an": ref_an,
        "reference_energy_xrot": ref_x,
        "num_modules_positive_recovery": num_positive,
        "fraction_modules_positive_recovery": float(num_positive / len(rows)),
        "median_module_recovery": float(np.nanmedian(recoveries)),
        "min_module_recovery": float(np.nanmin(recoveries)),
        "max_module_recovery": float(np.nanmax(recoveries)),
        "zero_rate_an": _weighted_mean(rows, "zero_rate_an"),
        "zero_rate_hif4_std": _weighted_mean(rows, "zero_rate_hif4_std"),
        "zero_rate_hif4_theory": _weighted_mean(rows, "zero_rate_hif4_theory"),
        "an_nonzero_to_hif4_std_zero_rate": _an_nonzero_to_zero_aggregate(
            rows, "an_nonzero_to_hif4_std_zero_rate"
        ),
        "an_nonzero_to_hif4_theory_zero_rate": _an_nonzero_to_zero_aggregate(
            rows, "an_nonzero_to_hif4_theory_zero_rate"
        ),
    }


def aggregate_by_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in SPLITS:
        for proj in TARGET_PROJECTIONS:
            sub = [r for r in rows if r["split"] == split and r["projection"] == proj]
            if not sub:
                raise RuntimeError(f"no rows for split={split} projection={proj}")
            agg = aggregate_rows(sub)
            out.append(
                {
                    "split": split,
                    "projection": proj,
                    "num_modules": agg["num_modules"],
                    "num_elements": agg["num_elements"],
                    "global_mse_std_vs_an": agg["global_mse_std_vs_an"],
                    "global_mse_theory_vs_an": agg["global_mse_theory_vs_an"],
                    "global_nmse_std_vs_an": agg["global_nmse_std_vs_an"],
                    "global_nmse_theory_vs_an": agg["global_nmse_theory_vs_an"],
                    "recovery_vs_an": agg["global_recovery_vs_an"],
                    "recovery_vs_xrot": agg["global_recovery_vs_xrot"],
                    "zero_rate_delta_std_to_theory": (
                        agg["zero_rate_hif4_theory"] - agg["zero_rate_hif4_std"]
                    ),
                    "an_nonzero_to_zero_delta_std_to_theory": (
                        agg["an_nonzero_to_hif4_theory_zero_rate"]
                        - agg["an_nonzero_to_hif4_std_zero_rate"]
                    ),
                }
            )
    return out


def _same_sign(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    if a == 0.0 or b == 0.0:
        return a == b
    return (a > 0.0) == (b > 0.0)


def build_summary(
    *,
    run_id: str,
    source_capture_run_id: str,
    cal: dict[str, Any],
    val: dict[str, Any],
) -> dict[str, Any]:
    helpful = bool(val["global_recovery_vs_an"] > 0.0)
    vs_an = float(val["global_recovery_vs_an"])
    vs_x = float(val["global_recovery_vs_xrot"])
    if helpful:
        statement = (
            "固定的理论均权 scale 在真实 captured activation 上有直接迁移收益。"
        )
    else:
        statement = (
            "固定的理论均权 scale 在真实 captured activation 上没有直接迁移收益。"
        )
    return {
        "experiment": "theoretical_uniform_abs_mse_grid_scale_activation_validation",
        "run_id": run_id,
        "source_capture_run_id": source_capture_run_id,
        "theory": {
            "objective": "uniform unique NVFP4 grid points, absolute squared error",
            "grid_scale": THEORY_GRID_SCALE,
            "standard_s0_divisor": STANDARD_S0_DIVISOR,
            "theory_s0_divisor": THEORY_S0_DIVISOR,
            "scale_selected_from_activation_data": False,
            "implementation_note": (
                "injected via s0_divisor only; S0 still applies BF16/E6M2 rounding "
                "and e8/e4 are re-thresholded from the new S0, so this is not a "
                "pointwise equivalent of the continuous s·H lattice"
            ),
        },
        "cal": cal,
        "val": val,
        "conclusion": {
            "theory_scale_helpful_on_captured_activation": helpful,
            "val_global_mse_std_vs_an": val["global_mse_std_vs_an"],
            "val_global_mse_theory_vs_an": val["global_mse_theory_vs_an"],
            "val_global_recovery_vs_an": vs_an,
            "val_num_modules_positive_recovery_vs_an": val["num_modules_positive_recovery"],
            "val_fraction_modules_positive_recovery_vs_an": val[
                "fraction_modules_positive_recovery"
            ],
            "val_global_recovery_vs_xrot": vs_x,
            "val_vs_an_and_vs_xrot_same_sign": _same_sign(vs_an, vs_x),
            "statement": statement,
        },
    }


def _fmt(value: float, digits: int = 8) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:.{digits}g}"


def _pct(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{100.0 * value:.4f}%"


def write_report_cn(
    path: Path,
    *,
    summary: dict[str, Any],
    projection_rows: list[dict[str, Any]],
) -> None:
    val = summary["val"]
    cal = summary["cal"]
    theory = summary["theory"]
    conclusion = summary["conclusion"]
    val_proj = [r for r in projection_rows if r["split"] == "val"]
    vs_an = float(val["global_recovery_vs_an"])
    vs_x = float(val["global_recovery_vs_xrot"])
    if vs_an > 0.0 and vs_x > 0.0:
        direction = (
            "情况 A：相对 A_N 改善，同时相对 X_rot 也改善。"
            "理论 scale 不仅更贴近 NVFP4 source，也对原始 pre-quant activation 的重构更好。"
        )
    elif vs_an > 0.0 and vs_x <= 0.0:
        direction = (
            "情况 B：相对 A_N 改善，但相对 X_rot 恶化。"
            "理论 scale 主要是在匹配 NVFP4 lattice；对转换有利，但不是普通 HiF4 activation MSE 的统一最优。"
        )
    elif vs_an <= 0.0 and vs_x <= 0.0:
        direction = (
            "相对 A_N 与相对 X_rot 的全局 recovery 都不为正。"
            "该固定理论 scale 在这两个 reference 上都没有带来 conversion / reconstruction 收益。"
        )
    else:
        direction = (
            "相对 A_N 未改善，但相对 X_rot 改善。"
            "该固定理论 scale 没有降低 NVFP4→HiF4 转换误差，不能用 vs X_rot 覆盖主结论。"
        )

    cal_sign = "正" if cal["global_recovery_vs_an"] > 0.0 else "非正"
    val_sign = "正" if vs_an > 0.0 else "非正"
    if (cal["global_recovery_vs_an"] > 0.0) == (vs_an > 0.0):
        split_note = f"cal 与 val 的全局 recovery 方向一致（均为{val_sign}）。"
    else:
        split_note = (
            f"cal 全局 recovery 为{cal_sign}，val 为{val_sign}；"
            "理论 scale 没有拟合 cal，此处只如实记录，不做任何调参。"
        )

    proj_lines = [
        "| projection | recovery vs A_N | recovery vs X_rot | MSE_std vs A_N | MSE_theory vs A_N |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in val_proj:
        proj_lines.append(
            f"| {row['projection']} | {_pct(row['recovery_vs_an'])} | "
            f"{_pct(row['recovery_vs_xrot'])} | {_fmt(row['global_mse_std_vs_an'])} | "
            f"{_fmt(row['global_mse_theory_vs_an'])} |"
        )

    zero_delta = val["zero_rate_hif4_theory"] - val["zero_rate_hif4_std"]
    nz_delta = (
        val["an_nonzero_to_hif4_theory_zero_rate"]
        - val["an_nonzero_to_hif4_std_zero_rate"]
    )
    text = f"""# NVFP4→HiF4 理论均权网格缩放：真实捕获激活验证

## 1. 实验问题

固定理论均权绝对 MSE 最优 scale `s*={THEORY_GRID_SCALE}`，通过修改标准 HiF4 的 `s0_divisor` 注入当前硬件语义 quantizer 后，在已有 35 个真实 NVFP4 QAT post-rotation activation captures 的 validation split 上，相对原生 `A_N` 的全局 conversion MSE 是改善还是恶化？

主指标是 `A_H` 相对 `A_N` 的转换误差，不是 HiF4 相对 BF16 的普通量化误差。

## 2. 理论 scale 的来源与实际使用方式

`s*={THEORY_GRID_SCALE}` 来自前序数学推导：在 NVFP4 / HiF4 theoretical unique grid 上，每个 NVFP4 点等权，最小化绝对 squared error。本实验不再求 `s`，也不读取 validation 选择参数。

注入方式只改 S0 divisor：

```
s0_raw,std    = amax64 / 7
s0_raw,theory = s* × amax64 / 7 = amax64 / (7 / s*)
THEORY_S0_DIVISOR = {THEORY_S0_DIVISOR}
```

其余 HiF4 配置保持标准值：`group_size=64, group_dim=-1, e8_threshold=4.0, e4_threshold=2.0, s0_mode=hardware, enable_exp8=True, enable_exp4=True`。

这不是对连续网格 `s·H` 的逐点完全等价实现：`S0` 仍要做 BF16/E6M2 rounding，e8/e4 也会依据新的 S0 重新判定。

## 3. 数据来源

只复用已保存 capture，不重新加载 Qwen3-8B，不重新 forward。

- source capture run：`{summary['source_capture_run_id']}`
- 本实验 run：`{summary['run_id']}`
- capture_mode=formal，coverage=35/35，capture_point=post_rotation_pre_activation_quant
- source_semantic_version=native_nvfp4_rot_a4_v1
- 正式层 {list(REQUIRED_FORMAL_LAYERS)} × 7 projection = 35 Linear；cal / val 都评估，但不使用任一 split 搜索或修改理论 scale
- `scale_selected_from_activation_data` = {str(theory['scale_selected_from_activation_data']).lower()}

## 4. validation 全局结果

主结论只看 validation split 的 energy-weighted 全局量（总 error energy / 总元素数），不对 35 个 module 的 MSE 做算术平均。

- standard HiF4 vs A_N global MSE = `{_fmt(val['global_mse_std_vs_an'], 12)}`
- theory-scale HiF4 vs A_N global MSE = `{_fmt(val['global_mse_theory_vs_an'], 12)}`
- validation global recovery vs A_N = `{_fmt(vs_an, 12)}`（{_pct(vs_an)}）
- 35 个 module 中 recovery > 0 的个数 = `{val['num_modules_positive_recovery']}`
- median / min / max module recovery = `{_fmt(val['median_module_recovery'])}` / `{_fmt(val['min_module_recovery'])}` / `{_fmt(val['max_module_recovery'])}`
- global NMSE std / theory vs A_N = `{_fmt(val['global_nmse_std_vs_an'])}` / `{_fmt(val['global_nmse_theory_vs_an'])}`

calibration 仅作方向对照：global recovery vs A_N = `{_fmt(cal['global_recovery_vs_an'], 12)}`。{split_note}

## 5. 7 类 projection（validation）

{chr(10).join(proj_lines)}

## 6. vs A_N 与 vs X_rot 是否方向一致

- validation global recovery vs X_rot = `{_fmt(vs_x, 12)}`（{_pct(vs_x)}）
- 两个 reference 的 recovery 符号是否相同：`{conclusion['val_vs_an_and_vs_xrot_same_sign']}`

{direction}

vs X_rot 只用于诊断，不覆盖主结论。

## 7. zero / NV 非零 → HiF4 零

validation 上按元素加权：

- zero_rate A_N / HiF4_std / HiF4_theory = `{_fmt(val['zero_rate_an'])}` / `{_fmt(val['zero_rate_hif4_std'])}` / `{_fmt(val['zero_rate_hif4_theory'])}`
- zero_rate 从 std 到 theory 的变化 = `{_fmt(zero_delta)}`
- A_N 非零被 HiF4 打成零：std `{_fmt(val['an_nonzero_to_hif4_std_zero_rate'])}`，theory `{_fmt(val['an_nonzero_to_hif4_theory_zero_rate'])}`，变化 `{_fmt(nz_delta)}`

## 8. 明确结论

{conclusion['statement']}

validation 上：`theory_scale_helpful_on_captured_activation = {str(conclusion['theory_scale_helpful_on_captured_activation']).lower()}`，global recovery vs A_N = `{_fmt(vs_an, 12)}`。

这句话只回答“固定理论均权 scale 能否直接迁移到真实 captured activation”，不声称 `1.2993` 是真实 activation 最优 scale，也不是部署最优 scale，更没有证明 occupancy weighting 无需做。

## 9. 边界

- 没有从这些激活上重新搜索 scale，也没有读取 validation 选择参数。
- 没有做 Linear GEMM，没有加载权重，没有 output NMSE，没有端到端 benchmark。
- 理论 lattice phase 是通过 S0 divisor 注入当前硬件语义 HiF4；由于 S0 rounding 与 e8/e4 重判定，它不是连续网格 `s·H` 的逐点完全等价实现。
- 下一阶段若继续，应单独做 occupancy-weighted 最优 scale；不得把 occupancy 混入本次验证。
"""
    path.write_text(text, encoding="utf-8")


def plot_recovery_figure(
    path: Path,
    *,
    projection_rows: list[dict[str, Any]],
    val_global_recovery: float,
) -> None:
    val_proj = [r for r in projection_rows if r["split"] == "val"]
    order = {name: i for i, name in enumerate(TARGET_PROJECTIONS)}
    val_proj = sorted(val_proj, key=lambda r: order[r["projection"]])
    names = [r["projection"] for r in val_proj]
    heights = [100.0 * float(r["recovery_vs_an"]) for r in val_proj]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, heights, color="steelblue")
    ax.axhline(
        100.0 * val_global_recovery,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"global val recovery = {100.0 * val_global_recovery:.4f}%",
    )
    ax.set_ylabel("validation recovery vs A_N (%)")
    ax.set_xlabel("projection")
    ax.set_title("s=1.29929547, S0 divisor=5.38753514")
    ax.legend()
    ax.axhline(0.0, color="gray", linewidth=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def validate_outputs(
    run_dir: Path,
    *,
    config: AppConfig,
    source_capture_run_id: str,
) -> None:
    per_path = run_dir / "grid_scale_per_module.csv"
    proj_path = run_dir / "grid_scale_by_projection.csv"
    summary_path = run_dir / "grid_scale_summary.json"
    report_path = run_dir / "grid_scale_report_cn.md"
    fig_path = run_dir / "figures" / "fig01_grid_scale_conversion_mse_recovery.png"
    for path in (per_path, proj_path, summary_path, report_path, fig_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing output: {path}")

    per_df = pd.read_csv(per_path)
    proj_df = pd.read_csv(proj_path)
    summary = read_json(summary_path)

    if len(per_df) != 70:
        raise RuntimeError(f"grid_scale_per_module.csv must have 70 rows, got {len(per_df)}")
    if len(proj_df) != 14:
        raise RuntimeError(
            f"grid_scale_by_projection.csv must have 14 rows, got {len(proj_df)}"
        )
    if "cal" not in summary or "val" not in summary:
        raise RuntimeError("summary json must contain cal and val")

    theory = summary["theory"]
    if abs(float(theory["grid_scale"]) - THEORY_GRID_SCALE) > 1e-12:
        raise RuntimeError(f"theory.grid_scale mismatch: {theory['grid_scale']}")
    if abs(float(theory["theory_s0_divisor"]) - THEORY_S0_DIVISOR) > 1e-12:
        raise RuntimeError(
            f"theory.theory_s0_divisor mismatch: {theory['theory_s0_divisor']}"
        )
    if theory["scale_selected_from_activation_data"] is not False:
        raise RuntimeError("scale_selected_from_activation_data must be false")
    if summary["source_capture_run_id"] != source_capture_run_id:
        raise RuntimeError("source_capture_run_id mismatch")

    energy_cols = [
        "error_energy_std_vs_an",
        "error_energy_theory_vs_an",
        "error_energy_std_vs_xrot",
        "error_energy_theory_vs_xrot",
    ]
    for col in energy_cols:
        series = per_df[col].astype(float)
        if not np.isfinite(series).all() or (series < 0).any():
            raise RuntimeError(f"{col} must be finite and >= 0")

    expected_modules = set(config.formal_module_names)
    for split in SPLITS:
        sub = per_df[per_df["split"] == split]
        if len(sub) != REQUIRED_MODULE_COUNT:
            raise RuntimeError(f"{split} must have 35 module rows, got {len(sub)}")
        got = set(sub["module_name"].tolist())
        if got != expected_modules:
            raise RuntimeError(f"{split} module set mismatch")

    if not np.allclose(per_df["theory_grid_scale"].astype(float), THEORY_GRID_SCALE):
        raise RuntimeError("per-module theory_grid_scale mismatch")
    if not np.allclose(per_df["theory_s0_divisor"].astype(float), THEORY_S0_DIVISOR):
        raise RuntimeError("per-module theory_s0_divisor mismatch")


def standard_and_theory_configs(hif4: Hif4Config) -> tuple[HiF4QuantConfig, HiF4QuantConfig]:
    assert_standard_hif4(hif4)
    common = {
        "group_size": hif4.group_size,
        "e8_threshold": hif4.e8_threshold,
        "e4_threshold": hif4.e4_threshold,
        "s0_mode": hif4.s0_mode,
    }
    base = build_hif4_config(**common, s0_divisor=STANDARD_S0_DIVISOR)
    theory = build_hif4_config(**common, s0_divisor=theory_s0_divisor(STANDARD_S0_DIVISOR))
    if abs(theory.s0_divisor - THEORY_S0_DIVISOR) > 1e-12:
        raise RuntimeError(
            f"theory s0_divisor must be {THEORY_S0_DIVISOR}, got {theory.s0_divisor}"
        )
    if not configs_differ_only_in_s0_divisor(base, theory):
        raise RuntimeError("standard/theory HiF4 configs must differ only in s0_divisor")
    return base, theory


def run_grid_scale_validation(
    config: AppConfig,
    *,
    capture_run_id: str,
    run_id: str,
    device: str,
) -> dict[str, Any]:
    assert_distinct_run_ids(capture_run_id, run_id)
    capture_dir = results_dir(capture_run_id)
    manifest_path = capture_dir / "capture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing capture manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    validate_capture_manifest(config, capture_dir, manifest)

    hif4_base, hif4_theory = standard_and_theory_configs(config.hif4)
    nvfp4_group_size = int(config.nvfp4.activation_group_size)
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)

    rows: list[dict[str, Any]] = []
    for module_name in config.formal_module_names:
        for split in SPLITS:
            print(f"[grid_scale] {split} {module_name}", flush=True)
            capture = load_pt(
                capture_file_path(capture_dir, module_name, split),
                map_location="cpu",
            )
            metrics = evaluate_capture(
                capture,
                hif4_base_config=hif4_base,
                hif4_theory_config=hif4_theory,
                nvfp4_group_size=nvfp4_group_size,
                device=torch_device,
            )
            row = {
                "run_id": run_id,
                "source_capture_run_id": capture_run_id,
                "split": capture["split"],
                "layer_idx": int(capture["layer_idx"]),
                "projection": capture["projection"],
                "module_name": capture["module_name"],
                "theory_grid_scale": THEORY_GRID_SCALE,
                "standard_s0_divisor": STANDARD_S0_DIVISOR,
                "theory_s0_divisor": THEORY_S0_DIVISOR,
                **metrics,
            }
            if row["split"] != split or row["module_name"] != module_name:
                raise RuntimeError(
                    f"capture metadata mismatch: expected {module_name}/{split}, "
                    f"got {row['module_name']}/{row['split']}"
                )
            rows.append(row)
            del capture

    if len(rows) != 70:
        raise RuntimeError(f"expected 70 per-module rows, got {len(rows)}")

    cal_rows = [r for r in rows if r["split"] == "cal"]
    val_rows = [r for r in rows if r["split"] == "val"]
    cal_agg = aggregate_rows(cal_rows)
    val_agg = aggregate_rows(val_rows)
    projection_rows = aggregate_by_projection(rows)

    run_dir = ensure_dir(results_dir(run_id))
    per_df = pd.DataFrame(rows, columns=PER_MODULE_COLUMNS)
    per_df.to_csv(run_dir / "grid_scale_per_module.csv", index=False)
    proj_df = pd.DataFrame(projection_rows, columns=PROJECTION_COLUMNS)
    proj_df.to_csv(run_dir / "grid_scale_by_projection.csv", index=False)

    summary = build_summary(
        run_id=run_id,
        source_capture_run_id=capture_run_id,
        cal=cal_agg,
        val=val_agg,
    )
    write_json(run_dir / "grid_scale_summary.json", summary)
    write_report_cn(
        run_dir / "grid_scale_report_cn.md",
        summary=summary,
        projection_rows=projection_rows,
    )
    plot_recovery_figure(
        run_dir / "figures" / "fig01_grid_scale_conversion_mse_recovery.png",
        projection_rows=projection_rows,
        val_global_recovery=float(val_agg["global_recovery_vs_an"]),
    )
    validate_outputs(
        run_dir, config=config, source_capture_run_id=capture_run_id
    )
    print(f"GRID SCALE VALIDATION DONE -> {run_dir}", flush=True)
    print(
        "val global MSE std/theory/recovery/"
        f"positive={val_agg['global_mse_std_vs_an']:.12g}/"
        f"{val_agg['global_mse_theory_vs_an']:.12g}/"
        f"{val_agg['global_recovery_vs_an']:.12g}/"
        f"{val_agg['num_modules_positive_recovery']}/35",
        flush=True,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate theoretical NVFP4→HiF4 grid scale on saved activations"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--capture-run-id", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    run_grid_scale_validation(
        config,
        capture_run_id=args.capture_run_id,
        run_id=args.run_id,
        device=args.device,
    )


if __name__ == "__main__":
    main()
