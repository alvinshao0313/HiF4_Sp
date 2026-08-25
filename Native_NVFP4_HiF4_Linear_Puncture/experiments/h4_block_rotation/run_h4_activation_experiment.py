"""H4 G4 block-rotation experiment on saved Native NVFP4 Linear puncture tensors.

Does not recapture activations, does not search DIAG, and does not change
existing HiF4 / DIAG math. Identity / DIAG / H4_FP32 / H4_BF16 share the
same saved X_rot and W_N.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from HiFloat4.hif4_scale_threshold_optimization.src.formats import S1P2_MAX
from HiFloat4.hif4_scale_threshold_optimization.src.quantizer import (
    HiF4QuantResult,
    quantize_hif4,
)

from Native_NVFP4_HiF4_Linear_Puncture.experiments.h4_block_rotation.analyze_h4_results import (
    analyze_run,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.h4_block_rotation.h4_transform import (
    H4_UNNORMALIZED,
    HIF4_GROUP_SIZE,
    apply_h4_g4,
    assert_r4_orthogonal,
    linear_prequant_equivalence_error,
    r4_matrix,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import (
    load_packed_linear_state,
    resolve_local_snapshot,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.config import (
    EXPERIMENT_ROOT,
    AppConfig,
    load_config,
    results_dir,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import (
    STANDARD_HIF4_CONFIG,
    qdq_hif4_direct,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.grid_scale_validation import (
    REQUIRED_FORMAL_LAYERS,
    REQUIRED_MODULE_COUNT,
    capture_file_path,
    validate_capture_manifest,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import (
    ensure_dir,
    load_pt,
    module_capture_stem,
    read_json,
    write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.metrics import compare_tensors, compute_nmse
from Native_NVFP4_HiF4_Linear_Puncture.src.native_nvfp4 import (
    PackedNVFP4LinearState,
    dequantize_packed_weight,
    qdq_nvfp4_post_rotation,
)

DEFAULT_CAPTURE_RUN_ID = "20260812T103800Z_native_nvfp4_hif4_linear_puncture"
EVAL_SPLIT = "val"
SMOKE_MAX_ROWS = 256
PREQUANT_REL_MAX = 1e-6
NMSE_EPS = 1e-30
G4 = 4
G8 = 8
G64 = HIF4_GROUP_SIZE
CF4_BIN_EDGES = torch.linspace(1.0, 2.0, 41)

GROUP_COLUMNS = [
    "run_id",
    "module_name",
    "layer_idx",
    "projection",
    "split",
    "token_row",
    "group_id",
    "nmse_identity",
    "nmse_h4",
    "nmse_ratio",
    "cf4_mean_before",
    "cf4_mean_after",
    "cf64_before",
    "cf64_after",
    "amax4_mean_before",
    "amax4_mean_after",
    "amax8_mean_before",
    "amax8_mean_after",
    "amax64_before",
    "amax64_after",
    "amax64_ratio",
    "s0_before",
    "s0_after",
    "e8_rate_before",
    "e8_rate_after",
    "e4_rate_before",
    "e4_rate_after",
    "zero_rate_before",
    "zero_rate_after",
    "clip_rate_before",
    "clip_rate_after",
    "local_scale_mean_before",
    "local_scale_mean_after",
]

LAYER_COLUMNS = [
    "run_id",
    "module_name",
    "layer_idx",
    "projection",
    "split",
    "num_rows",
    "k_dim",
    "out_features",
    "prequant_rel_fp32",
    "prequant_rel_bf16",
    "h4_norm_rel_error",
    "h4_rotate_back_nmse",
    "h4_rotate_domain_nmse",
    "act_nmse_identity",
    "act_nmse_diag",
    "act_nmse_h4_fp32",
    "act_nmse_h4_bf16",
    "act_ratio_h4_fp32",
    "act_gain_h4_fp32",
    "act_ratio_h4_bf16",
    "act_ratio_diag",
    "act_gain_diag",
    "act_ratio_h4_vs_diag",
    "weight_nmse_identity",
    "weight_nmse_diag",
    "weight_nmse_h4_fp32",
    "weight_nmse_h4_bf16",
    "weight_ratio_h4_fp32",
    "weight_gain_h4_fp32",
    "weight_ratio_h4_bf16",
    "weight_ratio_diag",
    "weight_gain_diag",
    "weight_ratio_h4_vs_diag",
    "output_nmse_identity",
    "output_nmse_diag",
    "output_nmse_h4_fp32",
    "output_nmse_h4_bf16",
    "output_rel_l2_identity",
    "output_rel_l2_diag",
    "output_rel_l2_h4_fp32",
    "output_rel_l2_h4_bf16",
    "output_cosine_identity",
    "output_cosine_diag",
    "output_cosine_h4_fp32",
    "output_cosine_h4_bf16",
    "output_max_abs_identity",
    "output_max_abs_diag",
    "output_max_abs_h4_fp32",
    "output_max_abs_h4_bf16",
    "output_ratio_h4_fp32",
    "output_gain_h4_fp32",
    "output_ratio_h4_bf16",
    "output_ratio_diag",
    "output_gain_diag",
    "output_ratio_h4_vs_diag",
    "fraction_groups_improved",
    "group_nmse_ratio_median",
    "group_nmse_ratio_p90",
    "group_nmse_ratio_p99",
    "cf4_mean_before",
    "cf4_mean_after",
    "cf4_median_before",
    "cf4_median_after",
    "cf4_p90_before",
    "cf4_p90_after",
    "cf4_p99_before",
    "cf4_p99_after",
    "cf64_mean_before",
    "cf64_mean_after",
    "e4_nmse_csv",
    "e5_nmse_csv",
]


def h4_results_dir(run_id: str) -> Path:
    return EXPERIMENT_ROOT / "results" / "h4_block_rotation" / run_id


def _finite_float(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"{name} is not finite: {value}")
    return float(value)


def _assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.numel() == 0:
        raise RuntimeError(f"{name} is empty")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def _ratio(numer: float, denom: float, name: str) -> float:
    if denom == 0.0 and numer == 0.0:
        return 1.0
    if denom == 0.0:
        raise RuntimeError(f"{name}: denominator is 0 while numerator={numer}")
    return _finite_float(name, numer / denom)


def _gain(ratio: float) -> float:
    return float(1.0 - ratio)


def _tensor_info(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "path": str(path),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "numel": int(tensor.numel()),
    }


def _packed_state(
    snapshot: Path, weight_map: dict[str, str], module_name: str
) -> PackedNVFP4LinearState:
    packed = load_packed_linear_state(snapshot, weight_map, module_name)
    return PackedNVFP4LinearState(
        module_name=module_name,
        weight_packed=packed["weight_packed"],  # type: ignore[arg-type]
        weight_scale=packed["weight_scale"],  # type: ignore[arg-type]
        weight_global_scale=packed["weight_global_scale"].to(torch.float32),  # type: ignore[union-attr]
        input_global_scale=packed["input_global_scale"].to(torch.float32),  # type: ignore[union-attr]
        rotation_matrix=packed["rotation_matrix"].to(torch.bfloat16),  # type: ignore[union-attr]
        bias=packed["bias"],
    )


def _linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return F.linear(
        x.to(dtype=torch.float32),
        w.to(dtype=torch.float32),
        None if bias is None else bias.to(dtype=torch.float32),
    )


def _crest(groups: torch.Tensor) -> torch.Tensor:
    """Peak / RMS over the last axis. All-zero groups have CF=1 (constant vector)."""
    peak = groups.abs().amax(dim=-1)
    rms = groups.square().mean(dim=-1).sqrt()
    zero_rms = rms == 0
    if torch.any(zero_rms & (peak != 0)):
        raise RuntimeError("crest factor: nonzero peak with zero RMS")
    ones = torch.ones_like(peak)
    return torch.where(zero_rms, ones, peak / rms)


def _group_nmse(hat_groups: torch.Tensor, ref_groups: torch.Tensor) -> torch.Tensor:
    err = (hat_groups.to(torch.float64) - ref_groups.to(torch.float64)).square().sum(dim=-1)
    ref = ref_groups.to(torch.float64).square().sum(dim=-1)
    zero_ref = ref == 0
    if torch.any(zero_ref & (err != 0)):
        raise RuntimeError("group NMSE: nonzero error on a zero-energy group")
    return torch.where(zero_ref, torch.zeros_like(err), err / ref.clamp_min(NMSE_EPS)).to(
        torch.float64
    )


def _percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        raise RuntimeError("percentile on empty tensor")
    return float(torch.quantile(values.reshape(-1).to(torch.float64), q).item())


def quantize_hif4_fp32(x: torch.Tensor) -> HiF4QuantResult:
    _assert_finite_tensor("hif4_input", x)
    result = quantize_hif4(x.to(torch.float32), config=STANDARD_HIF4_CONFIG)
    _assert_finite_tensor("hif4_reconstruction", result.reconstruction)
    return result


def _reshape_groups(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected 2D [N,K], got shape {tuple(x.shape)}")
    n, k = x.shape
    if k % G64 != 0:
        raise ValueError(f"K={k} is not divisible by {G64}")
    return x.reshape(n, k // G64, G64)


def _structure_stats(x: torch.Tensor) -> dict[str, torch.Tensor]:
    xg = _reshape_groups(x)
    n, g, _ = xg.shape
    g4 = xg.reshape(n, g, G64 // G4, G4)
    g8 = xg.reshape(n, g, G64 // G8, G8)
    cf4 = _crest(g4)
    return {
        "xg": xg,
        "cf4": cf4,
        "cf4_mean": cf4.mean(dim=-1),
        "cf64": _crest(xg),
        "amax4_mean": g4.abs().amax(dim=-1).mean(dim=-1),
        "amax8_mean": g8.abs().amax(dim=-1).mean(dim=-1),
        "amax64": xg.abs().amax(dim=-1),
    }


def _quant_group_stats(x: torch.Tensor, result: HiF4QuantResult) -> dict[str, torch.Tensor]:
    xg = _reshape_groups(x)
    n, g, _ = xg.shape
    hat = result.reconstruction.to(torch.float32).reshape(n, g, G64)
    payload = result.payload.to(torch.float32).reshape(n, g, G64)
    local = result.local_scale.to(torch.float32).reshape(n, g, G64)
    s0 = result.s0.to(torch.float32).reshape(n, g)
    e8 = result.e8.to(torch.float32).reshape(n, g, G64 // G8)
    e4 = result.e4.to(torch.float32).reshape(n, g, G64 // G4)
    return {
        "nmse": _group_nmse(hat, xg),
        "s0": s0,
        "e8_rate": e8.mean(dim=-1),
        "e4_rate": e4.mean(dim=-1),
        "zero_rate": (payload == 0).to(torch.float64).mean(dim=-1),
        "clip_rate": (payload == S1P2_MAX).to(torch.float64).mean(dim=-1),
        "local_scale_mean": local.mean(dim=-1),
    }


def _flat_cpu(tensor: torch.Tensor) -> Any:
    return tensor.reshape(-1).detach().cpu().numpy()


def _amax64_ratio(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    zero_before = before == 0
    if torch.any(zero_before & (after != 0)):
        raise RuntimeError("amax64_after != 0 while amax64_before == 0")
    ones = torch.ones_like(before, dtype=torch.float64)
    b = before.to(torch.float64)
    a = after.to(torch.float64)
    return torch.where(zero_before, ones, a / b)


def _hist_cf4(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    edges = CF4_BIN_EDGES.to(device=values.device, dtype=torch.float32)
    h = torch.histc(
        values.to(torch.float32),
        bins=int(edges.numel() - 1),
        min=float(edges[0].item()),
        max=float(edges[-1].item()),
    )
    return counts.to(h.device) + h


def _apply_h4_fp32(x: torch.Tensor) -> torch.Tensor:
    return apply_h4_g4(
        x.to(torch.float32),
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )


def _apply_h4_bf16_carrier(x: torch.Tensor) -> torch.Tensor:
    """BF16 in/out, FP32 matmul. Same carrier convention as the existing block rotation oracle."""
    y = apply_h4_g4(
        x.to(torch.bfloat16),
        compute_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
    return y.to(torch.float32)


def _case_metrics(
    *,
    x_src: torch.Tensor,
    w_src: torch.Tensor,
    x_qsrc: torch.Tensor,
    w_qsrc: torch.Tensor,
    y_ref: torch.Tensor,
    bias: torch.Tensor | None,
) -> dict[str, Any]:
    qx = quantize_hif4_fp32(x_qsrc)
    qw = quantize_hif4_fp32(w_qsrc)
    y = _linear(qx.reconstruction, qw.reconstruction, bias)
    act_nmse = compute_nmse(qx.reconstruction, x_qsrc)
    weight_nmse = compute_nmse(qw.reconstruction, w_qsrc)
    out = compare_tensors(y, y_ref)
    _assert_finite_tensor("y_hat", y)
    return {
        "qx": qx,
        "qw": qw,
        "y": y,
        "act_nmse": _finite_float("act_nmse", act_nmse),
        "weight_nmse": _finite_float("weight_nmse", weight_nmse),
        "output": out,
    }


def _load_existing_linear_nmse(capture_run_dir: Path) -> dict[str, dict[str, float]]:
    path = capture_run_dir / "linear_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing existing linear_results.csv: {path}")
    df = pd.read_csv(path)
    out: dict[str, dict[str, float]] = {}
    for module_name, sub in df.groupby("module_name"):
        row = {}
        for vid in ("E4_WH_AH_RTN", "E5_WH_AH_DIAG"):
            hit = sub[sub["variant_id"] == vid]
            if len(hit) != 1:
                raise RuntimeError(
                    f"{path} expected one {vid} row for {module_name}, got {len(hit)}"
                )
            row[vid] = float(hit.iloc[0]["nmse"])
        out[str(module_name)] = row
    return out


def evaluate_module(
    *,
    run_id: str,
    module_name: str,
    capture: dict[str, Any],
    capture_path: Path,
    w_n: torch.Tensor,
    bias: torch.Tensor | None,
    d: torch.Tensor,
    existing_nmse: dict[str, float] | None,
    device: torch.device,
    max_rows: int | None,
    cf4_hist_before: torch.Tensor,
    cf4_hist_after: torch.Tensor,
) -> dict[str, Any]:
    x_rot = capture["x_rot_bf16"].to(device=device)
    if x_rot.ndim != 2:
        raise ValueError(f"{module_name}: expected 2D X_rot, got {tuple(x_rot.shape)}")
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError(f"max_rows must be positive, got {max_rows}")
        x_rot = x_rot[:max_rows]
    scale = capture["input_global_scale_fp32"].to(device=device, dtype=torch.float32)
    w = w_n.to(device=device, dtype=torch.float32)
    d_dev = d.to(device=device, dtype=torch.float32)
    bias_dev = None if bias is None else bias.to(device=device, dtype=torch.float32)

    x = x_rot.to(torch.float32)
    _assert_finite_tensor(f"{module_name}.x", x)
    _assert_finite_tensor(f"{module_name}.w", w)
    if x.shape[-1] != w.shape[-1]:
        raise RuntimeError(
            f"{module_name}: activation K={x.shape[-1]} != weight K={w.shape[-1]}"
        )
    if d_dev.numel() != w.shape[-1]:
        raise RuntimeError(
            f"{module_name}: DIAG d length {d_dev.numel()} != K={w.shape[-1]}"
        )

    prequant_fp32 = linear_prequant_equivalence_error(x, w, compute_dtype=torch.float32)
    if prequant_fp32 >= PREQUANT_REL_MAX:
        raise RuntimeError(
            f"{module_name}: FP32 pre-quant Linear equivalence failed: "
            f"relative Frobenius={prequant_fp32:.3e} >= {PREQUANT_REL_MAX}"
        )
    prequant_bf16 = linear_prequant_equivalence_error(
        x_rot.to(torch.bfloat16),
        w.to(torch.bfloat16),
        compute_dtype=torch.float32,
    )

    x_h = _apply_h4_fp32(x)
    w_h = _apply_h4_fp32(w)
    if tuple(x_h.shape) != tuple(x.shape) or tuple(w_h.shape) != tuple(w.shape):
        raise RuntimeError(f"{module_name}: H4 did not restore activation/weight shape")
    xn = torch.linalg.vector_norm(x.to(torch.float64))
    hn = torch.linalg.vector_norm(x_h.to(torch.float64))
    norm_rel = float((hn - xn).abs().item() / xn.clamp_min(NMSE_EPS).item())
    _finite_float(f"{module_name}.h4_norm_rel_error", norm_rel)

    a_n = qdq_nvfp4_post_rotation(x_rot, scale).to(torch.float32)
    y_ref = _linear(a_n, w, bias_dev)
    _assert_finite_tensor(f"{module_name}.y_ref", y_ref)

    identity = _case_metrics(x_src=x, w_src=w, x_qsrc=x, w_qsrc=w, y_ref=y_ref, bias=bias_dev)
    identity_ref = qdq_hif4_direct(x, output_dtype=torch.float32)
    id_diff = (identity["qx"].reconstruction - identity_ref).abs().max().item()
    if id_diff != 0.0:
        raise RuntimeError(
            f"{module_name}: Identity HiF4 reconstruction differs from "
            f"formats.qdq_hif4_direct, max_abs={id_diff}"
        )

    h4 = _case_metrics(
        x_src=x, w_src=w, x_qsrc=x_h, w_qsrc=w_h, y_ref=y_ref, bias=bias_dev
    )
    x_h_hat_back = _apply_h4_fp32(h4["qx"].reconstruction)
    nmse_rot = compute_nmse(h4["qx"].reconstruction, x_h)
    nmse_back = compute_nmse(x_h_hat_back, x)
    rot_gap = abs(nmse_rot - nmse_back) / max(nmse_rot, NMSE_EPS)
    if rot_gap >= 1e-5 and abs(nmse_rot - nmse_back) >= 1e-12:
        raise RuntimeError(
            f"{module_name}: rotate-back NMSE mismatch: "
            f"rot={nmse_rot:.8e} back={nmse_back:.8e} rel={rot_gap:.3e}"
        )

    x_h_bf16 = _apply_h4_bf16_carrier(x_rot)
    w_h_bf16 = _apply_h4_bf16_carrier(w)
    h4_bf16 = _case_metrics(
        x_src=x,
        w_src=w,
        x_qsrc=x_h_bf16,
        w_qsrc=w_h_bf16,
        y_ref=y_ref,
        bias=bias_dev,
    )

    x_d = x / d_dev
    w_d = w * d_dev
    diag = _case_metrics(
        x_src=x, w_src=w, x_qsrc=x_d, w_qsrc=w_d, y_ref=y_ref, bias=bias_dev
    )

    before = _structure_stats(x)
    after = _structure_stats(x_h)
    q_before = _quant_group_stats(x, identity["qx"])
    q_after = _quant_group_stats(x_h, h4["qx"])
    n, g, _ = before["xg"].shape
    nmse_i = q_before["nmse"]
    nmse_h = q_after["nmse"]
    nmse_ratio = torch.where(
        (nmse_i == 0) & (nmse_h == 0),
        torch.ones_like(nmse_i),
        nmse_h / nmse_i.clamp_min(NMSE_EPS),
    )
    if torch.any((nmse_i == 0) & (nmse_h != 0)):
        raise RuntimeError(f"{module_name}: H4 group NMSE > 0 on a zero-NMSE identity group")
    amax_ratio = _amax64_ratio(before["amax64"], after["amax64"])
    cf4_hist_before = _hist_cf4(before["cf4"].reshape(-1), cf4_hist_before)
    cf4_hist_after = _hist_cf4(after["cf4"].reshape(-1), cf4_hist_after)

    token_row = torch.arange(n, device=x.device).repeat_interleave(g)
    group_id = torch.arange(g, device=x.device).repeat(n)
    group_df = pd.DataFrame(
        {
            "run_id": run_id,
            "module_name": module_name,
            "layer_idx": int(capture["layer_idx"]),
            "projection": str(capture["projection"]),
            "split": str(capture["split"]),
            "token_row": _flat_cpu(token_row),
            "group_id": _flat_cpu(group_id),
            "nmse_identity": _flat_cpu(nmse_i),
            "nmse_h4": _flat_cpu(nmse_h),
            "nmse_ratio": _flat_cpu(nmse_ratio),
            "cf4_mean_before": _flat_cpu(before["cf4_mean"]),
            "cf4_mean_after": _flat_cpu(after["cf4_mean"]),
            "cf64_before": _flat_cpu(before["cf64"]),
            "cf64_after": _flat_cpu(after["cf64"]),
            "amax4_mean_before": _flat_cpu(before["amax4_mean"]),
            "amax4_mean_after": _flat_cpu(after["amax4_mean"]),
            "amax8_mean_before": _flat_cpu(before["amax8_mean"]),
            "amax8_mean_after": _flat_cpu(after["amax8_mean"]),
            "amax64_before": _flat_cpu(before["amax64"]),
            "amax64_after": _flat_cpu(after["amax64"]),
            "amax64_ratio": _flat_cpu(amax_ratio),
            "s0_before": _flat_cpu(q_before["s0"]),
            "s0_after": _flat_cpu(q_after["s0"]),
            "e8_rate_before": _flat_cpu(q_before["e8_rate"]),
            "e8_rate_after": _flat_cpu(q_after["e8_rate"]),
            "e4_rate_before": _flat_cpu(q_before["e4_rate"]),
            "e4_rate_after": _flat_cpu(q_after["e4_rate"]),
            "zero_rate_before": _flat_cpu(q_before["zero_rate"]),
            "zero_rate_after": _flat_cpu(q_after["zero_rate"]),
            "clip_rate_before": _flat_cpu(q_before["clip_rate"]),
            "clip_rate_after": _flat_cpu(q_after["clip_rate"]),
            "local_scale_mean_before": _flat_cpu(q_before["local_scale_mean"]),
            "local_scale_mean_after": _flat_cpu(q_after["local_scale_mean"]),
        }
    )
    missing = [c for c in GROUP_COLUMNS if c not in group_df.columns]
    if missing:
        raise RuntimeError(f"group_metrics missing columns: {missing}")
    group_df = group_df[GROUP_COLUMNS]
    if group_df.isna().any().any():
        raise RuntimeError(f"{module_name}: group_metrics contains NaN")

    ratio_x = _ratio(h4["act_nmse"], identity["act_nmse"], "act_ratio_h4_fp32")
    ratio_x_bf = _ratio(h4_bf16["act_nmse"], identity["act_nmse"], "act_ratio_h4_bf16")
    ratio_x_d = _ratio(diag["act_nmse"], identity["act_nmse"], "act_ratio_diag")
    ratio_w = _ratio(h4["weight_nmse"], identity["weight_nmse"], "weight_ratio_h4_fp32")
    ratio_w_bf = _ratio(
        h4_bf16["weight_nmse"], identity["weight_nmse"], "weight_ratio_h4_bf16"
    )
    ratio_w_d = _ratio(diag["weight_nmse"], identity["weight_nmse"], "weight_ratio_diag")
    y_i = float(identity["output"]["nmse"])
    y_h = float(h4["output"]["nmse"])
    y_bf = float(h4_bf16["output"]["nmse"])
    y_d = float(diag["output"]["nmse"])
    ratio_y = _ratio(y_h, y_i, "output_ratio_h4_fp32")
    ratio_y_bf = _ratio(y_bf, y_i, "output_ratio_h4_bf16")
    ratio_y_d = _ratio(y_d, y_i, "output_ratio_diag")
    ratio_y_h_vs_d = _ratio(y_h, y_d, "output_ratio_h4_vs_diag")
    group_ratio = nmse_ratio.reshape(-1).to(torch.float64)
    fraction_improved = float((group_ratio < 1.0).to(torch.float64).mean().item())

    e4_csv = float("nan")
    e5_csv = float("nan")
    if existing_nmse is not None:
        e4_csv = existing_nmse["E4_WH_AH_RTN"]
        e5_csv = existing_nmse["E5_WH_AH_DIAG"]
        e4_gap = abs(y_i - e4_csv) / max(abs(e4_csv), NMSE_EPS)
        e5_gap = abs(y_d - e5_csv) / max(abs(e5_csv), NMSE_EPS)
        if e4_gap >= 1e-4:
            raise RuntimeError(
                f"{module_name}: Identity output NMSE {y_i:.8e} != existing E4 {e4_csv:.8e} "
                f"(rel={e4_gap:.3e})"
            )
        if e5_gap >= 1e-4:
            raise RuntimeError(
                f"{module_name}: DIAG output NMSE {y_d:.8e} != existing E5 {e5_csv:.8e} "
                f"(rel={e5_gap:.3e})"
            )

    layer_row = {
        "run_id": run_id,
        "module_name": module_name,
        "layer_idx": int(capture["layer_idx"]),
        "projection": str(capture["projection"]),
        "split": str(capture["split"]),
        "num_rows": int(n),
        "k_dim": int(x.shape[-1]),
        "out_features": int(w.shape[0]),
        "prequant_rel_fp32": _finite_float("prequant_rel_fp32", prequant_fp32),
        "prequant_rel_bf16": _finite_float("prequant_rel_bf16", prequant_bf16),
        "h4_norm_rel_error": norm_rel,
        "h4_rotate_back_nmse": _finite_float("h4_rotate_back_nmse", nmse_back),
        "h4_rotate_domain_nmse": _finite_float("h4_rotate_domain_nmse", nmse_rot),
        "act_nmse_identity": identity["act_nmse"],
        "act_nmse_diag": diag["act_nmse"],
        "act_nmse_h4_fp32": h4["act_nmse"],
        "act_nmse_h4_bf16": h4_bf16["act_nmse"],
        "act_ratio_h4_fp32": ratio_x,
        "act_gain_h4_fp32": _gain(ratio_x),
        "act_ratio_h4_bf16": ratio_x_bf,
        "act_ratio_diag": ratio_x_d,
        "act_gain_diag": _gain(ratio_x_d),
        "act_ratio_h4_vs_diag": _ratio(
            h4["act_nmse"], diag["act_nmse"], "act_ratio_h4_vs_diag"
        ),
        "weight_nmse_identity": identity["weight_nmse"],
        "weight_nmse_diag": diag["weight_nmse"],
        "weight_nmse_h4_fp32": h4["weight_nmse"],
        "weight_nmse_h4_bf16": h4_bf16["weight_nmse"],
        "weight_ratio_h4_fp32": ratio_w,
        "weight_gain_h4_fp32": _gain(ratio_w),
        "weight_ratio_h4_bf16": ratio_w_bf,
        "weight_ratio_diag": ratio_w_d,
        "weight_gain_diag": _gain(ratio_w_d),
        "weight_ratio_h4_vs_diag": _ratio(
            h4["weight_nmse"], diag["weight_nmse"], "weight_ratio_h4_vs_diag"
        ),
        "output_nmse_identity": y_i,
        "output_nmse_diag": y_d,
        "output_nmse_h4_fp32": y_h,
        "output_nmse_h4_bf16": y_bf,
        "output_rel_l2_identity": float(identity["output"]["relative_l2"]),
        "output_rel_l2_diag": float(diag["output"]["relative_l2"]),
        "output_rel_l2_h4_fp32": float(h4["output"]["relative_l2"]),
        "output_rel_l2_h4_bf16": float(h4_bf16["output"]["relative_l2"]),
        "output_cosine_identity": float(identity["output"]["cosine"]),
        "output_cosine_diag": float(diag["output"]["cosine"]),
        "output_cosine_h4_fp32": float(h4["output"]["cosine"]),
        "output_cosine_h4_bf16": float(h4_bf16["output"]["cosine"]),
        "output_max_abs_identity": float(identity["output"]["max_abs_error"]),
        "output_max_abs_diag": float(diag["output"]["max_abs_error"]),
        "output_max_abs_h4_fp32": float(h4["output"]["max_abs_error"]),
        "output_max_abs_h4_bf16": float(h4_bf16["output"]["max_abs_error"]),
        "output_ratio_h4_fp32": ratio_y,
        "output_gain_h4_fp32": _gain(ratio_y),
        "output_ratio_h4_bf16": ratio_y_bf,
        "output_ratio_diag": ratio_y_d,
        "output_gain_diag": _gain(ratio_y_d),
        "output_ratio_h4_vs_diag": ratio_y_h_vs_d,
        "fraction_groups_improved": fraction_improved,
        "group_nmse_ratio_median": _percentile(group_ratio, 0.5),
        "group_nmse_ratio_p90": _percentile(group_ratio, 0.9),
        "group_nmse_ratio_p99": _percentile(group_ratio, 0.99),
        "cf4_mean_before": float(before["cf4"].mean().item()),
        "cf4_mean_after": float(after["cf4"].mean().item()),
        "cf4_median_before": _percentile(before["cf4"], 0.5),
        "cf4_median_after": _percentile(after["cf4"], 0.5),
        "cf4_p90_before": _percentile(before["cf4"], 0.9),
        "cf4_p90_after": _percentile(after["cf4"], 0.9),
        "cf4_p99_before": _percentile(before["cf4"], 0.99),
        "cf4_p99_after": _percentile(after["cf4"], 0.99),
        "cf64_mean_before": float(before["cf64"].mean().item()),
        "cf64_mean_after": float(after["cf64"].mean().item()),
        "e4_nmse_csv": e4_csv,
        "e5_nmse_csv": e5_csv,
    }
    missing_layer = [c for c in LAYER_COLUMNS if c not in layer_row]
    if missing_layer:
        raise RuntimeError(f"layer_metrics missing columns: {missing_layer}")
    for key in LAYER_COLUMNS:
        if key in {
            "run_id",
            "module_name",
            "projection",
            "split",
            "e4_nmse_csv",
            "e5_nmse_csv",
        }:
            continue
        if isinstance(layer_row[key], float):
            _finite_float(f"{module_name}.{key}", float(layer_row[key]))

    resolved = {
        "module_name": module_name,
        "layer_idx": int(capture["layer_idx"]),
        "projection": str(capture["projection"]),
        "split": str(capture["split"]),
        "activation": _tensor_info(capture_path, capture["x_rot_bf16"]),
        "activation_rows_used": int(n),
        "activation_source_tensor": "x_rot_bf16",
        "capture_point": "post_rotation_pre_activation_quant",
        "weight_shape": list(w.shape),
        "weight_dtype": "float32",
        "weight_source": "dequantize_packed_nvfp4_W_N",
        "diag_d_shape": list(d_dev.shape),
        "y_ref": "Linear(A_N, W_N, bias) with A_N = Q_NVFP4(X_rot)",
        "online_rotation": "retained; not inverted",
    }

    return {
        "layer_row": layer_row,
        "group_df": group_df,
        "resolved": resolved,
        "cf4_hist_before": cf4_hist_before.detach().cpu(),
        "cf4_hist_after": cf4_hist_after.detach().cpu(),
    }


def _write_cf4_hist(path: Path, before: torch.Tensor, after: torch.Tensor) -> None:
    edges = CF4_BIN_EDGES.cpu()
    rows = []
    for i in range(before.numel()):
        rows.append(
            {
                "bin_left": float(edges[i].item()),
                "bin_right": float(edges[i + 1].item()),
                "count_before": int(before[i].item()),
                "count_after": int(after[i].item()),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def run_h4_experiment(
    config: AppConfig,
    *,
    capture_run_id: str,
    run_id: str,
    device: str,
    smoke: bool,
    modules: list[str] | None = None,
    max_rows: int | None = None,
    split: str = EVAL_SPLIT,
) -> dict[str, Any]:
    if split != EVAL_SPLIT:
        raise ValueError(
            f"H4 experiment must evaluate the DIAG validation split {EVAL_SPLIT!r}, got {split!r}"
        )
    capture_dir = results_dir(capture_run_id)
    manifest_path = capture_dir / "capture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"saved activations not found: {manifest_path}. Refusing to recapture."
        )
    manifest = read_json(manifest_path)
    validate_capture_manifest(config, capture_dir, manifest)
    if tuple(config.experiment.formal_layers) != REQUIRED_FORMAL_LAYERS:
        raise ValueError("config formal_layers mismatch")
    if len(config.formal_module_names) != REQUIRED_MODULE_COUNT:
        raise ValueError("config module count mismatch")

    snapshot = resolve_local_snapshot(config.model.model_id)
    index = read_json(snapshot / "model.safetensors.index.json")
    weight_map = index["weight_map"]

    if smoke:
        module_names = [config.formal_module_names[0]]
        row_limit = SMOKE_MAX_ROWS if max_rows is None else max_rows
        check_existing = False
    else:
        module_names = list(modules) if modules else list(config.formal_module_names)
        row_limit = max_rows
        check_existing = row_limit is None
        missing = [m for m in module_names if m not in config.formal_module_names]
        if missing:
            raise ValueError(f"unknown modules: {missing}")

    for module_name in module_names:
        cap_path = capture_file_path(capture_dir, module_name, split)
        if not cap_path.is_file():
            raise FileNotFoundError(f"missing capture {cap_path}. Refusing to recapture.")
        diag_path = capture_dir / "diagonal_scales" / f"{module_capture_stem(module_name)}.pt"
        if not diag_path.is_file():
            raise FileNotFoundError(
                f"missing DIAG scale {diag_path}. Refusing to re-search DIAG."
            )

    existing = _load_existing_linear_nmse(capture_dir) if check_existing else None
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)

    assert_r4_orthogonal()
    r4 = r4_matrix(dtype=torch.float64)
    out_dir = ensure_dir(h4_results_dir(run_id))
    fig_dir = ensure_dir(out_dir / "figures")
    write_json(
        out_dir / "config.json",
        {
            "run_id": run_id,
            "capture_run_id": capture_run_id,
            "smoke": smoke,
            "split": split,
            "max_rows": row_limit,
            "modules": module_names,
            "device": str(torch_device),
            "h4": {
                "H4_unnormalized": [list(row) for row in H4_UNNORMALIZED],
                "note": "operator is R4 = H4/2; unnormalized H4 is never applied",
                "R4": [list(row) for row in r4.tolist()],
                "layout": "last_dim contiguous G4, aligned to HiF4 group 64",
            },
            "cases": ["identity", "diag", "h4_fp32", "h4_bf16"],
            "hif4": {
                "group_size": STANDARD_HIF4_CONFIG.group_size,
                "group_dim": STANDARD_HIF4_CONFIG.group_dim,
                "s0_divisor": STANDARD_HIF4_CONFIG.s0_divisor,
                "e8_threshold": STANDARD_HIF4_CONFIG.e8_threshold,
                "e4_threshold": STANDARD_HIF4_CONFIG.e4_threshold,
                "s0_mode": STANDARD_HIF4_CONFIG.s0_mode,
            },
            "y_ref": "Linear(A_N, W_N, bias); A_N=Q_NVFP4(X_rot); W_N=dequant packed NVFP4",
            "activation_source": "x_rot_bf16 post online 16x16 rotation, pre NVFP4 quant",
            "diag_source": str(capture_dir / "diagonal_scales"),
            "capture_manifest": str(manifest_path),
        },
    )

    group_csv = out_dir / "group_metrics.csv"
    layer_rows: list[dict[str, Any]] = []
    resolved_list: list[dict[str, Any]] = []
    cf4_before = torch.zeros(CF4_BIN_EDGES.numel() - 1, dtype=torch.float32)
    cf4_after = torch.zeros_like(cf4_before)
    wrote_group_header = False

    for module_name in module_names:
        print(f"[h4] {module_name}", flush=True)
        cap_path = capture_file_path(capture_dir, module_name, split)
        capture = load_pt(cap_path, map_location="cpu")
        if capture["module_name"] != module_name:
            raise RuntimeError(
                f"capture module_name {capture['module_name']!r} != {module_name!r}"
            )
        diag_path = capture_dir / "diagonal_scales" / f"{module_capture_stem(module_name)}.pt"
        diag_obj = load_pt(diag_path, map_location="cpu")
        state = _packed_state(snapshot, weight_map, module_name)
        w_n = dequantize_packed_weight(state).to(dtype=torch.float32)
        result = evaluate_module(
            run_id=run_id,
            module_name=module_name,
            capture=capture,
            capture_path=cap_path,
            w_n=w_n,
            bias=state.bias if state.bias is None else state.bias.to(torch.float32),
            d=diag_obj["d"].to(torch.float32),
            existing_nmse=None if existing is None else existing[module_name],
            device=torch_device,
            max_rows=row_limit,
            cf4_hist_before=cf4_before,
            cf4_hist_after=cf4_after,
        )
        result["resolved"]["weight_checkpoint"] = str(snapshot)
        result["resolved"]["diag_path"] = str(diag_path)
        cf4_before = result["cf4_hist_before"]
        cf4_after = result["cf4_hist_after"]
        layer_rows.append(result["layer_row"])
        resolved_list.append(result["resolved"])
        result["group_df"].to_csv(
            group_csv,
            mode="w" if not wrote_group_header else "a",
            header=not wrote_group_header,
            index=False,
        )
        wrote_group_header = True
        del capture, w_n, result
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    layer_df = pd.DataFrame(layer_rows)[LAYER_COLUMNS]
    layer_df.to_csv(out_dir / "layer_metrics.csv", index=False)
    write_json(out_dir / "resolved_inputs.json", {"modules": resolved_list})
    _write_cf4_hist(out_dir / "cf4_hist.csv", cf4_before, cf4_after)

    if smoke:
        _run_smoke_field_checks(out_dir, layer_df)

    summary = analyze_run(out_dir)
    print(f"H4 EXPERIMENT DONE smoke={smoke} -> {out_dir}", flush=True)
    return summary


def _run_smoke_field_checks(out_dir: Path, layer_df: pd.DataFrame) -> None:
    group_df = pd.read_csv(out_dir / "group_metrics.csv")
    missing_g = [c for c in GROUP_COLUMNS if c not in group_df.columns]
    missing_l = [c for c in LAYER_COLUMNS if c not in layer_df.columns]
    if missing_g or missing_l:
        raise RuntimeError(f"smoke CSV fields missing group={missing_g} layer={missing_l}")
    if group_df.empty or layer_df.empty:
        raise RuntimeError("smoke produced empty metrics")
    if group_df.isna().any().any() or layer_df.isna().any().any():
        nan_cols = [c for c in layer_df.columns if layer_df[c].isna().any()]
        # e4/e5 csv are NaN in smoke because Identity/DIAG are not compared to full-val E4/E5.
        allowed = {"e4_nmse_csv", "e5_nmse_csv"}
        bad = [c for c in nan_cols if c not in allowed]
        if bad or group_df.isna().any().any():
            raise RuntimeError(f"smoke NaN in columns {bad}")
    required_files = [
        "config.json",
        "resolved_inputs.json",
        "group_metrics.csv",
        "layer_metrics.csv",
        "cf4_hist.csv",
    ]
    missing_f = [name for name in required_files if not (out_dir / name).is_file()]
    if missing_f:
        raise RuntimeError(f"smoke missing files: {missing_f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="H4 G4 block-rotation HiF4 experiment")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--capture-run-id", type=str, default=DEFAULT_CAPTURE_RUN_ID)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--modules", type=str, nargs="*", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    run_h4_experiment(
        config,
        capture_run_id=args.capture_run_id,
        run_id=args.run_id,
        device=args.device,
        smoke=bool(args.smoke),
        modules=args.modules,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
