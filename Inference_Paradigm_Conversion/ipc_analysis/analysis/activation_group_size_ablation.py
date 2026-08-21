"""AX2: HiF4 group-size ablation and sub16 dispersion sweep."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.analysis.synthetic_mechanisms import (
    apply_dispersion_dose,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics

_GROUP_VARIANTS = {
    16: "group16_full_hierarchy",
    32: "group32_full_hierarchy",
    64: "group64_full_hierarchy",
}


def sub16_dispersion_stats(x_groups_64: torch.Tensor) -> dict[str, torch.Tensor | float]:
    """x_groups_64 [G, 64] -> sub16 dispersion tensors/stats."""
    if x_groups_64.shape[-1] != 64:
        raise ValueError("expected last dim 64")
    sub = x_groups_64.reshape(-1, 4, 16).float()
    amax = sub.abs().amax(dim=-1)
    rms = torch.sqrt((sub * sub).mean(dim=-1).clamp_min(1e-12))
    energy = (sub * sub).sum(dim=-1)
    amax_max = amax.max(dim=-1).values
    amax_mean = amax.mean(dim=-1).clamp_min(1e-12)
    rms_max = rms.max(dim=-1).values
    rms_mean = rms.mean(dim=-1).clamp_min(1e-12)
    amax_pos = amax[amax > 0]
    log2_range = torch.tensor(0.0)
    if amax_pos.numel() > 1:
        log2_range = amax_pos.max().log2() - amax_pos.min().log2()
    return {
        "sub16_amax": amax,
        "sub16_rms": rms,
        "sub16_energy": energy,
        "sub16_amax_ratio": amax_max / amax_mean,
        "sub16_rms_ratio": rms_max / rms_mean,
        "sub16_log2_amax_range": log2_range,
        "sub16_energy_share_max": energy.max(dim=-1).values / energy.sum(dim=-1).clamp_min(1e-12),
        "sub16_cv_amax": amax.std(dim=-1) / amax_mean,
        "sub16_cv_rms": rms.std(dim=-1) / rms_mean,
    }


def apply_sub16_dispersion(x: torch.Tensor, d: float) -> torch.Tensor:
    """Apply [2^{-d}, 2^{-d/3}, 2^{d/3}, 2^d] on 4×16 blocks; restore 64-group RMS."""
    orig = x.shape
    flat = x.reshape(-1, 64).float()
    out = torch.stack([apply_dispersion_dose(flat[i].reshape(4, 16), d).reshape(64) for i in range(flat.shape[0])])
    return out.reshape(orig).to(x.dtype)


def _quantize_variant(x_bf16: torch.Tensor, group_size: int) -> torch.Tensor:
    variant = _GROUP_VARIANTS[group_size]
    view = quantize_hif4_tensor(
        x_bf16.float(), group_dim=-1, variant=variant, output_dtype=torch.float32
    )
    return view.metadata["values_fp32"].float()


def _row_metrics(
    *,
    variant: str,
    group_size: int,
    a_v: torch.Tensor,
    a_n: torch.Tensor,
    a_h: torch.Tensor,
    w_n: torch.Tensor,
    is_standard: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    e_full_a = float(((a_h - a_n) ** 2).sum().item())
    e_full_y = float((F.linear(a_h - a_n, w_n.float()) ** 2).sum().item())
    e_a = float(((a_v - a_n) ** 2).sum().item())
    e_y = float((F.linear(a_v - a_n, w_n.float()) ** 2).sum().item())
    m_act = compute_pair_metrics(a_n, a_v)
    row: dict[str, Any] = {
        "variant": variant,
        "group_size": group_size,
        "is_standard_hif4": is_standard,
        "activation_nmse": m_act["nmse"],
        "activation_error_energy": e_a,
        "output_error_energy": e_y,
        "R_A": 1.0 - e_a / e_full_a if e_full_a > 0 else 0.0,
        "R_Y": 1.0 - e_y / e_full_y if e_full_y > 0 else 0.0,
    }
    if extra:
        row.update(extra)
    return row


@torch.no_grad()
def run_group_size_ablation(
    x_bf16: torch.Tensor,
    a_n: torch.Tensor,
    w_n: torch.Tensor,
) -> list[dict[str, Any]]:
    """Rows for G16/G32/G64 using full_hierarchy variants."""
    a_n_f = a_n.float()
    a_h = _quantize_variant(x_bf16, 64)
    rows: list[dict[str, Any]] = []
    for gs in (16, 32, 64):
        a_v = _quantize_variant(x_bf16, gs)
        rows.append(
            _row_metrics(
                variant=f"G{gs}",
                group_size=gs,
                a_v=a_v,
                a_n=a_n_f,
                a_h=a_h,
                w_n=w_n,
                is_standard=gs == 64,
            )
        )
    return rows


@torch.no_grad()
def run_dispersion_sweep(
    x_bf16: torch.Tensor,
    a_n: torch.Tensor,
    w_n: torch.Tensor,
    d_values: list[float] | tuple[float, ...],
) -> list[dict[str, Any]]:
    """Dispersion dose sweep with G16/G32/G64 on perturbed x."""
    a_n_f = a_n.float()
    a_h_base = _quantize_variant(x_bf16, 64)
    rows: list[dict[str, Any]] = []
    for d in d_values:
        x_p = apply_sub16_dispersion(x_bf16, d)
        stats = sub16_dispersion_stats(x_p.reshape(-1, 64))
        stat_row = {k: float(v.mean().item()) if torch.is_tensor(v) else float(v) for k, v in stats.items()}
        for gs in (16, 32, 64):
            a_v = _quantize_variant(x_p, gs)
            row = _row_metrics(
                variant=f"G{gs}",
                group_size=gs,
                a_v=a_v,
                a_n=a_n_f,
                a_h=a_h_base,
                w_n=w_n,
                is_standard=gs == 64,
                extra={"dispersion_d": d, **stat_row},
            )
            rows.append(row)
    return rows
