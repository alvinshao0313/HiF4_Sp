"""Reconstruction and stage-decision metrics for HiF4 experiments."""

from __future__ import annotations

import math
from typing import Any

import torch

from .formats import S1P2_MAX, quantize_s1p2_magnitude
from .quantizer import HiF4QuantConfig, HiF4QuantResult, quantize_hif4


def nmse(reference: torch.Tensor, approx: torch.Tensor) -> float:
    ref = reference.detach().to(torch.float64).reshape(-1)
    apx = approx.detach().to(torch.float64).reshape(-1)
    denom = float((ref * ref).sum().item())
    if denom == 0.0:
        return 0.0
    err = ref - apx
    return float((err * err).sum().item() / denom)


def sqnr_db(reference: torch.Tensor, approx: torch.Tensor) -> float:
    n = nmse(reference, approx)
    if n <= 0.0:
        return float("inf")
    return -10.0 * math.log10(n)


def mae(reference: torch.Tensor, approx: torch.Tensor) -> float:
    ref = reference.detach().to(torch.float64)
    apx = approx.detach().to(torch.float64)
    return float((ref - apx).abs().mean().item())


def max_abs_error(reference: torch.Tensor, approx: torch.Tensor) -> float:
    ref = reference.detach().to(torch.float64)
    apx = approx.detach().to(torch.float64)
    return float((ref - apx).abs().max().item())


def detailed_quant_metrics(
    values: torch.Tensor,
    result: HiF4QuantResult,
    *,
    reference_result: HiF4QuantResult | None = None,
) -> dict[str, Any]:
    """Full reconstruction metrics plus clipping / stage statistics."""
    x = values.detach().to(torch.float32)
    recon = result.reconstruction.detach().to(torch.float32)
    err = x - recon

    # Payload / local scale live in original layout; work in group-flat space.
    payload = result.payload.detach().to(torch.float32)
    local_scale = result.local_scale.detach().to(torch.float32)
    abs_x = x.abs()
    # Ideal continuous magnitude on the same local scale.
    safe_local = torch.where(local_scale > 0, local_scale, torch.ones_like(local_scale))
    ideal_mag = abs_x / safe_local
    clipped = ideal_mag > S1P2_MAX
    clip_rate = float(clipped.float().mean().item())

    # Clipping error energy: contribution from magnitudes forced to 1.75.
    clipped_target = torch.where(clipped, S1P2_MAX * local_scale, abs_x)
    clip_err = abs_x - clipped_target
    clip_err_energy = float((clip_err * clip_err).sum().item())

    # In-range S1P2 rounding error (on unclipped elements).
    in_range = ~clipped
    rounded_mag = quantize_s1p2_magnitude(ideal_mag)
    round_err = torch.where(
        in_range,
        abs_x - rounded_mag * local_scale,
        torch.zeros_like(abs_x),
    )
    round_err_energy = float((round_err * round_err).sum().item())

    e8_rate = float(result.e8.float().mean().item()) if result.e8.numel() else 0.0
    e4_rate = float(result.e4.float().mean().item()) if result.e4.numel() else 0.0

    out: dict[str, Any] = {
        "nmse": nmse(x, recon),
        "sqnr_db": sqnr_db(x, recon),
        "mae": mae(x, recon),
        "max_absolute_error": max_abs_error(x, recon),
        "clipping_rate": clip_rate,
        "clipping_error_energy": clip_err_energy,
        "inrange_round_error_energy": round_err_energy,
        "e8_trigger_rate": e8_rate,
        "e4_trigger_rate": e4_rate,
        "total_error_energy": float((err * err).sum().item()),
        "reference_energy": float((x * x).sum().item()),
        "numel": int(x.numel()),
    }

    if reference_result is not None:
        e8_flip = float((result.e8 != reference_result.e8).float().mean().item())
        e4_flip = float((result.e4 != reference_result.e4).float().mean().item())
        out["e8_bit_flip_rate_vs_ref"] = e8_flip
        out["e4_bit_flip_rate_vs_ref"] = e4_flip

    return out


def evaluate_config(values: torch.Tensor, config: HiF4QuantConfig) -> dict[str, Any]:
    standard = quantize_hif4(values, config=HiF4QuantConfig(s0_mode=config.s0_mode))
    result = quantize_hif4(values, config=config)
    metrics = detailed_quant_metrics(values, result, reference_result=standard)
    metrics["s0_divisor"] = config.s0_divisor
    metrics["e8_threshold"] = config.e8_threshold
    metrics["e4_threshold"] = config.e4_threshold
    metrics["s0_mode"] = config.s0_mode
    return metrics
