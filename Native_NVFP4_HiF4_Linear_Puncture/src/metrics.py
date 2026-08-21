"""Metric helpers for Linear puncture experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _as_float(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float64).reshape(-1)


def error_energy(y_hat: torch.Tensor, y_ref: torch.Tensor) -> float:
    d = _as_float(y_hat) - _as_float(y_ref)
    return float((d * d).sum().item())


def reference_energy(y_ref: torch.Tensor) -> float:
    r = _as_float(y_ref)
    return float((r * r).sum().item())


def compute_nmse(y_hat: torch.Tensor, y_ref: torch.Tensor, eps: float = 1e-30) -> float:
    ref = reference_energy(y_ref)
    err = error_energy(y_hat, y_ref)
    return float(err / max(ref, eps))


# Alias used throughout the experiment package.
nmse = compute_nmse


def compute_sqnr_db(nmse_value: float, eps: float = 1e-30) -> float:
    return float(-10.0 * math.log10(max(nmse_value, eps)))


sqnr_db = compute_sqnr_db


def compute_recovery(direct_error_energy: float, improved_error_energy: float) -> float:
    """``(direct - improved) / direct``; NaN if direct_error_energy == 0."""
    if direct_error_energy == 0.0:
        return float("nan")
    return float((direct_error_energy - improved_error_energy) / direct_error_energy)


def recovery_ratio(direct_error: float, improved_error: float) -> float:
    return compute_recovery(direct_error, improved_error)


def aggregate_global_nmse(
    error_energies: Sequence[float],
    reference_energies: Sequence[float],
    eps: float = 1e-30,
) -> float:
    err = float(sum(error_energies))
    ref = float(sum(reference_energies))
    return err / max(ref, eps)


def relative_l2(y_hat: torch.Tensor, y_ref: torch.Tensor, eps: float = 1e-30) -> float:
    return float(math.sqrt(compute_nmse(y_hat, y_ref, eps=eps)))


def mae(y_hat: torch.Tensor, y_ref: torch.Tensor) -> float:
    d = (_as_float(y_hat) - _as_float(y_ref)).abs()
    return float(d.mean().item())


def max_abs_error(y_hat: torch.Tensor, y_ref: torch.Tensor) -> float:
    d = (_as_float(y_hat) - _as_float(y_ref)).abs()
    return float(d.max().item())


def cosine_similarity(y_hat: torch.Tensor, y_ref: torch.Tensor, eps: float = 1e-30) -> float:
    a = _as_float(y_hat)
    b = _as_float(y_ref)
    denom = float(a.norm().item() * b.norm().item())
    if denom <= eps:
        return float("nan")
    return float(torch.dot(a, b).item() / denom)


def bias_mean(y_hat: torch.Tensor, y_ref: torch.Tensor) -> float:
    d = _as_float(y_hat) - _as_float(y_ref)
    return float(d.mean().item())


def compare_tensors(y_hat: torch.Tensor, y_ref: torch.Tensor, eps: float = 1e-30) -> dict[str, float]:
    ref_e = reference_energy(y_ref)
    err_e = error_energy(y_hat, y_ref)
    nmse_v = err_e / max(ref_e, eps)
    return {
        "num_output_elements": float(y_ref.numel()),
        "reference_energy": ref_e,
        "error_energy": err_e,
        "nmse": nmse_v,
        "sqnr_db": compute_sqnr_db(nmse_v, eps=eps),
        "relative_l2": math.sqrt(nmse_v),
        "mae": mae(y_hat, y_ref),
        "max_abs_error": max_abs_error(y_hat, y_ref),
        "cosine": cosine_similarity(y_hat, y_ref, eps=eps),
        "bias_mean": bias_mean(y_hat, y_ref),
    }


def zero_rate(x: torch.Tensor) -> float:
    xf = x.detach().reshape(-1)
    return float((xf == 0).to(torch.float64).mean().item())
