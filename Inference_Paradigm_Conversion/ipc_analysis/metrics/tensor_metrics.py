"""Pairwise tensor / group metrics with FP64 accumulation."""

from __future__ import annotations

import math
from typing import Any

import torch


def _to_f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().reshape(-1).to(dtype=torch.float64, device="cpu")


def _safe_div(num: float, den: float) -> float:
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def _percentile(sorted_vals: torch.Tensor, q: float) -> float:
    if sorted_vals.numel() == 0:
        return 0.0
    # q in [0, 100]
    n = sorted_vals.numel()
    if n == 1:
        return float(sorted_vals[0].item())
    pos = (q / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo].item())
    w = pos - lo
    return float((sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w).item())


def compute_pair_metrics(
    reference: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    if reference.shape != target.shape:
        raise ValueError(
            f"shape mismatch: reference {tuple(reference.shape)} vs target {tuple(target.shape)}"
        )
    ref = _to_f64(reference)
    tgt = _to_f64(target)
    err = tgt - ref
    ref_energy = float(torch.dot(ref, ref).item())
    tgt_energy = float(torch.dot(tgt, tgt).item())
    err_energy = float(torch.dot(err, err).item())
    dot = float(torch.dot(ref, tgt).item())
    abs_err = err.abs()
    mae = float(abs_err.mean().item()) if ref.numel() else 0.0
    mean_signed = float(err.mean().item()) if ref.numel() else 0.0
    max_abs = float(abs_err.max().item()) if ref.numel() else 0.0
    nmse = _safe_div(err_energy, ref_energy)
    if err_energy == 0.0:
        sqnr = float("inf") if ref_energy > 0.0 else 0.0
    elif ref_energy == 0.0:
        sqnr = float("-inf")
    else:
        sqnr = 10.0 * math.log10(ref_energy / err_energy)
    ref_norm = math.sqrt(ref_energy)
    tgt_norm = math.sqrt(tgt_energy)
    if ref_norm == 0.0 and tgt_norm == 0.0:
        cosine = 1.0
    elif ref_norm == 0.0 or tgt_norm == 0.0:
        cosine = 0.0
    else:
        cosine = dot / (ref_norm * tgt_norm)
    relative_norm_change = _safe_div(tgt_norm - ref_norm, ref_norm) if ref_norm != 0.0 else (
        0.0 if tgt_norm == 0.0 else float("inf")
    )

    sorted_abs = torch.sort(abs_err).values
    # top-1% error-energy share
    sq = err * err
    if sq.numel() == 0:
        top1 = 0.0
    else:
        k = max(1, int(math.ceil(0.01 * sq.numel())))
        top_vals = torch.topk(sq, k=k, largest=True).values
        top1 = _safe_div(float(top_vals.sum().item()), err_energy) if err_energy > 0 else 0.0

    def _finite(v: float) -> float:
        # JSON forbid NaN; keep inf as a sentinel string-compatible via json allow_nan=False
        # so callers must replace non-finite before dump. We use large finite sentinels.
        if math.isnan(v):
            raise ValueError("metric produced NaN")
        if math.isinf(v):
            return 1.0e300 if v > 0 else -1.0e300
        return v

    return {
        "nmse": _finite(nmse),
        "sqnr_db": _finite(sqnr),
        "cosine": _finite(cosine),
        "mae": _finite(mae),
        "mean_signed_error": _finite(mean_signed),
        "max_abs_error": _finite(max_abs),
        "relative_norm_change": _finite(relative_norm_change),
        "reference_energy": _finite(ref_energy),
        "target_energy": _finite(tgt_energy),
        "error_energy": _finite(err_energy),
        "error_p50": _finite(_percentile(sorted_abs, 50)),
        "error_p90": _finite(_percentile(sorted_abs, 90)),
        "error_p99": _finite(_percentile(sorted_abs, 99)),
        "error_p99_9": _finite(_percentile(sorted_abs, 99.9)),
        "top1pct_error_energy_share": _finite(top1),
        "numel": float(ref.numel()),
    }


def compute_group_metrics(
    reference: torch.Tensor,
    target: torch.Tensor,
    group_size: int,
    group_dim: int = -1,
) -> list[dict[str, float]]:
    if reference.shape != target.shape:
        raise ValueError("shape mismatch")
    if reference.shape[group_dim] % group_size != 0:
        raise ValueError(
            f"dim {group_dim} length {reference.shape[group_dim]} not divisible by {group_size}"
        )
    ref = reference.movedim(group_dim, -1).reshape(-1, group_size)
    tgt = target.movedim(group_dim, -1).reshape(-1, group_size)
    out: list[dict[str, float]] = []
    for i in range(ref.shape[0]):
        m = compute_pair_metrics(ref[i], tgt[i])
        m["group_index"] = float(i)
        out.append(m)
    return out
