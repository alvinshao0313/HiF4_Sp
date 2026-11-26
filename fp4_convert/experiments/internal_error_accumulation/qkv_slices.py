"""Q/K/V fused-slice helpers using runtime q_size/kv_size only."""
from __future__ import annotations

import torch


def runtime_qkv_slice_bounds(attn_module) -> dict[str, tuple[int, int]]:
    """Return rank-local fused qkv slice bounds from the live Attention module."""
    q_size = int(getattr(attn_module, "q_size"))
    kv_size = int(getattr(attn_module, "kv_size"))
    if q_size <= 0 or kv_size <= 0:
        raise RuntimeError(f"invalid runtime q_size/kv_size: {q_size}/{kv_size}")
    return {
        "Q": (0, q_size),
        "K": (q_size, q_size + kv_size),
        "V": (q_size + kv_size, q_size + 2 * kv_size),
    }


def apply_qkv_slice_repair(
    actual: torch.Tensor,
    source: torch.Tensor,
    *,
    which: str,
    bounds: dict[str, tuple[int, int]],
) -> torch.Tensor:
    if which not in bounds:
        raise ValueError(f"unknown QKV slice {which}")
    if actual.shape != source.shape:
        raise RuntimeError("QKV actual/source shape mismatch")
    lo, hi = bounds[which]
    if actual.shape[-1] < hi:
        raise RuntimeError(
            f"fused qkv last-dim {actual.shape[-1]} smaller than required slice end {hi}"
        )
    out = actual.clone()
    out[..., lo:hi] = source[..., lo:hi]
    return out


def assert_captured_e1_slice_noop(
    actual_e1: torch.Tensor,
    *,
    which: str,
    bounds: dict[str, tuple[int, int]],
) -> dict:
    """Replacing an E1 slice with itself must be exact."""
    repaired = apply_qkv_slice_repair(actual_e1, actual_e1, which=which, bounds=bounds)
    if not torch.equal(repaired, actual_e1):
        raise RuntimeError(f"captured-E1 {which} slice no-op identity failed")
    return {"pass": True, "which": which, "bounds": bounds[which]}
