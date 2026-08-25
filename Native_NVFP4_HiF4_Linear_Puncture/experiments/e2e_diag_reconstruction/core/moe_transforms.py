"""No-H16 DIAG/R64 transforms for Qwen3-MoE.

All callers (training, fold, materialization and vLLM sidecar generation) use
these operations.  There is intentionally no compatibility branch for the
legacy dense checkpoint's forward Hadamard matrix.
"""

from __future__ import annotations

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.diag_gradient.r64_transform import (
    R64_GROUP_SIZE,
    apply_r64_g64,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.transforms import (
    expand_vo_scale,
)


def _vector(x: torch.Tensor, expected: int, name: str) -> torch.Tensor:
    result = x.reshape(-1).to(dtype=torch.float32)
    if result.numel() != expected:
        raise ValueError(f"{name} length={result.numel()} != {expected}")
    return result


def apply_r64_no_cross_head(x: torch.Tensor, *, head_dim: int | None = None) -> torch.Tensor:
    """Apply fixed R64 inside contiguous groups, optionally inside each head."""
    width = int(x.shape[-1])
    if head_dim is None:
        if width % R64_GROUP_SIZE:
            raise ValueError(f"width={width} is not divisible by G64")
        return apply_r64_g64(x, dim=-1, compute_dtype=torch.float32, output_dtype=torch.float32)
    if head_dim <= 0 or head_dim % R64_GROUP_SIZE:
        raise ValueError(f"head_dim={head_dim} must be a positive multiple of G64")
    if width % head_dim:
        raise ValueError(f"width={width} is not divisible by head_dim={head_dim}")
    shape = x.shape
    y = x.reshape(-1, width // head_dim, head_dim)
    y = apply_r64_g64(y, dim=-1, compute_dtype=torch.float32, output_dtype=torch.float32)
    return y.reshape(shape)


def online_weight_transform_no_h(
    weight: torch.Tensor,
    d: torch.Tensor,
    *,
    use_r64: bool,
    rot_order: str,
    head_dim: int | None = None,
) -> torch.Tensor:
    """Transform a linear weight for online DIAG with no native H16."""
    w = weight.to(torch.float32)
    scale = _vector(d, int(w.shape[-1]), "online D")
    if rot_order == "diag_then_rot":
        result = w / scale
        return apply_r64_no_cross_head(result, head_dim=head_dim) if use_r64 else result
    if rot_order == "rot_then_diag":
        result = apply_r64_no_cross_head(w, head_dim=head_dim) if use_r64 else w
        return result / scale
    raise ValueError(f"invalid rot_order={rot_order!r}")


def fusable_weight_transform_no_h(
    weight: torch.Tensor,
    d_in: torch.Tensor,
    d_out: torch.Tensor,
    *,
    use_r64: bool,
    head_dim: int | None = None,
) -> torch.Tensor:
    """``D_out @ W @ D_in^-1 [@ R64]`` for the fusable MoE design."""
    w = weight.to(torch.float32)
    d_in_v = _vector(d_in, int(w.shape[-1]), "fusable D_in")
    d_out_v = _vector(d_out, int(w.shape[0]), "fusable D_out")
    result = d_out_v[:, None] * (w / d_in_v)
    return apply_r64_no_cross_head(result, head_dim=head_dim) if use_r64 else result


def expand_vo_scale_qwen3_moe(d_vo: torch.Tensor) -> torch.Tensor:
    return expand_vo_scale(d_vo, num_attention_heads=32, num_key_value_heads=4, head_dim=128)


def transform_router_weight(router_weight: torch.Tensor, d_gu: torch.Tensor) -> torch.Tensor:
    """Strict router inverse compensation for folding post-attention RMSNorm."""
    return router_weight.to(torch.float32) / _vector(d_gu, int(router_weight.shape[-1]), "D_GU")
