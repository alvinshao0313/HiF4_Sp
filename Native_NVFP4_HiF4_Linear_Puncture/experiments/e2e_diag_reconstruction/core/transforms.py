"""DIAG / R64 / GQA scale math. Single source of truth for matrix order."""

from __future__ import annotations

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.diag_gradient.r64_transform import (
    R64_GROUP_SIZE,
    apply_r64_g64,
)

H16_GROUP_SIZE = 16


def apply_block_right_fp32(
    x: torch.Tensor,
    matrix: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Right-multiply contiguous last-dim groups: ``[..., G, g] @ matrix`` in FP32."""
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    k = int(x.shape[-1])
    if k % group_size != 0:
        raise ValueError(f"last dim K={k} must be divisible by group_size={group_size}")
    if tuple(matrix.shape) != (group_size, group_size):
        raise ValueError(
            f"matrix shape must be ({group_size}, {group_size}), got {tuple(matrix.shape)}"
        )
    orig_shape = x.shape
    xf = x.reshape(-1, k // group_size, group_size).to(torch.float32)
    h = matrix.to(device=x.device, dtype=torch.float32)
    y = torch.matmul(xf, h)
    return y.reshape(orig_shape)


def apply_r64(x: torch.Tensor) -> torch.Tensor:
    """Right-multiply every contiguous G64 on the last dim by the fixed R64."""
    return apply_r64_g64(
        x,
        dim=-1,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )


def expand_vo_scale(
    d_vo: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Repeat each KV-head scale block ``r`` times into Q-head space. No extra learning."""
    if num_attention_heads <= 0 or num_key_value_heads <= 0 or head_dim <= 0:
        raise ValueError("head counts and head_dim must be positive")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            f"num_attention_heads={num_attention_heads} is not divisible by "
            f"num_key_value_heads={num_key_value_heads}"
        )
    if head_dim % R64_GROUP_SIZE != 0:
        raise ValueError(
            f"head_dim={head_dim} is not divisible by {R64_GROUP_SIZE}; "
            "G64 must not cross an attention head"
        )
    expected = num_key_value_heads * head_dim
    if int(d_vo.numel()) != expected:
        raise ValueError(
            f"d_vo numel={int(d_vo.numel())} != num_kv_heads*head_dim={expected}"
        )
    r = num_attention_heads // num_key_value_heads
    blocks = d_vo.reshape(num_key_value_heads, head_dim)
    expanded = blocks.unsqueeze(1).expand(num_key_value_heads, r, head_dim)
    return expanded.reshape(-1)


def _as_row_scale(d: torch.Tensor, k: int, name: str) -> torch.Tensor:
    vec = d.reshape(-1).to(torch.float32)
    if int(vec.numel()) != k:
        raise ValueError(f"{name} length {int(vec.numel())} != expected {k}")
    return vec


def online_weight_transform(
    w_n: torch.Tensor,
    d: torch.Tensor,
    use_r64: bool,
    rot_order: str,
) -> torch.Tensor:
    """``diag_then_rot``: W / D then optional R64. ``rot_then_diag``: optional R64 then W / D."""
    w = w_n.to(torch.float32)
    scale = _as_row_scale(d, int(w.shape[-1]), "online D")
    if rot_order == "diag_then_rot":
        w = w / scale
        if use_r64:
            w = apply_r64(w)
        return w
    if rot_order == "rot_then_diag":
        if use_r64:
            w = apply_r64(w)
        return w / scale
    raise ValueError(f"invalid rot_order={rot_order!r}")


def fusable_weight_transform(
    w_n: torch.Tensor,
    h16: torch.Tensor,
    d_in: torch.Tensor,
    d_out: torch.Tensor,
    use_r64: bool,
) -> torch.Tensor:
    """``W_N @ H^T / D_in @ H`` then optional R64 then optional row-scale ``D_out``."""
    w = w_n.to(torch.float32)
    h = h16.to(device=w.device, dtype=torch.float32)
    d_in_vec = _as_row_scale(d_in, int(w.shape[-1]), "fusable D_in")
    d_out_vec = _as_row_scale(d_out, int(w.shape[0]), "fusable D_out")
    w = apply_block_right_fp32(w, h.T, H16_GROUP_SIZE)
    w = w / d_in_vec
    w = apply_block_right_fp32(w, h, H16_GROUP_SIZE)
    if use_r64:
        w = apply_r64(w)
    return w * d_out_vec[:, None]


def scale_from_log2(z: torch.Tensor) -> torch.Tensor:
    return torch.exp2(z.to(torch.float32))
