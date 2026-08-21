"""Offline activation format oracles (MXFP8 / HiF4). No rotation parameters."""

from __future__ import annotations

import torch

from HiFloat4.hif4_scale_threshold_optimization.src.quantizer import (
    HiF4QuantConfig,
    quantize_hif4,
)
from NVFP4.torch_fake import FP8_E4M3FN_MAX, cast_to_fp8_e4m3fn

STANDARD_HIF4_CONFIG = HiF4QuantConfig(
    group_size=64,
    group_dim=-1,
    s0_divisor=7.0,
    e8_threshold=4.0,
    e4_threshold=2.0,
    s0_mode="hardware",
)

MXFP8_BLOCK_SIZE = 32


def qdq_mxfp8_post_rotation(
    x_rot_bf16: torch.Tensor,
    *,
    block_size: int = MXFP8_BLOCK_SIZE,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """MXFP8 QDQ on already-rotated activations: block32 / E8M0 / E4M3FN."""
    if x_rot_bf16.shape[-1] % block_size != 0:
        raise ValueError(
            f"MXFP8 requires last dim divisible by {block_size}, "
            f"got {x_rot_bf16.shape[-1]}"
        )
    original_shape = x_rot_bf16.shape
    xf = x_rot_bf16.to(torch.float32)
    grouped = xf.reshape(*original_shape[:-1], -1, block_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    safe_amax = torch.where(amax == 0, torch.ones_like(amax), amax)
    scale_exp = torch.ceil(torch.log2(safe_amax / FP8_E4M3FN_MAX))
    scale_exp = torch.clamp(scale_exp, min=-127.0, max=127.0)
    scale = torch.where(amax == 0, torch.ones_like(amax), torch.exp2(scale_exp))
    payload = cast_to_fp8_e4m3fn(grouped / scale).to(torch.float32)
    recon = (payload * scale).reshape(original_shape)
    return recon.to(dtype=output_dtype)


def qdq_hif4_direct(
    x: torch.Tensor,
    *,
    config: HiF4QuantConfig | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """HiF4 direct RTN reconstruction along the last dimension."""
    cfg = config or STANDARD_HIF4_CONFIG
    result = quantize_hif4(x.to(torch.float32), config=cfg)
    out_dtype = output_dtype if output_dtype is not None else x.dtype
    return result.reconstruction.to(dtype=out_dtype)
