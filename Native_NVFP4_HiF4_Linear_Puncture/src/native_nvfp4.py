"""Packed NVFP4 weight decode and post-rotation activation QDQ."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from NVFP4.dequantize import dequantize_nvfp4_weight
from NVFP4.torch_fake import fake_quant_nvfp4_activation
import NVFP4.torch_fake as nvfp4_fake

from Native_NVFP4_HiF4_Linear_Puncture.src.rotation import apply_block_rotation


@dataclass(frozen=True)
class PackedNVFP4LinearState:
    module_name: str
    weight_packed: torch.Tensor
    weight_scale: torch.Tensor
    weight_global_scale: torch.Tensor
    input_global_scale: torch.Tensor
    rotation_matrix: torch.Tensor
    bias: torch.Tensor | None


def decode_weight_scale_uint8(scales_uint8: torch.Tensor) -> torch.Tensor:
    """Interpret checkpoint ``scales`` (uint8 storage) as float8_e4m3fn → float32."""
    if scales_uint8.dtype != torch.uint8:
        raise TypeError(
            f"checkpoint scales must be uint8 storage, got {scales_uint8.dtype}"
        )
    return scales_uint8.view(torch.float8_e4m3fn).to(torch.float32)


def dequantize_packed_weight(
    state: PackedNVFP4LinearState,
    *,
    dtype: torch.dtype = torch.bfloat16,
    group_size: int = 16,
) -> torch.Tensor:
    scale = state.weight_scale
    if scale.dtype == torch.uint8:
        scale = decode_weight_scale_uint8(scale)
    elif scale.dtype == torch.float8_e4m3fn:
        scale = scale.to(torch.float32)
    else:
        scale = scale.to(torch.float32)

    return dequantize_nvfp4_weight(
        weight_packed=state.weight_packed,
        weight_scale=scale,
        weight_global_scale=state.weight_global_scale.to(torch.float32),
        dtype=dtype,
        group_size=group_size,
    )


def qdq_nvfp4_post_rotation(
    x_rot_bf16: torch.Tensor,
    input_global_scale: torch.Tensor,
    *,
    group_size: int = 16,
) -> torch.Tensor:
    """NVFP4 activation QDQ on already-rotated activations (no extra rotation)."""
    prev = nvfp4_fake.USE_TRITON_NVFP4_KERNEL
    use_triton = bool(prev and x_rot_bf16.is_cuda)
    nvfp4_fake.USE_TRITON_NVFP4_KERNEL = use_triton
    try:
        return fake_quant_nvfp4_activation(
            x_rot_bf16,
            input_global_scale=input_global_scale.to(torch.float32),
            group_size=group_size,
            output_dtype=torch.bfloat16,
        )
    finally:
        nvfp4_fake.USE_TRITON_NVFP4_KERNEL = prev


def source_linear_semantic(
    x_pre: torch.Tensor,
    state: PackedNVFP4LinearState,
    *,
    rotation_group_size: int = 16,
    activation_group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(y, x_rot, a_n)`` for the native NVFP4 Linear semantic."""
    x_rot = apply_block_rotation(
        x_pre, state.rotation_matrix, group_size=rotation_group_size
    )
    a_n = qdq_nvfp4_post_rotation(x_rot, state.input_global_scale)
    w_n = dequantize_packed_weight(state)
    bias = state.bias
    if bias is not None:
        bias = bias.to(device=a_n.device, dtype=w_n.dtype)
    y = F.linear(a_n, w_n.to(device=a_n.device), bias)
    return y, x_rot, a_n
