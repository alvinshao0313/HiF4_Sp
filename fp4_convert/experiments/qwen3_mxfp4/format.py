"""CPU format contract shared by conversion and the independent GPU reference."""
from __future__ import annotations

import torch

from ..diag_gradient.r64_transform import apply_r64_g64, r64_matrix
from .fht32 import apply_fht32, fht32_reference

VARIANTS = ("direct", "r64", "fht32")

VALUES = (0., .5, 1., 1.5, 2., 3., 4., 6.)
MIDPOINTS = (.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)
SCALE_RULE = "clamp(floor(log2(amax))-2,-127,127); zero=0; E2M1 RNE"
FORMAT_VERSION = "qwen3-routed-mxfp4-v1"


def unpack_codes(packed):
    if packed.dtype != torch.uint8:
        raise ValueError("E2M1 packed storage must be uint8")
    return torch.stack((packed & 15, packed >> 4), -1).flatten(-2)


def unpack(packed):
    codes = unpack_codes(packed)
    table = torch.tensor(VALUES, dtype=torch.float32, device=packed.device)
    return torch.copysign(table[(codes & 7).long()], torch.where(codes >= 8, -1., 1.))


def decode_scales(scales):
    if scales.dtype != torch.uint8 or torch.any(scales == 255):
        raise ValueError("E8M0 requires finite scale bytes 0..254")
    return torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.int() - 127)


def quantize(x):
    if x.dtype not in (torch.bfloat16, torch.float32) or x.ndim < 1 or x.shape[-1] % 32:
        raise ValueError("Expected BF16/FP32 input with K divisible by 32")
    if not torch.isfinite(x).all():
        raise ValueError("Cannot quantize nonfinite input")
    blocks = x.float().reshape(*x.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(-1)
    # frexp avoids log2 rounding to the next integer immediately below powers of 2.
    _, binary_exponent = torch.frexp(maximum)
    exponent = torch.where(maximum == 0, 0, binary_exponent - 3).clamp(-127, 127)
    scales = (exponent + 127).to(torch.uint8)
    normalized = blocks / decode_scales(scales).unsqueeze(-1)
    magnitude = normalized.abs().contiguous()
    midpoints = torch.tensor(MIDPOINTS, device=x.device)
    code = torch.bucketize(magnitude, midpoints)
    tie = magnitude == midpoints[code.clamp_max(6)]
    code += (tie & (code % 2 == 1)).long()
    code = (code.to(torch.uint8) | (torch.signbit(normalized).to(torch.uint8) << 3)).reshape_as(x)
    return (code[..., ::2] | (code[..., 1::2] << 4)).contiguous(), scales.contiguous()


def dequantize(packed, scales):
    values = unpack(packed)
    if scales.shape != (*values.shape[:-1], values.shape[-1] // 32) or values.shape[-1] % 32:
        raise ValueError("MXFP4 value/scale shapes disagree")
    return (values.reshape(*scales.shape, 32) * decode_scales(scales).unsqueeze(-1)).reshape_as(values).bfloat16()


def nvfp4_to_bf16(packed, scales, scale_2):
    values = unpack(packed)
    if scales.dtype != torch.float8_e4m3fn or scale_2.numel() != 1:
        raise ValueError("Expected Qwen NVFP4 E4M3 group scales and scalar weight_scale_2")
    if scales.shape != (*values.shape[:-1], values.shape[-1] // 16):
        raise ValueError("NVFP4 value/scale shapes disagree")
    if not torch.isfinite(scales.float()).all() or not torch.all(scales.float() >= 0):
        raise ValueError("Invalid NVFP4 block scales")
    if not torch.isfinite(scale_2).all() or not torch.all(scale_2 > 0):
        raise ValueError("Invalid Qwen weight_scale_2")
    # Qwen's packed emulation kernel multiplies these scales BEFORE the E2M1
    # value, then rounds the GEMM operand to BF16. scale_2 is NOT reciprocal.
    effective = scales.float() * scale_2.float()
    return (values.reshape(*scales.shape, 16) * effective.unsqueeze(-1)).reshape_as(values).bfloat16()


def rotate(x, variant="r64"):
    if variant == "fht32":
        return apply_fht32(x)
    if variant == "direct":
        return x.float()
    if variant != "r64":
        raise ValueError(f"Unknown rotation: {variant}")
    return apply_r64_g64(x, compute_dtype=torch.float32, output_dtype=torch.float32)


def convert_weight(packed, scales, scale_2, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    b = nvfp4_to_bf16(packed, scales, scale_2)
    return quantize(rotate(b, variant))


def activation_reference(x, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    transformed = fht32_reference(x) if variant == "fht32" else rotate(x, variant)
    return dequantize(*quantize(transformed))
