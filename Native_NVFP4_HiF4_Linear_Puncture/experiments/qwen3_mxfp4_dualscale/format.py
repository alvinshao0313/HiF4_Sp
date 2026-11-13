"""FP32 outer absmax/6 scales and unchanged group32 floor-rule MXFP4."""
import torch
import torch.nn.functional as F

from ..qwen3_mxfp4.format import (VALUES, MIDPOINTS, quantize as mx_quantize,
    dequantize as mx_dequantize, unpack_codes, decode_scales, nvfp4_to_bf16)

GROUPS = (256, 512)
VARIANTS = ("dualscale_g256", "dualscale_g512")
MARKER = "qwen3_routed_mxfp4_dualscale"
FORMAT_VERSION = "qwen3-routed-mxfp4-dualscale-v1"
SCALE_RULE = "outer=FP32(absmax/6),zero=1; inner=clamp(floor(log2(amax))-2,-127,127),zero=0; E2M1 RNE"


def group_size(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown dual-scale variant: {variant}")
    return int(variant.rsplit("g", 1)[1])


def outer_scales(x, group):
    if group not in GROUPS or x.ndim < 1 or x.shape[-1] % 32 or not torch.isfinite(x).all():
        raise ValueError("Expected finite group32-aligned input and outer group256/512")
    padding = (-x.shape[-1]) % group
    values = F.pad(x.float(), (0, padding)) if padding else x.float()
    maximum = values.reshape(*x.shape[:-1], -1, group).abs().amax(-1)
    return torch.where(maximum == 0, 1., maximum / 6.)


def quantize(x, group):
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Expected BF16/FP32 quantizer input")
    outer = outer_scales(x, group)
    # Normalize in FP32; there is no BF16 rounding before the inner quantizer.
    normalized = x.float() / outer.repeat_interleave(group, -1)[..., :x.shape[-1]]
    packed, scales = mx_quantize(normalized)
    return packed, scales, outer.contiguous()


def convert_weight(packed, scales, scale_2, variant):
    return quantize(nvfp4_to_bf16(packed, scales, scale_2), group_size(variant))


def normalized_reference(x, group):
    packed, scales, outer = quantize(x, group)
    return mx_dequantize(packed, scales), outer


def reconstruction(packed, scales, outer, group):
    normalized = mx_dequantize(packed, scales).float()
    return normalized * outer.repeat_interleave(group, -1)[..., :normalized.shape[-1]]
