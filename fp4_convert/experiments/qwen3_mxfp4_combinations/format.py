"""Rotate in FP32, optionally normalize G512, then quantize group32 once."""
import torch

from ..qwen3_mxfp4.format import (quantize as floor_quantize, dequantize,
    decode_scales, unpack_codes, nvfp4_to_bf16)
from ..qwen3_mxfp4_uos.format import quantize as uos_quantize
from ..qwen3_mxfp4_dualscale.format import outer_scales
from ..qwen3_mxfp4_r16x2.transform import apply_r16x2, r16x2_matrix

VARIANTS = ("r16x2_uos_c7p25", "dualscale_g512_uos_c7p25",
            "r16x2_dualscale_g512", "r16x2_dualscale_g512_uos_c7p25")
MARKER = "qwen3_routed_mxfp4_combinations"
FORMAT_VERSION = "qwen3-routed-mxfp4-combinations-v1"


def options(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown combination: {variant}")
    return variant.startswith("r16x2"), "g512" in variant, "uos" in variant


def metadata(variant):
    rotate, outer, uos = options(variant)
    return {"format": FORMAT_VERSION, "variant": variant, "group_size": 32,
            "rotation": "r16x2_fp32" if rotate else "none", "outer_group_size": 512 if outer else None,
            "outer_scale": "FP32(absmax/6); zero=1" if outer else None,
            "inner_scale": "ceil(log2(amax/7.25))" if uos else "floor(log2(amax))-2",
            "exponent_clamp": [-127, 127], "zero_scale": 1, "rounding": "E2M1 RNE; saturate +/-6",
            "order": "NVFP4->BF16(weight only)->FP32 rotation->FP32 outer normalization->MXFP4",
            "gemm": "BF16 operands; FP32 accumulation; outer scales on group partial sums"}


def quantize(x, variant):
    rotate, outer, uos = options(variant)
    values = apply_r16x2(x) if rotate else x.float()
    scales_fp32 = outer_scales(values, 512) if outer else None
    if outer:
        values = values / scales_fp32.repeat_interleave(512, -1)[..., :values.shape[-1]]
    packed, scales = (uos_quantize if uos else floor_quantize)(values)
    return packed, scales, scales_fp32


def normalized_reference(x, variant):
    packed, scales, outer = quantize(x, variant)
    return dequantize(packed, scales), outer


def convert_weight(packed, scales, scale_2, variant):
    return quantize(nvfp4_to_bf16(packed, scales, scale_2), variant)
