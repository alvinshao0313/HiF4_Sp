"""The approved floor-scale MXFP4 format, with the isolated R16x2 transform."""
from ..qwen3_mxfp4.format import (
    SCALE_RULE, decode_scales, dequantize, nvfp4_to_bf16, quantize, unpack_codes,
)
from .transform import apply_r16x2, r16x2_reference

FORMAT_VERSION = "qwen3-routed-mxfp4-r16x2-v1"
VARIANTS = ("r16x2",)


def convert_weight(packed, scales, scale_2, variant):
    if variant not in VARIANTS:
        raise ValueError("This isolated converter only supports r16x2")
    bf16 = nvfp4_to_bf16(packed, scales, scale_2)
    return quantize(apply_r16x2(bf16))


def activation_reference(x, variant):
    if variant not in VARIANTS:
        raise ValueError("This isolated reference only supports r16x2")
    return dequantize(*quantize(r16x2_reference(x)))
