"""CPU format contract shared by conversion and the independent GPU reference."""
from __future__ import annotations

import torch


VARIANT = "direct_uos_c7p25"
VARIANTS = (VARIANT,)
ALGORITHM = "UOS c7.25"
C = 7.25
MARKER = "qwen3_routed_mxfp4_uos"

VALUES = (0., .5, 1., 1.5, 2., 3., 4., 6.)
MIDPOINTS = (.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)
SCALE_RULE = "clamp(ceil(log2(amax/7.25)),-127,127); zero=0; E2M1 RNE"
FORMAT_VERSION = "qwen3-routed-mxfp4-uos-c7p25-v1"


from ..qwen3_mxfp4.format import unpack_codes, unpack, decode_scales, dequantize, nvfp4_to_bf16


def quantize(x):
    if x.dtype not in (torch.bfloat16, torch.float32) or x.ndim < 1 or x.shape[-1] % 32:
        raise ValueError("Expected BF16/FP32 input with K divisible by 32")
    if not torch.isfinite(x).all():
        raise ValueError("Cannot quantize nonfinite input")
    blocks = x.float().reshape(*x.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(-1)
    # 7.25 = (29/32)*2**3. The strict comparison preserves ceiling
    # at exact thresholds; no approximate logarithm enters exponent selection.
    mantissa, binary_exponent = torch.frexp(maximum)
    raw = binary_exponent - 3 + (mantissa > 29/32).int()
    exponent = torch.where(maximum == 0, 0, raw).clamp(-127, 127)
    scales = (exponent + 127).to(torch.uint8)
    normalized = blocks / decode_scales(scales).unsqueeze(-1)
    magnitude = normalized.abs().contiguous()
    midpoints = torch.tensor(MIDPOINTS, device=x.device)
    code = torch.bucketize(magnitude, midpoints)
    tie = magnitude == midpoints[code.clamp_max(6)]
    code += (tie & (code % 2 == 1)).long()
    code = (code.to(torch.uint8) | (torch.signbit(normalized).to(torch.uint8) << 3)).reshape_as(x)
    return (code[..., ::2] | (code[..., 1::2] << 4)).contiguous(), scales.contiguous()



def convert_weight(packed, scales, scale_2, variant=VARIANT):
    if variant != VARIANT:
        raise ValueError(f"Unknown UOS variant: {variant}")
    return quantize(nvfp4_to_bf16(packed, scales, scale_2))


def activation_reference(x, variant=VARIANT):
    if variant != VARIANT:
        raise ValueError(f"Unknown UOS variant: {variant}")
    return dequantize(*quantize(x))
