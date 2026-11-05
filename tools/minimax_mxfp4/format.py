"""E2M1 values and OCP MXFP4 group-32 E8M0 scales, without optional packages."""
import torch

VALUES = (0., .5, 1., 1.5, 2., 3., 4., 6.)
MIDPOINTS = (.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)
BLOCK_SIZE = 32
SCALE_RULE = 'floor(log2(amax))-2; RNE E2M1; zero-block exponent=0'


def unpack(packed):
    if packed.dtype != torch.uint8:
        raise ValueError('Packed E2M1 values must be uint8')
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    table = torch.tensor(VALUES, device=packed.device)
    return table[(codes & 7).long()] * torch.where(codes >= 8, -1., 1.)


def decode_scales(scales):
    if scales.dtype != torch.uint8 or torch.any(scales == 255):
        raise ValueError('Expected finite E8M0 scale bytes, with no reserved 255')
    return torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.int() - 127)


def quantize(x):
    if x.dtype != torch.bfloat16 or x.shape[-1] % BLOCK_SIZE:
        raise ValueError('MXFP4 conversion requires BF16 input and K divisible by 32')
    if not torch.isfinite(x).all():
        raise ValueError('Cannot quantize nonfinite values')
    blocks = x.float().reshape(*x.shape[:-1], -1, BLOCK_SIZE)
    maximum = blocks.abs().amax(-1)
    exponent = torch.floor(torch.log2(maximum.clamp_min(2. ** -125))) - 2
    exponent = torch.where(maximum == 0, 0., exponent).clamp(-127, 127).int()
    scales = (exponent + 127).to(torch.uint8)
    normalized = blocks / torch.ldexp(torch.ones_like(maximum), exponent).unsqueeze(-1)
    magnitude = normalized.abs().contiguous()
    midpoints = torch.tensor(MIDPOINTS, device=x.device)
    code = torch.bucketize(magnitude, midpoints)
    # An exact midpoint belongs to the representable code with an even LSB.
    tie = magnitude == midpoints[code.clamp_max(6)]
    code += (tie & (code % 2 == 1)).long()
    code = code.to(torch.uint8) | (torch.signbit(normalized).to(torch.uint8) << 3)
    code = code.reshape_as(x)
    packed = code[..., ::2] | (code[..., 1::2] << 4)
    return packed.contiguous(), scales.contiguous()


def dequantize(packed, scales):
    values = unpack(packed)
    if values.shape[-1] % 32 or scales.shape != (*values.shape[:-1], values.shape[-1] // 32):
        raise ValueError('Packed MXFP4 value/scale shapes disagree')
    return (values.reshape(*scales.shape, 32) * decode_scales(scales).unsqueeze(-1)).reshape_as(values).bfloat16()


def nvfp4_to_bf16(packed, scales, multiplier):
    if scales.dtype != torch.float8_e4m3fn or multiplier.numel() != 1:
        raise ValueError('Expected NVFP4 E4M3 block scales and one global multiplier')
    if not torch.isfinite(multiplier).all() or not torch.all(multiplier > 0):
        raise ValueError('Invalid NVFP4 global multiplier')
    values = unpack(packed)
    if scales.shape != (*values.shape[:-1], values.shape[-1] // 16):
        raise ValueError('Packed NVFP4 value/scale shapes disagree')
    effective = scales.float() * multiplier.float().reciprocal()
    # Match the existing packed kernel, including its explicit BF16 rounding.
    return (values.reshape(*scales.shape, 16) * effective.unsqueeze(-1)).reshape_as(values).bfloat16()
