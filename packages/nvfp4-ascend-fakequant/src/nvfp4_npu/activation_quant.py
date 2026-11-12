"""NVFP4 activation and E4M3 KV fake quantization; outputs remain BF16.

NVFP4 uses checkpoint global scales, dynamic 16-value block amax, E4M3
block-scale rounding, and E2M1 round-to-nearest-even. Zero/underflowed
block scales produce zeros. Reference: the pinned NVIDIA Model Optimizer
sources in artifacts/provenance/modelopt_nvfp4/sources.json.
"""
import torch
import triton
import triton.language as tl


def _validate(x, scale, block):
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % block:
        raise ValueError('Expected a BF16 matrix with aligned columns')
    if scale.dtype != torch.float32 or scale.device != x.device:
        raise ValueError('Scale must be FP32 on the input device')
    if scale.numel() not in (1, x.shape[0]):
        raise ValueError('Expected one global scale or one scale per row')


def nvfp4_reference(x, scale):
    """CPU reference uses PyTorch's native E4M3 conversion."""
    _validate(x, scale, 16)
    assert x.device.type == 'cpu'
    assert torch.isfinite(x).all() and torch.isfinite(scale).all() and (scale > 0).all()
    values = x.float().reshape(x.shape[0], -1, 16)
    global_scale = scale.reshape(-1, 1, 1)
    block_amax = values.abs().amax(-1, keepdim=True)
    block_scale = (block_amax / (6.0 * global_scale)).clamp(max=448).to(torch.float8_e4m3fn).float() * global_scale
    normalized = values.abs() / torch.where(block_scale == 0, 1.0, block_scale)
    bounds = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0])
    code = torch.bucketize(normalized.contiguous(), bounds, right=False)
    code += ((normalized == .75) | (normalized == 1.75) | (normalized == 3.5)).long()
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    rounded = torch.copysign(table[code] * block_scale, values)
    return torch.where(block_scale == 0, 0., rounded).reshape_as(x).to(x.dtype)


def fp8_reference(x, scale):
    _validate(x, scale, 1)
    assert x.device.type == 'cpu'
    assert torch.isfinite(x).all() and torch.isfinite(scale).all() and (scale > 0).all()
    scale = scale.reshape(-1, 1)
    return ((x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)


@triton.jit
def _e4m3_round(value):
    magnitude = tl.minimum(tl.abs(value), 448.0)
    bits = magnitude.to(tl.int32, bitcast=True)
    exponent = tl.maximum(((bits >> 23) & 255) - 127, -6)
    step = ((exponent - 3 + 127) << 23).to(tl.float32, bitcast=True)
    scaled = magnitude / step
    lower = tl.floor(scaled)
    fraction = scaled - lower
    increment = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    rounded = (lower + increment.to(tl.float32)) * step
    return tl.where(value < 0, -rounded, rounded)


@triton.jit
def _e2m1_round(magnitude):
    return tl.where(magnitude <= .25, 0.,
           tl.where(magnitude < .75, .5,
           tl.where(magnitude <= 1.25, 1.,
           tl.where(magnitude < 1.75, 1.5,
           tl.where(magnitude <= 2.5, 2.,
           tl.where(magnitude < 3.5, 3.,
           tl.where(magnitude <= 5., 4., 6.)))))))


@triton.jit
def _nvfp4_kernel(X, S, Y, NB: tl.constexpr, K: tl.constexpr, PER_ROW: tl.constexpr, B: tl.constexpr):
    block = tl.program_id(0) * B + tl.arange(0, B)
    offset = block[:, None] * 16 + tl.arange(0, 16)[None, :]
    value = tl.load(X + offset, mask=block[:, None] < NB, other=0).to(tl.float32)
    row = block // (K // 16)
    scale = tl.load(S + tl.where(PER_ROW, row, 0), mask=block < NB, other=1.)
    amax = tl.max(tl.abs(value), axis=1)
    block_scale = _e4m3_round(amax / (6.0 * scale)) * scale
    normalized = tl.abs(value) / tl.where(block_scale[:, None] == 0, 1., block_scale[:, None])
    result = _e2m1_round(normalized) * block_scale[:, None]
    result = tl.where(value < 0, -result, result)
    result = tl.where(block_scale[:, None] == 0, 0., result)
    tl.store(Y + offset, result, mask=block[:, None] < NB)


@triton.jit
def _fp8_kernel(X, S, Y, N: tl.constexpr, K: tl.constexpr, PER_ROW: tl.constexpr, B: tl.constexpr):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(X + offset, mask=offset < N, other=0).to(tl.float32)
    scale = tl.load(S + tl.where(PER_ROW, offset // K, 0), mask=offset < N, other=1.)
    result = _e4m3_round(value / scale) * scale
    tl.store(Y + offset, result, mask=offset < N)


def nvfp4_fake_quant(x, scale):
    _validate(x, scale, 16)
    if x.device.type == 'cpu':
        return nvfp4_reference(x, scale)
    if x.device.type != 'npu':
        raise ValueError('Only explicit CPU reference and Ascend implementations are supported')
    x = x.contiguous(); scale = scale.contiguous()
    out = torch.empty_like(x)
    if x.numel():
        _nvfp4_kernel[(triton.cdiv(x.numel() // 16, 32),)](x, scale, out, x.numel() // 16, x.shape[1], scale.numel() != 1, 32, enable_fp_fusion=False)
    return out


def fp8_fake_quant(x, scale):
    _validate(x, scale, 1)
    if x.device.type == 'cpu':
        return fp8_reference(x, scale)
    if x.device.type != 'npu':
        raise ValueError('Only explicit CPU reference and Ascend implementations are supported')
    x = x.contiguous(); scale = scale.contiguous()
    out = torch.empty_like(x)
    if x.numel():
        _fp8_kernel[(triton.cdiv(x.numel(), 512),)](x, scale, out, x.numel(), x.shape[1], scale.numel() != 1, 512, enable_fp_fusion=False)
    return out
