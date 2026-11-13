"""Fused FP32 R16x2 and floor-scale group32 QDQ; shared packed BF16 GEMM."""
import torch
from vllm.triton_utils import triton, tl
from ..qwen3_mxfp4.kernels import _h4_stage, _h2_stage, packed_gemm
from .transform import INV_SQRT_2

R16X2_NORMALIZATION = tl.constexpr(INV_SQRT_2)


@triton.jit
def activation_kernel(x_ptr, out_ptr, codes_ptr, scales_ptr, rotated_ptr,
                      K: tl.constexpr, DIAGNOSTICS: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    BLOCK: tl.constexpr = 32
    lane = tl.arange(0, BLOCK)
    offset = row * K + block * BLOCK + lane
    x = tl.load(x_ptr + offset).to(tl.float32)
    x = _h4_stage(x, lane, 1)
    x = _h4_stage(x, lane, 4)
    x = _h2_stage(x, lane, 16) * R16X2_NORMALIZATION
    blocks = tl.reshape(x, (BLOCK // 32, 32))
    maximum = tl.max(tl.abs(blocks), 1)
    # The FP32 exponent gives floor(log2()) exactly for normal values; every
    # subnormal falls below the required clamp. This avoids approximate log2.
    exponent = ((maximum.to(tl.int32, bitcast=True) >> 23) & 255) - 129
    exponent = tl.where(maximum == 0, 0, tl.minimum(127, tl.maximum(-127, exponent)))
    scale_bits = tl.where(exponent == -127, 0x00400000, (exponent + 127) << 23)
    scale = scale_bits.to(tl.float32, bitcast=True)
    normalized = tl.div_rn(blocks, scale[:, None])
    a = tl.abs(normalized)
    code = ((a > .25).to(tl.int32) + (a >= .75).to(tl.int32)
            + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
            + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32) + (a > 5.).to(tl.int32))
    value = tl.where(code <= 4, code.to(tl.float32) * .5,
                     tl.where(code == 5, 3., tl.where(code == 6, 4., 6.)))
    sign = (normalized.to(tl.int32, bitcast=True) >> 31) & 1
    # Preserve signed zero as well as the low-nibble-first encoding contract.
    value = (value.to(tl.int32, bitcast=True) | (sign << 31)).to(tl.float32, bitcast=True)
    tl.store(out_ptr + offset, tl.reshape(value * scale[:, None], (BLOCK,)))
    if DIAGNOSTICS:
        tl.store(codes_ptr + offset, tl.reshape(code | (sign << 3), (BLOCK,)))
        tl.store(scales_ptr + row * (K // 32) + block * (BLOCK // 32) + tl.arange(0, BLOCK // 32), exponent + 127)
        tl.store(rotated_ptr + offset, x)


def activation_qdq(x, variant, diagnostics=False):
    block = 32
    if variant != "r16x2" or not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % block:
        raise ValueError("Qwen MXFP4 requires CUDA BF16 matrices aligned to the variant's transform block")
    x = x.contiguous()
    output = torch.empty_like(x)
    codes = torch.empty_like(x, dtype=torch.uint8) if diagnostics else output
    scales = torch.empty((x.shape[0], x.shape[1] // 32), device=x.device, dtype=torch.uint8) if diagnostics else output
    rotated = torch.empty_like(x, dtype=torch.float32) if diagnostics else output
    if x.numel():
        activation_kernel[(x.shape[0], x.shape[1] // block)](
            x, output, codes, scales, rotated, x.shape[1], diagnostics,
            num_warps=1, enable_fp_fusion=False)
    return (output, codes, scales, rotated) if diagnostics else output

