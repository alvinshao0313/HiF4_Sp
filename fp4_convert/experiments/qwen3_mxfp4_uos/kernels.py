"""SM80 software MXFP4 QDQ and packed E2M1/BF16 GEMM, without Quark."""
import torch

from vllm.triton_utils import triton, tl
from .format import VARIANT
from ..qwen3_mxfp4.kernels import packed_gemm


@triton.jit
def activation_kernel(x_ptr, out_ptr, codes_ptr, scales_ptr, rotated_ptr,
                      K: tl.constexpr, DIAGNOSTICS: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    offset = row * K + block * BLOCK + lane
    x = tl.load(x_ptr + offset).to(tl.float32)
    blocks = tl.reshape(x, (BLOCK // 32, 32))
    maximum = tl.max(tl.abs(blocks), 1)
    # Exact ceiling threshold: 7.25 = 1.8125 * 2**2, mantissa bits 0x680000.
    # All FP32 subnormals are below the minimum exponent and clamp to -127.
    bits = maximum.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) - 129 + ((bits & 0x7fffff) > 0x680000).to(tl.int32)
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
    block = 64 if x.shape[-1] % 64 == 0 else 32
    if variant != VARIANT or not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % block:
        raise ValueError("Qwen MXFP4 requires CUDA BF16 matrices aligned to the variant's transform block")
    x = x.contiguous()
    output = torch.empty_like(x)
    codes = torch.empty_like(x, dtype=torch.uint8) if diagnostics else output
    scales = torch.empty((x.shape[0], x.shape[1] // 32), device=x.device, dtype=torch.uint8) if diagnostics else output
    rotated = torch.empty_like(x, dtype=torch.float32) if diagnostics else output
    if x.numel():
        activation_kernel[(x.shape[0], x.shape[1] // block)](
            x, output, codes, scales, rotated, x.shape[1], diagnostics, BLOCK=block,
            num_warps=1, enable_fp_fusion=False)
    return (output, codes, scales, rotated) if diagnostics else output

