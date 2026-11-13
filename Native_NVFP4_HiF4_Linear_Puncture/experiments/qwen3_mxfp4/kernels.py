"""SM80 software MXFP4 QDQ and packed E2M1/BF16 GEMM, without Quark."""
import torch

from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import fused_moe_nvfp4_emulation_kernel
from vllm.triton_utils import triton, tl
from .fht32 import INV_SQRT_32

FHT32_NORMALIZATION = tl.constexpr(INV_SQRT_32)


@triton.jit
def _h4_stage(x, lane, STRIDE: tl.constexpr):
    digit = (lane // STRIDE) % 4
    base = (lane // (4 * STRIDE)) * (4 * STRIDE) + lane % STRIDE
    a = tl.gather(x, base, 0)
    b = tl.gather(x, base + STRIDE, 0)
    c = tl.gather(x, base + 2 * STRIDE, 0)
    d = tl.gather(x, base + 3 * STRIDE, 0)
    return tl.where(digit == 0, a+b+c-d,
                    tl.where(digit == 1, a+b-c+d,
                             tl.where(digit == 2, a-b+c+d, -a+b+c+d))) * .5


@triton.jit
def _h2_stage(x, lane, STRIDE: tl.constexpr):
    peer = tl.gather(x, lane ^ STRIDE, 0)
    return tl.where((lane & STRIDE) == 0, x + peer, peer - x)


@triton.jit
def activation_kernel(x_ptr, out_ptr, codes_ptr, scales_ptr, rotated_ptr,
                      K: tl.constexpr, ROTATE: tl.constexpr, DIAGNOSTICS: tl.constexpr,
                      FHT32: tl.constexpr = False):
    row, block = tl.program_id(0), tl.program_id(1)
    BLOCK: tl.constexpr = 32 if FHT32 else 64
    lane = tl.arange(0, BLOCK)
    offset = row * K + block * BLOCK + lane
    x = tl.load(x_ptr + offset).to(tl.float32)
    if ROTATE:
        x = _h4_stage(x, lane, 1)
        x = _h4_stage(x, lane, 4)
        x = _h4_stage(x, lane, 16)
    elif FHT32:
        x = _h2_stage(x, lane, 1)
        x = _h2_stage(x, lane, 2)
        x = _h2_stage(x, lane, 4)
        x = _h2_stage(x, lane, 8)
        x = _h2_stage(x, lane, 16)
        x = x * FHT32_NORMALIZATION
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
    block = 32 if variant == "fht32" else 64
    if variant not in ("direct", "r64", "fht32") or not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % block:
        raise ValueError("Qwen MXFP4 requires CUDA BF16 matrices aligned to the variant's transform block")
    x = x.contiguous()
    output = torch.empty_like(x)
    codes = torch.empty_like(x, dtype=torch.uint8) if diagnostics else output
    scales = torch.empty((x.shape[0], x.shape[1] // 32), device=x.device, dtype=torch.uint8) if diagnostics else output
    rotated = torch.empty_like(x, dtype=torch.float32) if diagnostics else output
    if x.numel():
        activation_kernel[(x.shape[0], x.shape[1] // block)](
            x, output, codes, scales, rotated, x.shape[1], variant == "r64", diagnostics,
            FHT32=variant == "fht32",
            num_warps=1, enable_fp_fusion=False)
    return (output, codes, scales, rotated) if diagnostics else output


def packed_gemm(a, weight, scales, ones, output, ids, experts, padded,
                top_k, route_weights, multiply, config):
    """Call the existing generic core, explicitly bypassing its NVFP4 wrapper."""
    n, k = weight.shape[1], a.shape[1]
    em = ids.numel()
    if a.shape[0] < config["BLOCK_SIZE_M"]:
        em = min(em, a.shape[0] * top_k * config["BLOCK_SIZE_M"])
    grid = (triton.cdiv(em, config["BLOCK_SIZE_M"]) * triton.cdiv(n, config["BLOCK_SIZE_N"]),)
    fused_moe_nvfp4_emulation_kernel[grid](
        a, weight, output, scales, ones, route_weights, ids, experts, padded,
        n, k, em, a.shape[0] * top_k, a.stride(0), a.stride(1),
        weight.stride(0), weight.stride(2), weight.stride(1),
        output.stride(1), output.stride(2), scales.stride(0), scales.stride(2), scales.stride(1),
        block_k_diviable=k % config["BLOCK_SIZE_K"] == 0,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"], BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"], GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=multiply, top_k=top_k, compute_type=tl.bfloat16,
        group_size=32, SCALES_ARE_FP32=True)
