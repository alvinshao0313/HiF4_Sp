"""Fused combination QDQ and BF16 Tensor Core GEMM with FP32 group scaling."""
import torch
import torch.distributed as dist
from vllm.triton_utils import triton, tl
from ..qwen3_mxfp4.kernels import _h4_stage, _h2_stage, fused_moe_nvfp4_emulation_kernel
from ..qwen3_mxfp4_r16x2.transform import INV_SQRT_2
from ..qwen3_mxfp4_dualscale.kernels import packed_gemm as outer_gemm
from .format import options

R32_NORM = tl.constexpr(INV_SQRT_2)


@triton.jit
def rotate(x, lane, ROTATE: tl.constexpr):
    if ROTATE:
        x = _h4_stage(x, lane, 1)
        x = _h4_stage(x, lane, 4)
        x = _h2_stage(x, lane, 16) * R32_NORM
    return x


@triton.jit
def _mx(blocks, UOS: tl.constexpr):
    maximum = tl.max(tl.abs(blocks), 1)
    bits = maximum.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) - 129
    if UOS:
        exponent += ((bits & 0x7fffff) > 0x680000).to(tl.int32)
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
    value = (value.to(tl.int32, bitcast=True) | (sign << 31)).to(tl.float32, bitcast=True)
    return value * scale[:, None], code | (sign << 3), exponent + 127


@triton.jit
def local_qdq_kernel(X, Y, OUTER, CODES, SCALES, K: tl.constexpr, G: tl.constexpr, DIAG: tl.constexpr, ROTATE: tl.constexpr, OUTER_ENABLED: tl.constexpr, UOS: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    k = group * G + tl.arange(0, G)
    x = tl.load(X + row*K+k, k<K, other=0.).to(tl.float32)
    x = rotate(x, tl.arange(0, G), ROTATE)
    outer = 1.
    if OUTER_ENABLED:
        maximum = tl.max(tl.abs(x), 0)
        outer = tl.where(maximum == 0, 1., tl.div_rn(maximum, 6.))
    normalized = tl.div_rn(x, outer)
    values, codes, scales = _mx(tl.reshape(normalized, (G//32, 32)), UOS)
    tl.store(Y+row*K+k, tl.reshape(values, (G,)), k<K)
    if OUTER_ENABLED:
        tl.store(OUTER+row*tl.cdiv(K,G)+group, outer)
    if DIAG:
        tl.store(CODES+row*K+k, tl.reshape(codes, (G,)), k<K)
        j = group*(G//32)+tl.arange(0,G//32)
        tl.store(SCALES+row*(K//32)+j, scales, j<K//32)


@triton.jit
def local_max_kernel(X, MAXIMUM, K: tl.constexpr, OFFSET: tl.constexpr,
                     GLOBAL_K: tl.constexpr, G: tl.constexpr, ROTATE: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    global_k = group*G+tl.arange(0,G)
    local_k = global_k-OFFSET
    x = tl.load(X+row*K+local_k, (local_k>=0)&(local_k<K)&(global_k<GLOBAL_K), other=0.).to(tl.float32)
    x = rotate(x, tl.arange(0, G), ROTATE)
    tl.store(MAXIMUM+row*tl.cdiv(GLOBAL_K,G)+group, tl.max(tl.abs(x),0))


@triton.jit
def distributed_qdq_kernel(X, MAXIMUM, Y, OUTER, CODES, SCALES,
                           K: tl.constexpr, OFFSET: tl.constexpr, GLOBAL_K: tl.constexpr,
                           G: tl.constexpr, DIAG: tl.constexpr, BLOCK: tl.constexpr, ROTATE: tl.constexpr, UOS: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    k = block*BLOCK+tl.arange(0,BLOCK)
    group = (OFFSET+block*BLOCK)//G
    count: tl.constexpr = (GLOBAL_K+G-1)//G
    maximum = tl.load(MAXIMUM+row*count+group)
    outer = tl.where(maximum == 0, 1., tl.div_rn(maximum,6.))
    x = tl.load(X+row*K+k).to(tl.float32)
    x = rotate(x, tl.arange(0, BLOCK), ROTATE)
    values, codes, scales = _mx(tl.reshape(tl.div_rn(x,outer),(BLOCK//32,32)), UOS)
    tl.store(Y+row*K+k,tl.reshape(values,(BLOCK,)))
    # A separate scale output avoids racing readers of the reduced maxima.
    if block == 0:
        groups = tl.arange(0,triton.next_power_of_2(count))
        maxima = tl.load(MAXIMUM+row*count+groups,groups<count,other=0.)
        tl.store(OUTER+row*count+groups,tl.where(maxima==0,1.,tl.div_rn(maxima,6.)),groups<count)
    if DIAG:
        tl.store(CODES+row*K+k,tl.reshape(codes,(BLOCK,)))
        tl.store(SCALES+row*(K//32)+block*(BLOCK//32)+tl.arange(0,BLOCK//32),scales)



def activation_qdq(x, variant, *, global_k=None, offset=0, process_group=None, diagnostics=False):
    rotation, outer_enabled, uos = options(variant)
    group = 512 if outer_enabled else 64
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1]%32:
        raise ValueError("Expected CUDA BF16 matrix with K aligned to group32")
    x=x.contiguous()
    rows,k=x.shape
    global_k=k if global_k is None else global_k
    if offset%32 or offset<0 or offset+k>global_k:
        raise ValueError("Invalid global-K slice")
    output=torch.empty_like(x)
    outer=torch.empty((rows,triton.cdiv(global_k,group)),device=x.device,dtype=torch.float32) if outer_enabled else None
    codes=torch.empty_like(x,dtype=torch.uint8) if diagnostics else output
    scales=torch.empty((rows,k//32),device=x.device,dtype=torch.uint8) if diagnostics else output
    if rows:
        if not outer_enabled or (global_k==k and offset==0):
            local_qdq_kernel[(rows,triton.cdiv(k,group))](x,output,outer if outer_enabled else output,codes,scales,k,group,diagnostics,rotation,outer_enabled,uos,
                num_warps=4,enable_fp_fusion=False)
        else:
            if not dist.is_initialized() or dist.get_world_size(process_group)!=2:
                raise ValueError("Sharded down activation requires the real TP2 process group")
            maximum=torch.empty_like(outer)
            local_max_kernel[(rows,outer.shape[1])](x,maximum,k,offset,global_k,group,rotation,num_warps=4,enable_fp_fusion=False)
            dist.all_reduce(maximum,op=dist.ReduceOp.MAX,group=process_group)
            block=64 if k%64==0 and offset%64==0 else 32
            distributed_qdq_kernel[(rows,k//block)](x,maximum,output,outer,codes,scales,k,offset,global_k,
                group,diagnostics,block,rotation,uos,num_warps=1,enable_fp_fusion=False)
    return (output,outer,codes,scales) if diagnostics else (output,outer)



def single_gemm(a, weight, scales, ones, output, ids, experts, padded,
                top_k, route_weights, multiply, config):
    """Call the existing generic core, explicitly bypassing its NVFP4 wrapper."""
    n, k = weight.shape[1], a.shape[1]
    em = ids.numel()
    if a.shape[0] < config["BLOCK_SIZE_M"]:
        em = min(em, a.shape[0] * top_k * config["BLOCK_SIZE_M"])
    grid = (triton.cdiv(em, config["BLOCK_SIZE_M"]) * triton.cdiv(n, config["BLOCK_SIZE_N"]),)
    return fused_moe_nvfp4_emulation_kernel[grid](
        a, weight, output, scales, ones, route_weights, ids, experts, padded,
        n, k, em, a.shape[0] * top_k, a.stride(0), a.stride(1),
        weight.stride(0), weight.stride(2), weight.stride(1),
        output.stride(1), output.stride(2), scales.stride(0), scales.stride(2), scales.stride(1),
        block_k_diviable=k % config["BLOCK_SIZE_K"] == 0,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"], BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"], GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=multiply, top_k=top_k, compute_type=tl.bfloat16,
        group_size=32, SCALES_ARE_FP32=True)


def packed_gemm(a, weight, scales, outer_a, outer_w, output, ids, experts, padded, top_k, routes, multiply, config, variant, offset=0):
    if options(variant)[1]:
        return outer_gemm(a, weight, scales, outer_a, outer_w, output, ids, experts, padded, top_k, routes, multiply, config, 512, offset)
    return single_gemm(a, weight, scales, outer_w, output, ids, experts, padded, top_k, routes, multiply, config)
