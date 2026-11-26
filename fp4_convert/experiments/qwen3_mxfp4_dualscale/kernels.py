"""Fused dual-scale QDQ and BF16 Tensor Core GEMM with FP32 group scaling."""
import torch
import torch.distributed as dist
from vllm.triton_utils import triton, tl
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import _e2m1_inline


@triton.jit
def _mx(blocks):
    maximum = tl.max(tl.abs(blocks), 1)
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
    value = (value.to(tl.int32, bitcast=True) | (sign << 31)).to(tl.float32, bitcast=True)
    return value * scale[:, None], code | (sign << 3), exponent + 127


@triton.jit
def local_qdq_kernel(X, Y, OUTER, CODES, SCALES, K: tl.constexpr, G: tl.constexpr, DIAG: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    k = group * G + tl.arange(0, G)
    x = tl.load(X + row*K+k, k<K, other=0.).to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    outer = tl.where(maximum == 0, 1., tl.div_rn(maximum, 6.))
    normalized = tl.div_rn(x, outer)
    values, codes, scales = _mx(tl.reshape(normalized, (G//32, 32)))
    tl.store(Y+row*K+k, tl.reshape(values, (G,)), k<K)
    tl.store(OUTER+row*tl.cdiv(K,G)+group, outer)
    if DIAG:
        tl.store(CODES+row*K+k, tl.reshape(codes, (G,)), k<K)
        j = group*(G//32)+tl.arange(0,G//32)
        tl.store(SCALES+row*(K//32)+j, scales, j<K//32)


@triton.jit
def local_max_kernel(X, MAXIMUM, K: tl.constexpr, OFFSET: tl.constexpr,
                     GLOBAL_K: tl.constexpr, G: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    global_k = group*G+tl.arange(0,G)
    local_k = global_k-OFFSET
    x = tl.load(X+row*K+local_k, (local_k>=0)&(local_k<K)&(global_k<GLOBAL_K), other=0.).to(tl.float32)
    tl.store(MAXIMUM+row*tl.cdiv(GLOBAL_K,G)+group, tl.max(tl.abs(x),0))


@triton.jit
def distributed_qdq_kernel(X, MAXIMUM, Y, OUTER, CODES, SCALES,
                           K: tl.constexpr, OFFSET: tl.constexpr, GLOBAL_K: tl.constexpr,
                           G: tl.constexpr, DIAG: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    k = block*BLOCK+tl.arange(0,BLOCK)
    group = (OFFSET+block*BLOCK)//G
    count: tl.constexpr = (GLOBAL_K+G-1)//G
    maximum = tl.load(MAXIMUM+row*count+group)
    outer = tl.where(maximum == 0, 1., tl.div_rn(maximum,6.))
    x = tl.load(X+row*K+k).to(tl.float32)
    values, codes, scales = _mx(tl.reshape(tl.div_rn(x,outer),(BLOCK//32,32)))
    tl.store(Y+row*K+k,tl.reshape(values,(BLOCK,)))
    # A separate scale output avoids racing readers of the reduced maxima.
    if block == 0:
        groups = tl.arange(0,triton.next_power_of_2(count))
        maxima = tl.load(MAXIMUM+row*count+groups,groups<count,other=0.)
        tl.store(OUTER+row*count+groups,tl.where(maxima==0,1.,tl.div_rn(maxima,6.)),groups<count)
    if DIAG:
        tl.store(CODES+row*K+k,tl.reshape(codes,(BLOCK,)))
        tl.store(SCALES+row*(K//32)+block*(BLOCK//32)+tl.arange(0,BLOCK//32),scales)


def activation_qdq(x, group, *, global_k=None, offset=0, process_group=None, diagnostics=False):
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1]%32 or group not in (256,512):
        raise ValueError("Expected CUDA BF16 matrix, K/group32, outer group256/512")
    x=x.contiguous()
    rows,k=x.shape
    global_k=k if global_k is None else global_k
    if offset%32 or offset<0 or offset+k>global_k:
        raise ValueError("Invalid global-K slice")
    output=torch.empty_like(x)
    outer=torch.empty((rows,triton.cdiv(global_k,group)),device=x.device,dtype=torch.float32)
    codes=torch.empty_like(x,dtype=torch.uint8) if diagnostics else output
    scales=torch.empty((rows,k//32),device=x.device,dtype=torch.uint8) if diagnostics else output
    if rows:
        if global_k==k and offset==0:
            local_qdq_kernel[(rows,triton.cdiv(k,group))](x,output,outer,codes,scales,k,group,diagnostics,
                num_warps=4,enable_fp_fusion=False)
        else:
            if not dist.is_initialized() or dist.get_world_size(process_group)!=4:
                raise ValueError("Sharded down activation requires the real TP4 process group")
            maximum=torch.empty_like(outer)
            local_max_kernel[(rows,outer.shape[1])](x,maximum,k,offset,global_k,group,num_warps=4)
            dist.all_reduce(maximum,op=dist.ReduceOp.MAX,group=process_group)
            block=64 if k%64==0 and offset%64==0 else 32
            distributed_qdq_kernel[(rows,k//block)](x,maximum,output,outer,codes,scales,k,offset,global_k,
                group,diagnostics,block,num_warps=1,enable_fp_fusion=False)
    return (output,outer,codes,scales) if diagnostics else (output,outer)


@triton.jit
def gemm_kernel(A,B,S,TA,TW,C,ROUTES,IDS,EXPERTS,PADDED,
                N:tl.constexpr,K:tl.constexpr,NG:tl.constexpr,G:tl.constexpr,OFFSET:tl.constexpr,
                EM,VALID,TOPK:tl.constexpr,MULTIPLY:tl.constexpr,
                BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GM:tl.constexpr):
    pid=tl.program_id(0)
    nm=tl.cdiv(EM,BM);nn=tl.cdiv(N,BN)
    group=pid//(GM*nn);first=group*GM;size=tl.minimum(nm-first,GM)
    pm=first+(pid%(GM*nn))%size;pn=(pid%(GM*nn))//size
    if pm*BM>=tl.load(PADDED):
        return
    tokens=tl.load(IDS+pm*BM+tl.arange(0,BM)).to(tl.int64)
    mask=tokens<VALID
    expert=tl.load(EXPERTS+pm)
    columns=pn*BN+tl.arange(0,BN)
    # EP is disabled; each scheduled expert must be locally available.
    columns_wrapped=columns%N
    total=tl.zeros((BM,BN),tl.float32)
    for outer_group in range(OFFSET//G,tl.cdiv(OFFSET+K,G)):
        begin=tl.maximum(outer_group*G-OFFSET,0)
        end=tl.minimum((outer_group+1)*G-OFFSET,K)
        partial=tl.zeros((BM,BN),tl.float32)
        for tile in range(tl.cdiv(end-begin,BK)):
            ks=begin+tile*BK+tl.arange(0,BK)
            kp=(begin+tile*BK)//2+tl.arange(0,BK//2)
            a=tl.load(A+(tokens[:,None]//TOPK)*K+ks[None,:],mask[:,None]&(ks[None,:]<end),other=0.).to(tl.bfloat16)
            raw=tl.load(B+expert*(N*(K//2))+columns_wrapped[:,None]*(K//2)+kp[None,:],kp[None,:]<end//2,other=0)
            scale=tl.load(S+expert*(N*(K//32))+columns_wrapped[:,None]*(K//32)+kp[None,:]//16,kp[None,:]<end//2,other=0.)
            low=_e2m1_inline(raw&15)*scale
            high=_e2m1_inline(raw>>4)*scale
            b=tl.trans(tl.interleave(low,high)).to(tl.bfloat16)
            partial=tl.dot(a,b,partial)
        ta=tl.load(TA+(tokens//TOPK)*NG+outer_group,mask,other=0.)
        tw=tl.load(TW+expert*(N*NG)+columns_wrapped*NG+outer_group)
        # FP32 scalar scaling follows each BF16 MMA group; no FP32/TF32 dot.
        total=total+partial*(ta[:,None]*tw[None,:])
    if MULTIPLY:
        total=total*tl.load(ROUTES+tokens,mask,other=0.)[:,None]
    tl.store(C+tokens[:,None]*N+columns[None,:],total.to(tl.bfloat16),mask[:,None]&(columns[None,:]<N))


def packed_gemm(a,weight,scales,outer_a,outer_w,output,ids,experts,padded,
                top_k,route_weights,multiply,config,group,offset=0):
    n,k=weight.shape[1],a.shape[1]
    em=ids.numel()
    if a.shape[0]<config["BLOCK_SIZE_M"]:
        em=min(em,a.shape[0]*top_k*config["BLOCK_SIZE_M"])
    grid=(triton.cdiv(em,config["BLOCK_SIZE_M"])*triton.cdiv(n,config["BLOCK_SIZE_N"]),)
    return gemm_kernel[grid](a,weight,scales,outer_a,outer_w,output,route_weights,ids,experts,padded,
        n,k,outer_a.shape[1],group,offset,em,a.shape[0]*top_k,top_k,multiply,
        config["BLOCK_SIZE_M"],config["BLOCK_SIZE_N"],64,config["GROUP_SIZE_M"],enable_fp_fusion=False)
