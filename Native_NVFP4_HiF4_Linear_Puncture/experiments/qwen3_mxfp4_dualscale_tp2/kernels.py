"""Use the original Triton kernels with a real two-rank MAX collective."""
import torch
import torch.distributed as dist
from vllm.triton_utils import triton
from ..qwen3_mxfp4_dualscale.kernels import (local_qdq_kernel, local_max_kernel, distributed_qdq_kernel, packed_gemm)


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
            if not dist.is_initialized() or dist.get_world_size(process_group)!=2:
                raise ValueError("Sharded down activation requires the real TP2 process group")
            maximum=torch.empty_like(outer)
            local_max_kernel[(rows,outer.shape[1])](x,maximum,k,offset,global_k,group,num_warps=4)
            dist.all_reduce(maximum,op=dist.ReduceOp.MAX,group=process_group)
            block=64 if k%64==0 and offset%64==0 else 32
            distributed_qdq_kernel[(rows,k//block)](x,maximum,output,outer,codes,scales,k,offset,global_k,
                group,diagnostics,block,num_warps=1,enable_fp_fusion=False)
    return (output,outer,codes,scales) if diagnostics else (output,outer)

