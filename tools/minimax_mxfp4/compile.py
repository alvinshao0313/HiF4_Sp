"""Compile the actual MXFP4 kernel specializations for SM80 without a GPU."""
import argparse
import hashlib
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from tools.minimax_m3_eval.download import save
from .runtime import _qdq_kernel, fused_moe_nvfp4_emulation_kernel


def compile_kernels(output):
    records=[]
    target=GPUTarget('cuda',80,32)
    for k in (6144,1536):
        compiled=triton.compile(ASTSource(_qdq_kernel,{'x_ptr':'*bf16','out_ptr':'*bf16'},
            constexprs={'K':k,'GROUPS':k//32}),target=target,options={'num_warps':1})
        records.append({'kernel':'mxfp4_activation_qdq','K':k,'status':'passed',
                        'cubin_sha256':hashlib.sha256(compiled.asm['cubin']).hexdigest()})
    fn=fused_moe_nvfp4_emulation_kernel
    from vllm.triton_utils import tl
    for multiply,topk in ((False,4),(True,1)):
        constants={'block_k_diviable':True,'BLOCK_SIZE_M':16,'BLOCK_SIZE_N':64,
            'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8,'MUL_ROUTED_WEIGHT':multiply,
            'top_k':topk,'compute_type':tl.bfloat16,'group_size':32,'SCALES_ARE_FP32':True}
        signature={name:('*bf16' if name in ('a_ptr','c_ptr') else '*u8' if name=='b_ptr'
            else '*fp32' if name in ('b_scale_ptr','w_global_scale_ptr','topk_weights_ptr')
            else '*i32' if name.endswith('_ptr') else 'i32') for name in fn.arg_names if name not in constants}
        compiled=triton.compile(ASTSource(fn,signature,constexprs=constants),target=target,
            options={'num_warps':4})
        records.append({'kernel':'packed_E2M1_GEMM_group32','routing_weight':multiply,
            'status':'passed','cubin_sha256':hashlib.sha256(compiled.asm['cubin']).hexdigest()})
    save(Path(output),{'status':'passed','target':'cuda:sm80','gpu_execution':False,'kernels':records})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    compile_kernels(parser.parse_args().output)
