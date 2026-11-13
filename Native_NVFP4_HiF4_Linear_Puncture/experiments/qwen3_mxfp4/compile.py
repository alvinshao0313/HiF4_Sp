"""Compile actual SM80 specializations without allocating a CUDA context."""
import argparse
import hashlib

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from vllm.triton_utils import tl

from .artifacts import save
from .kernels import activation_kernel, fused_moe_nvfp4_emulation_kernel


def compile_kernels(output):
    records = []
    target = GPUTarget("cuda", 80, 32)
    for k in (2048, 192):
        for rotate in (False, True):
            for diagnostics in (False, True):
                signature = {"x_ptr": "*bf16", "out_ptr": "*bf16", "codes_ptr": "*u8",
                             "scales_ptr": "*u8", "rotated_ptr": "*fp32"}
                compiled = triton.compile(ASTSource(activation_kernel, signature,
                    constexprs={"K": k, "ROTATE": rotate, "DIAGNOSTICS": diagnostics}),
                    target=target, options={"num_warps": 1, "enable_fp_fusion": False})
                records.append({"kernel": "activation", "K": k, "rotate": rotate, "diagnostics": diagnostics,
                                "cubin_sha256": hashlib.sha256(compiled.asm["cubin"]).hexdigest()})
    fn = fused_moe_nvfp4_emulation_kernel
    for multiply, topk in ((False, 8), (True, 1)):
        constants = {"block_k_diviable": True, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
                     "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "MUL_ROUTED_WEIGHT": multiply,
                     "top_k": topk, "compute_type": tl.bfloat16, "group_size": 32, "SCALES_ARE_FP32": True}
        signature = {name: ("*bf16" if name in ("a_ptr", "c_ptr") else "*u8" if name == "b_ptr"
                     else "*fp32" if name in ("b_scale_ptr", "w_global_scale_ptr", "topk_weights_ptr")
                     else "*i32" if name.endswith("_ptr") else "i32") for name in fn.arg_names if name not in constants}
        compiled = triton.compile(ASTSource(fn, signature, constexprs=constants), target=target, options={"num_warps": 4})
        records.append({"kernel": "packed_gemm", "routing_weight": multiply,
                        "cubin_sha256": hashlib.sha256(compiled.asm["cubin"]).hexdigest()})
    save(output, {"status": "passed", "target": "cuda:sm80", "gpu_execution": False, "kernels": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    compile_kernels(parser.parse_args().output)
