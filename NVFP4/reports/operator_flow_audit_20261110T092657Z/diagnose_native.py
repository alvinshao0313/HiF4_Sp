"""Isolate actual native_nvfp4_linear versus fused MoE arithmetic."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from vllm.triton_utils import tl
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import invoke_fused_moe_nvfp4_emulation_kernel
from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils as nv
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import native_nvfp4_linear
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.nvfp4_operator_parity.common import load_raw_linear_parts
from audit import metrics, sha


def main():
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.02, device)
    nv.kE2M1ToFloat_handle.val = nv.kE2M1ToFloat_handle.val.to(device)
    prior = json.loads(Path(__file__).with_name("results.json").read_text())
    snapshot = Path(prior["provenance"]["snapshot"])
    result = {"router": [], "w2": []}
    started = time.time()
    for path in prior["provenance"]["input_sha256"]:
        pack = torch.load(ROOT/path, map_location="cpu", weights_only=False)
        meta = {k:pack[k] for k in ("prompt_key","layer","decode_index")}
        logits = pack["router_logits"].reshape(1,-1).to(device)
        hidden = pack["moe_input"].reshape(1,-1).to(device)
        fused_w, fused_ids, _ = fused_topk(hidden, logits, 8, True)
        python_w, python_ids = torch.topk(F.softmax(logits, dtype=torch.float32, dim=-1), 8, dim=-1)
        python_w /= python_w.sum(-1,keepdim=True)
        python_bf16 = python_w.to(logits.dtype)
        ids_equal = torch.equal(fused_ids,python_ids)
        result["router"].append({**meta,"same_ids_order":ids_equal,"same_ids_set":torch.equal(fused_ids.sort().values,python_ids.sort().values),"vllm_dtype":str(fused_w.dtype),"semantic_dtype":str(python_bf16.dtype),"before_bf16_cast":metrics(fused_w,python_w),"after_bf16_cast":metrics(fused_w,python_bf16)})
        if not ids_equal:
            raise RuntimeError("Different routing order; do not compare unmatched expert slots")
        expert = pack["experts"][0]
        eid = expert["expert_id"]
        slot = (fused_ids[0] == eid).nonzero().item()
        r32 = fused_w[:,slot:slot+1].contiguous()
        rb = python_bf16[:,slot:slot+1].contiguous()
        prefix = f"model.layers.{pack['layer']}.mlp.experts.{eid}.down_proj"
        parts = load_raw_linear_parts(snapshot,prefix)
        b = parts["weight"].to(device).unsqueeze(0)
        s = parts["weight_scale"].to(device).unsqueeze(0)
        gw = torch.tensor([expert["w2_global"]],device=device,dtype=torch.float32)
        ga = torch.tensor(expert["a2_inv"],device=device,dtype=torch.float32)
        x = expert["w2_input"].reshape(1,-1).to(device)
        a = nv.ref_nvfp4_quant_dequant(x,ga,16)
        wd = nv.dequantize_to_dtype(b,s,gw,torch.bfloat16,swizzle=False)
        config_name = nvfp4_moe_quant_config(g1_alphas=gw,g2_alphas=gw,a1_gscale=ga,a2_gscale=ga,w1_scale=s,w2_scale=s).config_name(torch.bfloat16)
        # Use the same tuning lookup shapes as a full TP1 Qwen3 MoE layer.
        config = try_get_optimal_moe_config((128,1536,1024),(128,2048,384),8,config_name,1)
        ids = torch.zeros((1,1),dtype=torch.int32,device=device)
        sorted_ids,expert_ids,padded = moe_align_block_size(ids,config["BLOCK_SIZE_M"],1)
        def packed(route):
            out = torch.empty((1,1,2048),dtype=torch.bfloat16,device=device)
            invoke_fused_moe_nvfp4_emulation_kernel(a,b,out,s,ga,gw,route,sorted_ids,expert_ids,padded,route is not None,1,config,tl.bfloat16)
            return out.reshape(1,-1)
        plain = packed(None)
        weighted32 = packed(r32)
        weighted_b = packed(rb)
        unpacked_fused = torch.empty((1,1,2048),dtype=torch.bfloat16,device=device)
        invoke_fused_moe_triton_kernel(a,wd,unpacked_fused,None,None,r32,sorted_ids,expert_ids,padded,True,1,config,tl.bfloat16,False,False,False,False,False)
        actual_native = native_nvfp4_linear(x,wd[0].float(),ga)
        dense_native = nv.run_nvfp4_emulations(x,ga,b[0],s[0],gw,swizzle=False)
        result["w2"].append({**meta,"expert":eid,"config":config,"route_fp32":float(r32.item()),"route_bf16":float(rb.item()),"native_function_vs_dense_emulation":metrics(dense_native,actual_native),"packed_vs_predecoded_same_fused_gemm":metrics(weighted32,unpacked_fused.reshape(1,-1)),"fused_vs_torch_unweighted_gemm":metrics(plain,actual_native),"route_weight_dtype_only":metrics(weighted32,weighted_b),"route_before_vs_after_bf16_only":metrics(weighted_b,plain*rb),"production_weighted_vs_semantic":metrics(weighted32,actual_native*rb),"weight_tensor_sha256":{k:hashlib.sha256(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in parts.items()}})
    torch.cuda.synchronize()
    result["provenance"] = {"source_result_sha256":sha(Path(__file__).with_name("results.json")),"script_sha256":sha(Path(__file__)),"elapsed_seconds":time.time()-started,"peak_allocated_bytes":torch.cuda.max_memory_allocated(device),"scope":"Six single-expert W2 probes on saved real inputs; full K/N, TP1 tuning, fixed routing; not full-model or TP2/TP4 inference"}
    with Path(__file__).with_name("native_diagnosis.json").open("x") as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
