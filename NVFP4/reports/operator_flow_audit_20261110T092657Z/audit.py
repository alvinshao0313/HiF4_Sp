"""Compare current production operators on saved real Qwen3 activations.

This is a bounded diagnostic, not a full-model accuracy evaluation.
Run from the repository root with hif4 and one explicitly selected CUDA device.
"""

from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from NVFP4.torch_fake import fake_quant_nvfp4_activation
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.nvfp4_operator_parity.common import load_raw_linear_parts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation
from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils as native


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(reference, candidate):
    a, b = reference.detach().float(), candidate.detach().float()
    delta = a - b
    return {
        "numel": a.numel(),
        "different": int((a != b).sum().item()),
        "max_abs": float(delta.abs().max().item()),
        "nmse": float(delta.square().sum().item() / max(a.square().sum().item(), 1e-30)),
    }


def main():
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.02, device)
    native.kE2M1ToFloat_handle.val = native.kE2M1ToFloat_handle.val.to(device)
    source = ROOT / "Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/e0_nvfp4_operator_parity"
    manifest = json.loads((source / "frozen_inputs_manifest.json").read_text())
    paths = []
    for sample in manifest["samples"]:
        for layer in (0, 24, 47):
            entry = min((e for e in sample["layers"] if e["layer"] == layer), key=lambda e: e["decode_index"])
            paths.append(source / entry["path"])
    packs = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    result = {"scope": "six frozen decode packs; no new model generation or accuracy measurement", "qdq": [], "silu": [], "expert_sum": [], "dense": []}
    started = time.time()
    for pack in packs:
        meta = {k: pack[k] for k in ("prompt_key", "layer", "decode_index")}
        activations = [("qkv", pack["qkv_input"], pack["scales"]["qkv_input_inv"]), ("o", pack["o_input"], pack["scales"]["o_input_inv"]), ("moe_input", pack["moe_input"], pack["experts"][0]["a13_inv"])]
        activations += [(f"w2_expert_{e['expert_id']}", e["w2_input"], e["a2_inv"]) for e in pack["experts"]]
        for name, activation, scale_value in activations:
            x = activation.reshape(1, -1).to(device)
            scale = torch.tensor(scale_value, dtype=torch.float32, device=device)
            a = native.ref_nvfp4_quant_dequant(x, scale, block_size=16)
            b = fake_quant_nvfp4_activation(x, scale)
            result["qdq"].append({**meta, "input": name, "scale_inv": scale_value, **metrics(a, b)})
        for expert in pack["experts"]:
            gate, up = [expert[k].reshape(1, -1).to(device) for k in ("gate", "up")]
            fused = torch.empty_like(gate)
            apply_moe_activation(MoEActivation.SILU, fused, torch.cat((gate, up), dim=-1))
            eager = F.silu(gate) * up
            fp32_fused = (F.silu(gate.float()) * up.float()).to(gate.dtype)
            result["silu"].append({**meta, "expert": expert["expert_id"], "bf16_eager": metrics(fused, eager), "fp32_one_round": metrics(fused, fp32_fused)})
        weighted = torch.stack([e["weighted_down"] for e in pack["experts"]]).to(device)
        sequential = torch.zeros_like(weighted[0])
        for expert in sorted(pack["experts"], key=lambda e: e["expert_id"]):
            sequential += expert["weighted_down"].to(device)
        # The current top-8 production moe_sum uses at::sum_out.
        one_sum = weighted.sum(dim=0)
        result["expert_sum"].append({**meta, **metrics(one_sum, sequential)})

    snapshot = Path("/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3")
    pack = packs[0]
    expert = pack["experts"][0]
    dense_cases = [("model.layers.0.self_attn.q_proj", pack["qkv_input"], pack["scales"]["qkv_input_inv"], pack["scales"]["qkv_weight_global"]), (f"model.layers.0.mlp.experts.{expert['expert_id']}.down_proj", expert["w2_input"], expert["a2_inv"], expert["w2_global"])]
    for prefix, activation, input_inv, weight_global in dense_cases:
        parts = load_raw_linear_parts(snapshot, prefix)
        w, s = parts["weight"].to(device), parts["weight_scale"].to(device)
        gw = torch.tensor(weight_global, dtype=torch.float32, device=device)
        ga = torch.tensor(input_inv, dtype=torch.float32, device=device)
        x = activation.reshape(1, -1).to(device)
        dense_weight = native.dequantize_to_dtype(w, s, gw, torch.bfloat16, swizzle=False)
        packed_y = native.run_nvfp4_emulations(x, ga, w, s, gw, swizzle=False)
        predecoded_y = F.linear(native.ref_nvfp4_quant_dequant(x, ga, 16), dense_weight)
        fake_y = F.linear(fake_quant_nvfp4_activation(x, ga), dense_weight)
        result["dense"].append({"prefix": prefix, "weight_shape": list(dense_weight.shape), "predecoded_same_qdq": metrics(packed_y, predecoded_y), "predecoded_fake_qdq": metrics(packed_y, fake_y), "raw_tensor_hashes": {k: hashlib.sha256(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() for k, v in parts.items()}})
        del w, s, dense_weight, packed_y, predecoded_y, fake_y, parts
    torch.cuda.synchronize()
    code_paths = ["NVFP4/torch_fake.py", "NVFP4/triton_fake.py", "NVFP4/dequantize.py", "3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py", "3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py", "3rdparty/vllm/vllm/model_executor/layers/linear.py", "Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/moe_semantic_hif4.py"]
    result["provenance"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "snapshot": str(snapshot), "seed": "none; fixed saved real inputs", "elapsed_seconds": time.time()-started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths}, "source_sha256": {p: sha(ROOT / p) for p in code_paths}, "script_sha256": sha(Path(__file__)), "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, "tf32": torch.backends.cuda.matmul.allow_tf32}
    out = Path(__file__).with_name("results.json")
    with out.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"qdq_cases": len(result["qdq"]), "qdq_different": sum(r["different"] for r in result["qdq"]), "qdq_numel": sum(r["numel"] for r in result["qdq"]), "qdq_max_abs": max(r["max_abs"] for r in result["qdq"]), "silu_bf16_different": sum(r["bf16_eager"]["different"] for r in result["silu"]), "silu_fp32_different": sum(r["fp32_one_round"]["different"] for r in result["silu"]), "sum_max_abs": max(r["max_abs"] for r in result["expert_sum"]), "dense": result["dense"], "peak_allocated_bytes": result["provenance"]["peak_allocated_bytes"], "elapsed_seconds": result["provenance"]["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
