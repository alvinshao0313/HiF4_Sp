"""Locate QDQ mismatches from audit.py without changing either operator."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from NVFP4.torch_fake import _fake_quant_nvfp4_activation_torch, cast_to_fp8_e4m3fn, cast_to_fp4_e2m1, fake_quant_nvfp4_activation
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant, ref_nvfp4_quant_dequant, get_reciprocal


def main():
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.01, device)
    source = ROOT / "Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/e0_nvfp4_operator_parity"
    audit = json.loads(Path(__file__).with_name("results.json").read_text())
    manifest = json.loads((source / "frozen_inputs_manifest.json").read_text())
    examples, rows = [], []
    for row in audit["qdq"]:
        if not row["different"]:
            continue
        sample = next(s for s in manifest["samples"] if s["prompt_key"] == row["prompt_key"])
        entry = next(e for e in sample["layers"] if e["layer"] == row["layer"] and e["decode_index"] == row["decode_index"])
        pack = torch.load(source / entry["path"], map_location="cpu", weights_only=False)
        name = row["input"]
        if name.startswith("w2_expert_"):
            eid = int(name.removeprefix("w2_expert_"))
            tensor = next(e["w2_input"] for e in pack["experts"] if e["expert_id"] == eid)
        else:
            tensor = pack[{"qkv": "qkv_input", "o": "o_input", "moe_input": "moe_input"}[name]]
        x = tensor.reshape(1, -1).to(device)
        g = torch.tensor(row["scale_inv"], dtype=torch.float32, device=device)
        native = ref_nvfp4_quant_dequant(x, g, 16)
        triton = fake_quant_nvfp4_activation(x, g)
        eager = _fake_quant_nvfp4_activation_torch(x, g, output_dtype=x.dtype)
        q_native, s_native = ref_nvfp4_quant(x, g, 16)
        xg = x.float().reshape(-1, 16)
        amax = xg.abs().amax(dim=-1, keepdim=True)
        raw_mul = g * (amax * (1.0 / 6.0))
        raw_div = g * (amax / 6.0)
        s_builtin = raw_div.clamp(-448, 448).to(torch.float8_e4m3fn).float()
        s_manual = cast_to_fp8_e4m3fn(raw_div.clamp(-448,448))
        sn = s_native.reshape(-1, 1)
        reciprocal = get_reciprocal(sn * get_reciprocal(g))
        direct = g / sn
        q_direct = cast_to_fp4_e2m1((xg * direct).clamp(-6, 6)).reshape_as(x)
        direct_out = (q_direct.reshape(-1,16) * (sn/g)).reshape_as(x).to(x.dtype)
        rows.append({**row, "native_vs_eager_different": int((native != eager).sum()), "eager_vs_triton_different": int((eager != triton).sum()), "native_vs_direct_same_scale_different": int((native != direct_out).sum()), "direct_same_scale_vs_triton_different": int((direct_out != triton).sum()), "multiply_vs_divide_pre_fp8_different": int((raw_mul != raw_div).sum()), "native_vs_eager_block_scale_different": int((sn != s_manual).sum()), "builtin_vs_manual_fp8_scale_different": int((s_builtin != s_manual).sum()), "reciprocal_vs_direct_multiplier_different": int((reciprocal != direct).sum())})
        for idx in (native != triton).reshape(-1).nonzero().flatten()[:4].tolist():
            block = idx // 16
            examples.append({"layer":row["layer"],"input":name,"index":idx,"x":float(x.flatten()[idx]),"amax":float(amax[block]),"global_inv":float(g),"block_scale_native":float(sn[block]),"block_scale_eager":float(s_manual[block]),"multiplier_native":float(reciprocal[block]),"multiplier_direct":float(direct[block]),"scaled_native":float(x.flatten()[idx].float()*reciprocal[block]),"scaled_direct":float(x.flatten()[idx].float()*direct[block]),"q_native":float(q_native.flatten()[idx]),"q_direct":float(q_direct.flatten()[idx]),"out_native":float(native.flatten()[idx]),"out_eager":float(eager.flatten()[idx]),"out_triton":float(triton.flatten()[idx])})
    result = {"rows":rows,"examples":examples,"source_result_sha256":hashlib.sha256(Path(__file__).with_name("results.json").read_bytes()).hexdigest(),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with Path(__file__).with_name("qdq_diagnosis.json").open("x") as f:
        json.dump(result,f,indent=2)
    print(json.dumps({"affected_tensors":len(rows),"counts":{k:sum(r[k] for r in rows) for k in rows[0] if k.endswith("different")},"examples":examples[:5]},indent=2),flush=True)


if __name__ == "__main__":
    main()
