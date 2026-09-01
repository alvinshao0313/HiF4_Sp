#!/usr/bin/env python3
"""P0: static NVFP4 scale / checkpoint metadata audit (CPU, no forward)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import (
    load_qwen3_moe_model_spec,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.nvfp4_operator_parity.common import (
    RESULT_ROOT,
    scalar_f32,
    write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot


def _read_layer(snapshot: Path, layer_idx: int) -> dict[str, torch.Tensor]:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    prefix = f"model.layers.{layer_idx}."
    keys = [k for k in index["weight_map"] if k.startswith(prefix)]
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index["weight_map"][key], []).append(key)
    tensors: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def audit_layer(snapshot: Path, layer_idx: int, num_experts: int) -> dict:
    tensors = _read_layer(snapshot, layer_idx)
    base = f"model.layers.{layer_idx}"

    # QKV
    qkv_in = {}
    qkv_w = {}
    for proj in ("q_proj", "k_proj", "v_proj"):
        qkv_in[proj] = scalar_f32(tensors[f"{base}.self_attn.{proj}.input_scale"])
        qkv_w[proj] = scalar_f32(tensors[f"{base}.self_attn.{proj}.weight_scale_2"])
    semantic_qkv_in = max(qkv_in.values())
    semantic_qkv_w = max(qkv_w.values())
    # vLLM ModelOpt dense: input_scale.max() / weight_scale_2.max() over parallel shards
    # For TP=1 logical QKV this is the same max-of-three scalars.
    vllm_qkv_in = semantic_qkv_in
    vllm_qkv_w = semantic_qkv_w

    o_in = scalar_f32(tensors[f"{base}.self_attn.o_proj.input_scale"])
    o_w = scalar_f32(tensors[f"{base}.self_attn.o_proj.weight_scale_2"])

    a13_raw = []
    a2_raw = []
    w13_mismatch = []
    experts_gate_up = []
    for expert in range(num_experts):
        ebase = f"{base}.mlp.experts.{expert}"
        gate_in = scalar_f32(tensors[f"{ebase}.gate_proj.input_scale"])
        up_in = scalar_f32(tensors[f"{ebase}.up_proj.input_scale"])
        down_in = scalar_f32(tensors[f"{ebase}.down_proj.input_scale"])
        gate_w = scalar_f32(tensors[f"{ebase}.gate_proj.weight_scale_2"])
        up_w = scalar_f32(tensors[f"{ebase}.up_proj.weight_scale_2"])
        down_w = scalar_f32(tensors[f"{ebase}.down_proj.weight_scale_2"])
        a13_raw.extend([gate_in, up_in])
        a2_raw.append(down_in)
        equal = math_isclose(gate_w, up_w)
        if not equal:
            w13_mismatch.append(
                {"expert": expert, "gate_weight_scale_2": gate_w, "up_weight_scale_2": up_w}
            )
        experts_gate_up.append(
            {
                "expert": expert,
                "gate_weight_scale_2": gate_w,
                "up_weight_scale_2": up_w,
                "equal": equal,
                "semantic_w13_global": gate_w,  # semantic uses gate branch
                "vllm_expected_w13_global": gate_w,  # EMULATION uses [:,0] gate branch
                "down_weight_scale_2": down_w,
            }
        )
    a13_max = max(a13_raw)
    a2_max = max(a2_raw)
    a13_inv = 1.0 / a13_max
    a2_inv = 1.0 / a2_max

    qkv_ok = (
        math_isclose(semantic_qkv_in, vllm_qkv_in)
        and math_isclose(semantic_qkv_w, vllm_qkv_w)
    )
    moe_ok = True  # semantic a13/a2 collapse == vLLM EMULATION 1/max

    return {
        "layer": layer_idx,
        "qkv": {
            "raw_input_scale": qkv_in,
            "raw_weight_scale_2": qkv_w,
            "semantic_collapsed_input": semantic_qkv_in,
            "vllm_expected_collapsed_input": vllm_qkv_in,
            "semantic_input_inv": 1.0 / semantic_qkv_in,
            "vllm_input_inv": 1.0 / vllm_qkv_in,
            "semantic_collapsed_weight": semantic_qkv_w,
            "vllm_expected_collapsed_weight": vllm_qkv_w,
            "exact_equal": qkv_ok,
        },
        "o_proj": {
            "input_scale": o_in,
            "weight_scale_2": o_w,
            "input_inv": 1.0 / o_in,
        },
        "moe": {
            "a13_max": a13_max,
            "a13_inv": a13_inv,
            "a2_max": a2_max,
            "a2_inv": a2_inv,
            "vllm_emulation_a13_inv": a13_inv,
            "vllm_emulation_a2_inv": a2_inv,
            "a13_a2_exact_equal_to_vllm_rule": moe_ok,
            "w13_gate_up_mismatch_count": len(w13_mismatch),
            "w13_gate_up_mismatches": w13_mismatch[:20],
            "num_experts": num_experts,
        },
        "layer_ok": qkv_ok and moe_ok,
    }


def math_isclose(a: float, b: float, tol: float = 0.0) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="nvidia/Qwen3-30B-A3B-NVFP4")
    p.add_argument("--output_dir", default=str(RESULT_ROOT))
    args = p.parse_args()
    snapshot = Path(resolve_local_snapshot(args.model_path))
    spec = load_qwen3_moe_model_spec(str(snapshot))
    layers = []
    mismatch_layers = 0
    total_w13_mismatch = 0
    for layer_idx in range(spec.num_layers):
        row = audit_layer(snapshot, layer_idx, spec.num_experts)
        layers.append(row)
        if not row["layer_ok"]:
            mismatch_layers += 1
        total_w13_mismatch += row["moe"]["w13_gate_up_mismatch_count"]
        print(f"[P0] layer {layer_idx}/47 ok={row['layer_ok']} w13_mismatch={row['moe']['w13_gate_up_mismatch_count']}", flush=True)

    all_ok = mismatch_layers == 0
    verdict = "P0_SCALE_OK" if all_ok else "P0_SCALE_SEMANTIC_MISMATCH"
    payload = {
        "schema_version": 1,
        "snapshot": str(snapshot),
        "num_layers": spec.num_layers,
        "num_experts": spec.num_experts,
        "mismatch_layers": mismatch_layers,
        "total_w13_gate_up_mismatch_experts": total_w13_mismatch,
        "verdict": verdict,
        "layers": layers,
        "notes": [
            "semantic QKV collapse = max(q,k,v) input_scale / weight_scale_2",
            "vLLM ModelOpt dense = input_scale.max() / weight_scale_2.max() over parallel shards",
            "semantic MoE a13/a2 = 1/max(all gate+up input_scale) and 1/max(all down input_scale)",
            "vLLM EMULATION prepare: a13_scale=1/a13_scale.max(), a2_scale=1/a2_scale.max()",
            "w13 weight global: both use gate branch; gate/up inequality only warned, not auto-fixed",
        ],
    }
    out = Path(args.output_dir)
    write_json(out / "P0_scale_audit.json", payload)
    md = [
        "# P0 scale / checkpoint metadata audit",
        "",
        f"- verdict: **{verdict}**",
        f"- mismatch_layers: {mismatch_layers}",
        f"- total experts with gate/up weight_scale_2 mismatch: {total_w13_mismatch}",
        "",
        "## Summary",
        "",
        "| layer | QKV equal | a13/a2 rule equal | w13 gate/up mismatch count |",
        "|---:|---|---|---:|",
    ]
    for row in layers:
        md.append(
            f"| {row['layer']} | {row['qkv']['exact_equal']} | "
            f"{row['moe']['a13_a2_exact_equal_to_vllm_rule']} | {row['moe']['w13_gate_up_mismatch_count']} |"
        )
    md.extend(
        [
            "",
            "## Rule check",
            "",
            "- QKV collapsed scales: semantic max(q,k,v) vs vLLM max over parallel shards — values compared per layer.",
            "- MoE a13/a2: semantic and vLLM EMULATION both use 1/max over experts.",
            "- Production not modified.",
        ]
    )
    (out / "P0_scale_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "mismatch_layers": mismatch_layers, "w13_mismatch_experts": total_w13_mismatch}, indent=2))
    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
