#!/usr/bin/env python3
"""P4: single-expert W13/W2 parity on frozen routed experts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.nvfp4_operator_parity.common import (
    RESULT_ROOT,
    compare_tensors,
    ensure_cuda_lut,
    load_raw_linear_parts,
    scalar_f32,
    write_jsonl,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    ref_nvfp4_quant_dequant,
    run_nvfp4_emulations,
)


def iter_packs(result_root: Path):
    manifest = json.loads((result_root / "frozen_inputs_manifest.json").read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        for entry in sample["layers"]:
            path = result_root / entry["path"]
            pack = torch.load(path, map_location="cpu", weights_only=False)
            yield pack, path


def _d0_linear(x: torch.Tensor, parts: dict, input_inv: float, weight_global: float, device: torch.device) -> torch.Tensor:
    x = x.to(device=device, dtype=torch.bfloat16).reshape(1, -1)
    inv = torch.tensor(input_inv, device=device, dtype=torch.float32)
    gw = torch.tensor(weight_global, device=device, dtype=torch.float32)
    packed = parts["weight"].to(device)
    scale = parts["weight_scale"].to(device)
    x_qdq = ref_nvfp4_quant_dequant(x, inv, block_size=16)
    w_fp32 = dequantize_to_dtype(
        packed.view(torch.uint8), scale, gw, dtype=torch.float32, block_size=16, swizzle=False
    )
    return F.linear(x_qdq, w_fp32.to(dtype=x_qdq.dtype))


def _d1_linear(x: torch.Tensor, parts: dict, input_inv: float, weight_global: float, device: torch.device) -> torch.Tensor:
    x = x.to(device=device, dtype=torch.bfloat16).reshape(1, -1)
    inv = torch.tensor(input_inv, device=device, dtype=torch.float32)
    gw = torch.tensor(weight_global, device=device, dtype=torch.float32)
    return run_nvfp4_emulations(
        x=x,
        input_global_scale=inv,
        weight=parts["weight"].to(device),
        weight_scale_swizzled=parts["weight_scale"].to(device),
        weight_global_scale=gw,
        swizzle=False,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=str(RESULT_ROOT))
    p.add_argument("--model_path", default="nvidia/Qwen3-30B-A3B-NVFP4")
    args = p.parse_args()

    device = torch.device(args.device)
    ensure_cuda_lut(device)
    out = Path(args.output_dir)
    snapshot = Path(resolve_local_snapshot(args.model_path))
    rows: list[dict] = []
    parts_cache: dict[tuple[int, int, str], dict] = {}

    def parts_for(layer: int, expert_id: int, proj: str) -> dict:
        key = (layer, expert_id, proj)
        if key not in parts_cache:
            parts_cache[key] = load_raw_linear_parts(
                snapshot, f"model.layers.{layer}.mlp.experts.{expert_id}.{proj}"
            )
        return parts_cache[key]

    for pack, path in iter_packs(out):
        layer = int(pack["layer"])
        moe_in = pack["moe_input"]
        for expert in pack["experts"]:
            eid = int(expert["expert_id"])
            a13 = float(expert["a13_inv"])
            a2 = float(expert["a2_inv"])
            w13_global = float(expert["w13_global"])
            w2_global = float(expert["w2_global"])
            gate_parts = parts_for(layer, eid, "gate_proj")
            up_parts = parts_for(layer, eid, "up_proj")
            down_parts = parts_for(layer, eid, "down_proj")
            gate_w2 = scalar_f32(gate_parts["weight_scale_2"])
            up_w2 = scalar_f32(up_parts["weight_scale_2"])

            gate_d0 = _d0_linear(moe_in, gate_parts, a13, w13_global, device)
            gate_d1 = _d1_linear(moe_in, gate_parts, a13, w13_global, device)
            up_d0 = _d0_linear(moe_in, up_parts, a13, w13_global, device)
            up_d1 = _d1_linear(moe_in, up_parts, a13, w13_global, device)

            hidden_sem = F.silu(gate_d0) * up_d0
            hidden_prim = F.silu(gate_d1) * up_d1

            # W2 with semantic (frozen-path) hidden on both sides.
            w2_sem_d0 = _d0_linear(hidden_sem, down_parts, a2, w2_global, device)
            w2_sem_d1 = _d1_linear(hidden_sem, down_parts, a2, w2_global, device)
            # W2 with primitive hidden (propagated W13 error).
            w2_prim_d0 = _d0_linear(hidden_prim, down_parts, a2, w2_global, device)
            w2_prim_d1 = _d1_linear(hidden_prim, down_parts, a2, w2_global, device)

            row = {
                "prompt_key": pack["prompt_key"],
                "decode_index": int(pack["decode_index"]),
                "layer": layer,
                "bucket": pack["bucket"],
                "expert_id": eid,
                "slot": int(expert["slot"]),
                "routing_weight": float(expert["routing_weight"]),
                "gate_up_weight_scale_2_equal": gate_w2 == up_w2,
                "gate_weight_scale_2": gate_w2,
                "up_weight_scale_2": up_w2,
                "a13_inv": a13,
                "a2_inv": a2,
                "w13_global": w13_global,
                "w2_global": w2_global,
                "pack_path": str(path.relative_to(out)),
            }
            for tag, a, b in (
                ("w13_gate", gate_d0, gate_d1),
                ("w13_up", up_d0, up_d1),
                ("post_activation", hidden_sem, hidden_prim),
                ("w2_semantic_hidden", w2_sem_d0, w2_sem_d1),
                ("w2_primitive_hidden", w2_prim_d0, w2_prim_d1),
            ):
                metrics = compare_tensors(a, b)
                for k, v in metrics.items():
                    row[f"{tag}_{k}"] = v
            rows.append(row)
        print(
            f"[P4] {pack['prompt_key']} decode={pack['decode_index']} layer={layer} experts={len(pack['experts'])}",
            flush=True,
        )

    write_jsonl(out / "P4_single_expert_rows.jsonl", rows)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)

    stages = ("w13_gate", "w13_up", "post_activation", "w2_semantic_hidden", "w2_primitive_hidden")
    md = [
        "# P4 single-expert W13/W2 parity",
        "",
        f"- rows: {len(rows)}",
        f"- gate_up_weight_scale_2_mismatch: {sum(1 for r in rows if not r['gate_up_weight_scale_2_equal'])}",
        "",
        "## Stage summary",
        "",
        "| stage | max_abs_max | rel_l2_max | exact_fraction_min |",
        "|---|---:|---:|---:|",
    ]
    for stage in stages:
        md.append(
            f"| {stage} | {max(r[f'{stage}_max_abs'] for r in rows):.6g} | "
            f"{max(r[f'{stage}_rel_l2'] for r in rows):.6g} | "
            f"{min(r[f'{stage}_exact_fraction'] for r in rows):.6g} |"
        )
    md.extend(["", "## By bucket", "", "| bucket | n | w13_gate_max_abs | w2_semantic_max_abs | w2_primitive_max_abs |", "|---|---:|---:|---:|---:|"])
    for bucket in ("focus_low_margin", "uniform_control", "post_tp_divergence_control_only"):
        group = by_bucket.get(bucket, [])
        if not group:
            continue
        md.append(
            f"| {bucket} | {len(group)} | "
            f"{max(r['w13_gate_max_abs'] for r in group):.6g} | "
            f"{max(r['w2_semantic_hidden_max_abs'] for r in group):.6g} | "
            f"{max(r['w2_primitive_hidden_max_abs'] for r in group):.6g} |"
        )
    md.append("")
    (out / "P4_single_expert_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
