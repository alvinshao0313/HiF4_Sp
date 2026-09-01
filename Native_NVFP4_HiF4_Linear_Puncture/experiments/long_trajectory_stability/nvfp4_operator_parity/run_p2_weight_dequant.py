#!/usr/bin/env python3
"""P2: packed weight dequant parity (_decode_weight vs dequantize_to_dtype)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    _decode_weight,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import (
    load_qwen3_moe_model_spec,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.nvfp4_operator_parity.common import (
    DEFAULT_LAYERS,
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
)


ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")
EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")
CONTROL_NEVER_ROUTED = 3


def _global_weight_for_attn(parts: dict[str, torch.Tensor], proj: str, qkv_w: float) -> torch.Tensor:
    if proj in ("q_proj", "k_proj", "v_proj"):
        return torch.tensor(qkv_w, dtype=torch.float32)
    return parts["weight_scale_2"].reshape(()).to(dtype=torch.float32)


def _compare_decode(
    *,
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_w: torch.Tensor,
    device: torch.device,
    meta: dict,
) -> list[dict]:
    ensure_cuda_lut(device)
    a_fp32 = _decode_weight(packed, scale, global_w, device=device)
    b_fp32 = dequantize_to_dtype(
        packed.to(device),
        scale.to(device),
        global_w.to(device=device, dtype=torch.float32),
        dtype=torch.float32,
        block_size=16,
        swizzle=False,
    )
    rows = []
    for dtype_name, a, b in (
        ("fp32", a_fp32, b_fp32),
        ("bf16", a_fp32.to(torch.bfloat16), b_fp32.to(torch.bfloat16)),
    ):
        metrics = compare_tensors(a, b)
        rows.append(
            {
                **meta,
                "dtype": dtype_name,
                "hard_ok": metrics["exact_fraction"] == 1.0 and metrics["max_abs"] == 0.0,
                **metrics,
            }
        )
    return rows


def _routed_experts_by_layer(result_root: Path) -> dict[int, set[int]]:
    manifest = json.loads((result_root / "frozen_inputs_manifest.json").read_text(encoding="utf-8"))
    out: dict[int, set[int]] = defaultdict(set)
    for sample in manifest["samples"]:
        for entry in sample["layers"]:
            pack = torch.load(result_root / entry["path"], map_location="cpu", weights_only=False)
            layer = int(pack["layer"])
            for expert in pack["experts"]:
                out[layer].add(int(expert["expert_id"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=str(RESULT_ROOT))
    p.add_argument("--model_path", default="nvidia/Qwen3-30B-A3B-NVFP4")
    p.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    args = p.parse_args()

    device = torch.device(args.device)
    ensure_cuda_lut(device)
    out = Path(args.output_dir)
    snapshot = Path(resolve_local_snapshot(args.model_path))
    spec = load_qwen3_moe_model_spec(str(snapshot))
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    routed = _routed_experts_by_layer(out)
    rows: list[dict] = []

    for layer in layers:
        # Attention
        attn_parts = {
            proj: load_raw_linear_parts(snapshot, f"model.layers.{layer}.self_attn.{proj}")
            for proj in ATTN_PROJS
        }
        qkv_w = max(scalar_f32(attn_parts[p]["weight_scale_2"]) for p in ("q_proj", "k_proj", "v_proj"))
        for proj in ATTN_PROJS:
            parts = attn_parts[proj]
            gw = _global_weight_for_attn(parts, proj, qkv_w)
            rows.extend(
                _compare_decode(
                    packed=parts["weight"],
                    scale=parts["weight_scale"],
                    global_w=gw,
                    device=device,
                    meta={
                        "layer": layer,
                        "kind": "attention",
                        "proj": proj,
                        "expert_id": None,
                        "control": False,
                        "bucket": "all_layers",
                    },
                )
            )

        routed_ids = sorted(routed.get(layer, set()))
        all_ids = list(range(spec.num_experts))
        never = [e for e in all_ids if e not in routed.get(layer, set())][:CONTROL_NEVER_ROUTED]
        targets = [(eid, False) for eid in routed_ids] + [(eid, True) for eid in never]

        for expert_id, is_control in targets:
            for proj in EXPERT_PROJS:
                parts = load_raw_linear_parts(
                    snapshot, f"model.layers.{layer}.mlp.experts.{expert_id}.{proj}"
                )
                if proj in ("gate_proj", "up_proj"):
                    gate_parts = load_raw_linear_parts(
                        snapshot, f"model.layers.{layer}.mlp.experts.{expert_id}.gate_proj"
                    )
                    gw = gate_parts["weight_scale_2"].reshape(()).to(dtype=torch.float32)
                else:
                    gw = parts["weight_scale_2"].reshape(()).to(dtype=torch.float32)
                rows.extend(
                    _compare_decode(
                        packed=parts["weight"],
                        scale=parts["weight_scale"],
                        global_w=gw,
                        device=device,
                        meta={
                            "layer": layer,
                            "kind": "expert",
                            "proj": proj,
                            "expert_id": expert_id,
                            "control": is_control,
                            "bucket": "never_routed_control" if is_control else "routed",
                        },
                    )
                )
        print(f"[P2] layer {layer} routed={len(routed_ids)} never_control={never}", flush=True)

    write_jsonl(out / "P2_weight_dequant_rows.jsonl", rows)
    hard_fail = [r for r in rows if not r["hard_ok"]]
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)

    md = [
        "# P2 packed weight dequant parity",
        "",
        f"- rows: {len(rows)}",
        f"- hard_ok_all: {len(hard_fail) == 0}",
        f"- hard_fail_count: {len(hard_fail)}",
        "",
        "## By bucket",
        "",
        "| bucket | n | exact_fraction_min | max_abs_max | hard_ok |",
        "|---|---:|---:|---:|---|",
    ]
    for bucket, group in sorted(by_bucket.items()):
        exact_min = min(r["exact_fraction"] for r in group)
        max_abs = max(r["max_abs"] for r in group)
        ok = all(r["hard_ok"] for r in group)
        md.append(f"| {bucket} | {len(group)} | {exact_min:.6g} | {max_abs:.6g} | {ok} |")
    md.extend(
        [
            "",
            f"- verdict: **{'P2_WEIGHT_DEQUANT_OK' if not hard_fail else 'P2_WEIGHT_DEQUANT_MISMATCH'}**",
            "",
        ]
    )
    (out / "P2_weight_dequant_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "hard_fail": len(hard_fail)}, indent=2))
    if hard_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
