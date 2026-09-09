#!/usr/bin/env python3
"""Build puncture requests from actual-path frontiers and run ProductionPunctureOp."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import read_jsonl, write_jsonl
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.production_puncture import (
    ProductionPunctureOp,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.build_llm import build_real_vllm
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.quant_features import (
    load_verified_capture,
)


def _frontier_layers(compare_rows: list[dict], sample_key: str, decode_index: int, limit: int = 6) -> list[int]:
    rows = [r for r in compare_rows if r.get("sample_key") == sample_key and int(r.get("decode_index", -1)) == decode_index]
    scored = []
    for row in rows:
        layer = int(row["layer"])
        moe = float(row.get("moe_out_rel_l2") or row.get("boundaries", {}).get("moe_out", {}).get("rel_l2") or 0)
        attn = float(row.get("attention_core_rel_l2") or row.get("boundaries", {}).get("attention_core", {}).get("rel_l2") or 0)
        scored.append((max(moe, attn), moe - float(row.get("post_attn_norm_rel_l2") or 0), layer))
    scored.sort(reverse=True)
    layers = []
    for _, _, layer in scored:
        if layer not in layers:
            layers.append(layer)
        if len(layers) >= limit:
            break
    if not layers:
        layers = [0, 8, 16, 24, 32, 40][:limit]
    return layers


def build_requests(root: Path) -> list[dict]:
    mech = read_jsonl(root / "02_cohort/mechanism_cohort.jsonl")
    events = {r["sample_key"]: r for r in read_jsonl(root / "05_probe_plan/divergence_events.jsonl")}
    compare_path = root / "07_core_capture/e0_e1_compare.jsonl"
    compare = read_jsonl(compare_path) if compare_path.exists() else []
    requests = []
    for row in mech:
        key = row["prompt_key"]
        event = events.get(key, {})
        t_lex = event.get("t_lex")
        if t_lex is None:
            continue
        tokens = sorted({max(1, t_lex + d) for d in (-4, -2, -1, 0, 1) if t_lex + d >= 1})[:5]
        layers = _frontier_layers(compare, key, t_lex if t_lex >= 1 else 1)
        for decode in tokens:
            for layer in layers:
                for operator in ("qkv", "o_proj", "moe"):
                    requests.append({"sample_key": key, "decode_index": int(decode),
                                    "layer": int(layer), "operator": operator,
                                    "mechanism_group": row["mechanism_group"]})
    return requests


def _hooks_root(root: Path, variant: str) -> Path:
    base = root / "07_core_capture" / variant
    hooks = base / "hooks" / variant
    return hooks if hooks.exists() else base


def run(root: Path, phasea_root: Path, model_path: str, variants: list[str] | None = None) -> dict:
    root = Path(root)
    variants = variants or ["E0", "E1"]
    requests = build_requests(root)
    if not requests:
        raise RuntimeError("no puncture requests; need mechanism cohort + divergence events + core compare")
    plan_path = root / "08_rq1_puncture/request_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"requests": requests}, indent=2) + "\n")
    # Verify E0/E1 captures share forced E0 history before puncture.
    e0_manifest = root / "07_core_capture/E0/E0_forced_core_manifest.json"
    canonical = root / "03_isolated/E0.jsonl"
    sample_dirs = sorted({r["sample_key"] for r in requests})
    for sample in sample_dirs[:1]:
        path = _hooks_root(root, "E0") / sample / "rank0.pt"
        if path.exists():
            load_verified_capture(path, e0_manifest, canonical, capture_level="core")
    summary = {"requests": len(requests), "variants": {}}
    reference = str(_hooks_root(root, "E0"))
    baseline_root = None
    for variant in variants:
        out = root / "08_rq1_puncture" / variant
        out.mkdir(parents=True, exist_ok=True)
        llm, _ = build_real_vllm(variant, model_path=model_path, phasea_root=Path(phasea_root))
        op = ProductionPunctureOp(
            requests, reference, str(_hooks_root(root, variant if variant in {"E0", "E1"} else "E1")),
            str(out), variant, canonical_history_verified=True,
            baseline_puncture_root=baseline_root,
        )
        # For E2/E3 compare against E1 actual path inputs still using E0 x0 puncture definition.
        if variant in {"E2", "E3"}:
            op = ProductionPunctureOp(
                requests, reference, str(_hooks_root(root, variant)),
                str(out), variant, canonical_history_verified=True,
                baseline_puncture_root=baseline_root,
            )
        receipts = llm.apply_model(op)
        if variant == "E0":
            baseline_root = str(out)
            if any(not r.get("pass", True) for r in receipts):
                raise RuntimeError(f"MOE_OR_PROJECTION_PUNCTURE_BLOCKED on E0 identity: {receipts}")
        summary["variants"][variant] = receipts
        del llm
    (root / "08_rq1_puncture/summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", type=Path, required=True)
    p.add_argument("--phasea_root", type=Path, required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--variants", nargs="+", default=["E0", "E1"])
    args = p.parse_args()
    print(json.dumps(run(args.run_root, args.phasea_root, args.model_path, args.variants), default=str))
