#!/usr/bin/env python3
"""Reduced RQ1 (2026-09-08 replan): MoE-first same-input puncture only.

Skips: feature scan, semantic frontier, full QKV/O default puncture, E2/E3.
Gate B: stop after MoE e=q+p (+ optional frozen-router if q_M large).
Gate A: after MoE, emit RQ2 unlock marker for limited state intervention.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import (
    read_jsonl,
    write_jsonl,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion import (
    run_go,
    run_puncture_go,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.production_puncture import (
    ProductionPunctureOp,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.build_llm import (
    build_real_vllm,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.quant_features import (
    load_verified_capture,
)


def _frontier_moe_layers(compare_rows: list[dict], sample_key: str, decode_index: int, limit: int = 3) -> list[int]:
    rows = [
        r
        for r in compare_rows
        if r.get("sample_key") == sample_key and int(r.get("decode_index", -1)) == decode_index
    ]
    scored = []
    for row in rows:
        layer = int(row["layer"])
        moe = float(
            row.get("moe_out_rel_l2")
            or row.get("boundaries", {}).get("moe_out", {}).get("rel_l2")
            or 0
        )
        scored.append((moe, layer))
    scored.sort(reverse=True)
    layers: list[int] = []
    for _, layer in scored:
        if layer not in layers:
            layers.append(layer)
        if len(layers) >= limit:
            break
    return layers or [0, 16, 32][:limit]


def build_moe_requests(root: Path) -> list[dict]:
    """Plan §9: probe={t*-8,t*-2,t*-1,t*}; Top3 MoE layers; operator=moe only."""
    mech = read_jsonl(root / "02_cohort/mechanism_cohort.jsonl")
    events = {r["sample_key"]: r for r in read_jsonl(root / "05_probe_plan/divergence_events.jsonl")}
    compare_path = root / "07_core_capture/e0_e1_compare.jsonl"
    compare = read_jsonl(compare_path) if compare_path.exists() else []
    requests = []
    for row in mech:
        if row.get("mechanism_group") != "mechanism_regression":
            # still puncture robust controls at matched relative depth via event if present
            pass
        key = row["prompt_key"]
        event = events.get(key, {})
        t_lex = event.get("t_lex")
        if t_lex is None:
            continue
        tokens = sorted({max(1, int(t_lex) + d) for d in (-8, -2, -1, 0)})
        layers = _frontier_moe_layers(compare, key, int(t_lex) if int(t_lex) >= 1 else 1, limit=3)
        for decode in tokens:
            for layer in layers:
                requests.append(
                    {
                        "sample_key": key,
                        "decode_index": int(decode),
                        "layer": int(layer),
                        "operator": "moe",
                        "mechanism_group": row["mechanism_group"],
                    }
                )
    return requests


def run_moe_puncture(root: Path, phasea_root: Path, model_path: str) -> dict:
    root = Path(root)
    requests = build_moe_requests(root)
    if not requests:
        raise RuntimeError("no MoE puncture requests; need mechanism cohort + divergence + core compare")
    plan_path = root / "08_rq1_puncture/request_plan_moe_only.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"policy": "moe_first_replan_20260908", "requests": requests}, indent=2) + "\n")
    e0_manifest = root / "07_core_capture/E0/E0_forced_core_manifest.json"
    canonical = root / "03_isolated/E0.jsonl"
    hooks_e0 = run_puncture_go._hooks_root(root, "E0")
    for sample in sorted({r["sample_key"] for r in requests})[:1]:
        path = hooks_e0 / sample / "rank0.pt"
        if path.exists():
            load_verified_capture(path, e0_manifest, canonical, capture_level="core")
    summary = {"requests": len(requests), "variants": {}, "operators": ["moe"]}
    reference = str(hooks_e0)
    baseline_root = None
    for variant in ("E0", "E1"):
        out = root / "08_rq1_puncture" / variant
        out.mkdir(parents=True, exist_ok=True)
        llm, _ = build_real_vllm(variant, model_path=model_path, phasea_root=Path(phasea_root))
        op = ProductionPunctureOp(
            requests,
            reference,
            str(run_puncture_go._hooks_root(root, variant)),
            str(out),
            variant,
            canonical_history_verified=True,
            baseline_puncture_root=baseline_root,
        )
        receipts = llm.apply_model(op)
        if variant == "E0":
            baseline_root = str(out)
            if any(not r.get("pass", True) for r in receipts):
                raise RuntimeError(f"MOE_PUNCTURE_BLOCKED on E0 identity: {receipts}")
        summary["variants"][variant] = receipts
        del llm
    (root / "08_rq1_puncture/summary_moe_only.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    return summary


def prepare_layout(bridge_layout: Path, mechanism_root: Path) -> None:
    """Copy greedy-bridge layout into 13_reduced_rq1 go-shaped tree."""
    mechanism_root.mkdir(parents=True, exist_ok=True)
    for rel in (
        "02_cohort/benchmark_cohort.jsonl",
        "03_isolated/E0.jsonl",
        "03_isolated/E1.jsonl",
        "04_greedy_judge/greedy_task_matrix.jsonl",
        "04_greedy_judge/greedy_task_summary.json",
    ):
        src = bridge_layout / rel
        if not src.exists():
            # judge may write flat into bridge dir
            alt = {
                "04_greedy_judge/greedy_task_matrix.jsonl": bridge_layout.parent / "greedy_task_matrix.jsonl",
                "04_greedy_judge/greedy_task_summary.json": bridge_layout.parent / "greedy_task_summary.json",
            }.get(rel)
            src = alt if alt and alt.exists() else src
        if not src.exists():
            raise FileNotFoundError(f"missing {rel} (looked at {src})")
        dst = mechanism_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mech_root", type=Path, required=True, help="13_reduced_rq1 go-shaped root")
    p.add_argument("--phasea_root", type=Path, required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--gate", required=True, choices=["A", "B"])
    p.add_argument("--max_pairs", type=int, default=8)
    args = p.parse_args()
    root = args.mech_root
    # mechanism cohort from bridge judge
    run_go.select_mechanism_cohort(root, target=args.max_pairs)
    print("[reduced-rq1] probe plan", flush=True)
    heavy, _feature = run_go.task_probe_plan(root, args.model_path)
    print("[reduced-rq1] forced_core capture (skip feature scan)", flush=True)
    run_go.task_capture(root, heavy, ["E0", "E1"], "forced_core", "07_core_capture")
    run_go.task_compare_core(root)
    print("[reduced-rq1] MoE-only production puncture", flush=True)
    summary = run_moe_puncture(root, args.phasea_root, args.model_path)
    unlock = {
        "gate": args.gate,
        "moe_requests": summary["requests"],
        "rq2_state_allowed": args.gate == "A",
        "e2_diag_probe_allowed": False,  # only after >=2 cases share mechanism
        "note": "semantic frontier cancelled; QKV/O only if MoE q insufficient (manual follow-up)",
    }
    (root / "RQ1_MOE_DONE.json").write_text(json.dumps(unlock, indent=2) + "\n")
    if args.gate == "A":
        (root / "RQ2_UNLOCK.json").write_text(
            json.dumps({"status": "READY", "max_layers_per_case": 3, "alphas": [0.0, 1.0, 0.5]}, indent=2)
            + "\n"
        )
    print(json.dumps({"status": "PASS", "unlock": unlock}, ensure_ascii=False))


if __name__ == "__main__":
    main()
