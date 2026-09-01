#!/usr/bin/env python3
"""Aggregate P0–P6 artifacts into E0_NVFP4_OPERATOR_PARITY_REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.nvfp4_operator_parity.common import (
    RESULT_ROOT,
)


ROOT_CAUSE_LABELS = (
    "SCALE_METADATA_MISMATCH",
    "ACTIVATION_QDQ_PATH_MISMATCH",
    "WEIGHT_DEQUANT_PATH_MISMATCH",
    "DENSE_GEMM_NUMERIC_DIFFERENCE",
    "SINGLE_EXPERT_GEMM_NUMERIC_DIFFERENCE",
    "FUSED_MOE_ACCUMULATION_DIFFERENCE",
    "TP_PARTITION_REDUCTION_DIFFERENCE",
    "MIXED_SMALL_NUMERIC_DIFFERENCES",
    "INCONCLUSIVE",
)

PROD_RECS = (
    "NO_PRODUCTION_CHANGE_NEEDED",
    "DIAGNOSTIC_TP2_ADAPTER_NEEDED",
    "PRODUCTION_BUG_CANDIDATE",
)


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _max_field(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return max(float(r[key]) for r in rows if key in r)


def _min_field(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return min(float(r[key]) for r in rows if key in r)


def _classify(out: Path) -> tuple[str, str, list[str]]:
    notes: list[str] = []
    p0 = _load_json(out / "P0_scale_audit.json")
    p1 = _load_jsonl(out / "P1_qdq_rows.jsonl")
    p2 = _load_jsonl(out / "P2_weight_dequant_rows.jsonl")
    p3 = _load_jsonl(out / "P3_dense_linear_rows.jsonl")
    p4 = _load_jsonl(out / "P4_single_expert_rows.jsonl")
    p5 = _load_jsonl(out / "P5_fused_moe_rows.jsonl")
    p6 = _load_jsonl(out / "P6_manual_tp2_rows.jsonl")

    if p0 and p0.get("verdict") == "P0_SCALE_SEMANTIC_MISMATCH":
        return "SCALE_METADATA_MISMATCH", "PRODUCTION_BUG_CANDIDATE", [
            "P0 collapsed scale rules disagree with vLLM expectations."
        ]

    p1_fail = any(not r.get("hard_ok", False) for r in p1)
    if p1_fail:
        return "ACTIVATION_QDQ_PATH_MISMATCH", "PRODUCTION_BUG_CANDIDATE", [
            "P1 QDQ paths differ under identical input/scale."
        ]

    p2_fail = any(not r.get("hard_ok", False) for r in p2)
    if p2_fail:
        return "WEIGHT_DEQUANT_PATH_MISMATCH", "PRODUCTION_BUG_CANDIDATE", [
            "P2 weight dequant paths differ under identical packed tensors."
        ]

    def nonzero(rows, keys):
        for r in rows:
            for k in keys:
                if k in r and float(r[k]) != 0.0:
                    return True
        return False

    p3_nz = nonzero(p3, ["D0_D1_max_abs", "D0_D2_max_abs", "D1_D2_max_abs"])
    p4_nz = nonzero(
        p4,
        [
            "w13_gate_max_abs",
            "w13_up_max_abs",
            "w2_semantic_hidden_max_abs",
            "w2_primitive_hidden_max_abs",
        ],
    )
    p5_nz = nonzero(p5, ["semantic_vs_fused_max_abs", "p4_weighted_sum_vs_fused_max_abs"])
    p6_nz = nonzero(p6, ["max_abs"])

    # First-stage attribution priority (earliest nonzero wins).
    if p3_nz:
        notes.append("First clear nonzero appears at dense GEMM (P3).")
        if p6_nz and (_max_field(p6, "max_abs") or 0) > (_max_field(p3, "D0_D1_max_abs") or 0):
            notes.append("Manual TP2 reduction error dominates dense pair diffs.")
            return "TP_PARTITION_REDUCTION_DIFFERENCE", "DIAGNOSTIC_TP2_ADAPTER_NEEDED", notes
        return "DENSE_GEMM_NUMERIC_DIFFERENCE", "NO_PRODUCTION_CHANGE_NEEDED", notes

    if p4_nz:
        notes.append("Single-expert GEMM already differs before fused MoE.")
        return "SINGLE_EXPERT_GEMM_NUMERIC_DIFFERENCE", "NO_PRODUCTION_CHANGE_NEEDED", notes

    if p5_nz:
        notes.append("P4 single-expert aligned; fused MoE introduces additional error.")
        return "FUSED_MOE_ACCUMULATION_DIFFERENCE", "PRODUCTION_BUG_CANDIDATE", notes

    if p6_nz:
        notes.append("Full GEMM vs manual TP2 shard/reduce is the primary difference.")
        return "TP_PARTITION_REDUCTION_DIFFERENCE", "DIAGNOSTIC_TP2_ADAPTER_NEEDED", notes

    if not any((p1, p2, p3, p4, p5, p6)):
        return "INCONCLUSIVE", "NO_PRODUCTION_CHANGE_NEEDED", ["Missing stage artifacts."]

    notes.append("P0–P6 show no hard semantic mismatch under frozen identical inputs.")
    return "INCONCLUSIVE", "NO_PRODUCTION_CHANGE_NEEDED", notes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=str(RESULT_ROOT))
    args = p.parse_args()
    out = Path(args.output_dir)

    p0 = _load_json(out / "P0_scale_audit.json") or {}
    p1 = _load_jsonl(out / "P1_qdq_rows.jsonl")
    p2 = _load_jsonl(out / "P2_weight_dequant_rows.jsonl")
    p3 = _load_jsonl(out / "P3_dense_linear_rows.jsonl")
    p4 = _load_jsonl(out / "P4_single_expert_rows.jsonl")
    p5 = _load_jsonl(out / "P5_fused_moe_rows.jsonl")
    p6 = _load_jsonl(out / "P6_manual_tp2_rows.jsonl")

    root_cause, prod_rec, notes = _classify(out)
    assert root_cause in ROOT_CAUSE_LABELS
    assert prod_rec in PROD_RECS

    first_nonzero = None
    for name in ("D0_D1", "D0_D2", "D1_D2"):
        if any(float(r.get(f"{name}_max_abs", 0)) != 0.0 for r in p3):
            first_nonzero = name
            break

    lines = [
        "# E0 NVFP4 Operator Parity Report",
        "",
        "Frozen identical input operator parity (diagnostic only). Production runtime not modified.",
        "",
        "## 1. P0 scale audit",
        "",
        f"- QKV collapse equal layers: mismatch_layers={p0.get('mismatch_layers', 'n/a')}",
        f"- a13/a2 rule: see P0_scale_audit.md",
        f"- w13 gate/up global scale mismatch experts: {p0.get('total_w13_gate_up_mismatch_experts', 'n/a')}",
        f"- verdict: **{p0.get('verdict', 'MISSING')}**",
        "",
        "## 2. P1 QDQ",
        "",
        f"- rows: {len(p1)}",
        f"- exact_fraction_min: {_min_field(p1, 'exact_fraction')}",
        f"- max_abs_max: {_max_field(p1, 'max_abs')}",
        f"- focus/control: see P1_qdq_summary.md (focus_low_margin / uniform_control / post_tp_divergence_control_only)",
        f"- verdict: **{'P1_QDQ_OK' if p1 and all(r.get('hard_ok') for r in p1) else ('P1_QDQ_MISMATCH' if p1 else 'MISSING')}**",
        "",
        "## 3. P2 weight dequant",
        "",
        f"- rows: {len(p2)}",
        f"- exact_fraction_min: {_min_field(p2, 'exact_fraction')}",
        f"- max_abs_max: {_max_field(p2, 'max_abs')}",
        f"- verdict: **{'P2_WEIGHT_DEQUANT_OK' if p2 and all(r.get('hard_ok') for r in p2) else ('P2_WEIGHT_DEQUANT_MISMATCH' if p2 else 'MISSING')}**",
        "",
        "## 4. P3 dense linear",
        "",
        f"- rows: {len(p3)}",
        f"- D0-D1 max_abs: {_max_field(p3, 'D0_D1_max_abs')}",
        f"- D0-D2 max_abs: {_max_field(p3, 'D0_D2_max_abs')}",
        f"- D1-D2 max_abs: {_max_field(p3, 'D1_D2_max_abs')}",
        f"- first nonzero stage/pair: **{first_nonzero}**",
        "",
        "| proj | D0_D1_max_abs | D0_D2_max_abs | D1_D2_max_abs |",
        "|---|---:|---:|---:|",
    ]
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        group = [r for r in p3 if r.get("proj") == proj]
        if not group:
            continue
        lines.append(
            f"| {proj} | {_max_field(group, 'D0_D1_max_abs'):.6g} | "
            f"{_max_field(group, 'D0_D2_max_abs'):.6g} | {_max_field(group, 'D1_D2_max_abs'):.6g} |"
        )
    lines.extend(
        [
            "",
            "## 5. P4 single expert",
            "",
            f"- rows: {len(p4)}",
            f"- W13 gate max_abs: {_max_field(p4, 'w13_gate_max_abs')}",
            f"- W13 up max_abs: {_max_field(p4, 'w13_up_max_abs')}",
            f"- post activation max_abs: {_max_field(p4, 'post_activation_max_abs')}",
            f"- W2 frozen-hidden (semantic) max_abs: {_max_field(p4, 'w2_semantic_hidden_max_abs')}",
            f"- W2 propagated-hidden (primitive) max_abs: {_max_field(p4, 'w2_primitive_hidden_max_abs')}",
            "",
            "## 6. P5 fused MoE",
            "",
            f"- rows: {len(p5)}",
            f"- semantic vs fused max_abs: {_max_field(p5, 'semantic_vs_fused_max_abs')}",
            f"- P4 weighted-sum vs fused max_abs: {_max_field(p5, 'p4_weighted_sum_vs_fused_max_abs')}",
            "- single-expert vs fused: compare P4 vs P5 magnitudes in summaries",
            "- sorting/padding/accumulation delta: inferred if P4≈0 but P5≠0",
            "",
            "## 7. P6 TP2",
            "",
            f"- rows: {len(p6)}",
            f"- full vs shard/reduce max_abs: {_max_field(p6, 'max_abs')}",
            f"- row/column parallel covered: {sorted({r.get('parallel') for r in p6})}",
            f"- reduction dtypes: {sorted({r.get('reduce_dtype') for r in p6})}",
            "",
            "## 8. Root-cause classification",
            "",
            f"- primary: **{root_cause}**",
            "- allowed labels: " + ", ".join(ROOT_CAUSE_LABELS),
            "- notes:",
        ]
    )
    for note in notes:
        lines.append(f"  - {note}")
    lines.extend(
        [
            "",
            "## 9. Production change recommendation",
            "",
            f"- recommendation: **{prod_rec}**",
            "- options: " + " / ".join(PROD_RECS),
            "",
            "Only `PRODUCTION_BUG_CANDIDATE` warrants a follow-up production runtime change.",
            "",
        ]
    )
    report = out / "E0_NVFP4_OPERATOR_PARITY_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(report), "root_cause": root_cause, "recommendation": prod_rec}, indent=2))


if __name__ == "__main__":
    main()
