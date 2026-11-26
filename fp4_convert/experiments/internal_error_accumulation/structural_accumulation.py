"""Structural accumulation analysis (RQ1) over residual ledgers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import NUM_LAYERS
from .math_utils import exact_kl, fisher_quadratic_kl, sample_bootstrap_ci
from .residual_ledger import coherence_ratio, cross_variant_ledger, extract_layer_tensors, index_capture_records
from .run_state import atomic_write_json, write_jsonl


def analyze_state_pair(
    *,
    sample_meta: dict,
    e0_records: list[dict],
    e1_records: list[dict],
    e0_logits: torch.Tensor,
    e1_logits: torch.Tensor,
) -> dict[str, Any]:
    idx0 = index_capture_records(e0_records)
    idx1 = index_capture_records(e1_records)
    t0 = extract_layer_tensors(idx0, sample_key=sample_meta["sample_key"], decode_index=int(sample_meta["decode_index"]), rank=0)
    t1 = extract_layer_tensors(idx1, sample_key=sample_meta["sample_key"], decode_index=int(sample_meta["decode_index"]), rank=0)
    ledger = cross_variant_ledger(t0, t1)
    kl_exact = exact_kl(e0_logits, e1_logits)
    d_logits = e1_logits.detach().to(dtype=torch.float64).reshape(-1) - e0_logits.detach().to(dtype=torch.float64).reshape(-1)
    kl_fisher = fisher_quadratic_kl(e0_logits, d_logits)
    branch_vecs = []
    attn_vecs = []
    moe_vecs = []
    for layer in range(NUM_LAYERS):
        attn_vecs.append(ledger["delta_A"][layer])
        moe_vecs.append(ledger["delta_M"][layer])
        branch_vecs.append(ledger["delta_A"][layer])
        branch_vecs.append(ledger["delta_M"][layer])
    early = branch_vecs[0:32]
    middle = branch_vecs[32:64]
    late = branch_vecs[64:96]
    return {
        "sample_key": sample_meta["sample_key"],
        "calibration_sample_id": sample_meta["calibration_sample_id"],
        "source": sample_meta["source"],
        "split": sample_meta["split"],
        "prefix_length_j": sample_meta["prefix_length_j"],
        "delta_R0_l2": ledger["delta_R0_l2"],
        "delta_R48_l2": ledger["delta_R48_l2"],
        "final_hidden_rel_l2": ledger["final_hidden_rel_l2"],
        "kl_exact": kl_exact,
        "kl_fisher": kl_fisher,
        "fisher_over_exact": (kl_fisher / kl_exact) if kl_exact else None,
        "fisher_abs_err": abs(kl_fisher - kl_exact),
        "kappa_all": coherence_ratio(branch_vecs),
        "kappa_attn": coherence_ratio(attn_vecs),
        "kappa_moe": coherence_ratio(moe_vecs),
        "kappa_early": coherence_ratio(early),
        "kappa_middle": coherence_ratio(middle),
        "kappa_late": coherence_ratio(late),
        "rows": ledger["rows"],
        "attributions": ledger["attributions"],
        "final_closure_max_abs": ledger["final_closure_max_abs"],
        "ledger_pass": ledger["pass"],
    }


def write_structural_reports(structural_dir: Path, state_rows: list[dict]) -> dict:
    structural_dir = Path(structural_dir)
    structural_dir.mkdir(parents=True, exist_ok=True)
    flat = []
    for state in state_rows:
        for row in state["rows"]:
            flat.append(
                {
                    "sample_key": state["sample_key"],
                    "calibration_sample_id": state["calibration_sample_id"],
                    "source": state["source"],
                    "split": state["split"],
                    "prefix_length_j": state["prefix_length_j"],
                    "kl_exact": state["kl_exact"],
                    **row,
                }
            )
    write_jsonl(structural_dir / "structural_metrics.jsonl", flat)
    write_jsonl(structural_dir / "residual_ledger_rows.jsonl", flat)
    # sample-level aggregation for CI
    by_sample: dict[str, list[float]] = {}
    for state in state_rows:
        by_sample.setdefault(state["calibration_sample_id"], []).append(float(state["kl_exact"]))
    sample_means = [sum(v) / len(v) for v in by_sample.values()]
    summary = {
        "n_states": len(state_rows),
        "n_samples": len(by_sample),
        "kl_exact_sample_bootstrap": sample_bootstrap_ci(sample_means, seed=20260909) if sample_means else None,
        "median_kappa_all": float(torch.tensor([s["kappa_all"] for s in state_rows if s["kappa_all"] is not None]).median())
        if any(s["kappa_all"] is not None for s in state_rows)
        else None,
    }
    atomic_write_json(structural_dir / "residual_ledger_summary.json", summary)
    report = structural_dir / "STRUCTURAL_ACCUMULATION_REPORT.md"
    report.write_text(
        "# Structural Accumulation Report\n\n"
        f"- states: {summary['n_states']}\n"
        f"- samples: {summary['n_samples']}\n"
        f"- median kappa_all: {summary['median_kappa_all']}\n"
        f"- KL sample bootstrap: {summary['kl_exact_sample_bootstrap']}\n",
        encoding="utf-8",
    )
    gate = structural_dir / "RESIDUAL_LEDGER_GATE.md"
    gate.write_text(
        "# Residual Ledger Gate\n\n"
        f"- all_states_pass: {all(bool(s['ledger_pass']) for s in state_rows)}\n"
        f"- n_states: {len(state_rows)}\n",
        encoding="utf-8",
    )
    return summary
