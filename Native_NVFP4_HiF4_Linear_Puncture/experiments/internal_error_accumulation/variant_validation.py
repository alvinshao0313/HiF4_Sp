"""E1–E4 structural validation on the same 64-state calibration cohort."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .capture_states import capture_variant_states, load_rank_records, load_raw_logits
from .run_state import atomic_write_json, read_jsonl
from .structural_accumulation import analyze_state_pair, write_structural_reports


def write_variant_validation_stub(output_dir: Path, variants: list[str] | None = None) -> dict[str, Any]:
    """Historical S0 plan artifact only — S5 must not treat this as completion."""
    variants = variants or ["E1", "E2", "E3", "E4"]
    payload = {
        "variants": variants,
        "metrics": [
            "final_residual_nmse",
            "sum_positive_G",
            "kappa",
            "moe_positive_gain",
            "attention_positive_gain",
            "final_direction_concentration",
            "logit_kl",
        ],
        "note": "Plan-only stub for S0; S5 must call run_variant_structural_validation.",
        "status": "PLAN_ONLY_NOT_COMPLETE",
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_json(Path(output_dir) / "variant_validation_plan.json", payload)
    (Path(output_dir) / "E1_E4_STRUCTURAL_VALIDATION.md").write_text(
        "# E1–E4 Structural Validation\n\nPending S5 execution.\n",
        encoding="utf-8",
    )
    return payload


def run_variant_structural_validation(
    *,
    run_root: Path,
    model_path: str,
    phasea_root: Path,
    variants: list[str] | None = None,
    best_candidate_variants: list[str] | None = None,
) -> dict[str, Any]:
    """Capture + structural analyze E1/E2/E3/E4 (+ optional best candidates) on same 64-state cohort."""
    run_root = Path(run_root)
    out_dir = run_root / "70_variant_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = run_root / "00_protocol" / "internal_error_state_cohort.jsonl"
    if not cohort_path.is_file():
        raise RuntimeError(f"missing cohort for variant validation: {cohort_path}")
    states = read_jsonl(cohort_path)
    if len(states) != 64:
        raise RuntimeError(f"expected 64-state cohort, got {len(states)}")

    variants = list(variants or ["E1", "E2", "E3", "E4"])
    best_candidate_variants = list(best_candidate_variants or [])
    all_variants = list(dict.fromkeys(variants + best_candidate_variants))

    # E0 reference is required for pair analysis; reuse S1 capture if present.
    e0_capture = run_root / "10_capture" / "E0"
    if not (e0_capture / "hooks").is_dir():
        capture_variant_states(
            variant="E0",
            cohort_path=cohort_path,
            output_root=e0_capture,
            model_path=model_path,
            phasea_root=Path(phasea_root),
            capture_level="core",
            gpu_memory_utilization=0.90,
        )

    summary: dict[str, Any] = {"variants": {}, "n_states": len(states)}
    for variant in all_variants:
        v_root = out_dir / "captures" / variant
        capture_variant_states(
            variant=variant,
            cohort_path=cohort_path,
            output_root=v_root,
            model_path=model_path,
            phasea_root=Path(phasea_root),
            capture_level="core",
            gpu_memory_utilization=0.90,
        )
        structural_rows = []
        for meta in states:
            key = meta["sample_key"]
            di = int(meta["decode_index"])
            e0_recs = load_rank_records(e0_capture / "hooks", "E0", key, 0)
            v_recs = load_rank_records(v_root / "hooks", variant, key, 0)
            e0_logits = load_raw_logits(e0_capture / "raw_logits", "E0", key, di, 0)
            v_logits = load_raw_logits(v_root / "raw_logits", variant, key, di, 0)
            structural_rows.append(
                analyze_state_pair(
                    sample_meta=meta,
                    e0_records=e0_recs,
                    e1_records=v_recs,
                    e0_logits=e0_logits,
                    e1_logits=v_logits,
                )
            )
        write_structural_reports(out_dir / "structural" / variant, structural_rows)
        # Aggregate key metrics for the comparison table.
        kappas = [float(r.get("kappa_all") or r.get("kappa") or 0.0) for r in structural_rows]
        kls = [float(r.get("kl_exact") or r.get("logit_kl") or 0.0) for r in structural_rows]
        summary["variants"][variant] = {
            "n_states": len(structural_rows),
            "mean_kappa": sum(kappas) / max(len(kappas), 1),
            "mean_logit_kl": sum(kls) / max(len(kls), 1),
            "status": "COMPLETE",
        }

    atomic_write_json(out_dir / "variant_validation_results.json", summary)
    lines = [
        "# E1–E4 Structural Validation\n",
        f"- n_states: {len(states)}\n",
        f"- variants: {all_variants}\n",
        "- status: COMPLETE (actual-path capture + structural analysis)\n",
    ]
    for v, payload in summary["variants"].items():
        lines.append(
            f"- {v}: mean_kappa={payload['mean_kappa']:.6g}, "
            f"mean_logit_kl={payload['mean_logit_kl']:.6g}\n"
        )
    (out_dir / "E1_E4_STRUCTURAL_VALIDATION.md").write_text("".join(lines), encoding="utf-8")
    return summary
