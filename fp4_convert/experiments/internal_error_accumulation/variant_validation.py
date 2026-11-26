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
    best_candidate_variants: list[dict] | None = None,
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
    candidate_by_name = {v['name']: v for v in best_candidate_variants}
    if len(candidate_by_name) != len(best_candidate_variants) or set(candidate_by_name) & set(variants):
        raise ValueError('duplicate candidate names')
    all_variants = variants + list(candidate_by_name)
    from .objective_holdout import run_capture_job
    from .candidate_runtime import sha256

    # E0 reference is required for pair analysis; reuse S1 capture if present.
    e0_capture = run_root / "10_capture" / "E0"
    if not (e0_capture / "hooks").is_dir():
        raise RuntimeError('S5 requires completed S1 E0 capture')

    summary: dict[str, Any] = {"variants": {}, "n_states": len(states)}
    for variant in all_variants:
        v_root = out_dir / "captures" / variant
        runtime_variant = 'E1' if variant in candidate_by_name else variant
        job = {'variant': runtime_variant, 'label': variant, 'model_path': model_path,
               'phasea_root': str(phasea_root), 'cohort': str(cohort_path),
               'cohort_sha256': sha256(cohort_path), 'output_root': str(v_root),
               'reference_root': str(e0_capture), 'layers': [], 'causal_source_by_layer': {}}
        if variant in candidate_by_name:
            candidate = candidate_by_name[variant]
            for entry in candidate['checkpoints'].values():
                if sha256(Path(entry['path'])) != entry['sha256']:
                    raise RuntimeError('candidate checkpoint changed before validation')
            job['materialized_model_path'] = candidate['model_dir']
            job['checkpoints'] = candidate['checkpoints']
        run_capture_job(job, log_path=run_root / f'logs/structural_{variant}.log')
        structural_rows = []
        for meta in states:
            key = meta["sample_key"]
            di = int(meta["decode_index"])
            e0_recs = load_rank_records(e0_capture / "hooks", "E0", key, 0)
            v_recs = load_rank_records(v_root / "hooks", runtime_variant, key, 0)
            e0_logits = load_raw_logits(e0_capture / "raw_logits", "E0", key, di, 0)
            v_logits = load_raw_logits(v_root / "raw_logits", runtime_variant, key, di, 0)
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
        if not all(r['ledger_pass'] for r in structural_rows):
            raise RuntimeError('structural residual identity failed')
        kappas = [float(r['kappa_all']) for r in structural_rows]
        kls = [float(r['kl_exact']) for r in structural_rows]
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
