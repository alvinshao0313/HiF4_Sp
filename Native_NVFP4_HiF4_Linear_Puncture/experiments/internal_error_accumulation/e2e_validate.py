"""S6 external MMLU-Pro300 validation for at most 1–2 selected candidates."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .run_state import atomic_write_json, read_json


def run_e2e_mmlu_pro(
    *,
    run_root: Path,
    model_path: str,
    phasea_root: Path,
) -> dict[str, Any]:
    """Actually invoke audited MMLU-Pro300 runner. Plan-only JSON is not completion."""
    run_root = Path(run_root)
    gate_path = run_root / "70_variant_validation" / "e2e_candidate_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("missing e2e_candidate_gate.json")
    gate = read_json(gate_path)
    if gate.get("status") != "ALLOW_E2E":
        raise RuntimeError(f"E2E gate not ALLOW_E2E: {gate.get('status')}")
    candidates = list(gate.get("candidates") or [])
    if not 1 <= len(candidates) <= 2:
        raise RuntimeError(f"E2E allows 1–2 candidates, got {len(candidates)}")

    plan = {
        "task": "mmlu_pro|0",
        "max_samples": 300,
        "tool": "lighteval_via_main.py",
        "candidates": candidates,
        "no_default_lcb": True,
        "model_path": model_path,
        "phasea_root": str(phasea_root),
        "status": "RUNNING",
    }
    out_root = run_root / "70_variant_validation" / "e2e_mmlu_pro"
    out_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_root / "70_variant_validation" / "e2e_mmlu_pro_plan.json", plan)

    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner import (
        run_mmlu_pro_300_vllm,
    )

    results: dict[str, Any] = {"candidates": {}}
    for cand in candidates:
        if isinstance(cand, dict):
            name = str(cand.get("name") or cand.get("variant") or cand.get("id"))
            eval_variant = str(cand.get("eval_variant") or cand.get("variant") or "direct_hif4")
            artifact_path = cand.get("artifact_path")
            artifact_diag_variant = str(cand.get("artifact_diag_variant") or "adopted")
        else:
            name = str(cand)
            eval_variant = "direct_hif4"
            artifact_path = None
            artifact_diag_variant = "adopted"
            # Map formal E-names to eval variants when provided as plain strings.
            mapping = {
                "E0": "native_nvfp4",
                "E1": "direct_hif4",
                "E2": "r64_only",
                "E3": "artifact",
                "E4": "artifact",
                "native_nvfp4": "native_nvfp4",
                "direct_hif4": "direct_hif4",
                "r64_only": "r64_only",
            }
            if name in mapping:
                eval_variant = mapping[name]

        cand_out = out_root / name
        cand_out.mkdir(parents=True, exist_ok=True)
        metrics = run_mmlu_pro_300_vllm(
            variant=eval_variant,
            output_dir=cand_out,
            model_path=model_path,
            artifact_path=artifact_path,
            artifact_diag_variant=artifact_diag_variant,
            batch_size=128,
        )
        metrics_path = cand_out / "eval" / "mmlu_pro" / "metrics.json"
        if not metrics_path.is_file():
            raise RuntimeError(f"S6 incomplete: missing MMLU-Pro metrics for {name}")
        results["candidates"][name] = {
            "eval_variant": eval_variant,
            "metrics_path": str(metrics_path),
            "metrics": metrics,
            "status": "COMPLETE",
        }

    results["status"] = "COMPLETE"
    results["task"] = "mmlu_pro|0"
    results["max_samples"] = 300
    atomic_write_json(out_root / "e2e_mmlu_pro_results.json", results)
    plan["status"] = "COMPLETE"
    atomic_write_json(run_root / "70_variant_validation" / "e2e_mmlu_pro_plan.json", plan)

    # Completeness gate: plan-only is not enough.
    if not (out_root / "e2e_mmlu_pro_results.json").is_file():
        raise RuntimeError("S6 incomplete: missing e2e_mmlu_pro_results.json")
    return results
