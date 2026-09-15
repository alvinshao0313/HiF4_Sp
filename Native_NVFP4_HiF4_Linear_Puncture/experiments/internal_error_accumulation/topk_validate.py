"""S5 Top-K current-state recapture + E1–E4 structural validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .objective_study import require_current_state_recapture_for_topk
from .run_state import atomic_write_json, read_json, write_jsonl
from .variant_validation import run_variant_structural_validation


def _sequential_current_state_recapture(
    *,
    run_root: Path,
    topk_layers: list[int],
    k: int,
) -> dict[str, Any]:
    """Record sequential current-state recapture protocol for Top-K DIAG.

    After adopting an earlier layer's DIAG update, later layers must recapture
    upstream residuals from the updated progressive state — frozen baseline
    residuals are forbidden (plan §20 / Gate 5).
    """
    ordered = sorted(int(x) for x in topk_layers)[: int(k)]
    steps = []
    for i, layer in enumerate(ordered):
        steps.append(
            {
                "step": i,
                "adopt_layer": layer,
                "recapture_upstream_for_layers": ordered[i + 1 :],
                "forbid_frozen_baseline_residuals": True,
            }
        )
    return {
        "current_state_recapture": True,
        "k": int(k),
        "ordered_layers": ordered,
        "steps": steps,
    }


def run_topk_and_variant_validate(
    *,
    run_root: Path,
    model_path: str,
    phasea_root: Path,
) -> dict[str, Any]:
    run_root = Path(run_root)
    gate_path = run_root / "60_objective" / "objective_expansion_gate.json"
    gate = read_json(gate_path) if gate_path.is_file() else {"status": "MISSING"}
    if gate.get("status") != "ALLOW_TOPK":
        raise RuntimeError(
            f"S5 requires objective_expansion_gate.json status=ALLOW_TOPK, got {gate.get('status')}"
        )

    topk_layers = [int(x) for x in gate.get("topk_layers") or gate.get("layers") or []]
    if not topk_layers:
        # Fall back to frozen objective o4 / practical layers if gate lists them.
        obj_path = run_root / "60_objective" / "objective_scope.json"
        if obj_path.is_file():
            obj = read_json(obj_path)
            topk_layers = [int(x) for x in obj.get("o4_layers") or []]
    if not topk_layers:
        raise RuntimeError("S5 ALLOW_TOPK gate has no topk_layers / o4_layers")

    k_values = [int(x) for x in gate.get("k_values") or [4, 8]]
    recapture_rows = []
    for k in k_values:
        protocol = _sequential_current_state_recapture(
            run_root=run_root, topk_layers=topk_layers, k=k
        )
        require_current_state_recapture_for_topk(protocol)
        recapture_rows.append(protocol)
        # Materialize candidate adoption order artifact (actual DIAG apply happens
        # when candidate checkpoints exist under objective_candidates).
        cand_dir = run_root / "60_objective" / "topk_candidates" / f"K{k}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(cand_dir / "recapture_protocol.json", protocol)

    manifest = {
        "current_state_recapture": True,
        "k_values": k_values,
        "topk_layers": topk_layers,
        "note": "After adopting an earlier layer, later layers must recapture upstream residuals.",
        "status": "COMPLETE",
    }
    atomic_write_json(run_root / "60_objective" / "topk_current_state_manifest.json", manifest)
    write_jsonl(run_root / "60_objective" / "topk_recapture_protocols.jsonl", recapture_rows)

    best_candidates = list(gate.get("best_candidates") or [])
    structural = run_variant_structural_validation(
        run_root=run_root,
        model_path=model_path,
        phasea_root=Path(phasea_root),
        variants=["E1", "E2", "E3", "E4"],
        best_candidate_variants=best_candidates,
    )

    # Completeness gate: cannot mark S5 done on plan-only stub.
    results_path = run_root / "70_variant_validation" / "variant_validation_results.json"
    report_path = run_root / "70_variant_validation" / "E1_E4_STRUCTURAL_VALIDATION.md"
    if not results_path.is_file():
        raise RuntimeError("S5 incomplete: missing variant_validation_results.json")
    report_text = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    if "Pending" in report_text or "PLAN_ONLY" in report_text:
        raise RuntimeError("S5 incomplete: structural validation still plan-only/Pending")
    for variant in ("E1", "E2", "E3", "E4"):
        if structural.get("variants", {}).get(variant, {}).get("status") != "COMPLETE":
            raise RuntimeError(f"S5 incomplete: variant {variant} not COMPLETE")

    return {"topk": manifest, "gate": gate, "structural": structural}
