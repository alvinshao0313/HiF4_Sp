"""S4 practical protection + formal O0/O1/O2/O3_full/O3_topk/O4 scaffolding."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .objective_study import (
    assert_formal_objective_scope,
    decide_objective_scopes,
)
from .run_state import atomic_write_json, read_json


def _require_frozen(path: Path, name: str) -> dict:
    payload = read_json(path)
    if payload.get("status") != "FROZEN":
        raise RuntimeError(f"{name} must be FROZEN before S4, got {payload.get('status')}")
    return payload


def run_protection_and_objectives(
    *,
    run_root: Path,
    model_path: str,
    phasea_root: Path,
) -> dict[str, Any]:
    run_root = Path(run_root)
    practical = _require_frozen(
        run_root / "50_protection" / "practical_protection_scope.json",
        "practical_protection_scope",
    )
    objective = _require_frozen(run_root / "60_objective" / "objective_scope.json", "objective_scope")
    _require_frozen(
        run_root / "00_protocol" / "objective_split_manifest.json",
        "objective_split_manifest",
    )
    o3_eligibility = assert_formal_objective_scope(objective)
    selected_objective_layers = [int(x) for x in objective.get("selected_objective_layers", [])]
    if not selected_objective_layers:
        raise RuntimeError("FROZEN objective_scope missing selected_objective_layers")
    # Full o3_layers is eligibility only; recipes/teacher-cache use focused router subset.
    router_o3 = [int(x) for x in objective.get("router_objective_layers", [])]
    if not router_o3:
        router_o3 = sorted(set(o3_eligibility) & set(selected_objective_layers))
    if set(router_o3) - set(o3_eligibility):
        raise RuntimeError(
            "router_objective_layers must be subset of o3_layers eligibility; "
            f"router={router_o3} eligibility={o3_eligibility}"
        )
    if set(router_o3) - set(selected_objective_layers):
        raise RuntimeError(
            "router_objective_layers must be subset of selected_objective_layers; "
            f"router={router_o3} selected={selected_objective_layers}"
        )
    causal = {int(k): v for k, v in objective["causal_source_by_layer"].items()}
    if set(causal) != set(selected_objective_layers):
        raise RuntimeError(
            "objective causal_source_by_layer must contain exactly selected_objective_layers; "
            f"causal={sorted(causal)} selected={sorted(selected_objective_layers)}"
        )
    scopes = decide_objective_scopes(causal, o3_layers=router_o3)
    atomic_write_json(
        run_root / "60_objective" / "decided_diag_scopes.json",
        {
            "scopes": {str(k): v for k, v in scopes.items()},
            "o3_layers_eligibility": o3_eligibility,
            "o3_layers": router_o3,
            "router_objective_layers": router_o3,
        },
    )

    # Practical protection: if local deployable path cannot be guaranteed, hard-block.
    try:
        from .practical_protection import run_practical_protection_budget

        prot = run_practical_protection_budget(
            run_root=run_root,
            practical=practical,
            model_path=model_path,
            phasea_root=phasea_root,
        )
    except Exception as exc:  # noqa: BLE001
        from .layer_protection import mark_practical_protection_blocked

        mark_practical_protection_blocked(run_root / "50_protection", str(exc))
        prot = {"status": "PRACTICAL_PROTECTION_BLOCKED", "reason": str(exc)}

    from .objective_train import run_formal_objective_ablation

    obj = run_formal_objective_ablation(
        run_root=run_root,
        scopes=scopes,
        o3_layers=router_o3,
        o4_layers=[int(x) for x in objective.get("o4_layers", [])][:4],
        model_path=model_path,
        phasea_root=phasea_root,
        router_lambda=float(objective.get("router_lambda", 0.1)),
    )
    (run_root / "60_objective" / "OBJECTIVE_TRAINING_ABLATION_REPORT.md").write_text(
        "# Objective Training Ablation\n\n"
        f"- practical: {prot.get('status')}\n"
        f"- objectives: {obj.get('completed_objectives')}\n"
        f"- o3_layers_eligibility: {o3_eligibility}\n"
        f"- o3_recipe_layers: {router_o3}\n"
        f"- o4_layers: {objective.get('o4_layers')}\n",
        encoding="utf-8",
    )
    return {"protection": prot, "objectives": obj}
