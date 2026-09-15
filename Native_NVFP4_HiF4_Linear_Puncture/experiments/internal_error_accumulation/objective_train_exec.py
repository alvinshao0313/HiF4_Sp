"""Execute objective recipes via selected-layer trainer under frozen fairness constraints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .run_state import atomic_write_json, read_json
from .router_teacher_cache import build_router_teacher_cache


def _candidate_dir(run_root: Path, layer: int, loss_name: str) -> Path:
    return (
        Path(run_root)
        / "60_objective"
        / "objective_candidates"
        / f"L{int(layer):02d}"
        / str(loss_name)
    )


def _recipe_artifacts_complete(cand: Path) -> bool:
    required = [
        cand / "checkpoint.pt",
        cand / "train_metrics.jsonl",
        cand / "val_metrics.json",
        cand / "cost.json",
        cand / "recipe.json",
    ]
    return all(p.is_file() for p in required)


def _reuse_router_teacher_cache(
    *,
    cache_dir: Path,
    split: dict[str, Any],
    layers: list[int],
) -> dict[str, Any] | None:
    """Reuse COMPLETE cache only when layers and frozen split IDs match exactly."""
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.is_file():
        return None
    existing = read_json(manifest_path)
    if existing.get("status") != "COMPLETE":
        return None
    if sorted(int(x) for x in (existing.get("layers") or [])) != sorted(int(x) for x in layers):
        return None
    if list(existing.get("train_ids") or []) != list(split.get("train_ids") or []):
        return None
    if list(existing.get("val_ids") or []) != list(split.get("val_ids") or []):
        return None
    return existing


def execute_recipes(
    *,
    recipes: list[dict],
    output_dir: Path,
    model_path: str,
    phasea_root: Path,
    run_root: Path,
    o3_layers: list[int] | None = None,
    teacher_batch_size: int = 4,
) -> dict[str, Any]:
    """Run each formal recipe through train_selected_layer_recipe.

    O3 recipes require router teacher cache status=COMPLETE before training starts.
    A recipe is complete only when checkpoint + train/val metrics + cost exist.
    Already-complete recipes are skipped (resume-safe; no retrain).
    """
    output_dir = Path(output_dir)
    run_root = Path(run_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not recipes:
        return {"status": "EMPTY", "n_recipes": 0}

    # Teacher cache layers must follow actual O3 recipes, never full eligibility list.
    recipe_o3_layers = sorted(
        {
            int(r["layer"])
            for r in recipes
            if str(r.get("loss")) in {"O3_full", "O3_topk"}
        }
    )
    if o3_layers is not None:
        declared = sorted({int(x) for x in o3_layers})
        if set(recipe_o3_layers) - set(declared):
            raise RuntimeError(
                "O3 recipes outside declared o3_layers: "
                f"recipes={recipe_o3_layers} declared={declared}"
            )
    o3_layer_list = recipe_o3_layers

    split = read_json(run_root / "00_protocol" / "objective_split_manifest.json")
    if split.get("status") != "FROZEN":
        raise RuntimeError(
            f"objective_split_manifest must be FROZEN before training, got {split.get('status')}"
        )
    cache_dir = run_root / "60_objective" / "router_teacher_cache"
    cache_manifest = _reuse_router_teacher_cache(
        cache_dir=cache_dir,
        split=split,
        layers=o3_layer_list,
    )
    if cache_manifest is None:
        cache_manifest = build_router_teacher_cache(
            run_root=run_root,
            model_path=model_path,
            objective_split_manifest=split,
            layers=o3_layer_list,
            batch_size=int(teacher_batch_size),
        )
    else:
        cache_manifest = dict(cache_manifest)
        cache_manifest["reused"] = True

    configs = []
    for i, recipe in enumerate(recipes):
        cfg_path = output_dir / f"recipe_{i:03d}_L{recipe['layer']}_{recipe['loss']}.json"
        atomic_write_json(cfg_path, recipe)
        configs.append(str(cfg_path))

    plan: dict[str, Any] = {
        "status": "RUNNING",
        "n_recipes": len(recipes),
        "configs": configs,
        "model_path": model_path,
        "phasea_root": str(phasea_root),
        "router_teacher_cache": cache_manifest,
        "note": (
            "Selected-layer O0/O1/O2/O3_full/O3_topk/O4 share seed/init/steps/50:50 mix; "
            "final checkpoint only (no objective-specific best-epoch selection)."
        ),
    }
    atomic_write_json(output_dir / "exec_plan.json", plan)

    from .selected_layer_objective_trainer import train_selected_layer_recipe

    results = []
    skipped = []
    for recipe in recipes:
        loss_name = str(recipe["loss"])
        layer = int(recipe["layer"])
        cand = _candidate_dir(run_root, layer, loss_name)
        if _recipe_artifacts_complete(cand):
            skipped.append(f"L{layer}:{loss_name}")
            results.append(
                {
                    "status": "SKIPPED_COMPLETE",
                    "layer": layer,
                    "loss": loss_name,
                    "candidate_dir": str(cand),
                }
            )
            continue

        if loss_name in {"O3_full", "O3_topk"}:
            if cache_manifest.get("status") != "COMPLETE":
                raise RuntimeError(
                    f"O3 recipe requires router teacher cache COMPLETE, got {cache_manifest.get('status')}"
                )
            recipe = dict(recipe)
            recipe.setdefault("router_teacher_cache_dir", str(cache_dir))

        result = train_selected_layer_recipe(
            recipe=recipe,
            run_root=run_root,
            model_path=model_path,
            phasea_root=Path(phasea_root),
        )
        if not _recipe_artifacts_complete(cand):
            required = [
                cand / "checkpoint.pt",
                cand / "train_metrics.jsonl",
                cand / "val_metrics.json",
                cand / "cost.json",
                cand / "recipe.json",
            ]
            missing = [str(p) for p in required if not p.is_file()]
            raise RuntimeError(f"recipe incomplete, missing artifacts: {missing}")
        results.append(result)

    plan["status"] = "COMPLETE"
    plan["results"] = results
    plan["skipped_complete"] = skipped
    atomic_write_json(output_dir / "exec_plan.json", plan)
    return plan
