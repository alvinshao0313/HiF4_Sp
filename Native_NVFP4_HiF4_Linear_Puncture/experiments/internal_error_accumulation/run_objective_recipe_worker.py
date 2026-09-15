"""Parallel selected-layer recipe worker. Does not touch run_state / S5."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.objective_train_exec import (
    _candidate_dir,
    _recipe_artifacts_complete,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.run_state import (
    atomic_write_json,
    read_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.selected_layer_objective_trainer import (
    train_selected_layer_recipe,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--phasea-root", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--exclude-o4", action="store_true")
    p.add_argument("--only-o4", action="store_true")
    return p.parse_args()


def _select_recipes(recipes: list[dict], *, shard: int, shards: int, exclude_o4: bool, only_o4: bool) -> list[dict]:
    if shard < 0 or shards <= 0 or shard >= shards:
        raise ValueError(f"invalid shard={shard} shards={shards}")
    if exclude_o4 and only_o4:
        raise ValueError("exclude-o4 and only-o4 are mutually exclusive")
    selected: list[dict] = []
    for recipe in recipes:
        loss = str(recipe["loss"])
        if exclude_o4 and loss == "O4":
            continue
        if only_o4 and loss != "O4":
            continue
        selected.append(recipe)
    return [r for i, r in enumerate(selected) if i % shards == shard]


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    recipe_file = run_root / "60_objective" / "objective_candidates" / "training_recipes.json"
    payload = read_json(recipe_file)
    recipes = list(payload["recipes"])
    mine = _select_recipes(
        recipes,
        shard=int(args.shard),
        shards=int(args.shards),
        exclude_o4=bool(args.exclude_o4),
        only_o4=bool(args.only_o4),
    )
    worker_id = f"shard{args.shard}of{args.shards}_pid{os.getpid()}"
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "worker_id": worker_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "n_assigned": len(mine),
        "assigned": [f"L{int(r['layer'])}:{r['loss']}" for r in mine],
        "status": "RUNNING",
    }
    atomic_write_json(log_dir / f"recipe_worker_{args.shard}.json", status)
    print(f"[recipe_worker] {status}", flush=True)

    results = []
    for recipe in mine:
        layer = int(recipe["layer"])
        loss = str(recipe["loss"])
        cand = _candidate_dir(run_root, layer, loss)
        tag = f"L{layer}:{loss}"
        if _recipe_artifacts_complete(cand):
            print(f"[recipe_worker] skip complete {tag}", flush=True)
            results.append({"tag": tag, "status": "SKIPPED_COMPLETE"})
            continue
        print(f"[recipe_worker] start {tag}", flush=True)
        out = train_selected_layer_recipe(
            recipe=recipe,
            run_root=run_root,
            model_path=args.model_path,
            phasea_root=Path(args.phasea_root),
        )
        if not _recipe_artifacts_complete(cand) and out.get("status") != "SKIPPED_COMPLETE":
            raise RuntimeError(f"worker recipe incomplete: {tag} status={out.get('status')}")
        print(f"[recipe_worker] done {tag} status={out.get('status')}", flush=True)
        results.append({"tag": tag, "status": out.get("status")})

    status["status"] = "COMPLETE"
    status["results"] = results
    atomic_write_json(log_dir / f"recipe_worker_{args.shard}.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
