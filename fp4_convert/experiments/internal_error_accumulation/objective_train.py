"""Formal O0/O1/O2/O3_full/O3_topk/O4 training ablation on 50:50 WikiText2:S1K."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import OBJECTIVE_TRAIN_SEED
from .objective_study import cumulative_residual_nmse, local_q_nmse
from .run_state import atomic_write_json, read_json


DEFAULT_ROUTER_LAMBDA = 0.1


def run_formal_objective_ablation(
    *,
    run_root: Path,
    scopes: dict[int, dict],
    o3_layers: list[int],
    o4_layers: list[int],
    model_path: str,
    phasea_root: Path,
    router_lambda: float = DEFAULT_ROUTER_LAMBDA,
    seed: int = OBJECTIVE_TRAIN_SEED,
    epochs: int = 20,
    lr: float = 5e-3,
    weight_decay: float = 0.0,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Train selected-layer DIAG under fair protocol.

    O0 is always retrained on the frozen 50:50 mix (never reuse pure-S1K history).
    O3_full/O3_topk recipes are emitted only for layers in o3_layers (via scopes).
    O4 runs only on Top-4 layers with final-KL through real subsequent network.
    """
    run_root = Path(run_root)
    split = read_json(run_root / "00_protocol" / "objective_split_manifest.json")
    out = run_root / "60_objective" / "objective_candidates"
    out.mkdir(parents=True, exist_ok=True)
    o3_set = {int(x) for x in o3_layers}
    o4_set = {int(x) for x in o4_layers}
    cache_dir = run_root / "60_objective" / "router_teacher_cache"

    completed = []
    recipes = []
    for layer, scope in scopes.items():
        layer_i = int(layer)
        for loss_name in scope["losses"]:
            if loss_name == "O3_conditional":
                raise RuntimeError("O3_conditional is forbidden in formal recipes")
            if loss_name in {"O3_full", "O3_topk"} and layer_i not in o3_set:
                continue
            if loss_name == "O4" and layer_i not in o4_set:
                continue
            recipe: dict[str, Any] = {
                "layer": layer_i,
                "loss": loss_name,
                "params": list(scope["params"]),
                "train_ids": list(split["train_ids"]),
                "val_ids": list(split["val_ids"]),
                "source_ratio": split.get("source_ratio"),
                "seed": int(seed),
                "epochs": int(epochs),
                "optimizer": "AdamW",
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "batch_size": int(batch_size),
                "calib_note": "formal O0/O1/O2/O3_full/O3_topk/O4 share this 50:50 mix",
            }
            if loss_name in {"O3_full", "O3_topk"}:
                recipe["router_lambda"] = float(router_lambda)
                recipe["router_teacher_cache_dir"] = str(cache_dir)
            recipes.append(recipe)
            completed.append(f"L{layer_i}:{loss_name}")

    atomic_write_json(
        out / "training_recipes.json",
        {
            "recipes": recipes,
            "o3_layers": [int(x) for x in o3_layers],
            "o4_layers": [int(x) for x in o4_layers],
            "router_lambda": float(router_lambda),
        },
    )
    _ = local_q_nmse
    _ = cumulative_residual_nmse

    from .objective_train_exec import execute_recipes

    exec_result = execute_recipes(
        recipes=recipes,
        output_dir=out,
        model_path=model_path,
        phasea_root=phasea_root,
        run_root=run_root,
        o3_layers=[int(x) for x in o3_layers],
    )
    return {"completed_objectives": completed, "exec": exec_result}
