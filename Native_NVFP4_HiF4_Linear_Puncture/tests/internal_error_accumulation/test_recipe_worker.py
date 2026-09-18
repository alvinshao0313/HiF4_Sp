"""Recipe worker sharding and skip-complete lock."""
from __future__ import annotations

from pathlib import Path
import pytest

from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.run_objective_recipe_worker import (
    _select_recipes,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.run_state import (
    atomic_write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.selected_layer_objective_trainer import (
    train_selected_layer_recipe,
)


def test_select_recipes_excludes_o4_and_shards():
    recipes = [
        {"layer": 39, "loss": "O0"},
        {"layer": 39, "loss": "O4"},
        {"layer": 41, "loss": "O0"},
        {"layer": 41, "loss": "O1_A"},
        {"layer": 43, "loss": "O0"},
    ]
    s0 = _select_recipes(recipes, shard=0, shards=2, exclude_o4=True, only_o4=False)
    s1 = _select_recipes(recipes, shard=1, shards=2, exclude_o4=True, only_o4=False)
    tags0 = [f"L{r['layer']}:{r['loss']}" for r in s0]
    tags1 = [f"L{r['layer']}:{r['loss']}" for r in s1]
    assert "L39:O4" not in tags0 + tags1
    assert sorted(tags0 + tags1) == ["L39:O0", "L41:O0", "L41:O1_A", "L43:O0"]
    assert not set(tags0) & set(tags1)


def test_trainer_rejects_legacy_completion_without_current_path_provenance(tmp_path: Path):
    run_root = tmp_path / "run"
    cand = run_root / "60_objective" / "objective_candidates" / "L39" / "O0"
    cand.mkdir(parents=True)
    recipe = {"layer": 39, "loss": "O0", "params": ["D_GU", "D_UD"], "train_ids": ["a"], "val_ids": ["b"]}
    atomic_write_json(cand / "recipe.json", recipe)
    (cand / "train_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    atomic_write_json(cand / "val_metrics.json", {})
    atomic_write_json(cand / "cost.json", {})
    (cand / "checkpoint.pt").write_bytes(b"ckpt")
    with pytest.raises(RuntimeError, match="legacy objective training is closed"):
        train_selected_layer_recipe(
            recipe=recipe, run_root=run_root, model_path="unused", phasea_root=tmp_path,
        )


def test_corrected_completion_detects_modified_checkpoint(tmp_path: Path):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.corrected_phase import (
        completion_record, validated_completion,
    )
    recipe = {"training_phase": "corrected_objectives_v3", "layer": 31, "loss": "O2_M"}
    for name in ("checkpoint.pt", "train_metrics.jsonl", "val_metrics.json", "cost.json", "recipe.json"):
        (tmp_path / name).write_text("original")
    assert not validated_completion(tmp_path, recipe)
    atomic_write_json(tmp_path / "complete.json", completion_record(tmp_path, recipe))
    assert validated_completion(tmp_path, recipe)
    (tmp_path / "checkpoint.pt").write_text("changed")
    with pytest.raises(RuntimeError, match="content/provenance changed"):
        validated_completion(tmp_path, recipe)
