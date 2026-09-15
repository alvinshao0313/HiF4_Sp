"""Completeness gates: S4/S5/S6 must not mark completed on plan-only / stub paths."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.e2e_validate import (
    run_e2e_mmlu_pro,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.objective_train_exec import (
    execute_recipes,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.practical_protection import (
    run_practical_protection_budget,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.run_state import (
    atomic_write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.topk_validate import (
    run_topk_and_variant_validate,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.variant_validation import (
    write_variant_validation_stub,
)


def test_s4_skips_already_complete_recipes(tmp_path: Path):
    """Resume path must not retrain recipes that already have full artifacts."""
    run_root = tmp_path / "run"
    (run_root / "00_protocol").mkdir(parents=True)
    out = run_root / "60_objective" / "objective_candidates"
    atomic_write_json(
        run_root / "00_protocol" / "objective_split_manifest.json",
        {
            "status": "FROZEN",
            "train_ids": ["a"],
            "val_ids": ["b"],
            "source_ratio": {"wikitext2": 0.5, "s1k_original": 0.5},
        },
    )
    done = out / "L31" / "O2_M"
    done.mkdir(parents=True)
    recipe = {
        "layer": 31,
        "loss": "O2_M",
        "params": ["D_GU", "D_UD"],
        "train_ids": ["a"],
        "val_ids": ["b"],
        "steps": 1,
        "seed": 0,
        "lr": 1e-3,
    }
    atomic_write_json(done / "recipe.json", recipe)
    (done / "train_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    atomic_write_json(done / "val_metrics.json", {})
    atomic_write_json(done / "cost.json", {"steps": 1})
    (done / "checkpoint.pt").write_bytes(b"ckpt")

    train_calls: list[str] = []

    def _fake_train(*, recipe, run_root, model_path, phasea_root):
        train_calls.append(f"L{recipe['layer']}:{recipe['loss']}")
        raise RuntimeError("should not train complete recipe")

    with mock.patch(
        "Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation."
        "objective_train_exec.build_router_teacher_cache",
        return_value={"status": "EMPTY", "layers": []},
    ), mock.patch(
        "Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation."
        "selected_layer_objective_trainer.train_selected_layer_recipe",
        side_effect=_fake_train,
    ):
        plan = execute_recipes(
            recipes=[recipe],
            output_dir=out,
            model_path="nvidia/Qwen3-30B-A3B-NVFP4",
            phasea_root=tmp_path,
            run_root=run_root,
            o3_layers=[],
        )
    assert plan["status"] == "COMPLETE"
    assert plan["skipped_complete"] == ["L31:O2_M"]
    assert train_calls == []


def test_s4_cannot_recipe_only_mark_completed(tmp_path: Path):
    """execute_recipes must call real trainer and require checkpoint artifacts."""
    run_root = tmp_path / "run"
    (run_root / "00_protocol").mkdir(parents=True)
    (run_root / "60_objective").mkdir(parents=True)
    atomic_write_json(
        run_root / "00_protocol" / "objective_split_manifest.json",
        {
            "status": "FROZEN",
            "train_ids": ["a"],
            "val_ids": ["b"],
            "source_ratio": {"wikitext2": 0.5, "s1k_original": 0.5},
        },
    )
    out = run_root / "60_objective" / "objective_candidates"
    recipes = [
        {
            "layer": 31,
            "loss": "O2_M",
            "params": ["D_GU", "D_UD"],
            "train_ids": ["a"],
            "val_ids": ["b"],
            "steps": 1,
            "seed": 0,
            "lr": 1e-3,
        }
    ]

    def _fake_train(*, recipe, run_root, model_path, phasea_root):
        # Intentionally omit checkpoint → must fail completeness
        layer = int(recipe["layer"])
        loss = recipe["loss"]
        cand = run_root / "60_objective" / "objective_candidates" / f"L{layer:02d}" / loss
        cand.mkdir(parents=True, exist_ok=True)
        atomic_write_json(cand / "recipe.json", recipe)
        (cand / "train_metrics.jsonl").write_text("{}\n", encoding="utf-8")
        atomic_write_json(cand / "val_metrics.json", {})
        atomic_write_json(cand / "cost.json", {"steps": 1})
        return {"status": "INCOMPLETE_TEST"}

    with mock.patch(
        "Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation."
        "objective_train_exec.build_router_teacher_cache",
        return_value={"status": "EMPTY", "layers": []},
    ), mock.patch(
        "Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation."
        "selected_layer_objective_trainer.train_selected_layer_recipe",
        side_effect=_fake_train,
    ):
        with pytest.raises(RuntimeError, match="recipe incomplete"):
            execute_recipes(
                recipes=recipes,
                output_dir=out,
                model_path="nvidia/Qwen3-30B-A3B-NVFP4",
                phasea_root=tmp_path,
                run_root=run_root,
                o3_layers=[],
            )


def test_s4_practical_protection_not_bare_raise_stub(tmp_path: Path):
    """practical_protection must probe and record Gate-4 evidence, not only raise."""
    run_root = tmp_path / "run"
    (run_root / "50_protection").mkdir(parents=True)
    with mock.patch(
        "Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation."
        "practical_protection._probe_local_source_format_switch",
        return_value={
            "available": False,
            "has_per_layer_source_switch": False,
            "reason": "no verified local switch",
        },
    ):
        result = run_practical_protection_budget(
            run_root=run_root,
            practical={"layers": [44, 31], "scopes": ["moe_only"], "k_values": [1, 2]},
            model_path="nvidia/Qwen3-30B-A3B-NVFP4",
            phasea_root=tmp_path,
        )
    assert result["status"] == "PRACTICAL_PROTECTION_BLOCKED"
    assert (run_root / "50_protection" / "protection_budget_results.json").is_file()
    assert (run_root / "50_protection" / "practical_protection_path_probe.json").is_file()
    assert (run_root / "50_protection" / "PRACTICAL_PROTECTION_BLOCKED.json").is_file()


def test_s5_cannot_stub_mark_completed(tmp_path: Path):
    run_root = tmp_path / "run"
    (run_root / "60_objective").mkdir(parents=True)
    (run_root / "70_variant_validation").mkdir(parents=True)
    atomic_write_json(
        run_root / "60_objective" / "objective_expansion_gate.json",
        {"status": "ALLOW_TOPK", "topk_layers": [44, 31, 11, 12], "k_values": [4]},
    )
    # Only write plan stub — must not count as S5 complete.
    write_variant_validation_stub(run_root / "70_variant_validation")

    with mock.patch(
        "Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation."
        "topk_validate.run_variant_structural_validation",
        return_value={
            "variants": {
                "E1": {"status": "COMPLETE"},
                "E2": {"status": "COMPLETE"},
                "E3": {"status": "COMPLETE"},
                # E4 missing COMPLETE
            }
        },
    ):
        with pytest.raises(RuntimeError, match="S5 incomplete"):
            run_topk_and_variant_validate(
                run_root=run_root,
                model_path="nvidia/Qwen3-30B-A3B-NVFP4",
                phasea_root=tmp_path,
            )


def test_s6_cannot_plan_only_mark_completed(tmp_path: Path):
    run_root = tmp_path / "run"
    (run_root / "70_variant_validation").mkdir(parents=True)
    atomic_write_json(
        run_root / "70_variant_validation" / "e2e_candidate_gate.json",
        {"status": "ALLOW_E2E", "candidates": ["E1"]},
    )

    def _fake_mmlu(**kwargs):
        # Do not write metrics.json → must fail
        return {"acc": 0.0}

    with mock.patch(
        "Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction."
        "evaluation.vllm_runner.run_mmlu_pro_300_vllm",
        side_effect=_fake_mmlu,
    ):
        with pytest.raises(RuntimeError, match="S6 incomplete|missing MMLU-Pro metrics"):
            run_e2e_mmlu_pro(
                run_root=run_root,
                model_path="nvidia/Qwen3-30B-A3B-NVFP4",
                phasea_root=tmp_path,
            )
