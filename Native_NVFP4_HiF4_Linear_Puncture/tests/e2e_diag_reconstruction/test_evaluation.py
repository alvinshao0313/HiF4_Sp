from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.common import (
    REASONING_EVAL_NUM_GPUS,
    require_visible_cuda_count,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.runner import (
    _set_eval_seed,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner import (
    export_switchable_linear,
    needs_materialized_checkpoint,
    vllm_fake_act_for_variant,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.summarize import (
    structure_preset,
    summarize_runs,
)


def test_eval_seed_sets_same_torch_sequence():
    _set_eval_seed(42)
    a = torch.rand(4)
    _set_eval_seed(42)
    b = torch.rand(4)
    assert torch.equal(a, b)
    _set_eval_seed(0)
    c = torch.rand(4)
    assert not torch.equal(a, c)


def test_structure_preset_mapping_unambiguous():
    assert structure_preset({"method": "E0_native_nvfp4"}) == "native_nvfp4"
    assert structure_preset({"method": "E1_direct_hif4"}) == "direct_hif4"
    assert structure_preset({"method": "E2_r64_only"}) == "r64_only"
    assert structure_preset({"method": "E3", "diag_mode": "fusable", "use_r64": False}) == "fusable"
    assert structure_preset({"method": "E4", "diag_mode": "fusable", "use_r64": True}) == "fusable_r64"
    assert structure_preset({"method": "E5", "diag_mode": "online", "use_r64": False}) == "online"
    assert (
        structure_preset(
            {
                "method": "E6",
                "diag_mode": "online",
                "use_r64": True,
                "rot_order": "diag_then_rot",
            }
        )
        == "online_diag_then_r64"
    )
    assert (
        structure_preset(
            {
                "method": "E7",
                "diag_mode": "online",
                "use_r64": True,
                "rot_order": "rot_then_diag",
            }
        )
        == "online_r64_then_diag"
    )


def _write_run(root: Path, name: str, cfg: dict, scores: dict) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    eval_arc = d / "eval" / "arc"
    eval_mmlu = d / "eval" / "mmlu_pro"
    eval_arc.mkdir(parents=True)
    eval_mmlu.mkdir(parents=True)
    (eval_arc / "metrics.json").write_text(
        json.dumps({"scores": {"arc_easy": scores["arc_easy"], "arc_challenge": scores["arc_challenge"]}}),
        encoding="utf-8",
    )
    (eval_mmlu / "metrics.json").write_text(
        json.dumps({"results": {"mmlu_pro": {"acc": scores["mmlu_pro_300"]}}}),
        encoding="utf-8",
    )
    return d


def test_phase_a_best_presets_follow_three_level_rule(tmp_path):
    e3 = _write_run(
        tmp_path,
        "E3_fusable",
        {"diag_mode": "fusable", "use_r64": False, "rot_order": "diag_then_rot"},
        {"mmlu_pro_300": 0.40, "arc_challenge": 0.50, "arc_easy": 0.80},
    )
    e4 = _write_run(
        tmp_path,
        "E4_fusable_r64",
        {"diag_mode": "fusable", "use_r64": True, "rot_order": "diag_then_rot"},
        {"mmlu_pro_300": 0.40, "arc_challenge": 0.55, "arc_easy": 0.10},
    )
    e5 = _write_run(
        tmp_path,
        "E5_online",
        {"diag_mode": "online", "use_r64": False, "rot_order": "diag_then_rot"},
        {"mmlu_pro_300": 0.30, "arc_challenge": 0.90, "arc_easy": 0.90},
    )
    e6 = _write_run(
        tmp_path,
        "E6_online_diag_then_r64",
        {"diag_mode": "online", "use_r64": True, "rot_order": "diag_then_rot"},
        {"mmlu_pro_300": 0.31, "arc_challenge": 0.10, "arc_easy": 0.10},
    )
    e7 = _write_run(
        tmp_path,
        "E7_online_r64_then_diag",
        {"diag_mode": "online", "use_r64": True, "rot_order": "rot_then_diag"},
        {"mmlu_pro_300": 0.31, "arc_challenge": 0.10, "arc_easy": 0.20},
    )
    summary = summarize_runs(
        {
            "E3_fusable": str(e3),
            "E4_fusable_r64": str(e4),
            "E5_online": str(e5),
            "E6_online_diag_then_r64": str(e6),
            "E7_online_r64_then_diag": str(e7),
        }
    )
    assert summary["best_fusable_preset"] == "fusable_r64"
    assert summary["best_online_preset"] == "online"
    assert summary["best_overall_preset"] == "online"


def test_reasoning_eval_requires_two_gpus():
    assert REASONING_EVAL_NUM_GPUS == 2
    with pytest.raises(RuntimeError, match="need at least 2"):
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.device_count", return_value=1
        ):
            require_visible_cuda_count(REASONING_EVAL_NUM_GPUS)
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
        "torch.cuda.device_count", return_value=2
    ):
        assert require_visible_cuda_count(REASONING_EVAL_NUM_GPUS) == 2


def test_vllm_fake_act_mapping():
    assert vllm_fake_act_for_variant("native_nvfp4") == "nvfp4"
    assert vllm_fake_act_for_variant("direct_hif4") == "hif4"
    assert vllm_fake_act_for_variant("artifact") == "hif4"
    assert needs_materialized_checkpoint("direct_hif4") is False
    assert needs_materialized_checkpoint("artifact") is True


def test_export_switchable_linear_uses_weight_shape():
    mod = mock.Mock()
    mod._mode = "hif4_eval"
    mod._folded_weight_fp32 = None
    mod.transformed_master_weight.return_value = torch.ones(4, 8, dtype=torch.float32)
    mod.bias = None
    with mock.patch(
        "Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner.quant_weight",
        side_effect=lambda w, use_ste=False: w.to(torch.bfloat16),
    ):
        linear = export_switchable_linear(mod)
    assert tuple(linear.weight.shape) == (4, 8)
