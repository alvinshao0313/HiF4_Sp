from __future__ import annotations

import json
from pathlib import Path
from typing import Any
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
    run_aime25_avg5_vllm,
    run_mmlu_pro_300_vllm,
    vllm_fake_act_for_variant,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.lcb_runner import (
    run_livecodebench_vllm,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.lm_eval_vllm import (
    build_lm_eval_vllm_kwargs,
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
    assert summary["best_online_preset"] == "online_r64_then_diag"
    assert summary["best_overall_preset"] == "fusable_r64"


def test_candidate_diagnostic_excluded_from_best_ranking(tmp_path):
    adopted = _write_run(
        tmp_path,
        "E3_fusable",
        {
            "diag_mode": "fusable",
            "use_r64": False,
            "rot_order": "diag_then_rot",
            "artifact_diag_variant": "adopted",
            "loss_rollback": "on",
            "router_rollback": "on",
            "router_align_loss_weight": 0.0,
        },
        {"mmlu_pro_300": 0.20, "arc_challenge": 0.20, "arc_easy": 0.20},
    )
    candidate = _write_run(
        tmp_path,
        "E3_fusable_candidate",
        {
            "diag_mode": "fusable",
            "use_r64": False,
            "rot_order": "diag_then_rot",
            "artifact_diag_variant": "candidate",
            "loss_rollback": "on",
            "router_rollback": "on",
            "router_align_loss_weight": 0.0,
        },
        {"mmlu_pro_300": 0.99, "arc_challenge": 0.99, "arc_easy": 0.99},
    )
    summary = summarize_runs(
        {
            "E3_fusable": str(adopted),
            "E3_fusable_candidate": str(candidate),
        }
    )
    assert summary["best_fusable"]["method"] == "E3_fusable"
    assert summary["best_fusable_preset"] == "fusable"
    assert any(r["is_candidate_diagnostic"] for r in summary["rows"])


def test_layer_stats_split_rollback_fields(tmp_path):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.summarize import (
        layer_stats,
    )

    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "accepted": False,
                        "rollback": True,
                        "loss_would_rollback": False,
                        "loss_rollback_applied": False,
                        "router_would_rollback": True,
                        "router_rollback_applied": True,
                        "candidate_best_val_loss": 0.2,
                        "adopted_val_loss": 0.3,
                        "candidate_best_router_kl": 0.1,
                        "adopted_router_kl": 0.0,
                        "candidate_router_topk_mismatch_tokens": 4,
                        "candidate_router_topk_mismatch_ratio": 0.5,
                        "router_topk_mismatch_tokens": 0,
                        "router_topk_mismatch_ratio": 0.0,
                        "identity_val_loss": 0.4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    stats = layer_stats(run)
    assert stats["accepted_count"] == 0
    assert stats["rollback_count"] == 1
    assert stats["loss_would_rollback_count"] == 0
    assert stats["router_would_rollback_count"] == 1
    assert stats["router_rollback_applied_count"] == 1
    assert stats["candidate_router_topk_mismatch_tokens"] == 4
    assert stats["mean_recovery"] is not None


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


def test_reasoning_eval_uses_official_qwen3_thinking_sampling(tmp_path):
    spec = mock.Mock()
    spec.model_path = tmp_path / "model"
    spec.fake_act_quant = "none"
    spec.hif4_runtime_spec_path = None
    spec.native_nvfp4 = True
    module = (
        "Native_NVFP4_HiF4_Linear_Puncture.experiments."
        "e2e_diag_reconstruction.evaluation.vllm_runner"
    )
    with mock.patch(f"{module}.resolve_vllm_eval_spec", return_value=spec), mock.patch(
        f"{module}.run_main_py_lighteval", return_value={"results": {}}
    ) as run_eval:
        run_mmlu_pro_300_vllm(
            variant="native_nvfp4",
            output_dir=tmp_path / "mmlu",
        )
        mmlu = run_eval.call_args.kwargs
        assert mmlu["max_new_tokens"] == 32768
        assert mmlu["temperature"] == 0.6
        assert mmlu["top_p"] == 0.95
        assert mmlu["top_k"] == 20
        assert mmlu["min_p"] == 0.0
        assert mmlu["disable_thinking"] is False

        run_eval.reset_mock()
        run_aime25_avg5_vllm(
            variant="native_nvfp4",
            output_dir=tmp_path / "aime",
        )
        aime = run_eval.call_args.kwargs
        assert aime["max_new_tokens"] == 38912
        assert aime["temperature"] == 0.6
        assert aime["top_p"] == 0.95
        assert aime["top_k"] == 20
        assert aime["min_p"] == 0.0
        assert aime["disable_thinking"] is False


def test_livecodebench_uses_official_qwen3_thinking_sampling(tmp_path):
    spec = mock.Mock()
    spec.model_path = tmp_path / "model"
    spec.fake_act_quant = "none"
    spec.hif4_runtime_spec_path = None
    spec.native_nvfp4 = True
    module = (
        "Native_NVFP4_HiF4_Linear_Puncture.experiments."
        "e2e_diag_reconstruction.evaluation.lcb_runner"
    )
    with mock.patch(f"{module}.resolve_vllm_eval_spec", return_value=spec), mock.patch(
        f"{module}.run_main_py_lighteval", return_value={"results": {}}
    ) as run_eval, mock.patch(f"{module}.cleanup_materialized_eval_spec"):
        run_livecodebench_vllm(
            variant="native_nvfp4",
            output_dir=tmp_path / "lcb",
        )
        lcb = run_eval.call_args.kwargs
        assert lcb["datasets"] == "lcb:codegeneration_v6|0"
        assert lcb["max_samples"] is None
        assert lcb["max_new_tokens"] == 38912
        assert lcb["temperature"] == 0.6
        assert lcb["top_p"] == 0.95
        assert lcb["top_k"] == 20
        assert lcb["min_p"] == 0.0
        assert lcb["disable_thinking"] is False


def test_reasoning_lighteval_cli_locks_max_model_length_40960(tmp_path):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation import (
        vllm_runner,
    )

    out = tmp_path / "out"
    out.mkdir()
    (out / "results_dummy.json").write_text(
        json.dumps({"results": {"mmlu_pro": {"acc": 0.1}}}),
        encoding="utf-8",
    )
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return mock.Mock(returncode=0)

    with mock.patch.object(vllm_runner, "require_visible_cuda_count", return_value=2), mock.patch(
        "subprocess.run", side_effect=fake_run
    ):
        payload = vllm_runner.run_main_py_lighteval(
            model_path=tmp_path / "model",
            output_dir=out,
            datasets="mmlu_pro|0",
            max_samples=300,
            max_new_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            fake_act_quant="none",
            disable_thinking=False,
        )
    cmd = captured["cmd"]
    assert "--max_model_length" in cmd
    assert cmd[cmd.index("--max_model_length") + 1] == "40960"
    assert "--disable_thinking" not in cmd
    assert cmd[cmd.index("--temperature") + 1] == "0.6"
    assert cmd[cmd.index("--top_p") + 1] == "0.95"
    assert cmd[cmd.index("--top_k") + 1] == "20"
    assert cmd[cmd.index("--min_p") + 1] == "0.0"
    assert payload["max_model_length"] == 40960
    assert payload["enable_thinking"] is True


def test_vllm_fake_act_mapping():
    assert vllm_fake_act_for_variant("native_nvfp4") == "none"
    assert vllm_fake_act_for_variant("direct_hif4") == "none"
    assert vllm_fake_act_for_variant("artifact") == "none"
    assert needs_materialized_checkpoint("direct_hif4") is True
    assert needs_materialized_checkpoint("artifact") is True


def test_lm_eval_vllm_kwargs_native_and_sidecar(tmp_path):
    sidecar = tmp_path / "hif4_runtime_spec.pt"
    torch.save({"variant": "fusable"}, sidecar)
    native = build_lm_eval_vllm_kwargs(model_path="native", native_nvfp4=True)
    assert native["tensor_parallel_size"] == 2
    assert native["kv_cache_dtype"] == "bfloat16"
    assert native["enforce_eager"] is True
    assert native["linear_backend"] == "emulation"
    assert native["moe_backend"] == "emulation"
    assert native["seed"] == 42
    converted = build_lm_eval_vllm_kwargs(
        model_path="converted", hif4_runtime_spec_path=str(sidecar), seed=7
    )
    assert converted["additional_config"]["hif4_runtime_spec_path"] == str(sidecar)
    assert converted["moe_backend"] == "triton"
    assert converted["seed"] == 7


def test_lm_eval_vllm_generate_compat_maps_prompt_token_ids():
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.lm_eval_vllm import (
        patch_lm_eval_vllm_generate_compat,
    )

    calls: list[tuple[Any, dict[str, Any]]] = []

    class _FakeLLM:
        def generate(self, prompts=None, sampling_params=None, **kwargs):
            calls.append((prompts, {"sampling_params": sampling_params, **kwargs}))
            return ["ok"]

    import vllm

    original = vllm.LLM.generate
    try:
        vllm.LLM.generate = _FakeLLM.generate  # type: ignore[method-assign]
        patch_lm_eval_vllm_generate_compat()
        out = vllm.LLM.generate(
            _FakeLLM(),
            prompt_token_ids=[[1, 2, 3], [4, 5]],
            sampling_params="sp",
            use_tqdm=False,
        )
        assert out == ["ok"]
        assert calls[0][0] == [[1, 2, 3], [4, 5]]
        assert calls[0][1]["sampling_params"] == "sp"
        assert calls[0][1]["use_tqdm"] is False
        # idempotent
        patch_lm_eval_vllm_generate_compat()
    finally:
        vllm.LLM.generate = original  # type: ignore[method-assign]


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
