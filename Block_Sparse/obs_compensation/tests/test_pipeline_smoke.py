from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from obs_compensation.config import OBSCompensationConfig
from obs_compensation.pipeline import run_obs_compensation
from obs_compensation.serialization import (
    get_incomplete_output_dir,
    save_obs_artifacts,
    save_obs_package_atomically,
    validate_atomic_output_paths,
    verify_fixed_masks_and_weights,
)
from obs_compensation.solver import ResolvedOBSOrderPolicy, resolve_obs_order_policy
from obs_compensation.tests.helpers import (
    TinyCausalLM,
    TinyTokenizer,
    make_block_masks_for_targets,
    make_descending_permutation_payload,
    make_targets_from_tiny,
    write_source_artifacts,
)


def _cfg(tmp_path, source, out, **kwargs):
    base = dict(
        model_path="tiny-model",
        source_artifacts_dir=source,
        output_dir=out,
        calibration_dataset="s1k",
        calibration_samples=2,
        sequence_length=8,
        obs_percdamp=0.01,
        solver_block_size=8,
        obs_order_policy="auto",
        dtype="float32",
        device="cpu",
        seed=0,
        trust_remote_code=True,
    )
    base.update(kwargs)
    return OBSCompensationConfig(**base)


def test_pipeline_call_order(monkeypatch, tmp_path):
    import obs_compensation.pipeline as pipe

    calls: list[str] = []

    class FakeArts:
        class metadata:
            model_path = "tiny-model"
            block_size = "2x2"
            block_height = 2
            block_width = 2
            target_block_sparsity = 0.1
            actual_block_sparsity = 0.1
            score_type = "magnitude"
            mlp_permutation = "none"
            residual_permutation = "none"
            num_pruning_rounds = 1

        root = tmp_path / "src"
        masks = {}
        permutation_payload = None
        raw_summary = {"model_path": "tiny-model"}

    def load_source_artifacts(root):
        calls.append("load_source_artifacts")
        return FakeArts()

    def resolve(requested_policy, mlp_permutation):
        calls.append("resolve_obs_order_policy")
        assert "load_model_and_tokenizer" not in calls
        return ResolvedOBSOrderPolicy(
            requested_policy, "standard", "left_to_right", "left_to_right"
        )

    def load_model_and_tokenizer(cfg):
        calls.append("load_model_and_tokenizer")
        assert "resolve_obs_order_policy" in calls
        model = TinyCausalLM()
        return model, TinyTokenizer()

    def collect_mlp_linears(model, block_height, block_width):
        calls.append("collect_mlp_linears")
        return make_targets_from_tiny(model, block_height, block_width)

    def validate_source_artifacts_against_targets(artifacts, targets):
        calls.append("validate_source_artifacts_against_targets")

    def group_mlp_projection_triplets(targets):
        calls.append("group_mlp_projection_triplets")
        from obs_compensation.permutation import group_mlp_projection_triplets as real

        return real(targets)

    def build_calibration_samples(tokenizer, config):
        calls.append("build_calibration_samples")
        from obs_compensation.calibration import make_calibration_sample

        return [
            make_calibration_sample(torch.tensor([[1, 2, 3, 4]], dtype=torch.long)),
            make_calibration_sample(torch.tensor([[5, 6, 7, 8]], dtype=torch.long)),
        ]

    def capture_first_decoder_layer_inputs(model, samples):
        calls.append("capture_first_decoder_layer_inputs")
        from obs_compensation.model_adapter import CapturedLayerInputs

        return CapturedLayerInputs(
            hidden_states=[torch.randn(1, 4, 4) for _ in samples],
            layer_kwargs=[{} for _ in samples],
        )

    def run_layerwise_mlp_obs(**kwargs):
        calls.append("run_layerwise_mlp_obs")
        from obs_compensation.layerwise import LayerOBSReport, LayerwiseOBSResult, ModuleOBSReport

        return LayerwiseOBSResult(
            module_reports=[
                ModuleOBSReport(
                    module_name="m",
                    layer_index=0,
                    projection_type="gate_proj",
                    solver_direction="left_to_right",
                    solver_applied=True,
                    skip_reason="",
                    num_total_blocks=1,
                    num_pruned_blocks=0,
                    block_sparsity=0.0,
                    num_fully_pruned_block_rows=0,
                    num_fully_pruned_output_rows=0,
                    num_hessian_tokens=1,
                    hessian_diagonal_mean=1.0,
                    hessian_damp_value=0.01,
                    hessian_dead_columns=0,
                    kept_delta_l2=0.0,
                    kept_delta_max_abs=0.0,
                    original_pruned_l2=0.0,
                    original_max_abs=1.0,
                    compensated_max_abs=1.0,
                )
            ],
            layer_reports=[
                LayerOBSReport(0, 1, 0.0, 0.0, 0.0, 0, 0)
            ],
        )

    def verify_fixed_masks_and_weights(**kwargs):
        calls.append("verify_fixed_masks_and_weights")

    def validate_atomic_output_paths(output_dir):
        calls.append("validate_atomic_output_paths")
        return Path(output_dir).with_name(f".{Path(output_dir).name}.incomplete")

    def save_obs_package_atomically(**kwargs):
        calls.append("save_obs_package_atomically")
        assert "verify_fixed_masks_and_weights" in calls
        output_dir = kwargs["config"].output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return Path(output_dir)

    monkeypatch.setattr(pipe, "load_source_artifacts", load_source_artifacts)
    monkeypatch.setattr(pipe, "resolve_obs_order_policy", resolve)
    monkeypatch.setattr(pipe, "load_model_and_tokenizer", load_model_and_tokenizer)
    monkeypatch.setattr(pipe, "collect_mlp_linears", collect_mlp_linears)
    monkeypatch.setattr(
        pipe, "validate_source_artifacts_against_targets", validate_source_artifacts_against_targets
    )
    monkeypatch.setattr(pipe, "group_mlp_projection_triplets", group_mlp_projection_triplets)
    monkeypatch.setattr(pipe, "build_calibration_samples", build_calibration_samples)
    monkeypatch.setattr(
        pipe, "capture_first_decoder_layer_inputs", capture_first_decoder_layer_inputs
    )
    monkeypatch.setattr(pipe, "run_layerwise_mlp_obs", run_layerwise_mlp_obs)
    monkeypatch.setattr(pipe, "verify_fixed_masks_and_weights", verify_fixed_masks_and_weights)
    monkeypatch.setattr(pipe, "validate_atomic_output_paths", validate_atomic_output_paths)
    monkeypatch.setattr(pipe, "save_obs_package_atomically", save_obs_package_atomically)

    # Need real masks for final forward - FakeArts.masks empty and collect returns real targets
    # Final verify is patched. Final model forward uses real TinyCausalLM - OK.
    # But validate is patched so empty masks OK. group creates triplets from real targets.
    # apply permutations skipped for none.
    # run_layerwise patched so weights unchanged.
    source = tmp_path / "src"
    source.mkdir()
    out = tmp_path / "out"
    cfg = _cfg(tmp_path, source, out)
    run_obs_compensation(cfg)

    expected_prefix = [
        "validate_atomic_output_paths",
        "load_source_artifacts",
        "resolve_obs_order_policy",
        "load_model_and_tokenizer",
        "collect_mlp_linears",
        "validate_source_artifacts_against_targets",
        "group_mlp_projection_triplets",
        "build_calibration_samples",
        "capture_first_decoder_layer_inputs",
        "run_layerwise_mlp_obs",
        "verify_fixed_masks_and_weights",
        "save_obs_package_atomically",
    ]
    assert calls[: len(expected_prefix)] == expected_prefix


def test_permutation_aware_unsorted_never_loads_model(monkeypatch, tmp_path):
    import obs_compensation.pipeline as pipe

    model = TinyCausalLM()
    targets = make_targets_from_tiny(model)
    masks = make_block_masks_for_targets(targets, 2, 2)
    source = write_source_artifacts(
        tmp_path / "src",
        masks=masks,
        mlp_permutation="none",
        model_path="tiny-model",
    )
    loaded = {"model": False}

    def boom(*args, **kwargs):
        loaded["model"] = True
        raise AssertionError("should not load model")

    monkeypatch.setattr(pipe, "load_model_and_tokenizer", boom)
    out = tmp_path / "out"
    cfg = _cfg(tmp_path, source, out, obs_order_policy="permutation_aware")
    with pytest.raises(ValueError, match="requires mlp_permutation=wanda_shared"):
        run_obs_compensation(cfg)
    assert loaded["model"] is False
    assert not out.exists()


def test_verify_fixed_masks_and_weights_strict_zero():
    model = TinyCausalLM()
    targets = make_targets_from_tiny(model)
    masks = make_block_masks_for_targets(targets, 2, 2)
    # manually zero pruned
    for t in targets:
        elem = masks[t.module_name].repeat_interleave(2, 0).repeat_interleave(2, 1)
        with torch.no_grad():
            t.module.weight[~elem] = 0
    verify_fixed_masks_and_weights(masks, targets, 2, 2)
    # break one
    with torch.no_grad():
        targets[0].module.weight[0, 0] = 1.0 if not masks[targets[0].module_name][0, 0] else targets[0].module.weight[0, 0]
        # force a pruned nonzero
        elem = masks[targets[0].module_name].repeat_interleave(2, 0).repeat_interleave(2, 1)
        targets[0].module.weight[~elem] = 1.0
    with pytest.raises(RuntimeError, match="exactly zero"):
        verify_fixed_masks_and_weights(masks, targets, 2, 2)


def _patch_model_and_dataset(monkeypatch, model_factory):
    import obs_compensation.pipeline as pipe
    import obs_compensation.calibration as cal

    def load_model_and_tokenizer(cfg):
        return model_factory(), TinyTokenizer()

    rows = [{"text": "abcdefghijklmnop"}, {"text": "qrstuvwxyzabcdef"}]

    monkeypatch.setattr(pipe, "load_model_and_tokenizer", load_model_and_tokenizer)
    monkeypatch.setattr(cal, "load_dataset", lambda *a, **k: rows)


def test_full_integration_unsorted_and_sorted(monkeypatch, tmp_path):
    # Unsorted
    model_u = TinyCausalLM(num_layers=2, d_model=4, d_ff=8)
    targets_u = make_targets_from_tiny(model_u)
    masks_u = make_block_masks_for_targets(targets_u, 2, 2)
    source_u = write_source_artifacts(
        tmp_path / "src_u",
        masks=masks_u,
        mlp_permutation="none",
        model_path="tiny-model",
    )
    before_u = {p.name: p.read_bytes() for p in source_u.iterdir()}
    out_u = tmp_path / "out_u"
    _patch_model_and_dataset(monkeypatch, lambda: TinyCausalLM(num_layers=2, d_model=4, d_ff=8))
    # Keep dense weights snapshot from a fresh model for change check
    dense = TinyCausalLM(num_layers=2, d_model=4, d_ff=8)
    dense_sd = {k: v.detach().clone() for k, v in dense.state_dict().items()}

    cfg_u = _cfg(tmp_path, source_u, out_u, obs_order_policy="auto")
    # Re-patch to return a fresh clone each time
    def factory():
        m = TinyCausalLM(num_layers=2, d_model=4, d_ff=8)
        m.load_state_dict(dense_sd)
        return m

    _patch_model_and_dataset(monkeypatch, factory)
    path_u = run_obs_compensation(cfg_u)
    assert path_u == out_u
    assert (out_u / "tiny_model_marker.txt").is_file()
    art_u = out_u / "obs_artifacts"
    for name in [
        "source_pruning_summary.json",
        "block_masks.pt",
        "obs_config.json",
        "obs_summary.json",
        "per_module_obs.csv",
        "per_layer_reconstruction.csv",
    ]:
        assert (art_u / name).is_file()
    assert not (art_u / "mlp_permutations.pt").exists()
    after_u = {p.name: p.read_bytes() for p in source_u.iterdir()}
    assert before_u == after_u
    summary_u = json.loads((art_u / "obs_summary.json").read_text())
    assert summary_u["resolved_obs_order_policy"] == "standard"
    assert summary_u["gate_up_direction"] == "left_to_right"
    assert summary_u["down_direction"] == "left_to_right"
    assert summary_u["num_layers"] == 2
    assert summary_u["num_modules"] == 6
    assert summary_u["source_dense_model_reloaded"] is True
    assert summary_u["fixed_mask"] is True
    assert summary_u["pruned_weights_exact_zero"] is True
    csv_u = (art_u / "per_module_obs.csv").read_text()
    assert "solver_direction" in csv_u
    assert "right_to_left" not in csv_u

    # Sorted
    model_s = TinyCausalLM(num_layers=2, d_model=4, d_ff=8)
    targets_s = make_targets_from_tiny(model_s)
    masks_s = make_block_masks_for_targets(targets_s, 2, 2)
    # ensure down right-half has more pruned for diagnostics
    for name, mask in masks_s.items():
        if name.endswith("down_proj"):
            mask[:, :] = True
            mask[0, -1] = False
            if mask.shape[1] > 2:
                mask[0, -2] = False
            masks_s[name] = mask
    payload = make_descending_permutation_payload(targets_s, intermediate_size=8)
    source_s = write_source_artifacts(
        tmp_path / "src_s",
        masks=masks_s,
        mlp_permutation="wanda_shared",
        permutation_payload=payload,
        model_path="tiny-model",
    )
    before_s = {p.name: p.read_bytes() for p in source_s.iterdir()}
    out_s = tmp_path / "out_s"

    def factory_s():
        m = TinyCausalLM(num_layers=2, d_model=4, d_ff=8)
        m.load_state_dict(dense_sd)
        return m

    _patch_model_and_dataset(monkeypatch, factory_s)
    cfg_s = _cfg(tmp_path, source_s, out_s, obs_order_policy="auto")
    path_s = run_obs_compensation(cfg_s)
    assert path_s == out_s
    art_s = out_s / "obs_artifacts"
    assert (art_s / "mlp_permutations.pt").is_file()
    after_s = {p.name: p.read_bytes() for p in source_s.iterdir()}
    assert before_s == after_s
    summary_s = json.loads((art_s / "obs_summary.json").read_text())
    assert summary_s["resolved_obs_order_policy"] == "permutation_aware"
    assert summary_s["down_direction"] == "right_to_left"
    assert summary_s["down_pruned_blocks_right_total"] > summary_s["down_pruned_blocks_left_total"]
    csv_s = (art_s / "per_module_obs.csv").read_text(encoding="utf-8")
    assert "right_to_left" in csv_s
    # right_to_left only on down rows
    for line in csv_s.splitlines()[1:]:
        if "right_to_left" in line:
            assert "down_proj" in line

    # Failure case: unsorted + permutation_aware
    out_bad = tmp_path / "out_bad"
    cfg_bad = _cfg(
        tmp_path, source_u, out_bad, obs_order_policy="permutation_aware"
    )
    with pytest.raises(ValueError, match="requires mlp_permutation=wanda_shared"):
        run_obs_compensation(cfg_bad)
    assert not out_bad.exists()


def test_save_obs_artifacts_schema(tmp_path):
    from obs_compensation.artifacts import load_source_artifacts
    from obs_compensation.layerwise import LayerOBSReport, LayerwiseOBSResult, ModuleOBSReport

    model = TinyCausalLM()
    targets = make_targets_from_tiny(model)
    masks = make_block_masks_for_targets(targets, 2, 2)
    source = write_source_artifacts(tmp_path / "src", masks=masks, model_path="tiny-model")
    arts = load_source_artifacts(source)
    cfg = _cfg(tmp_path, source, tmp_path / "out")
    policy = resolve_obs_order_policy("auto", "none")
    result = LayerwiseOBSResult(
        module_reports=[
            ModuleOBSReport(
                module_name=t.module_name,
                layer_index=t.layer_index,
                projection_type=t.projection_type,
                solver_direction="left_to_right",
                solver_applied=True,
                skip_reason="",
                num_total_blocks=int(masks[t.module_name].numel()),
                num_pruned_blocks=int((~masks[t.module_name]).sum().item()),
                block_sparsity=float((~masks[t.module_name]).float().mean().item()),
                num_fully_pruned_block_rows=0,
                num_fully_pruned_output_rows=0,
                num_hessian_tokens=10,
                hessian_diagonal_mean=1.0,
                hessian_damp_value=0.01,
                hessian_dead_columns=0,
                kept_delta_l2=0.1,
                kept_delta_max_abs=0.2,
                original_pruned_l2=0.3,
                original_max_abs=1.0,
                compensated_max_abs=1.1,
            )
            for t in targets
        ],
        layer_reports=[
            LayerOBSReport(0, 8, 0.1, 0.2, 0.3, 0, 1),
            LayerOBSReport(1, 8, 0.1, 0.2, 0.3, 0, 1),
        ],
    )
    out = tmp_path / "out"
    out.mkdir()
    art_dir = save_obs_artifacts(out, cfg, arts, policy, result)
    summary = json.loads((art_dir / "obs_summary.json").read_text())
    assert summary["requested_obs_order_policy"] == "auto"
    assert summary["resolved_obs_order_policy"] == "standard"
    assert "down_pruned_blocks_left_total" in summary
    assert "num_solver_applied_modules" in summary
    assert "num_solver_skipped_modules" in summary
    assert "total_fully_pruned_block_rows" in summary
    assert "total_fully_pruned_output_rows" in summary
    assert (
        summary["num_solver_applied_modules"] + summary["num_solver_skipped_modules"]
        == summary["num_modules"]
    )
    cfg_json = json.loads((art_dir / "obs_config.json").read_text())
    assert cfg_json["obs_order_policy"] == "auto"
    layer_csv = (art_dir / "per_layer_reconstruction.csv").read_text()
    assert "down_pruned_blocks_left" in layer_csv
    assert "down_pruned_blocks_right" in layer_csv
    module_csv = (art_dir / "per_module_obs.csv").read_text(encoding="utf-8")
    assert "solver_applied" in module_csv
    assert "skip_reason" in module_csv
    assert "num_fully_pruned_block_rows" in module_csv
    assert "num_fully_pruned_output_rows" in module_csv


def _minimal_serialization_inputs(tmp_path):
    from obs_compensation.artifacts import load_source_artifacts
    from obs_compensation.layerwise import (
        LayerOBSReport,
        LayerwiseOBSResult,
        ModuleOBSReport,
    )

    model = TinyCausalLM(num_layers=1, d_model=4, d_ff=8)
    targets = make_targets_from_tiny(model)
    masks = make_block_masks_for_targets(targets, 2, 2)
    source = write_source_artifacts(
        tmp_path / "src_atomic",
        masks=masks,
        model_path="tiny-model",
    )
    artifacts = load_source_artifacts(source)
    config = _cfg(tmp_path, source, tmp_path / "final_atomic")
    policy = resolve_obs_order_policy("auto", "none")
    module_reports = []
    for target in targets:
        mask = masks[target.module_name]
        module_reports.append(
            ModuleOBSReport(
                module_name=target.module_name,
                layer_index=target.layer_index,
                projection_type=target.projection_type,
                solver_direction="left_to_right",
                solver_applied=True,
                skip_reason="",
                num_total_blocks=int(mask.numel()),
                num_pruned_blocks=int((~mask).sum().item()),
                block_sparsity=float((~mask).float().mean().item()),
                num_fully_pruned_block_rows=int((~mask).all(dim=1).sum().item()),
                num_fully_pruned_output_rows=int((~mask).all(dim=1).sum().item()) * 2,
                num_hessian_tokens=8,
                hessian_diagonal_mean=1.0,
                hessian_damp_value=0.01,
                hessian_dead_columns=0,
                kept_delta_l2=0.1,
                kept_delta_max_abs=0.1,
                original_pruned_l2=0.2,
                original_max_abs=1.0,
                compensated_max_abs=1.1,
            )
        )
    result = LayerwiseOBSResult(
        module_reports=module_reports,
        layer_reports=[LayerOBSReport(0, 8, 0.1, 0.1, 0.1, 0, 1)],
    )
    return model, TinyTokenizer(), config, artifacts, policy, result


def test_atomic_output_path_validation(tmp_path):
    final = tmp_path / "obs_output"
    staging = get_incomplete_output_dir(final)
    assert staging == tmp_path / ".obs_output.incomplete"
    assert validate_atomic_output_paths(final) == staging

    staging.mkdir()
    with pytest.raises(ValueError, match="incomplete output directory already exists"):
        validate_atomic_output_paths(final)


def test_atomic_save_failure_never_creates_final_output(monkeypatch, tmp_path):
    import obs_compensation.serialization as serialization_module

    model, tokenizer, config, artifacts, policy, result = (
        _minimal_serialization_inputs(tmp_path)
    )
    staging = get_incomplete_output_dir(config.output_dir)

    def fail_after_model_save(*args, **kwargs):
        raise RuntimeError("intentional artifact failure")

    monkeypatch.setattr(
        serialization_module,
        "save_obs_artifacts",
        fail_after_model_save,
    )

    with pytest.raises(RuntimeError, match="intentional artifact failure"):
        save_obs_package_atomically(
            model=model,
            tokenizer=tokenizer,
            config=config,
            artifacts=artifacts,
            order_policy=policy,
            layerwise_result=result,
        )

    assert not config.output_dir.exists()
    assert staging.is_dir()
    assert (staging / "tiny_model_marker.txt").is_file()


def test_atomic_save_commits_complete_output(tmp_path):
    model, tokenizer, config, artifacts, policy, result = (
        _minimal_serialization_inputs(tmp_path)
    )
    staging = get_incomplete_output_dir(config.output_dir)

    saved = save_obs_package_atomically(
        model=model,
        tokenizer=tokenizer,
        config=config,
        artifacts=artifacts,
        order_policy=policy,
        layerwise_result=result,
    )

    assert saved == config.output_dir
    assert config.output_dir.is_dir()
    assert not staging.exists()
    assert (config.output_dir / "tiny_model_marker.txt").is_file()
    assert (config.output_dir / "obs_artifacts" / "obs_summary.json").is_file()
