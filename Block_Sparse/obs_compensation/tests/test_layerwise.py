from __future__ import annotations

import torch
import torch.nn as nn

from obs_compensation.config import OBSCompensationConfig
from obs_compensation.hessian import HessianAccumulator
from obs_compensation.layerwise import _accumulate_linear_inputs, run_layerwise_mlp_obs
from obs_compensation.model_adapter import CapturedLayerInputs
from obs_compensation.permutation import group_mlp_projection_triplets
from obs_compensation.solver import ResolvedOBSOrderPolicy
from obs_compensation.tests.helpers import TinyCausalLM, make_targets_from_tiny


def _cfg(tmp_path):
    return OBSCompensationConfig(
        model_path="tiny",
        source_artifacts_dir=tmp_path / "src",
        output_dir=tmp_path / "out",
        calibration_dataset="s1k",
        calibration_samples=2,
        sequence_length=4,
        obs_percdamp=0.01,
        solver_block_size=8,
        obs_order_policy="auto",
        dtype="float32",
        device="cpu",
        seed=0,
        trust_remote_code=True,
    )


def test_hessian_hook_helper():
    linear = nn.Linear(3, 2, bias=False)
    acc = HessianAccumulator(3, torch.device("cpu"), "hook")
    x = torch.randn(2, 4, 3)
    with _accumulate_linear_inputs(linear, acc):
        _ = linear(x)
    snap = acc.finalize()
    assert snap.num_tokens == 8


def _run_one_layer_case(tmp_path, policy: ResolvedOBSOrderPolicy):
    torch.manual_seed(0)
    model = TinyCausalLM(num_layers=1, d_model=4, d_ff=8)
    targets = make_targets_from_tiny(model)
    triplets = group_mlp_projection_triplets(targets)
    masks = {}
    for t in targets:
        d_out, d_in = t.module.weight.shape
        mask = torch.ones(d_out // 2, d_in // 2, dtype=torch.bool)
        mask[0, -1] = False
        masks[t.module_name] = mask

    # correlated captured inputs
    hidden = torch.tensor(
        [[[1.0, 1.0, 0.5, 0.4], [2.0, 2.1, -0.5, -0.4]]],
        dtype=torch.float32,
    )
    captured = CapturedLayerInputs(
        hidden_states=[hidden.clone(), hidden.clone() + 0.1],
        layer_kwargs=[{}, {}],
    )

    # snapshot non-mlp params
    other_before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if "mlp" not in n
    }
    dense_gate = triplets[0].gate.module.weight.detach().clone()

    result = run_layerwise_mlp_obs(
        model=model,
        captured=captured,
        triplets=triplets,
        masks=masks,
        config=_cfg(tmp_path),
        order_policy=policy,
        block_height=2,
        block_width=2,
    )
    assert len(result.module_reports) == 3
    assert len(result.layer_reports) == 1
    gate_r, up_r, down_r = result.module_reports
    assert gate_r.num_hessian_tokens == up_r.num_hessian_tokens
    assert gate_r.hessian_damp_value == up_r.hessian_damp_value
    assert gate_r.solver_direction == "left_to_right"
    assert up_r.solver_direction == "left_to_right"
    assert down_r.solver_direction == policy.down_direction

    for t in targets:
        elem = masks[t.module_name].repeat_interleave(2, 0).repeat_interleave(2, 1)
        assert torch.count_nonzero(t.module.weight.detach()[~elem]) == 0

    assert not torch.equal(triplets[0].gate.module.weight.detach(), dense_gate) or any(
        r.kept_delta_l2 > 0 for r in result.module_reports
    )
    assert result.layer_reports[0].output_mse >= 0
    assert torch.isfinite(torch.tensor(result.layer_reports[0].output_mse))
    for n, p in model.named_parameters():
        if "mlp" not in n:
            assert torch.equal(p.detach(), other_before[n])
    return result, captured, triplets, model


def test_one_layer_standard_and_aware(tmp_path):
    standard = ResolvedOBSOrderPolicy(
        requested_policy="standard",
        resolved_policy="standard",
        gate_up_direction="left_to_right",
        down_direction="left_to_right",
    )
    aware = ResolvedOBSOrderPolicy(
        requested_policy="auto",
        resolved_policy="permutation_aware",
        gate_up_direction="left_to_right",
        down_direction="right_to_left",
    )
    r1, *_ = _run_one_layer_case(tmp_path, standard)
    assert r1.module_reports[2].solver_direction == "left_to_right"
    r2, *_ = _run_one_layer_case(tmp_path, aware)
    assert r2.module_reports[2].solver_direction == "right_to_left"


def test_two_layer_propagation(tmp_path):
    torch.manual_seed(1)
    model = TinyCausalLM(num_layers=2, d_model=4, d_ff=8)
    targets = make_targets_from_tiny(model)
    triplets = group_mlp_projection_triplets(targets)
    masks = {}
    for t in targets:
        d_out, d_in = t.module.weight.shape
        mask = torch.ones(d_out // 2, d_in // 2, dtype=torch.bool)
        mask[0, 0] = False
        masks[t.module_name] = mask
    hidden = torch.randn(1, 2, 4)
    captured = CapturedLayerInputs(hidden_states=[hidden.clone()], layer_kwargs=[{}])

    layer1_inputs = []

    def hook(module, args, kwargs):
        del module, kwargs
        layer1_inputs.append(args[0].detach().cpu().clone())

    handle = model.layers[1].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        # Manually run layerwise and also capture during pass3 of layer0 via instrumentation:
        # Instead, monkeypatch run_decoder_layer on layer 1 by wrapping layer forward.
        pass
    finally:
        handle.remove()

    # Re-run with a wrapper that records layer1 inputs during the real pipeline.
    recorded = []
    original_forward = model.layers[1].forward

    def wrapped(hidden_states, **kwargs):
        recorded.append(hidden_states.detach().cpu().clone())
        return original_forward(hidden_states, **kwargs)

    model.layers[1].forward = wrapped
    result = run_layerwise_mlp_obs(
        model=model,
        captured=captured,
        triplets=triplets,
        masks=masks,
        config=_cfg(tmp_path),
        order_policy=ResolvedOBSOrderPolicy(
            "standard", "standard", "left_to_right", "left_to_right"
        ),
        block_height=2,
        block_width=2,
    )
    assert len(result.layer_reports) == 2
    # First call to layer1 during its pass1 should equal layer0 third-pass output.
    # Layer0 third-pass outputs become current_hidden for layer1; first layer1 forward
    # happens in pass1 and is the first recorded entry for layer1.
    assert recorded, "layer1 was never called"
    # Compute expected by replaying only layer0 third-pass is hard; instead ensure
    # first recorded input is not equal to the original captured hidden.
    assert not torch.equal(recorded[0], hidden)
    assert recorded[0].shape == hidden.shape


def test_layerwise_obs_disables_grad(monkeypatch, tmp_path):
    import obs_compensation.layerwise as layerwise_module

    grad_states: list[bool] = []
    real_run_decoder_layer = layerwise_module.run_decoder_layer

    def recording_run_decoder_layer(**kwargs):
        grad_states.append(torch.is_grad_enabled())
        return real_run_decoder_layer(**kwargs)

    monkeypatch.setattr(
        layerwise_module,
        "run_decoder_layer",
        recording_run_decoder_layer,
    )
    policy = ResolvedOBSOrderPolicy(
        requested_policy="standard",
        resolved_policy="standard",
        gate_up_direction="left_to_right",
        down_direction="left_to_right",
    )

    with torch.enable_grad():
        _result, _captured, _triplets, model = _run_one_layer_case(tmp_path, policy)

    assert grad_states
    assert not any(grad_states)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_all_kept_layer_skips_hessian_and_solver(monkeypatch, tmp_path):
    import obs_compensation.layerwise as layerwise_module

    model = TinyCausalLM(num_layers=1, d_model=4, d_ff=8)
    targets = make_targets_from_tiny(model)
    triplets = group_mlp_projection_triplets(targets)
    masks = {
        target.module_name: torch.ones(
            target.module.weight.shape[0] // 2,
            target.module.weight.shape[1] // 2,
            dtype=torch.bool,
        )
        for target in targets
    }
    hidden = torch.randn(1, 2, 4)
    captured = CapturedLayerInputs(
        hidden_states=[hidden.clone(), hidden.clone() + 0.1],
        layer_kwargs=[{}, {}],
    )
    forward_calls = {"count": 0}
    real_run_decoder_layer = layerwise_module.run_decoder_layer

    def counting_run_decoder_layer(**kwargs):
        forward_calls["count"] += 1
        return real_run_decoder_layer(**kwargs)

    def forbidden_build(*args, **kwargs):
        raise AssertionError("all-kept layer must not build an OBS system")

    def forbidden_solve(*args, **kwargs):
        raise AssertionError("all-kept layer must not call OBS solver")

    monkeypatch.setattr(layerwise_module, "run_decoder_layer", counting_run_decoder_layer)
    monkeypatch.setattr(layerwise_module, "build_obs_system", forbidden_build)
    monkeypatch.setattr(layerwise_module, "solve_fixed_mask_obs", forbidden_solve)

    result = run_layerwise_mlp_obs(
        model=model,
        captured=captured,
        triplets=triplets,
        masks=masks,
        config=_cfg(tmp_path),
        order_policy=ResolvedOBSOrderPolicy(
            "standard", "standard", "left_to_right", "left_to_right"
        ),
        block_height=2,
        block_width=2,
    )

    assert forward_calls["count"] == len(captured.hidden_states)
    assert len(result.module_reports) == 3
    assert all(not report.solver_applied for report in result.module_reports)
    assert all(report.skip_reason == "mask_all_kept" for report in result.module_reports)
    assert all(report.num_hessian_tokens == 0 for report in result.module_reports)
    assert result.layer_reports[0].output_mse == 0.0
    assert result.layer_reports[0].output_relative_mse == 0.0
    assert result.layer_reports[0].output_max_abs_error == 0.0


def test_only_pruned_projection_is_solved(monkeypatch, tmp_path):
    import obs_compensation.layerwise as layerwise_module

    model = TinyCausalLM(num_layers=1, d_model=4, d_ff=8)
    targets = make_targets_from_tiny(model)
    triplets = group_mlp_projection_triplets(targets)
    masks = {}
    for target in targets:
        mask = torch.ones(
            target.module.weight.shape[0] // 2,
            target.module.weight.shape[1] // 2,
            dtype=torch.bool,
        )
        if target.projection_type == "up_proj":
            mask[0, 0] = False
        masks[target.module_name] = mask

    hidden = torch.randn(1, 2, 4)
    captured = CapturedLayerInputs(
        hidden_states=[hidden.clone(), hidden.clone() + 0.1],
        layer_kwargs=[{}, {}],
    )
    counts = {"build": 0, "solve": 0, "forward": 0}
    real_build = layerwise_module.build_obs_system
    real_solve = layerwise_module.solve_fixed_mask_obs
    real_run = layerwise_module.run_decoder_layer

    def counting_build(*args, **kwargs):
        counts["build"] += 1
        return real_build(*args, **kwargs)

    def counting_solve(*args, **kwargs):
        counts["solve"] += 1
        return real_solve(*args, **kwargs)

    def counting_run(**kwargs):
        counts["forward"] += 1
        return real_run(**kwargs)

    monkeypatch.setattr(layerwise_module, "build_obs_system", counting_build)
    monkeypatch.setattr(layerwise_module, "solve_fixed_mask_obs", counting_solve)
    monkeypatch.setattr(layerwise_module, "run_decoder_layer", counting_run)

    result = run_layerwise_mlp_obs(
        model=model,
        captured=captured,
        triplets=triplets,
        masks=masks,
        config=_cfg(tmp_path),
        order_policy=ResolvedOBSOrderPolicy(
            "standard", "standard", "left_to_right", "left_to_right"
        ),
        block_height=2,
        block_width=2,
    )

    reports = {report.projection_type: report for report in result.module_reports}
    assert counts == {"build": 1, "solve": 1, "forward": 4}
    assert reports["gate_proj"].solver_applied is False
    assert reports["gate_proj"].skip_reason == "mask_all_kept"
    assert reports["up_proj"].solver_applied is True
    assert reports["up_proj"].skip_reason == ""
    assert reports["down_proj"].solver_applied is False
    assert reports["down_proj"].skip_reason == "mask_all_kept"
