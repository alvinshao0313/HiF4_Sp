from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.config import GradientBlockPruningConfig
from block_pruning.mlp_permutation import (
    apply_mlp_intermediate_permutations,
    compute_down_neuron_score,
    compute_mlp_shared_wanda_permutations,
    compute_up_or_gate_neuron_score,
    group_mlp_projection_triplets,
    normalize_projection_score,
    undo_mlp_intermediate_permutations,
)
from block_pruning.mlp_registry import MLPLinearTarget
from block_pruning.serialization import save_mlp_permutation_artifacts
from block_pruning.wanda_scorer import InputRMSRecord


def _target(
    name: str,
    linear: nn.Linear,
    layer: int,
    proj: str,
) -> MLPLinearTarget:
    return MLPLinearTarget(
        module_name=name,
        module=linear,
        layer_index=layer,
        projection_type=proj,
    )


def _make_layer_linears(layer: int, d_model: int = 4, d_ff: int = 6, bias: bool = False):
    gate = nn.Linear(d_model, d_ff, bias=bias)
    up = nn.Linear(d_model, d_ff, bias=bias)
    down = nn.Linear(d_ff, d_model, bias=bias)
    targets = [
        _target(f"model.layers.{layer}.mlp.gate_proj", gate, layer, "gate_proj"),
        _target(f"model.layers.{layer}.mlp.up_proj", up, layer, "up_proj"),
        _target(f"model.layers.{layer}.mlp.down_proj", down, layer, "down_proj"),
    ]
    return gate, up, down, targets


def test_group_mlp_projection_triplets_two_layers():
    _, _, _, t0 = _make_layer_linears(0)
    _, _, _, t1 = _make_layer_linears(1)
    # Shuffle to ensure sort by layer.
    targets = [t1[1], t0[2], t1[0], t0[0], t1[2], t0[1]]
    triplets = group_mlp_projection_triplets(targets)
    assert [t.layer_index for t in triplets] == [0, 1]
    assert triplets[0].gate.module_name.endswith("gate_proj")
    assert triplets[0].up is t0[1]
    assert triplets[1].down is t1[2]
    assert triplets[0].intermediate_size == 6


def test_group_rejects_missing_duplicate_shape():
    gate, up, down, targets = _make_layer_linears(0)
    try:
        group_mlp_projection_triplets(targets[:2])
        assert False, "expected missing projection error"
    except ValueError as e:
        assert "missing" in str(e).lower()

    dup = targets + [
        _target("model.layers.0.mlp.gate_proj.extra", gate, 0, "gate_proj")
    ]
    try:
        group_mlp_projection_triplets(dup)
        assert False, "expected duplicate error"
    except ValueError as e:
        assert "Duplicate" in str(e)

    bad_down = nn.Linear(5, 4, bias=False)
    bad_targets = [
        targets[0],
        targets[1],
        _target("model.layers.0.mlp.down_proj", bad_down, 0, "down_proj"),
    ]
    try:
        group_mlp_projection_triplets(bad_targets)
        assert False, "expected shape error"
    except ValueError as e:
        assert "incompatible" in str(e)


def test_up_gate_and_down_scores_hand_computed():
    weight = torch.tensor(
        [
            [1.0, -2.0],
            [3.0, -4.0],
            [5.0, -6.0],
        ]
    )
    rms = torch.tensor([2.0, 0.5])
    # row0: 1*2 + 2*0.5 = 3; row1: 3*2+4*0.5=8; row2: 5*2+6*0.5=13
    expected_up = torch.tensor([3.0, 8.0, 13.0], dtype=torch.float64)
    got_up = compute_up_or_gate_neuron_score(weight, rms)
    torch.testing.assert_close(got_up, expected_up)

    down_w = torch.tensor(
        [
            [1.0, -2.0, 3.0],
            [4.0, -5.0, 6.0],
        ]
    )
    down_rms = torch.tensor([1.0, 2.0, 0.5])
    # col sums abs: [5, 7, 9] * rms -> [5, 14, 4.5]
    expected_down = torch.tensor([5.0, 14.0, 4.5], dtype=torch.float64)
    got_down = compute_down_neuron_score(down_w, down_rms)
    torch.testing.assert_close(got_down, expected_down)


def test_normalize_and_score_validation():
    score = torch.tensor([1.0, 3.0], dtype=torch.float64)
    norm = normalize_projection_score(score, 0, "up_proj")
    torch.testing.assert_close(norm, torch.tensor([0.25, 0.75], dtype=torch.float64))

    try:
        normalize_projection_score(
            torch.tensor([0.0, 0.0], dtype=torch.float64), 1, "gate_proj"
        )
        assert False, "expected zero-total error"
    except ValueError as e:
        assert "positive" in str(e)

    try:
        compute_up_or_gate_neuron_score(torch.randn(2, 3), torch.randn(2))
        assert False, "expected channel mismatch"
    except ValueError as e:
        assert "d_in" in str(e)


def test_shared_permutation_equal_projection_and_stable_ties():
    gate = nn.Linear(2, 3, bias=False)
    up = nn.Linear(2, 3, bias=False)
    down = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        # Distinct projection preferences; after L1-equal combine, neuron ranking
        # must follow combined mass (gate-heavy 0, up-heavy 1, down-heavy 2).
        gate.weight.copy_(
            torch.tensor(
                [
                    [3.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            )
        )
        up.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [3.0, 0.0],
                    [0.0, 0.0],
                ]
            )
        )
        down.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 3.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        )
    targets = [
        _target("model.layers.0.mlp.gate_proj", gate, 0, "gate_proj"),
        _target("model.layers.0.mlp.up_proj", up, 0, "up_proj"),
        _target("model.layers.0.mlp.down_proj", down, 0, "down_proj"),
    ]
    triplets = group_mlp_projection_triplets(targets)
    rms_records = {
        t.module_name: InputRMSRecord(
            module_name=t.module_name,
            layer_index=0,
            projection_type=t.projection_type,
            num_tokens=1,
            channel_square_sum=torch.ones(t.module.weight.shape[1], dtype=torch.float64),
            input_rms=torch.ones(t.module.weight.shape[1], dtype=torch.float64),
        )
        for t in targets
    }
    records = compute_mlp_shared_wanda_permutations(triplets, rms_records)
    rec = records[0]
    # Each projection contributes mass 1 on a different neuron -> combined all ones.
    torch.testing.assert_close(
        rec.combined_score,
        torch.ones(3, dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    # Exact ties: stable argsort keeps original order.
    assert rec.permutation.tolist() == [0, 1, 2]
    assert rec.inverse_permutation.tolist() == [0, 1, 2]

    # Non-tie: make neuron 2 dominate after equal projection norm.
    with torch.no_grad():
        gate.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [4.0, 0.0],
                ]
            )
        )
        up.weight.copy_(gate.weight.clone())
        down.weight.copy_(
            torch.tensor(
                [
                    [1.0, 1.0, 4.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        )
    records2 = compute_mlp_shared_wanda_permutations(triplets, rms_records)
    assert records2[0].permutation.tolist()[0] == 2
    assert sorted(records2[0].permutation.tolist()) == [0, 1, 2]
    assert (
        records2[0].inverse_permutation[records2[0].permutation]
        == torch.arange(3)
    ).all()


def test_apply_undo_parameter_identity_and_mapping():
    gate, up, down, targets = _make_layer_linears(0, d_model=2, d_ff=4, bias=True)
    with torch.no_grad():
        gate.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2))
        up.weight.copy_(torch.arange(8, 16, dtype=torch.float32).reshape(4, 2))
        down.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4))
        gate.bias.copy_(torch.tensor([10.0, 20.0, 30.0, 40.0]))
        up.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        down.bias.copy_(torch.tensor([100.0, 200.0]))

    triplets = group_mlp_projection_triplets(targets)
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(4, dtype=torch.int64)
    from block_pruning.mlp_permutation import MLPIntermediatePermutationRecord

    zeros = torch.zeros(4, dtype=torch.float64)
    records = {
        0: MLPIntermediatePermutationRecord(
            layer_index=0,
            gate_module_name=targets[0].module_name,
            up_module_name=targets[1].module_name,
            down_module_name=targets[2].module_name,
            intermediate_size=4,
            gate_score=zeros.clone(),
            up_score=zeros.clone(),
            down_score=zeros.clone(),
            normalized_gate_score=zeros.clone(),
            normalized_up_score=zeros.clone(),
            normalized_down_score=zeros.clone(),
            combined_score=zeros.clone(),
            permutation=perm,
            inverse_permutation=inv,
        )
    }
    gate_w_id, up_w_id, down_w_id = id(gate.weight), id(up.weight), id(down.weight)
    gate_b_id, up_b_id, down_b_id = id(gate.bias), id(up.bias), id(down.bias)
    gate_w0 = gate.weight.detach().clone()
    up_w0 = up.weight.detach().clone()
    down_w0 = down.weight.detach().clone()
    gate_b0 = gate.bias.detach().clone()
    up_b0 = up.bias.detach().clone()
    down_b0 = down.bias.detach().clone()

    apply_mlp_intermediate_permutations(triplets, records)
    assert id(gate.weight) == gate_w_id
    assert id(up.weight) == up_w_id
    assert id(down.weight) == down_w_id
    assert id(gate.bias) == gate_b_id
    assert id(up.bias) == up_b_id
    assert id(down.bias) == down_b_id

    torch.testing.assert_close(gate.weight, gate_w0.index_select(0, perm))
    torch.testing.assert_close(up.weight, up_w0.index_select(0, perm))
    torch.testing.assert_close(down.weight, down_w0.index_select(1, perm))
    torch.testing.assert_close(gate.bias, gate_b0.index_select(0, perm))
    torch.testing.assert_close(up.bias, up_b0.index_select(0, perm))
    # down bias is on hidden dim; untouched
    torch.testing.assert_close(down.bias, down_b0)

    undo_mlp_intermediate_permutations(triplets, records)
    torch.testing.assert_close(gate.weight, gate_w0)
    torch.testing.assert_close(up.weight, up_w0)
    torch.testing.assert_close(down.weight, down_w0)
    torch.testing.assert_close(gate.bias, gate_b0)
    torch.testing.assert_close(up.bias, up_b0)
    torch.testing.assert_close(down.bias, down_b0)


def _swiglu(gate: nn.Linear, up: nn.Linear, down: nn.Linear, x: torch.Tensor):
    hidden = torch.nn.functional.silu(gate(x)) * up(x)
    return down(hidden)


def test_swiglu_equivalence_and_negative_controls():
    torch.manual_seed(0)
    gate, up, down, targets = _make_layer_linears(0, d_model=3, d_ff=5, bias=False)
    with torch.no_grad():
        gate.weight.normal_()
        up.weight.normal_()
        down.weight.normal_()
    x = torch.randn(7, 3)
    y0 = _swiglu(gate, up, down, x)

    triplets = group_mlp_projection_triplets(targets)
    perm = torch.tensor([4, 1, 0, 3, 2], dtype=torch.int64)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(5, dtype=torch.int64)
    from block_pruning.mlp_permutation import MLPIntermediatePermutationRecord

    zeros = torch.zeros(5, dtype=torch.float64)
    records = {
        0: MLPIntermediatePermutationRecord(
            layer_index=0,
            gate_module_name=targets[0].module_name,
            up_module_name=targets[1].module_name,
            down_module_name=targets[2].module_name,
            intermediate_size=5,
            gate_score=zeros.clone(),
            up_score=zeros.clone(),
            down_score=zeros.clone(),
            normalized_gate_score=zeros.clone(),
            normalized_up_score=zeros.clone(),
            normalized_down_score=zeros.clone(),
            combined_score=zeros.clone(),
            permutation=perm,
            inverse_permutation=inv,
        )
    }
    apply_mlp_intermediate_permutations(triplets, records)
    y1 = _swiglu(gate, up, down, x)
    torch.testing.assert_close(y1, y0, atol=1e-6, rtol=1e-5)

    undo_mlp_intermediate_permutations(triplets, records)
    y2 = _swiglu(gate, up, down, x)
    torch.testing.assert_close(y2, y0, atol=1e-6, rtol=1e-5)

    # Negative: only permute up rows
    apply_mlp_intermediate_permutations(triplets, records)
    undo_mlp_intermediate_permutations(triplets, records)
    with torch.no_grad():
        up.weight.copy_(up.weight.index_select(0, perm))
    y_bad_up = _swiglu(gate, up, down, x)
    assert not torch.allclose(y_bad_up, y0, atol=1e-5)

    with torch.no_grad():
        up.weight.copy_(up.weight.index_select(0, inv))
        gate.weight.copy_(gate.weight.index_select(0, perm))
        up.weight.copy_(up.weight.index_select(0, perm))
        # missing down
    y_bad_down = _swiglu(gate, up, down, x)
    assert not torch.allclose(y_bad_down, y0, atol=1e-5)

    # restore and different perms for up vs gate
    with torch.no_grad():
        gate.weight.copy_(gate.weight.index_select(0, inv))
        up.weight.copy_(up.weight.index_select(0, inv))
        other = torch.tensor([0, 4, 3, 2, 1], dtype=torch.int64)
        gate.weight.copy_(gate.weight.index_select(0, perm))
        up.weight.copy_(up.weight.index_select(0, other))
        down.weight.copy_(down.weight.index_select(1, perm))
    y_bad_diff = _swiglu(gate, up, down, x)
    assert not torch.allclose(y_bad_diff, y0, atol=1e-5)


def test_config_mlp_permutation_modes():
    cfg = GradientBlockPruningConfig(score_type="magnitude", mlp_permutation="none")
    cfg.validate()
    assert not cfg.requires_calibration()

    cfg2 = GradientBlockPruningConfig(
        score_type="magnitude", mlp_permutation="wanda_shared"
    )
    cfg2.validate()
    assert cfg2.requires_calibration()
    assert not cfg2.requires_gradient_checkpointing()

    cfg3 = GradientBlockPruningConfig(score_type="fisher", mlp_permutation="none")
    cfg3.validate()
    assert cfg3.requires_calibration()

    try:
        bad = GradientBlockPruningConfig(mlp_permutation="learned")
        bad.validate()
        assert False, "expected invalid mode"
    except ValueError as e:
        assert "mlp_permutation" in str(e)


def test_save_mlp_permutation_artifacts(tmp_path: Path):
    gate, up, down, targets = _make_layer_linears(0, d_model=2, d_ff=2, bias=False)
    with torch.no_grad():
        gate.weight.fill_(1.0)
        up.weight.fill_(1.0)
        down.weight.fill_(1.0)
    triplets = group_mlp_projection_triplets(targets)
    rms_records = {
        t.module_name: InputRMSRecord(
            module_name=t.module_name,
            layer_index=0,
            projection_type=t.projection_type,
            num_tokens=1,
            channel_square_sum=torch.ones(t.module.weight.shape[1], dtype=torch.float64),
            input_rms=torch.ones(t.module.weight.shape[1], dtype=torch.float64),
        )
        for t in targets
    }
    records = compute_mlp_shared_wanda_permutations(triplets, rms_records)
    cfg = GradientBlockPruningConfig(mlp_permutation="wanda_shared")
    cfg.validate()
    out = tmp_path / "artifacts"
    save_mlp_permutation_artifacts(out, records, cfg)
    assert (out / "mlp_permutations.pt").is_file()
    assert (out / "mlp_permutation_summary.json").is_file()
    payload = torch.load(out / "mlp_permutations.pt", map_location="cpu", weights_only=False)
    assert "0" in payload
    assert "permutation" in payload["0"]
    assert "combined_score" in payload["0"]
