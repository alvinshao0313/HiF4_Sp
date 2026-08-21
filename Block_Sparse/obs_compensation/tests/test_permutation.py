from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from block_pruning.mlp_registry import MLPLinearTarget
from obs_compensation.permutation import (
    apply_saved_mlp_permutations,
    group_mlp_projection_triplets,
)
from obs_compensation.tests.helpers import TinyDecoderLayer, TinyMLP


def _targets_from_layer(layer: TinyDecoderLayer, layer_index: int = 0):
    # Names must parse via registry conventions when using collect; here build manually.
    return [
        MLPLinearTarget(
            module_name=f"layers.{layer_index}.mlp.gate_proj",
            module=layer.mlp.gate_proj,
            layer_index=layer_index,
            projection_type="gate_proj",
        ),
        MLPLinearTarget(
            module_name=f"layers.{layer_index}.mlp.up_proj",
            module=layer.mlp.up_proj,
            layer_index=layer_index,
            projection_type="up_proj",
        ),
        MLPLinearTarget(
            module_name=f"layers.{layer_index}.mlp.down_proj",
            module=layer.mlp.down_proj,
            layer_index=layer_index,
            projection_type="down_proj",
        ),
    ]


def test_group_triplets_happy_path():
    layer = TinyDecoderLayer(d_model=4, d_ff=6)
    targets = _targets_from_layer(layer)
    triplets = group_mlp_projection_triplets(targets)
    assert len(triplets) == 1
    assert triplets[0].intermediate_size == 6


def test_group_rejects_missing_and_duplicate():
    layer = TinyDecoderLayer()
    targets = _targets_from_layer(layer)
    with pytest.raises(ValueError, match="missing"):
        group_mlp_projection_triplets(targets[:2])
    dup = targets + [
        MLPLinearTarget(
            module_name="layers.0.mlp.gate_proj.dup",
            module=nn.Linear(4, 6, bias=False),
            layer_index=0,
            projection_type="gate_proj",
        )
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        group_mlp_projection_triplets(dup)


def test_permutation_invariance_and_layout():
    torch.manual_seed(1)
    layer = TinyDecoderLayer(d_model=4, d_ff=6)
    targets = _targets_from_layer(layer)
    triplet = group_mlp_projection_triplets(targets)[0]
    x = torch.randn(2, 3, 4)
    before = layer.mlp(x).detach()

    perm = torch.tensor([2, 0, 5, 1, 4, 3], dtype=torch.int64)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(6, dtype=torch.int64)
    # Construct descending ordered score: score[perm] descends
    ordered = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    score = torch.empty(6)
    score[perm] = ordered

    gate_before = triplet.gate.module.weight.detach().clone()
    up_before = triplet.up.module.weight.detach().clone()
    down_before = triplet.down.module.weight.detach().clone()
    gate_id = id(triplet.gate.module.weight)

    payload = {
        "0": {
            "layer_index": 0,
            "gate_module_name": triplet.gate.module_name,
            "up_module_name": triplet.up.module_name,
            "down_module_name": triplet.down.module_name,
            "intermediate_size": 6,
            "combined_score": score,
            "permutation": perm,
            "inverse_permutation": inverse,
        }
    }
    apply_saved_mlp_permutations([triplet], payload)
    after = layer.mlp(x).detach()
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
    assert id(triplet.gate.module.weight) == gate_id
    torch.testing.assert_close(
        triplet.gate.module.weight.detach(), gate_before.index_select(0, perm)
    )
    torch.testing.assert_close(
        triplet.up.module.weight.detach(), up_before.index_select(0, perm)
    )
    torch.testing.assert_close(
        triplet.down.module.weight.detach(), down_before.index_select(1, perm)
    )


def test_rejects_non_descending_importance():
    layer = TinyDecoderLayer(d_model=4, d_ff=4)
    targets = _targets_from_layer(layer)
    triplet = group_mlp_projection_triplets(targets)[0]
    perm = torch.arange(4, dtype=torch.int64)
    inverse = perm.clone()
    score = torch.tensor([1.0, 2.0, 3.0, 4.0])  # ascending after identity perm
    payload = {
        "0": {
            "layer_index": 0,
            "gate_module_name": triplet.gate.module_name,
            "up_module_name": triplet.up.module_name,
            "down_module_name": triplet.down.module_name,
            "intermediate_size": 4,
            "combined_score": score,
            "permutation": perm,
            "inverse_permutation": inverse,
        }
    }
    with pytest.raises(ValueError, match="descending"):
        apply_saved_mlp_permutations([triplet], payload)


def test_rejects_module_name_mismatch():
    layer = TinyDecoderLayer(d_model=4, d_ff=4)
    targets = _targets_from_layer(layer)
    triplet = group_mlp_projection_triplets(targets)[0]
    perm = torch.arange(3, -1, -1, dtype=torch.int64)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(4, dtype=torch.int64)
    score = torch.arange(1, 5, dtype=torch.float32)
    payload = {
        "0": {
            "layer_index": 0,
            "gate_module_name": "wrong",
            "up_module_name": triplet.up.module_name,
            "down_module_name": triplet.down.module_name,
            "intermediate_size": 4,
            "combined_score": score,
            "permutation": perm,
            "inverse_permutation": inverse,
        }
    }
    with pytest.raises(ValueError, match="gate_module_name"):
        apply_saved_mlp_permutations([triplet], payload)
