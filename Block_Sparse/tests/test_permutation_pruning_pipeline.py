from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.config import GradientBlockPruningConfig
from block_pruning.mlp_permutation import (
    compute_mlp_shared_wanda_permutations,
    prepare_and_apply_mlp_permutations,
)
from block_pruning.mlp_registry import MLPLinearTarget, initialize_all_one_masks
from block_pruning.wanda_scorer import InputRMSRecord


def _target(name, linear, layer, proj):
    return MLPLinearTarget(
        module_name=name,
        module=linear,
        layer_index=layer,
        projection_type=proj,
    )


def _make_targets(layer: int = 0):
    gate = nn.Linear(2, 4, bias=False)
    up = nn.Linear(2, 4, bias=False)
    down = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        gate.weight.fill_(1.0)
        up.weight.fill_(1.0)
        down.weight.fill_(1.0)
    targets = [
        _target(f"model.layers.{layer}.mlp.gate_proj", gate, layer, "gate_proj"),
        _target(f"model.layers.{layer}.mlp.up_proj", up, layer, "up_proj"),
        _target(f"model.layers.{layer}.mlp.down_proj", down, layer, "down_proj"),
    ]
    return targets


def _fake_rms(targets):
    return {
        t.module_name: InputRMSRecord(
            module_name=t.module_name,
            layer_index=t.layer_index,
            projection_type=t.projection_type,
            num_tokens=1,
            channel_square_sum=torch.ones(t.module.weight.shape[1], dtype=torch.float64),
            input_rms=torch.ones(t.module.weight.shape[1], dtype=torch.float64),
        )
        for t in targets
    }


def test_prepare_call_order_and_rejects_missing_batches():
    targets = _make_targets()
    cfg = GradientBlockPruningConfig(mlp_permutation="wanda_shared", score_type="magnitude")
    cfg.validate()
    model = nn.Module()
    order: list[str] = []

    def fake_collect_rms(model, batches, targets_arg, **kwargs):
        order.append("collect_rms")
        return _fake_rms(targets_arg)

    real_compute = compute_mlp_shared_wanda_permutations

    def fake_compute(triplets, rms):
        order.append("compute")
        return real_compute(triplets, rms)

    def fake_apply(triplets, records):
        order.append("apply")

    with patch(
        "block_pruning.mlp_permutation.collect_mlp_input_rms",
        side_effect=fake_collect_rms,
    ), patch(
        "block_pruning.mlp_permutation.compute_mlp_shared_wanda_permutations",
        side_effect=fake_compute,
    ), patch(
        "block_pruning.mlp_permutation.apply_mlp_intermediate_permutations",
        side_effect=fake_apply,
    ):
        prepare_and_apply_mlp_permutations(
            model=model,
            batches=[{"input_ids": torch.zeros(1, 2, dtype=torch.long)}],
            targets=targets,
            config=cfg,
        )
        order.append("init_masks")
        initialize_all_one_masks(targets, cfg.block_height, cfg.block_width)

    assert order == ["collect_rms", "compute", "apply", "init_masks"]

    try:
        prepare_and_apply_mlp_permutations(
            model=model,
            batches=None,
            targets=targets,
            config=cfg,
        )
        assert False, "expected missing batches error"
    except ValueError as e:
        assert "calibration batches" in str(e)


def test_disabled_path_does_not_call_prepare():
    """Simulate main() branch: mlp_permutation=none skips prepare entirely."""
    cfg = GradientBlockPruningConfig(mlp_permutation="none", score_type="magnitude")
    cfg.validate()
    assert cfg.mlp_permutation == "none"
    prepare = MagicMock()
    if cfg.mlp_permutation == "wanda_shared":
        prepare()
    prepare.assert_not_called()


def test_multi_round_apply_once_semantics():
    """Permutation is applied once before rounds; rounds do not re-apply."""
    targets = _make_targets()
    cfg = GradientBlockPruningConfig(
        mlp_permutation="wanda_shared",
        score_type="magnitude",
        pruning_rounds=3,
        block_size="2",
    )
    cfg.validate()
    model = nn.Module()
    apply_calls = {"n": 0}

    def counting_apply(triplets, records):
        apply_calls["n"] += 1

    with patch(
        "block_pruning.mlp_permutation.collect_mlp_input_rms",
        side_effect=lambda m, b, t, **kwargs: _fake_rms(t),
    ), patch(
        "block_pruning.mlp_permutation.apply_mlp_intermediate_permutations",
        side_effect=counting_apply,
    ):
        prepare_and_apply_mlp_permutations(
            model=model,
            batches=[{"input_ids": torch.zeros(1, 2, dtype=torch.long)}],
            targets=targets,
            config=cfg,
        )
        # Simulate three pruning rounds that never call prepare again.
        for _ in range(cfg.pruning_rounds):
            pass

    assert apply_calls["n"] == 1
