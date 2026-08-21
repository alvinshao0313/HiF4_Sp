"""Tests for SearchConfig and related dataclasses."""

from __future__ import annotations

import pytest
import torch

from permutation_optimization.config import LayerSearchResult, MLPLayerSpec, SearchConfig


def test_default_config_ok():
    cfg = SearchConfig()
    assert cfg.activation_rows == 512
    assert cfg.validation_fraction == 0.2
    assert cfg.neighbor_k == 32
    assert cfg.seed == 42


def test_rejects_non_positive_ints():
    with pytest.raises(ValueError, match="activation_rows"):
        SearchConfig(activation_rows=0)
    with pytest.raises(ValueError, match="candidate_window"):
        SearchConfig(candidate_window=-1)
    with pytest.raises(ValueError, match="beam_width_g4"):
        SearchConfig(beam_width_g4=0)


def test_rejects_bad_validation_fraction():
    with pytest.raises(ValueError, match="validation_fraction"):
        SearchConfig(validation_fraction=0.0)
    with pytest.raises(ValueError, match="validation_fraction"):
        SearchConfig(validation_fraction=0.5)
    with pytest.raises(ValueError, match="validation_fraction"):
        SearchConfig(validation_fraction=0.9)


def test_rejects_weights_outside_unit_interval():
    with pytest.raises(ValueError, match="activation_loss_weight"):
        SearchConfig(activation_loss_weight=-0.1)
    with pytest.raises(ValueError, match="weight_loss_weight"):
        SearchConfig(weight_loss_weight=1.5)
    with pytest.raises(ValueError, match="range_loss_weight"):
        SearchConfig(range_loss_weight=2.0)


def test_rejects_neighbor_k_too_large():
    with pytest.raises(ValueError, match="neighbor_k"):
        SearchConfig(candidate_window=8, neighbor_k=32)


def test_mlp_layer_spec_requires_divisible_by_64():
    with pytest.raises(ValueError, match="divisible by 64"):
        MLPLayerSpec(
            name="layers.0.mlp",
            layer_index=0,
            gate_name="layers.0.mlp.gate_proj",
            up_name="layers.0.mlp.up_proj",
            down_name="layers.0.mlp.down_proj",
            intermediate_size=100,
        )


def test_validation_seeds_at_least_two():
    with pytest.raises(ValueError, match="validation_seeds"):
        SearchConfig(validation_seeds=(42,))
    cfg = SearchConfig(validation_seeds=(42, 43))
    assert cfg.validation_seeds == (42, 43)


def test_min_relative_improvement_must_be_positive():
    with pytest.raises(ValueError, match="min_relative_improvement"):
        SearchConfig(min_relative_improvement=0.0)
    with pytest.raises(ValueError, match="min_relative_improvement"):
        SearchConfig(min_relative_improvement=-0.1)


def test_min_validation_wins_within_seed_count():
    with pytest.raises(ValueError, match="min_validation_wins"):
        SearchConfig(validation_seeds=(42, 43), min_validation_wins=3)
    with pytest.raises(ValueError, match="min_validation_wins"):
        SearchConfig(validation_seeds=(42, 43, 44), min_validation_wins=0)


def test_max_bf16_reorder_drift_must_be_positive():
    with pytest.raises(ValueError, match="max_bf16_reorder_drift"):
        SearchConfig(max_bf16_reorder_drift=0.0)


def test_layer_search_result_fields():
    d = 64
    perm = torch.arange(d, dtype=torch.long)
    result = LayerSearchResult(
        layer_name="layers.0.mlp",
        permutation=perm,
        candidate_permutation=perm.clone(),
        baseline_metrics={"identity": {"hif4_loss": 1.0}},
        identity_hif4_loss=1.0,
        optimized_hif4_loss=0.5,
        identity_output_nrmse=0.1,
        optimized_output_nrmse=0.05,
        accepted=True,
    )
    assert result.accepted is True
    assert result.permutation.shape == (d,)
