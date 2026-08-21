"""Tests for the pure threshold-gating policy."""

from __future__ import annotations

import torch
import pytest

from permutation_optimization.threshold_policy import (
    build_threshold_gated_permutations,
    relative_output_gain_pct,
)


def _summary(identity: float, candidate: float) -> dict:
    return {
        "results": [
            {
                "layer_name": "model.layers.0.mlp",
                "identity_output_nrmse": identity,
                "optimized_output_nrmse": candidate,
            }
        ]
    }


def _candidate() -> dict[str, torch.Tensor]:
    return {"model.layers.0.mlp": torch.tensor([1, 0, 2, 3])}


def test_gain_equal_to_threshold_is_reordered():
    summary = {
        "results": [
            {
                "layer_name": "model.layers.0.mlp",
                "identity_output_nrmse": 0.100,
                "optimized_output_nrmse": 0.099,
            }
        ]
    }
    candidates = {"model.layers.0.mlp": torch.tensor([1, 0, 2, 3])}
    gated, report = build_threshold_gated_permutations(
        summary, candidates, threshold_pct=1.0
    )
    assert torch.equal(gated["model.layers.0.mlp"], candidates["model.layers.0.mlp"])
    assert report["n_reordered"] == 1


def test_below_threshold_uses_identity():
    gated, report = build_threshold_gated_permutations(
        _summary(0.100, 0.0995), _candidate(), threshold_pct=1.0
    )
    assert torch.equal(gated["model.layers.0.mlp"], torch.arange(4))
    assert report["n_reordered"] == 0


def test_negative_gain_always_uses_identity():
    gated, report = build_threshold_gated_permutations(
        _summary(0.100, 0.101), _candidate(), threshold_pct=0.0
    )
    assert torch.equal(gated["model.layers.0.mlp"], torch.arange(4))
    assert report["layers"][0]["relative_gain_pct"] < 0.0


def test_negative_threshold_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        build_threshold_gated_permutations(
            _summary(0.100, 0.099), _candidate(), threshold_pct=-0.1
        )


def test_missing_summary_layer_is_rejected():
    with pytest.raises(KeyError, match="model.layers.0.mlp"):
        build_threshold_gated_permutations(
            {"results": []}, _candidate(), threshold_pct=0.0
        )


def test_illegal_candidate_permutation_is_rejected():
    illegal = {"model.layers.0.mlp": torch.tensor([0, 0, 2, 3])}
    with pytest.raises(ValueError, match="exactly once"):
        build_threshold_gated_permutations(
            _summary(0.100, 0.099), illegal, threshold_pct=0.0
        )


def test_relative_output_gain_pct_basic():
    assert relative_output_gain_pct(0.100, 0.099) == pytest.approx(1.0)
    assert relative_output_gain_pct(0.100, 0.100) == pytest.approx(0.0)
    assert relative_output_gain_pct(0.100, 0.110) == pytest.approx(-10.0)
    assert relative_output_gain_pct(0.0, 0.0) == 0.0


def test_deterministic_same_inputs():
    s = _summary(0.100, 0.099)
    c = _candidate()
    g1, r1 = build_threshold_gated_permutations(s, c, 0.5)
    g2, r2 = build_threshold_gated_permutations(s, c, 0.5)
    assert torch.equal(g1["model.layers.0.mlp"], g2["model.layers.0.mlp"])
    assert r1 == r2
