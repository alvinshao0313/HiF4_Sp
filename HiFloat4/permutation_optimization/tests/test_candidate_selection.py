"""Tests for candidate pool aggregation and robust multi-split acceptance."""

from __future__ import annotations

import torch
import pytest

from permutation_optimization.candidate_selection import (
    CandidateDecision,
    CandidateMetrics,
    select_candidate,
)
from permutation_optimization.config import SearchConfig
from permutation_optimization.objective import DeploymentMetrics


def _cand(
    name: str,
    totals: tuple[float, ...],
    drifts: tuple[float, ...] = (0.0, 0.0, 0.0),
    residuals: tuple[float, ...] = (0.1, 0.1, 0.1),
    eligible: bool = True,
    d_ff: int = 64,
) -> CandidateMetrics:
    perms = torch.arange(d_ff)
    split_metrics = tuple(
        DeploymentMetrics(
            bf16_reorder_drift=d,
            quantization_residual_nrmse=r,
            total_nrmse=t,
        )
        for t, d, r in zip(totals, drifts, residuals)
    )
    return CandidateMetrics(
        name=name,
        permutation=perms,
        split_metrics=split_metrics,
        eligible_for_deployment=eligible,
    )


def _identity(totals=(0.10, 0.10, 0.10)) -> CandidateMetrics:
    return _cand("identity", totals, drifts=(0.0, 0.0, 0.0), eligible=True)


def test_q99_selected_when_it_beats_hierarchical():
    cfg = SearchConfig()
    identity = _identity()
    q99 = _cand("q99_sort_desc", (0.0995, 0.0994, 0.0996), drifts=(0.0005, 0.0005, 0.0005))
    hier = _cand("hierarchical", (0.101, 0.1005, 0.102), drifts=(0.0005, 0.0005, 0.0005))
    decision = select_candidate([identity, q99, hier], cfg)
    assert decision.accepted
    assert decision.selected_name == "q99_sort_desc"
    assert decision.rejection_reason == "accepted"


def test_random_best_never_deployed():
    cfg = SearchConfig()
    identity = _identity()
    hier = _cand("hierarchical", (0.0995, 0.0994, 0.0996), drifts=(0.0005, 0.0005, 0.0005))
    rand = _cand(
        "random_seed_43", (0.0990, 0.0990, 0.0990), drifts=(0.0005, 0.0005, 0.0005),
        eligible=False,
    )
    decision = select_candidate([identity, hier, rand], cfg)
    assert not decision.accepted
    assert torch.equal(decision.selected_permutation, identity.permutation)
    assert (
        decision.rejection_reason
        == "random_negative_control_outperformed_structured_candidates"
    )


def test_no_structured_beats_identity_falls_back():
    cfg = SearchConfig()
    identity = _identity()
    hier = _cand("hierarchical", (0.1005, 0.1002, 0.1008), drifts=(0.0005,) * 3)
    decision = select_candidate([identity, hier], cfg)
    assert not decision.accepted
    assert decision.selected_name == "identity"
    assert decision.rejection_reason == "no_structured_candidate_beats_identity"


def test_unstable_variance_rejected():
    """deltas [-0.5%, +0.1%, +0.1%]: 2 wins but mean below threshold and noisy."""
    cfg = SearchConfig()
    identity = _identity()
    hier = _cand("hierarchical", (0.1005, 0.0999, 0.0999), drifts=(0.0005,) * 3)
    decision = select_candidate([identity, hier], cfg)
    assert not decision.accepted
    assert decision.rejection_reason in {
        "relative_improvement_below_threshold",
        "improvement_not_above_split_variance",
    }


def test_consistent_small_gain_accepted():
    """deltas [+0.25%, +0.22%, +0.20%], drift 0.05% → accept."""
    cfg = SearchConfig()
    identity = _identity()
    hier = _cand(
        "hierarchical",
        (0.09975, 0.09978, 0.09980),
        drifts=(0.0005, 0.0005, 0.0005),
    )
    decision = select_candidate([identity, hier], cfg)
    assert decision.accepted
    assert decision.selected_name == "hierarchical"


def test_high_bf16_drift_rejected():
    cfg = SearchConfig()
    identity = _identity()
    hier = _cand(
        "hierarchical",
        (0.0995, 0.0995, 0.0995),
        drifts=(0.005, 0.005, 0.005),
    )
    decision = select_candidate([identity, hier], cfg)
    assert not decision.accepted
    assert decision.rejection_reason == "bf16_reorder_drift_above_threshold"


def test_insufficient_wins_rejected():
    """Only 1 of 3 splits improves."""
    cfg = SearchConfig()
    identity = _identity()
    hier = _cand("hierarchical", (0.0990, 0.1005, 0.1005), drifts=(0.0005,) * 3)
    decision = select_candidate([identity, hier], cfg)
    assert not decision.accepted
    assert decision.rejection_reason == "insufficient_validation_wins"


def test_aggregate_metrics_recorded():
    cfg = SearchConfig()
    identity = _identity()
    hier = _cand("hierarchical", (0.09975, 0.09978, 0.09980), drifts=(0.0005,) * 3)
    decision = select_candidate([identity, hier], cfg)
    agg = decision.aggregate_metrics
    assert "identity" in agg and "hierarchical" in agg
    for key in (
        "mean_total_nrmse",
        "std_total_nrmse",
        "mean_quantization_residual_nrmse",
        "mean_bf16_reorder_drift",
        "wins_vs_identity",
        "relative_improvement_pct",
    ):
        assert key in agg["hierarchical"]
    assert isinstance(decision, CandidateDecision)


def test_requires_identity_baseline():
    cfg = SearchConfig()
    hier = _cand("hierarchical", (0.09, 0.09, 0.09))
    with pytest.raises(ValueError, match="identity"):
        select_candidate([hier], cfg)
