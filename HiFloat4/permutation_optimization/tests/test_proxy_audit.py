"""Tests for G4-proxy vs real-G64 ranking correlation audit."""

from __future__ import annotations

import math

import pytest
import torch

from permutation_optimization.config import SearchConfig
from permutation_optimization.proxy_audit import (
    ProxyAuditResult,
    _rank_correlations,
    audit_g4_proxy_ranking,
)
from permutation_optimization.objective import build_channel_statistics


def test_identical_ranking_gives_correlation_one():
    proxy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    real = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    r = _rank_correlations(proxy, real)
    assert r.spearman == pytest.approx(1.0, abs=1e-12)
    assert r.pearson == pytest.approx(1.0, abs=1e-12)
    assert r.top1_match is True
    assert r.top5_overlap == 1.0
    assert r.n_candidates == 6


def test_reversed_ranking_gives_spearman_minus_one():
    proxy = [0.1, 0.2, 0.3, 0.4]
    real = [4.0, 3.0, 2.0, 1.0]
    r = _rank_correlations(proxy, real)
    assert r.spearman == pytest.approx(-1.0, abs=1e-12)
    assert r.top1_match is False
    assert r.top5_overlap == 1.0  # 4 candidates: top-5 covers all


def test_audit_returns_finite_and_correct_count():
    torch.manual_seed(37)
    d_ff = 64
    act = torch.randn(32, d_ff)
    w = torch.randn(32, d_ff)
    cfg = SearchConfig(candidate_window=32, neighbor_k=16)
    stats = build_channel_statistics(act, w, cfg)
    base_g64 = list(range(d_ff))
    pool = list(range(16))
    candidates = [tuple(pool[i : i + 4]) for i in range(0, 16, 4)]
    candidates += [(0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)]
    result = audit_g4_proxy_ranking(act, w, base_g64, candidates, stats, cfg)
    assert isinstance(result, ProxyAuditResult)
    assert result.n_candidates == len(candidates)
    for v in (result.spearman, result.pearson, result.top5_overlap):
        assert math.isfinite(v)
    assert -1.0 <= result.spearman <= 1.0
    assert -1.0 <= result.pearson <= 1.0
    assert 0.0 <= result.top5_overlap <= 1.0
    assert isinstance(result.top1_match, bool)


def test_audit_rejects_inconsistent_candidates():
    torch.manual_seed(41)
    d_ff = 64
    act = torch.randn(16, d_ff)
    w = torch.randn(16, d_ff)
    cfg = SearchConfig()
    stats = build_channel_statistics(act, w, cfg)
    # Candidate channel outside base G64 must fail loudly.
    bad_candidates = [(0, 1, 2, 99)]
    try:
        audit_g4_proxy_ranking(act, w, list(range(d_ff)), bad_candidates, stats, cfg)
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-block candidate")
