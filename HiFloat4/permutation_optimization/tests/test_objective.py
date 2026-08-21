"""Tests for objective (stats, C4/C64, NRMSE)."""

from __future__ import annotations

import torch
import pytest

from permutation_optimization.config import SearchConfig
from permutation_optimization.hif4_reference import hif4_fake_quantize, s1p2_oracle_quantize_rows
from permutation_optimization.objective import (
    build_channel_statistics,
    c4_cost,
    c64_cost,
    down_output_nrmse,
    pair_compatibility_cost,
    sample_weight_rows,
)


def test_same_scale_channels_are_neighbors():
    torch.manual_seed(0)
    d_ff = 128
    rows = 64
    # Three scale clusters interleaved.
    act = torch.zeros(rows, d_ff)
    w = torch.zeros(rows, d_ff)
    for c in range(d_ff):
        scale = [1.0, 4.0, 16.0][c % 3]
        act[:, c] = scale * torch.randn(rows)
        w[:, c] = scale * torch.randn(rows)
    cfg = SearchConfig(candidate_window=64, neighbor_k=8, activation_rows=rows, weight_rows=rows)
    stats = build_channel_statistics(act, w, cfg)
    # Channel 0 (scale1) should have neighbors mostly congruent mod 3.
    neigh = stats.neighbors[0].tolist()
    same = sum(1 for j in neigh if j % 3 == 0)
    assert same >= 4


def test_no_full_cdist(monkeypatch):
    called = {"n": 0}
    real_cdist = torch.cdist

    def wrapped(*args, **kwargs):
        called["n"] += 1
        return real_cdist(*args, **kwargs)

    monkeypatch.setattr(torch, "cdist", wrapped)
    d_ff = 128
    act = torch.randn(32, d_ff)
    w = torch.randn(32, d_ff)
    cfg = SearchConfig(candidate_window=16, neighbor_k=8)
    _ = build_channel_statistics(act, w, cfg)
    assert called["n"] == 0


def test_sample_weight_rows_deterministic():
    w = torch.randn(200, 64)
    a = sample_weight_rows(w, 32, seed=3)
    b = sample_weight_rows(w, 32, seed=3)
    c = sample_weight_rows(w, 32, seed=4)
    assert torch.equal(a, b)
    assert a.shape == (32, 64)
    assert not torch.equal(a, c)


def test_compatible_vs_alternating():
    rows = 64
    # Channels 0,1 share abs trajectory; channel 2 peaks when 0 is near zero.
    t = torch.linspace(0, 6.28, rows)
    a0 = torch.sin(t).abs() + 0.05
    a1 = a0 * 1.05
    a2 = torch.cos(t).abs() + 0.05  # phase-shifted abs peaks
    a3 = torch.ones(rows) * 0.1
    act = torch.stack([a0, a1, a2, a3], dim=1)
    act = torch.cat([act, torch.randn(rows, 60) * 0.01], dim=1)
    w = act.clone()
    cfg = SearchConfig()
    stats = build_channel_statistics(act, w, cfg)
    c01 = pair_compatibility_cost(0, 1, act, w, stats, cfg.eps)
    c02 = pair_compatibility_cost(0, 2, act, w, stats, cfg.eps)
    assert c01 < c02


def test_c4_similar_better_than_spread():
    cfg = SearchConfig()
    # Fake stats
    from permutation_optimization.objective import ChannelStatistics

    good = torch.tensor([[1.0, 0.9, 0.8, 0.7]], dtype=torch.float32)
    bad = torch.tensor([[1.0, 0.1, 0.01, 0.001]], dtype=torch.float32)
    # Expand to multiple rows
    good = good.repeat(16, 1)
    bad = bad.repeat(16, 1)
    # Pad channels to build stats on 64 dims
    act_g = torch.cat([good, torch.randn(16, 60) * 0.01], dim=1)
    w_g = act_g.clone()
    stats = build_channel_statistics(act_g, w_g, cfg)
    c_good = c4_cost([0, 1, 2, 3], good, good, stats, cfg)
    # For bad, reuse stats energies for first 4
    c_bad = c4_cost([0, 1, 2, 3], bad, bad, stats, cfg)
    assert c_good < c_bad


def test_c4_grid_near_zero_error():
    cfg = SearchConfig()
    pts = torch.tensor([[0.0, 0.25, 0.5, 1.75]]).repeat(8, 1)
    from permutation_optimization.objective import ChannelStatistics as CS

    e = torch.ones(4)
    stats4 = CS(
        activation_energy=e,
        weight_energy=e,
        output_sensitivity=e,
        features=torch.zeros(4, 12),
        primary_scale=torch.zeros(4),
        neighbors=torch.zeros(4, 2, dtype=torch.long),
    )
    cost = c4_cost([0, 1, 2, 3], pts, pts, stats4, cfg)
    assert cost < 1e-4


def test_c64_matches_manual():
    cfg = SearchConfig()
    torch.manual_seed(0)
    act = torch.randn(32, 64)
    w = torch.randn(32, 64)
    stats = build_channel_statistics(act, w, cfg)
    ch = list(range(64))
    cost = c64_cost(ch, act, w, stats, cfg)
    aq = hif4_fake_quantize(act)
    wq = hif4_fake_quantize(w)
    # Cross energy: activation error weighted by weight column energy, and
    # weight error weighted by activation energy.
    ea_err = (
        (stats.weight_energy * (act - aq) ** 2).sum()
        / ((stats.weight_energy * act * act).sum() + cfg.eps)
    ).item()
    ew_err = (
        (stats.activation_energy * (w - wq) ** 2).sum()
        / ((stats.activation_energy * w * w).sum() + cfg.eps)
    ).item()
    manual = 0.5 * ea_err + 0.5 * ew_err
    assert abs(cost - manual) < 1e-6


def _distinct_stats(d_ff: int, seed: int = 0) -> "ChannelStatistics":
    """Stats where activation/weight energy and output sensitivity all differ."""
    from permutation_optimization.objective import ChannelStatistics

    g = torch.Generator().manual_seed(seed)
    e_a = torch.rand(d_ff, generator=g) + 0.5
    e_w = torch.rand(d_ff, generator=g) * 100 + 0.5
    s = torch.rand(d_ff, generator=g) * 7 + 0.5
    return ChannelStatistics(
        activation_energy=e_a,
        weight_energy=e_w,
        output_sensitivity=s,
        features=torch.zeros(d_ff, 12),
        primary_scale=torch.zeros(d_ff),
        neighbors=torch.zeros(d_ff, 2, dtype=torch.long),
    )


def test_c4_activation_error_weighted_by_weight_energy():
    """Isolate activation term: cost must match manual weighting by weight_energy,
    not by output_sensitivity."""
    from permutation_optimization.objective import _weighted_nrmse

    cfg = SearchConfig(
        activation_loss_weight=1.0, weight_loss_weight=0.0, range_loss_weight=0.0
    )
    torch.manual_seed(1)
    act = torch.randn(16, 4)
    w = torch.randn(16, 4)
    stats = _distinct_stats(4)
    cost = c4_cost([0, 1, 2, 3], act, w, stats, cfg)
    act_q = s1p2_oracle_quantize_rows(act, eps=cfg.eps)
    manual = _weighted_nrmse(act, act_q, stats.weight_energy, cfg.eps)
    wrong = _weighted_nrmse(act, act_q, stats.output_sensitivity, cfg.eps)
    assert abs(cost - manual) < 1e-9
    assert abs(cost - wrong) > 1e-6


def test_c4_weight_error_weighted_by_activation_energy():
    """Isolate weight term: cost must match manual weighting by activation_energy,
    not by output_sensitivity."""
    from permutation_optimization.objective import _weighted_nrmse

    cfg = SearchConfig(
        activation_loss_weight=0.0, weight_loss_weight=1.0, range_loss_weight=0.0
    )
    torch.manual_seed(2)
    act = torch.randn(16, 4)
    w = torch.randn(16, 4)
    stats = _distinct_stats(4, seed=5)
    cost = c4_cost([0, 1, 2, 3], act, w, stats, cfg)
    w_q = s1p2_oracle_quantize_rows(w, eps=cfg.eps)
    manual = _weighted_nrmse(w, w_q, stats.activation_energy, cfg.eps)
    wrong = _weighted_nrmse(w, w_q, stats.output_sensitivity, cfg.eps)
    assert abs(cost - manual) < 1e-9
    assert abs(cost - wrong) > 1e-6


def test_pair_compatibility_uses_cross_energy_weights():
    """Pair cost: activation trajectory weighted by weight_energy, weight
    trajectory weighted by activation_energy — never output_sensitivity."""
    torch.manual_seed(3)
    rows = 32
    act = torch.randn(rows, 4)
    w = torch.randn(rows, 4)
    stats = _distinct_stats(4, seed=7)
    eps = 1e-8
    cost = pair_compatibility_cost(0, 1, act, w, stats, eps)

    def _traj(x: torch.Tensor, importance: torch.Tensor) -> float:
        xi, xj = x[:, 0], x[:, 1]
        large = torch.maximum(xi.abs(), xj.abs())
        small = torch.minimum(xi.abs(), xj.abs())
        log_ratio = torch.log2((large + eps) / (small + 0.01 * large + eps))
        row_w = importance[0] * xi * xi + importance[1] * xj * xj
        den = row_w.sum()
        return float(((log_ratio * log_ratio) * row_w).sum().item() / den.item())

    manual = 0.5 * _traj(act, stats.weight_energy) + 0.5 * _traj(w, stats.activation_energy)
    wrong = 0.5 * _traj(act, stats.output_sensitivity) + 0.5 * _traj(w, stats.output_sensitivity)
    assert abs(cost - manual) < 1e-6
    assert abs(cost - wrong) > 1e-5


def test_precompute_neighbor_pair_costs_matches_pair_cost():
    from permutation_optimization.objective import precompute_neighbor_pair_costs

    torch.manual_seed(4)
    d_ff = 64
    act = torch.randn(24, d_ff)
    w = torch.randn(24, d_ff)
    cfg = SearchConfig(candidate_window=32, neighbor_k=8)
    stats = build_channel_statistics(act, w, cfg)
    cache = precompute_neighbor_pair_costs(act, w, stats, cfg.eps)
    assert cache
    for (i, j), c in list(cache.items())[:8]:
        ref = pair_compatibility_cost(i, j, act, w, stats, cfg.eps)
        assert abs(c - ref) < 1e-6


def test_full_layout_matches_manual_cross_weighted():
    from permutation_optimization.objective import _weighted_nrmse, full_layout_hif4_loss

    cfg = SearchConfig()
    torch.manual_seed(6)
    d_ff = 64
    a = torch.randn(16, d_ff)
    w = torch.randn(16, d_ff)
    stats = _distinct_stats(d_ff, seed=9)
    identity = torch.arange(d_ff)
    loss, _blocks = full_layout_hif4_loss(identity, a, w, stats, cfg)
    qa = hif4_fake_quantize(a)
    qw = hif4_fake_quantize(w)
    a_err = _weighted_nrmse(a, qa, stats.weight_energy, cfg.eps)
    w_err = _weighted_nrmse(w, qw, stats.activation_energy, cfg.eps)
    manual = 0.5 * a_err + 0.5 * w_err
    assert abs(loss - manual) < 1e-6


def test_down_output_nrmse_identity():
    torch.manual_seed(0)
    a = torch.randn(16, 64)
    w = torch.randn(32, 64)
    perm = torch.arange(64)
    nrmse = down_output_nrmse(a, w, perm)
    # Manual
    y_fp = a @ w.T
    aq = hif4_fake_quantize(a)
    wq = hif4_fake_quantize(w)
    y_q = aq @ wq.T
    manual = (torch.linalg.norm(y_q - y_fp) / (torch.linalg.norm(y_fp) + 1e-8)).item()
    assert abs(nrmse - manual) < 1e-6


def test_zero_denom_returns_zero():
    cfg = SearchConfig()
    from permutation_optimization.objective import ChannelStatistics as CS

    z = torch.zeros(4, 4)
    e = torch.zeros(4)
    stats = CS(
        activation_energy=e,
        weight_energy=e,
        output_sensitivity=e,
        features=torch.zeros(4, 12),
        primary_scale=torch.zeros(4),
        neighbors=torch.zeros(4, 2, dtype=torch.long),
    )
    cost = c4_cost([0, 1, 2, 3], z, z, stats, cfg)
    assert cost == 0.0
    assert not (cost != cost)  # not NaN


def test_build_quantized_swiglu_activation_permutation_equivariance():
    from permutation_optimization.objective import build_quantized_swiglu_activation

    torch.manual_seed(0)
    d_model, d_ff = 64, 128
    x = torch.randn(8, d_model) * 0.1
    wu = torch.randn(d_ff, d_model) * 0.05
    wg = torch.randn(d_ff, d_model) * 0.05
    a = build_quantized_swiglu_activation(x, wu, wg)
    assert a.shape == (8, d_ff)
    a = a.detach().cpu()
    perm = torch.randperm(d_ff)
    a_perm = build_quantized_swiglu_activation(x, wu[perm], wg[perm]).detach().cpu()
    assert torch.allclose(a_perm, a.index_select(1, perm), rtol=1e-5, atol=1e-5)


def _make_deployment_context(d_model: int = 64, d_ff: int = 128, rows: int = 8,
                             device: str = "cpu", seed: int = 0):
    from permutation_optimization.objective import DeploymentMLPContext

    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, d_model, generator=g) * 0.5).to(torch.bfloat16)
    wu = (torch.randn(d_ff, d_model, generator=g) * 0.05).to(torch.bfloat16)
    wg = (torch.randn(d_ff, d_model, generator=g) * 0.05).to(torch.bfloat16)
    wd = (torch.randn(d_model, d_ff, generator=g) * 0.05).to(torch.bfloat16)
    ctx = DeploymentMLPContext(x, wu, wg, wd, torch.device(device))
    return ctx, x, wu, wg, wd


def test_deployment_context_identity_has_zero_reorder_drift():
    ctx, *_ = _make_deployment_context()
    metrics = ctx.evaluate(torch.arange(128))
    assert metrics.bf16_reorder_drift == pytest.approx(0.0, abs=1e-8)
    # For identity, total error must equal the quantization residual itself.
    assert metrics.total_nrmse == pytest.approx(metrics.quantization_residual_nrmse, abs=1e-8)


def test_deployment_context_metric_references():
    """residual uses candidate's own BF16 output; total uses identity BF16 output."""
    ctx, *_ = _make_deployment_context()
    perm = torch.randperm(128, generator=torch.Generator().manual_seed(3))
    y_bf16_p, y_w4a4_p = ctx._debug_forward(perm)
    y_bf16_i, _ = ctx._debug_forward(torch.arange(128))

    def _nrmse(a: torch.Tensor, ref: torch.Tensor) -> float:
        num = torch.linalg.norm(a.float() - ref.float(), ord="fro")
        den = torch.linalg.norm(ref.float(), ord="fro")
        return float((num / (den + 1e-8)).item())

    metrics = ctx.evaluate(perm)
    assert metrics.quantization_residual_nrmse == pytest.approx(
        _nrmse(y_w4a4_p, y_bf16_p), abs=1e-8
    )
    assert metrics.total_nrmse == pytest.approx(_nrmse(y_w4a4_p, y_bf16_i), abs=1e-8)
    assert metrics.bf16_reorder_drift == pytest.approx(_nrmse(y_bf16_p, y_bf16_i), abs=1e-8)


def test_deployment_context_reproducible():
    ctx, *_ = _make_deployment_context()
    perm = torch.randperm(128, generator=torch.Generator().manual_seed(5))
    m1 = ctx.evaluate(perm)
    m2 = ctx.evaluate(perm)
    assert m1 == m2


def test_deployment_context_rejects_shape_mismatch():
    from permutation_optimization.objective import DeploymentMLPContext

    x = torch.randn(4, 64, dtype=torch.bfloat16)
    wu = torch.randn(128, 64, dtype=torch.bfloat16)
    wg = torch.randn(128, 64, dtype=torch.bfloat16)
    wd = torch.randn(64, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        DeploymentMLPContext(x, wu, wg, wd[:-1], torch.device("cpu"))


def test_mlp_w4a4_output_nrmse_identity_finite_and_reproducible():
    from permutation_optimization.objective import mlp_w4a4_output_nrmse

    torch.manual_seed(0)
    d_model, d_ff = 64, 128
    x = torch.randn(16, d_model) * 0.1
    wu = torch.randn(d_ff, d_model) * 0.05
    wg = torch.randn(d_ff, d_model) * 0.05
    wd = torch.randn(d_model, d_ff) * 0.05
    identity = torch.arange(d_ff)
    n1 = mlp_w4a4_output_nrmse(x, wu, wg, wd, identity)
    n2 = mlp_w4a4_output_nrmse(x, wu, wg, wd, identity)
    assert n1 == n2
    assert 0.0 <= n1 < 10.0
