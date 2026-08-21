"""Unit tests for activation_distribution_residual (Task 1)."""

from __future__ import annotations

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_distribution_residual import (
    activation_quantile_residual_curve,
    build_token_group_residual_map,
    group64_residual_stats,
    residual_element_stats,
    residual_energy_concentration,
    zero_transition_stats,
)


def test_delta_stats_match_hand():
    an = torch.tensor([1.0, -2.0, 0.0, 3.0])
    ah = torch.tensor([1.5, -1.0, 0.5, 3.0])
    st = residual_element_stats(an, ah)
    delta = ah - an
    assert abs(st["mean_delta"] - float(delta.mean())) < 1e-6
    assert abs(st["mae"] - float(delta.abs().mean())) < 1e-6
    assert abs(st["rms"] - float(torch.sqrt((delta * delta).mean()))) < 1e-6
    assert st["numel"] == 4.0


def test_zero_transition_sums_to_one():
    an = torch.tensor([0.0, 0.0, 1.0, 2.0])
    ah = torch.tensor([0.0, 1.0, 0.0, 3.0])
    z = zero_transition_stats(an, ah)
    s = z["both_zero"] + z["nv_zero_hf_nonzero"] + z["nv_nonzero_hf_zero"] + z["both_nonzero"]
    assert abs(s - 1.0) < 1e-8
    assert abs(z["both_zero"] - 0.25) < 1e-8
    assert abs(z["nv_zero_hf_nonzero"] - 0.25) < 1e-8
    assert abs(z["nv_nonzero_hf_zero"] - 0.25) < 1e-8
    assert abs(z["both_nonzero"] - 0.25) < 1e-8


def test_identical_degenerates_to_zero():
    a = torch.randn(128, 64)
    st = residual_element_stats(a, a)
    z = zero_transition_stats(a, a)
    e = residual_energy_concentration(a - a)
    assert st["rms"] == 0.0
    assert st["nmse_hif4_vs_nvfp4"] == 0.0
    assert e["top_1pct_energy_share"] == 0.0
    assert abs(z["both_zero"] + z["both_nonzero"] - 1.0) < 1e-8


def test_outlier_dominates_top_energy():
    d = torch.zeros(10000)
    d[0] = 100.0
    e = residual_energy_concentration(d)
    assert e["top_0p1pct_energy_share"] > 0.99
    assert e["top_1pct_energy_share"] > 0.99


def test_quantile_curve_count_sum():
    an = torch.randn(1000)
    delta = torch.randn(1000) * 0.1
    c = activation_quantile_residual_curve(an, delta, num_bins=32)
    assert int(sum(c["count"])) == 1000


def test_group64_rows_and_token_map():
    t, k = 2, 128
    x = torch.randn(t, k)
    an = x + 0.01
    ah = x + 0.02
    rows = group64_residual_stats(x, an, ah)
    assert len(rows) == k // 64
    assert all(r["num_tokens"] == t for r in rows)
    m = build_token_group_residual_map(ah - an, group_size=64)
    assert tuple(m.shape) == (2, 2)
