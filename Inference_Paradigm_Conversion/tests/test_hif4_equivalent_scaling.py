"""Unit tests for HiF4 deployment-equivalent scaling primitives."""

from __future__ import annotations

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.hif4_equivalent_scaling import (
    ScalingSpec,
    apply_linear_equivalent_scaling,
    build_equalization_scale,
    build_weight_aware_equalization_scale,
    candidate_pts_scales,
    collapse_gqa_o_amplitude,
    expand_gqa_o_scale,
    expand_group_scales,
    finalize_channel_amplitude,
    shared_input_weight_stat,
    update_channel_stats,
)


def test_linear_equivalent_scaling_exact_fp32():
    torch.manual_seed(0)
    x = torch.randn(7, 128)
    w = torch.randn(64, 128)
    d = torch.exp2(torch.linspace(-1, 1, 128))
    xs, ws = apply_linear_equivalent_scaling(x, w, d)
    torch.testing.assert_close(xs @ ws.T, x @ w.T, rtol=1e-5, atol=1e-6)


def test_alpha_zero_is_identity():
    amp = torch.rand(256) + 0.1
    d = build_equalization_scale(amp, granularity=16, alpha=0.0)
    torch.testing.assert_close(d, torch.ones_like(d), rtol=0, atol=0)


def test_granularity16_repeats_each_scale_16_times():
    amp = torch.arange(1, 65, dtype=torch.float32)
    d = build_equalization_scale(amp, granularity=16, alpha=1.0)
    assert d.shape == (64,)
    for start in range(0, 64, 16):
        assert torch.unique(d[start : start + 16]).numel() == 1


def test_scale_is_bounded():
    amp = torch.logspace(-8, 8, 64)
    d = build_equalization_scale(amp, granularity=1, alpha=1.0)
    assert float(d.min()) >= 0.5
    assert float(d.max()) <= 2.0


def test_zero_units_do_not_shift_group_center():
    amp = torch.ones(64)
    amp[0] = 0.0
    d = build_equalization_scale(amp, granularity=1, alpha=1.0)
    assert d[0].item() == 1.0
    torch.testing.assert_close(d[1:], torch.ones_like(d[1:]), rtol=0, atol=0)


def test_all_zero_group_is_identity():
    amp = torch.zeros(64)
    d = build_equalization_scale(amp, granularity=4, alpha=1.0)
    torch.testing.assert_close(d, torch.ones_like(d), rtol=0, atol=0)


def test_invalid_granularity_is_rejected():
    amp = torch.ones(64)
    try:
        build_equalization_scale(amp, granularity=3, alpha=0.5)
    except ValueError as exc:
        assert "granularity" in str(exc)
    else:
        raise AssertionError("granularity=3 must be rejected")


def test_channel_stats_use_all_leading_dims_and_preserve_zero_channel():
    x1 = torch.tensor(
        [[[1.0, 0.0], [3.0, 0.0]], [[2.0, 0.0], [4.0, 0.0]]],
        dtype=torch.float32,
    )
    sum_sq = torch.zeros(2, dtype=torch.float64)
    max_abs = torch.zeros(2, dtype=torch.float64)
    sum_sq, max_abs, count = update_channel_stats(sum_sq, max_abs, 0, x1)
    assert count == 4
    torch.testing.assert_close(sum_sq, torch.tensor([30.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(max_abs, torch.tensor([4.0, 0.0], dtype=torch.float64))
    amp = finalize_channel_amplitude(sum_sq, max_abs, count)
    expected_rms0 = (30.0 / 4.0) ** 0.5
    expected_amp0 = (4.0 * expected_rms0) ** 0.5
    torch.testing.assert_close(amp[0], torch.tensor(expected_amp0, dtype=torch.float64))
    assert amp[1].item() == 0.0


def test_pts_grid_has_33_log2_uniform_values():
    c = candidate_pts_scales(log2_min=-1.0, log2_max=1.0, points=33)
    assert c.shape == (33,)
    torch.testing.assert_close(c[0], torch.tensor(0.5), rtol=0, atol=1e-7)
    torch.testing.assert_close(c[-1], torch.tensor(2.0), rtol=0, atol=1e-7)
    steps = torch.diff(torch.log2(c))
    torch.testing.assert_close(steps, torch.full_like(steps, 1.0 / 16.0), rtol=0, atol=1e-6)


def test_expand_group_scales_matches_contiguous_64_groups():
    g = torch.tensor([0.5, 1.0, 2.0])
    d = expand_group_scales(g, width=192, group_size=64)
    assert d.shape == (192,)
    torch.testing.assert_close(d[:64], torch.full((64,), 0.5))
    torch.testing.assert_close(d[64:128], torch.ones(64))
    torch.testing.assert_close(d[128:], torch.full((64,), 2.0))


def test_shared_input_weight_stat_takes_max_over_rows_and_modules():
    w1 = torch.tensor([[1.0, 2.0, 0.5], [4.0, 1.0, 3.0]])
    w2 = torch.tensor([[2.0, 5.0, 1.0]])
    stat = shared_input_weight_stat((w1, w2))
    torch.testing.assert_close(stat, torch.tensor([4.0, 5.0, 3.0]))


def test_weight_aware_equalization_is_identity_when_activation_matches_weight():
    amp = torch.arange(1, 65, dtype=torch.float32)
    weight_stat = amp.clone()
    d = build_weight_aware_equalization_scale(
        amp,
        weight_stat,
        granularity=1,
        beta=0.5,
    )
    torch.testing.assert_close(d, torch.ones_like(d), rtol=0, atol=1e-6)


def test_gqa_collapse_max_uses_all_query_head_copies():
    # Hq=4, Hkv=2, repeat=2, D=2. Query heads 0/1 map to KV0, 2/3 map to KV1.
    a = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 1.0],
            [4.0, 5.0],
            [2.0, 7.0],
        ]
    ).reshape(-1)
    unique = collapse_gqa_o_amplitude(
        a,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        reduction="max",
    )
    torch.testing.assert_close(unique, torch.tensor([[3.0, 2.0], [4.0, 7.0]]))


def test_gqa_expand_repeats_each_kv_head_scale_to_query_heads():
    d_unique = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    expanded = expand_gqa_o_scale(
        d_unique,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
    )
    expected = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [4.0, 8.0],
            [4.0, 8.0],
        ]
    ).reshape(-1)
    torch.testing.assert_close(expanded, expected)


def test_scaling_spec_is_explicit_about_kind_and_domain():
    spec = ScalingSpec(kind="equalize", domain="attn_in", granularity=16, alpha=0.5)
    assert spec.kind == "equalize"
    assert spec.domain == "attn_in"
    assert spec.granularity == 16
    assert spec.alpha == 0.5
