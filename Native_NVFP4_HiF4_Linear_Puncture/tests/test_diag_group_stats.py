"""K=64 group DIAG activation stats: formulas, rollback identity, zero-denominator."""

from __future__ import annotations

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.src.diag_group_stats import (
    GROUP_SIZE,
    apply_channel_diagonal,
    assert_rollback_identity,
    compute_group64_metrics,
    evaluate_module_split,
    reshape_to_groups,
)


def _hand_metrics(g: torch.Tensor) -> dict[str, float]:
    g = g.to(torch.float64).reshape(-1)
    if g.numel() != GROUP_SIZE:
        raise ValueError("hand metrics expect 64 values")
    mean = float(g.mean().item())
    centered = g - mean
    var = float((centered * centered).mean().item())
    m4 = float((centered ** 4).mean().item())
    abs_g = g.abs()
    sorted_abs = abs_g.sort().values
    median_abs = float((0.5 * (sorted_abs[31] + sorted_abs[32])).item())
    mean_abs = float(abs_g.mean().item())
    std_abs = float(((abs_g - mean_abs).pow(2).mean().sqrt()).item())
    return {
        "variance": var,
        "kurtosis": m4 / (var * var),
        "max_mid": float(abs_g.max().item()) / median_abs,
        "divergence": std_abs / mean_abs,
    }


def test_group64_metrics_match_hand_formula():
    g = torch.arange(1, 65, dtype=torch.float64)
    got = compute_group64_metrics(g)
    expected = _hand_metrics(g)
    for name in expected:
        assert abs(float(got[name].item()) - expected[name]) < 1e-12


def test_uniform_positive_scale_leaves_shape_metrics():
    g = torch.arange(1, 65, dtype=torch.float64)
    scaled = g / 2.0
    before = compute_group64_metrics(g)
    after = compute_group64_metrics(scaled)
    assert abs(float(after["variance"].item()) - float(before["variance"].item()) / 4.0) < 1e-12
    assert abs(float(after["kurtosis"].item()) - float(before["kurtosis"].item())) < 1e-12
    assert abs(float(after["max_mid"].item()) - float(before["max_mid"].item())) < 1e-12
    assert abs(float(after["divergence"].item()) - float(before["divergence"].item())) < 1e-12


def test_unequal_channel_scale_changes_all_four_metrics():
    g = torch.arange(1, 65, dtype=torch.float64)
    d = torch.ones(64, dtype=torch.float64)
    d[32:] = 2.0
    after = g / d
    b = compute_group64_metrics(g)
    a = compute_group64_metrics(after)
    expected = _hand_metrics(after)
    for name in expected:
        assert abs(float(a[name].item()) - expected[name]) < 1e-12
        assert abs(float(a[name].item()) - float(b[name].item())) > 1e-12


def test_rollback_d_eq_1_leaves_x_and_metrics_unchanged():
    x = torch.arange(1, 129, dtype=torch.float64).reshape(1, 128)
    d = torch.ones(128, dtype=torch.float64)
    kept = torch.tensor([False, False])
    x_d = apply_channel_diagonal(x, d)
    assert torch.equal(x_d, x)
    assert_rollback_identity(x, x_d, d, kept, module_name="m", split="val")
    before = compute_group64_metrics(reshape_to_groups(x))
    after = compute_group64_metrics(reshape_to_groups(x_d))
    for name in before:
        assert torch.equal(before[name], after[name])


def test_evaluate_module_split_rollback_delta_is_zero():
    x = torch.arange(1, 129, dtype=torch.float64).reshape(2, 64)
    d = torch.ones(64, dtype=torch.float64)
    kept = torch.tensor([False])
    out = evaluate_module_split(x, d, kept, module_name="m", split="val")
    rb = next(r for r in out["subset_rows"] if r["subset"] == "rollback")
    for name in ("variance", "kurtosis", "max_mid", "divergence"):
        assert rb[f"{name}_delta_median"] == 0.0
        assert rb[f"{name}_delta_mean"] == 0.0


def test_zero_median_abs_raises():
    g = torch.zeros(64, dtype=torch.float64)
    g[33:] = torch.arange(1, 32, dtype=torch.float64)
    with pytest.raises(RuntimeError, match=r"median\(\|g\|\)==0"):
        compute_group64_metrics(g, module_name="m", split="val")


def test_zero_variance_raises():
    g = torch.full((64,), 3.0, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="Var==0"):
        compute_group64_metrics(g, module_name="m", split="val")


def test_assert_distinct_capture_and_run_ids():
    from Native_NVFP4_HiF4_Linear_Puncture.src.grid_scale_validation import (
        assert_distinct_run_ids,
    )

    with pytest.raises(ValueError):
        assert_distinct_run_ids("same", "same")
    assert_distinct_run_ids("capture", "stats")
