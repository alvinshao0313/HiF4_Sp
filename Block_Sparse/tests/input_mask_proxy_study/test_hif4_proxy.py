from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.input_mask_proxy_study.hif4_proxy import (  # noqa: E402
    build_hif4_ternary_proxy,
    quantize_e6m2_nonnegative,
)


def test_e6m2_zero():
    x = torch.tensor([0.0], dtype=torch.float32)
    assert quantize_e6m2_nonnegative(x).item() == 0.0


def test_e6m2_min_normal():
    x = torch.tensor([2.0**-48], dtype=torch.float32)
    y = quantize_e6m2_nonnegative(x)
    assert y.item() == pytest.approx(2.0**-48)


def test_e6m2_power_of_two_boundary():
    x = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32)
    y = quantize_e6m2_nonnegative(x)
    assert torch.equal(y, x)


def test_e6m2_ties_to_even():
    # At exponent e=0: grid is m * 2^-2 with m in {4,5,6,7,...}
    # 1.125 is halfway between 1.0 (m=4) and 1.25 (m=5) -> even m=4 -> 1.0
    x = torch.tensor([1.125], dtype=torch.float32)
    y = quantize_e6m2_nonnegative(x)
    assert y.item() == pytest.approx(1.0)
    # 1.375 halfway between 1.25 (m=5) and 1.5 (m=6) -> even m=6 -> 1.5
    x2 = torch.tensor([1.375], dtype=torch.float32)
    y2 = quantize_e6m2_nonnegative(x2)
    assert y2.item() == pytest.approx(1.5)


def test_e6m2_max_clamp():
    x = torch.tensor([1.75 * (2.0**15), 1e30], dtype=torch.float32)
    y = quantize_e6m2_nonnegative(x)
    assert torch.all(y == 1.5 * (2.0**15))


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_e6m2_rejects_invalid(bad: float):
    x = torch.tensor([bad], dtype=torch.float32)
    with pytest.raises(ValueError):
        quantize_e6m2_nonnegative(x)


def test_zero_group_no_nan():
    x = torch.zeros(2, 64, dtype=torch.float32)
    out = build_hif4_ternary_proxy(x)
    assert torch.isfinite(out.proxy).all()
    assert torch.isfinite(out.hif4_dequant).all()
    assert torch.isfinite(out.local_scale).all()
    assert torch.equal(out.proxy, torch.zeros_like(out.proxy))
    assert torch.equal(out.ternary_code, torch.zeros_like(out.ternary_code))


def test_ternary_code_values():
    torch.manual_seed(0)
    x = torch.randn(3, 128, dtype=torch.float32)
    out = build_hif4_ternary_proxy(x)
    uniq = set(out.ternary_code.unique().tolist())
    assert uniq.issubset({-1.0, 0.0, 1.0})


def test_payload_properties():
    torch.manual_seed(1)
    x = torch.randn(4, 64, dtype=torch.float32)
    out = build_hif4_ternary_proxy(x)
    scaled = out.payload * 4.0
    assert torch.allclose(scaled, scaled.round(), atol=1e-6)
    assert torch.all(out.payload >= 0.0)
    assert torch.all(out.payload <= 1.75)


def test_proxy_and_dequant_identities():
    torch.manual_seed(2)
    x = torch.randn(2, 192, dtype=torch.float32)
    out = build_hif4_ternary_proxy(x)
    assert torch.allclose(out.proxy, out.local_scale * out.ternary_code, atol=0.0, rtol=0.0)
    sign = torch.sign(x)
    assert torch.allclose(
        out.hif4_dequant, out.local_scale * sign * out.payload, atol=1e-6, rtol=0.0
    )


def test_e8_e4_binary_and_local_scale_groups():
    torch.manual_seed(3)
    x = torch.randn(1, 64, dtype=torch.float32) * 10
    out = build_hif4_ternary_proxy(x)
    assert set(out.e8.unique().tolist()).issubset({0.0, 1.0})
    assert set(out.e4.unique().tolist()).issubset({0.0, 1.0})
    ls = out.local_scale.view(-1, 16, 4)
    assert torch.equal(ls, ls[:, :, :1].expand_as(ls))


def test_last_dim_not_multiple_of_64():
    x = torch.randn(2, 63, dtype=torch.float32)
    with pytest.raises(ValueError):
        build_hif4_ternary_proxy(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cross_check_hif4_fake_quantize():
    from HiFloat4.permutation_optimization.hif4_reference import hif4_fake_quantize

    torch.manual_seed(4)
    x = torch.randn(8, 256, device="cuda", dtype=torch.float32)
    out = build_hif4_ternary_proxy(x)
    ref = hif4_fake_quantize(x).to(torch.float32)
    tol = 1e-3 * ref.abs().max().clamp_min(0.0) + 1e-6
    assert torch.max(torch.abs(out.hif4_dequant - ref)).item() <= float(tol)
