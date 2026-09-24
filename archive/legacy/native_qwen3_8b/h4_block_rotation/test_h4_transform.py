"""Minimal tests for the fixed G4 H4 transform. No quantization."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from Native_NVFP4_HiF4_Linear_Puncture.experiments.h4_block_rotation.h4_transform import (
    HIF4_GROUP_SIZE,
    apply_h4_g4,
    assert_r4_orthogonal,
    linear_prequant_equivalence_error,
    r4_matrix,
    relative_frobenius,
)


def test_r4_is_normalized_hadamard_and_orthogonal():
    r4 = r4_matrix(dtype=torch.float64)
    h4 = torch.tensor(
        [
            [1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    assert torch.equal(r4, h4 * 0.5)
    assert_r4_orthogonal(r4)
    assert torch.allclose(r4, r4.T, atol=0.0, rtol=0.0)


def test_apply_h4_rejects_last_dim_not_divisible_by_64():
    x = torch.randn(3, 32, dtype=torch.float32)
    try:
        apply_h4_g4(x)
    except ValueError as exc:
        assert "64" in str(exc)
    else:
        raise AssertionError("expected ValueError for K not divisible by 64")


def test_shape_is_restored():
    x = torch.randn(2, 5, 128, dtype=torch.float32)
    y = apply_h4_g4(x)
    assert tuple(y.shape) == tuple(x.shape)
    assert y.dtype == x.dtype


def test_norm_preservation():
    torch.manual_seed(0)
    x = torch.randn(7, 192, dtype=torch.float64)
    y = apply_h4_g4(x, compute_dtype=torch.float64, output_dtype=torch.float64)
    xn = torch.linalg.vector_norm(x)
    yn = torch.linalg.vector_norm(y)
    rel = float((yn - xn).abs().item() / xn.clamp_min(1e-30).item())
    assert rel < 1e-12


def test_involution():
    torch.manual_seed(1)
    x = torch.randn(4, 64, dtype=torch.float64)
    y = apply_h4_g4(x, compute_dtype=torch.float64, output_dtype=torch.float64)
    z = apply_h4_g4(y, compute_dtype=torch.float64, output_dtype=torch.float64)
    assert relative_frobenius(z, x) < 1e-12


def test_group_boundary_no_cross_g4_or_g64_mix():
    x = torch.zeros(1, 128, dtype=torch.float64)
    x[0, 0] = 1.0
    y = apply_h4_g4(x, compute_dtype=torch.float64, output_dtype=torch.float64)
    r4 = r4_matrix(dtype=torch.float64)
    expected_g4 = r4[0]
    assert torch.allclose(y[0, :4], expected_g4, atol=1e-12, rtol=0.0)
    assert torch.equal(y[0, 4:], torch.zeros(124, dtype=torch.float64))

    x2 = torch.zeros(1, 128, dtype=torch.float64)
    x2[0, 64] = 1.0
    y2 = apply_h4_g4(x2, compute_dtype=torch.float64, output_dtype=torch.float64)
    assert torch.equal(y2[0, :64], torch.zeros(64, dtype=torch.float64))
    assert torch.allclose(y2[0, 64:68], expected_g4, atol=1e-12, rtol=0.0)
    assert torch.equal(y2[0, 68:], torch.zeros(60, dtype=torch.float64))


def test_linear_equivalence():
    torch.manual_seed(2)
    x = torch.randn(11, 128, dtype=torch.float32)
    w = torch.randn(9, 128, dtype=torch.float32)
    err = linear_prequant_equivalence_error(x, w)
    assert err < 1e-6
    y0 = F.linear(x, w)
    y1 = F.linear(apply_h4_g4(x), apply_h4_g4(w))
    assert tuple(y0.shape) == (11, 9)
    assert relative_frobenius(y1, y0) < 1e-6


def test_hif4_group_size_constant():
    assert HIF4_GROUP_SIZE == 64
