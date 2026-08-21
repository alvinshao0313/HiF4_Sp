"""Tests for S1P2 oracle and real HiF4 wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HIFLOAT4 = Path(__file__).resolve().parents[2]
if str(_HIFLOAT4) not in sys.path:
    sys.path.insert(0, str(_HIFLOAT4))

from hif4_gpu.quant_cy import QType, quant_dequant_float
from permutation_optimization.hif4_reference import hif4_fake_quantize, s1p2_oracle_quantize_rows


def test_s1p2_grid_points_unchanged():
    # Row max must be 1.75 so scale=1 and absolute grid points are fixed points.
    pts = torch.tensor(
        [[0.0, 0.25, 0.5, 1.75], [-1.0, -1.25, -1.5, -1.75]],
        dtype=torch.float32,
    )
    out = s1p2_oracle_quantize_rows(pts)
    assert torch.allclose(out, pts, atol=1e-6, rtol=0)


def test_s1p2_zero_row_and_tiny_values():
    x = torch.zeros(3, 4)
    out = s1p2_oracle_quantize_rows(x)
    assert torch.equal(out, x)
    assert not torch.isnan(out).any()

    tiny = torch.full((1, 4), 1e-20)
    out_tiny = s1p2_oracle_quantize_rows(tiny)
    assert not torch.isnan(out_tiny).any()


def test_s1p2_negative_and_bf16():
    x = torch.tensor([[-1.7, 0.3, -0.1, 1.2]], dtype=torch.bfloat16)
    out = s1p2_oracle_quantize_rows(x)
    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape
    # Does not modify input
    assert torch.equal(x, torch.tensor([[-1.7, 0.3, -0.1, 1.2]], dtype=torch.bfloat16))


def test_s1p2_does_not_modify_input():
    x = torch.randn(5, 4)
    x_clone = x.clone()
    _ = s1p2_oracle_quantize_rows(x)
    assert torch.equal(x, x_clone)


def test_hif4_matches_direct_quantizer():
    torch.manual_seed(0)
    x = torch.randn(17, 128, dtype=torch.float32)
    wrapped = hif4_fake_quantize(x)
    qtype = QType("hifx4").dim(-1)
    direct = quant_dequant_float(x, qtype, force_py=True, force_fp32=True)
    assert torch.equal(wrapped.cpu(), direct)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hif4_cuda_bitexact_vs_force_py():
    torch.manual_seed(1)
    x = torch.randn(32, 256, dtype=torch.float32)
    qtype = QType("hifx4").dim(-1)
    py = quant_dequant_float(x, qtype, force_py=True, force_fp32=True)
    cu = quant_dequant_float(
        x.cuda().contiguous(), qtype, force_py=False, force_fp32=True
    ).cpu()
    assert torch.equal(py, cu)
    assert torch.equal(hif4_fake_quantize(x), py)


def test_hif4_rejects_non_multiple_of_64():
    x = torch.randn(4, 63)
    with pytest.raises(ValueError, match="divisible by 64"):
        hif4_fake_quantize(x)


def test_s1p2_rejects_bad_last_dim():
    with pytest.raises(ValueError, match="last dim"):
        s1p2_oracle_quantize_rows(torch.randn(2, 5))
