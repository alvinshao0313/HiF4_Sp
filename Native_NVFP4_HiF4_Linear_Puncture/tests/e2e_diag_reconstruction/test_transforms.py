from __future__ import annotations

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.transforms import (
    apply_block_right_fp32,
    apply_r64,
    expand_vo_scale,
    fusable_weight_transform,
    online_weight_transform,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.metrics import relative_l2


def _rand_orthogonal(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float32))
    return q


def test_online_unquantized_diag_then_rot_equivalence():
    torch.manual_seed(0)
    b, k, o = 5, 128, 64
    x = torch.randn(b, k, dtype=torch.float32)
    w = torch.randn(o, k, dtype=torch.float32)
    d = torch.exp2(torch.randn(k, dtype=torch.float32) * 0.25)
    h16 = _rand_orthogonal(16, 1)

    x_native = apply_block_right_fp32(x, h16, 16)
    y0 = x_native @ w.T
    x_new = apply_r64(x_native * d)
    w_new = online_weight_transform(w, d, use_r64=True, rot_order="diag_then_rot")
    y1 = x_new @ w_new.T
    assert relative_l2(y1, y0) < 1e-6


def test_online_unquantized_rot_then_diag_equivalence():
    torch.manual_seed(1)
    b, k, o = 4, 128, 32
    x = torch.randn(b, k, dtype=torch.float32)
    w = torch.randn(o, k, dtype=torch.float32)
    d = torch.exp2(torch.randn(k, dtype=torch.float32) * 0.3)
    h16 = _rand_orthogonal(16, 2)

    x_native = apply_block_right_fp32(x, h16, 16)
    y0 = x_native @ w.T
    x_new = apply_r64(x_native) * d
    w_new = online_weight_transform(w, d, use_r64=True, rot_order="rot_then_diag")
    y1 = x_new @ w_new.T
    assert relative_l2(y1, y0) < 1e-6


def test_fusable_generic_unquantized_equivalence():
    torch.manual_seed(2)
    b, k, o = 6, 128, 64
    x = torch.randn(b, k, dtype=torch.float32)
    w = torch.randn(o, k, dtype=torch.float32)
    d_in = torch.exp2(torch.randn(k, dtype=torch.float32) * 0.2)
    d_out = torch.exp2(torch.randn(o, dtype=torch.float32) * 0.2)
    h16 = _rand_orthogonal(16, 3)

    y_expected = (apply_block_right_fp32(x, h16, 16) @ w.T) * d_out
    x_new = apply_r64(apply_block_right_fp32(x * d_in, h16, 16))
    w_new = fusable_weight_transform(w, h16, d_in, d_out, use_r64=True)
    y_actual = x_new @ w_new.T
    assert relative_l2(y_actual, y_expected) < 1e-6


def test_fusable_uses_h_transpose_not_symmetry_assumption():
    torch.manual_seed(3)
    k, o = 64, 32
    x = torch.randn(2, k, dtype=torch.float32)
    w = torch.randn(o, k, dtype=torch.float32)
    d_in = torch.ones(k)
    d_out = torch.ones(o)
    h16 = _rand_orthogonal(16, 4)
    assert (h16 - h16.T).abs().max() > 1e-3

    y_expected = apply_block_right_fp32(x, h16, 16) @ w.T
    x_new = apply_block_right_fp32(x, h16, 16)
    w_new = fusable_weight_transform(w, h16, d_in, d_out, use_r64=False)
    y_actual = x_new @ w_new.T
    assert relative_l2(y_actual, y_expected) < 1e-6


def test_expand_vo_scale_gqa_repeat():
    d = torch.arange(128, dtype=torch.float32)
    expanded = expand_vo_scale(d, 4, 2, 64)
    assert expanded.shape == (256,)
    assert torch.equal(expanded[:64], d[:64])
    assert torch.equal(expanded[64:128], d[:64])
    assert torch.equal(expanded[128:192], d[64:128])
    assert torch.equal(expanded[192:256], d[64:128])


def test_expand_vo_scale_rejects_bad_gqa_or_head_dim():
    d = torch.arange(96, dtype=torch.float32)
    with pytest.raises(ValueError, match="not divisible"):
        expand_vo_scale(d, 5, 2, 64)
    d64 = torch.arange(64, dtype=torch.float32)
    with pytest.raises(ValueError, match="64"):
        expand_vo_scale(d64, 2, 2, 32)
