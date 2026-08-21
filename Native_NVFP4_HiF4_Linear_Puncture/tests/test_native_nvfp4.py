"""Native NVFP4 semantic linear oracle tests (synthetic / monkeypatch)."""

from __future__ import annotations

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from Native_NVFP4_HiF4_Linear_Puncture.src import native_nvfp4 as nv_mod
from Native_NVFP4_HiF4_Linear_Puncture.src.native_nvfp4 import (
    PackedNVFP4LinearState,
    dequantize_packed_weight,
    qdq_nvfp4_post_rotation,
    source_linear_semantic,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.rotation import apply_block_rotation


def _make_state(
    *,
    k: int = 32,
    out: int = 16,
    group_size: int = 16,
    bias: Optional[torch.Tensor] = None,
) -> PackedNVFP4LinearState:
    h = torch.eye(group_size, dtype=torch.bfloat16)
    weight_packed = torch.zeros(out, k // 2, dtype=torch.uint8)
    weight_scale = torch.ones(out, k // group_size, dtype=torch.float32)
    return PackedNVFP4LinearState(
        module_name="synthetic.linear",
        weight_packed=weight_packed,
        weight_scale=weight_scale,
        weight_global_scale=torch.tensor(2.0, dtype=torch.float32),
        input_global_scale=torch.tensor(1.0, dtype=torch.float32),
        rotation_matrix=h,
        bias=bias,
    )


def test_post_rotation_nvfp4_qdq_does_not_rotate_again(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("qdq_nvfp4_post_rotation must not call apply_block_rotation")

    monkeypatch.setattr(nv_mod, "apply_block_rotation", boom)
    monkeypatch.setattr(
        nv_mod,
        "fake_quant_nvfp4_activation",
        lambda x_in, **kwargs: x_in,
    )

    x = torch.randn(4, 32, dtype=torch.bfloat16)
    out = qdq_nvfp4_post_rotation(x, torch.tensor(1.0))
    assert out.shape == x.shape


def test_packed_weight_dequant_uses_weight_global_scale(monkeypatch):
    seen = {}

    def fake_dequant(*, weight_packed, weight_scale, weight_global_scale, dtype, group_size=16):
        seen["weight_global_scale"] = weight_global_scale.detach().clone()
        out_f, packed_k = weight_packed.shape
        return torch.ones(out_f, packed_k * 2, dtype=dtype) * float(weight_global_scale.item())

    monkeypatch.setattr(nv_mod, "dequantize_nvfp4_weight", fake_dequant)

    state = _make_state()
    w = dequantize_packed_weight(state)
    assert torch.equal(seen["weight_global_scale"], state.weight_global_scale)
    assert torch.allclose(w.float(), torch.full_like(w.float(), 2.0))


def test_source_linear_matches_manual_rotate_qdq_linear(monkeypatch):
    group_size = 16
    state = _make_state(k=32, out=8, group_size=group_size, bias=None)
    w_n = torch.randn(8, 32, dtype=torch.bfloat16)
    monkeypatch.setattr(nv_mod, "dequantize_packed_weight", lambda s: w_n)

    def fake_qdq(x_rot, input_global_scale):
        return x_rot * 0.5

    monkeypatch.setattr(nv_mod, "qdq_nvfp4_post_rotation", fake_qdq)

    x_pre = torch.randn(5, 32, dtype=torch.bfloat16)
    y, x_rot, a_n = source_linear_semantic(x_pre, state)

    x_rot_ref = apply_block_rotation(x_pre, state.rotation_matrix, group_size=group_size)
    a_n_ref = fake_qdq(x_rot_ref, state.input_global_scale)
    y_ref = F.linear(a_n_ref, w_n, None)

    assert torch.equal(x_rot, x_rot_ref)
    assert torch.equal(a_n, a_n_ref)
    assert torch.equal(y, y_ref)


@pytest.mark.parametrize("with_bias", [False, True])
def test_zero_bias_and_nonzero_bias_both_match_manual_reference(monkeypatch, with_bias):
    group_size = 16
    bias = torch.randn(8, dtype=torch.bfloat16) if with_bias else None
    state = _make_state(k=32, out=8, group_size=group_size, bias=bias)
    w_n = torch.randn(8, 32, dtype=torch.bfloat16)
    monkeypatch.setattr(nv_mod, "dequantize_packed_weight", lambda s: w_n)
    monkeypatch.setattr(nv_mod, "qdq_nvfp4_post_rotation", lambda x, s: x)

    x_pre = torch.randn(3, 32, dtype=torch.bfloat16)
    y, x_rot, a_n = source_linear_semantic(x_pre, state)
    y_ref = F.linear(a_n, w_n, bias)
    assert torch.equal(y, y_ref)
    if with_bias:
        assert not torch.equal(y, F.linear(a_n, w_n, None))
