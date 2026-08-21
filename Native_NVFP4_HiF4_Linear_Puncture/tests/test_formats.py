"""MXFP8 / HiF4 offline format oracle tests."""

from __future__ import annotations

import torch

from Native_NVFP4_HiF4_Linear_Puncture.src import formats as formats_mod
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct, qdq_mxfp8_post_rotation


def test_mxfp8_block_size_is_32():
    x = torch.zeros(2, 64, dtype=torch.bfloat16)
    x[:, :32] = 1.0
    x[:, 32:] = 8.0
    y = qdq_mxfp8_post_rotation(x)
    assert y.shape == x.shape
    block0 = y[0, :32].float()
    block1 = y[0, 32:].float()
    assert torch.allclose(block0, block0[0].expand_as(block0), rtol=1e-2, atol=1e-2)
    assert torch.allclose(block1, block1[0].expand_as(block1), rtol=1e-2, atol=1e-2)
    assert not torch.allclose(block0[0], block1[0], rtol=1e-2, atol=1e-2)


def test_mxfp8_zero_block_stays_zero():
    x = torch.zeros(3, 32, dtype=torch.bfloat16)
    y = qdq_mxfp8_post_rotation(x)
    assert torch.equal(y, x)


def test_hif4_group_size_is_64(monkeypatch):
    seen = {}

    def fake_quantize_hif4(values, *, config=None):
        seen["config"] = config

        class _R:
            reconstruction = values

        return _R()

    monkeypatch.setattr(formats_mod, "quantize_hif4", fake_quantize_hif4)
    _ = qdq_hif4_direct(torch.randn(2, 128, dtype=torch.bfloat16))
    assert seen["config"].group_size == 64


def test_hif4_grouping_is_last_dim_for_both_activation_and_weight(monkeypatch):
    seen_dims = []

    def fake_quantize_hif4(values, *, config=None):
        seen_dims.append(config.group_dim)

        class _R:
            reconstruction = values

        return _R()

    monkeypatch.setattr(formats_mod, "quantize_hif4", fake_quantize_hif4)
    _ = qdq_hif4_direct(torch.randn(5, 128, dtype=torch.bfloat16))
    _ = qdq_hif4_direct(torch.randn(7, 128, dtype=torch.bfloat16))
    assert seen_dims == [-1, -1]


def test_offline_formats_never_apply_rotation(monkeypatch):
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("offline formats must not apply rotation")

    import Native_NVFP4_HiF4_Linear_Puncture.src.rotation as rot_mod

    monkeypatch.setattr(rot_mod, "apply_block_rotation", boom)

    x = torch.randn(4, 64, dtype=torch.bfloat16)
    _ = qdq_mxfp8_post_rotation(x)
    _ = qdq_hif4_direct(x)
    assert calls["n"] == 0
