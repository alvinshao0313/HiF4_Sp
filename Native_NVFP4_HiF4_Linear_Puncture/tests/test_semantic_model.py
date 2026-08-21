"""Semantic Linear wrapper tests with tiny mock modules (no 8B)."""

from __future__ import annotations

import torch
import torch.nn as nn

from Native_NVFP4_HiF4_Linear_Puncture.src import semantic_model as sm_mod
from Native_NVFP4_HiF4_Linear_Puncture.src.rotation import apply_block_rotation
from Native_NVFP4_HiF4_Linear_Puncture.src.semantic_model import NativeNVFP4SemanticLinear


def _make_wrapper(rotation_matrix: torch.Tensor | None = None, group_size: int = 16):
    k, o = 32, 8
    base = nn.Linear(k, o, bias=True)
    with torch.no_grad():
        base.weight.copy_(torch.randn(o, k))
        base.bias.copy_(torch.randn(o))
    h = (
        rotation_matrix
        if rotation_matrix is not None
        else torch.eye(group_size, dtype=torch.bfloat16)
    )
    return NativeNVFP4SemanticLinear(
        base_linear=base,
        module_name="mock.layers.0.self_attn.q_proj",
        input_global_scale=torch.tensor(1.0, dtype=torch.float32),
        rotation_matrix=h,
        rotation_group_size=group_size,
    )


def test_observer_records_after_rotation_before_qdq(monkeypatch):
    order: list[str] = []
    recorded = {}

    real_rotate = apply_block_rotation

    def spy_rotate(x, h, group_size):
        order.append("rotate")
        return real_rotate(x, h, group_size)

    def spy_qdq(x_rot, scale):
        order.append("qdq")
        return x_rot

    monkeypatch.setattr(sm_mod, "apply_block_rotation", spy_rotate)
    monkeypatch.setattr(sm_mod, "qdq_nvfp4_post_rotation", spy_qdq)

    wrap = _make_wrapper()

    def recorder(x_rot: torch.Tensor) -> None:
        order.append("record")
        recorded["x_rot"] = x_rot.detach().clone()

    wrap.set_recorder(recorder)
    _ = wrap(torch.randn(4, 32, dtype=torch.bfloat16))
    assert order == ["rotate", "record", "qdq"]
    assert "x_rot" in recorded


def test_observer_does_not_change_output(monkeypatch):
    monkeypatch.setattr(sm_mod, "qdq_nvfp4_post_rotation", lambda x, s: x)

    wrap = _make_wrapper()
    x = torch.randn(3, 32, dtype=torch.bfloat16)
    y0 = wrap(x).clone()

    buf = []
    wrap.set_recorder(lambda x_rot: buf.append(x_rot.detach().clone()))
    y1 = wrap(x)
    assert torch.equal(y0, y1)
    assert len(buf) == 1


def test_each_linear_uses_its_own_rotation(monkeypatch):
    monkeypatch.setattr(sm_mod, "qdq_nvfp4_post_rotation", lambda x, s: x)

    g = 16
    h0 = torch.eye(g, dtype=torch.bfloat16)
    h1 = -torch.eye(g, dtype=torch.bfloat16)
    base = nn.Linear(32, 4, bias=False)
    w0 = NativeNVFP4SemanticLinear(base, "m.q_proj", torch.tensor(1.0), h0, g)
    w1 = NativeNVFP4SemanticLinear(base, "m.k_proj", torch.tensor(1.0), h1, g)

    x = torch.randn(2, 32, dtype=torch.bfloat16)
    assert not torch.equal(w0(x), w1(x))
