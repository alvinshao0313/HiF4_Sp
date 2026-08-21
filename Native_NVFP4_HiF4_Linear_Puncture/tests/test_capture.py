"""Capture helpers: recorder semantics + deterministic token sampling."""

from __future__ import annotations

import torch
import torch.nn as nn

from Native_NVFP4_HiF4_Linear_Puncture.src import semantic_model as sm_mod
from Native_NVFP4_HiF4_Linear_Puncture.src.capture import select_token_indices
from Native_NVFP4_HiF4_Linear_Puncture.src.semantic_model import NativeNVFP4SemanticLinear


def test_token_sampling_is_deterministic_and_matches_linspace_rule():
    # seq_len <= 64: keep all
    idx_short = select_token_indices(seq_len=17, max_rows=64)
    assert torch.equal(idx_short, torch.arange(17))

    # seq_len > 64: fixed linspace round
    idx = select_token_indices(seq_len=200, max_rows=64)
    expected = torch.linspace(0, 199, 64).round().long()
    assert torch.equal(idx, expected)

    idx2 = select_token_indices(seq_len=200, max_rows=64)
    assert torch.equal(idx, idx2)


def test_observer_records_after_rotation_before_qdq(monkeypatch):
    order: list[str] = []

    def spy_rotate(x, h, group_size):
        order.append("rotate")
        return x

    def spy_qdq(x_rot, scale):
        order.append("qdq")
        return x_rot

    monkeypatch.setattr(sm_mod, "apply_block_rotation", spy_rotate)
    monkeypatch.setattr(sm_mod, "qdq_nvfp4_post_rotation", spy_qdq)

    base = nn.Linear(32, 8, bias=False)
    wrap = NativeNVFP4SemanticLinear(
        base_linear=base,
        module_name="mock.layers.2.self_attn.q_proj",
        input_global_scale=torch.tensor(1.0),
        rotation_matrix=torch.eye(16, dtype=torch.bfloat16),
        rotation_group_size=16,
    )
    wrap.set_recorder(lambda x_rot: order.append("record"))
    _ = wrap(torch.randn(2, 32, dtype=torch.bfloat16))
    assert order == ["rotate", "record", "qdq"]


def test_observer_does_not_change_output(monkeypatch):
    monkeypatch.setattr(sm_mod, "qdq_nvfp4_post_rotation", lambda x, s: x)
    base = nn.Linear(32, 8, bias=True)
    wrap = NativeNVFP4SemanticLinear(
        base_linear=base,
        module_name="mock.layers.18.mlp.down_proj",
        input_global_scale=torch.tensor(1.0),
        rotation_matrix=torch.eye(16, dtype=torch.bfloat16),
        rotation_group_size=16,
    )
    x = torch.randn(5, 32, dtype=torch.bfloat16)
    y0 = wrap(x).clone()
    wrap.set_recorder(lambda x_rot: None)
    y1 = wrap(x)
    assert torch.equal(y0, y1)
