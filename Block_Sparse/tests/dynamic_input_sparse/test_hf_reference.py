from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.dynamic_input_sparse.config import (  # noqa: E402
    DynamicInputMaskMethod,
    DynamicInputSparseConfig,
)
from Block_Sparse.dynamic_input_sparse.hf_reference import (  # noqa: E402
    DynamicInputSparseMLPReference,
)
from Block_Sparse.dynamic_input_sparse.masked_linear import apply_input_block_mask  # noqa: E402


class _TinyMLP(nn.Module):
    def __init__(self, hidden=128, intermediate=256):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_keep_one_dense_equivalence():
    torch.manual_seed(0)
    mlp = _TinyMLP()
    x = torch.randn(3, 128)
    dense = mlp(x)
    for method in (DynamicInputMaskMethod.M8_ENERGY, DynamicInputMaskMethod.M1_ORACLE):
        cfg = DynamicInputSparseConfig(method=method, keep_ratio=1.0)
        wrap = DynamicInputSparseMLPReference(mlp, cfg)
        out = wrap(x)
        assert torch.equal(out, dense)


def test_gate_up_same_mask():
    torch.manual_seed(1)
    mlp = _TinyMLP()
    x = torch.randn(4, 128)
    for method in (DynamicInputMaskMethod.M8_ENERGY, DynamicInputMaskMethod.M1_ORACLE):
        cfg = DynamicInputSparseConfig(method=method, keep_ratio=0.5)
        wrap = DynamicInputSparseMLPReference(mlp, cfg, capture_masks=True)
        _ = wrap(x)
        assert wrap.last_mx_gate_up is not None
        # shared mask applied once; instrument via re-predict
        mx = wrap._predict_gate_up(x)
        assert torch.equal(mx, wrap.last_mx_gate_up)


def test_down_uses_sparse_intermediate():
    torch.manual_seed(2)
    mlp = _TinyMLP()
    x = torch.randn(2, 128)
    cfg = DynamicInputSparseConfig(
        method=DynamicInputMaskMethod.M8_ENERGY, keep_ratio=0.25
    )
    wrap = DynamicInputSparseMLPReference(mlp, cfg, capture_masks=True)
    _ = wrap(x)
    # Dense H
    h_dense = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    # Sparse-path H from captured
    assert wrap.last_down_input is not None
    assert not torch.allclose(wrap.last_down_input, h_dense, atol=1e-5)
    # Down predictor on sparse H must match captured mask
    mx_from_sparse = wrap._predict_down(wrap.last_down_input)
    assert torch.equal(mx_from_sparse, wrap.last_mx_down)
    mx_from_dense = wrap._predict_down(h_dense)
    assert not torch.equal(mx_from_dense, wrap.last_mx_down)


def test_weights_unchanged():
    torch.manual_seed(3)
    mlp = _TinyMLP()
    before = {n: p.detach().clone() for n, p in mlp.named_parameters()}
    cfg = DynamicInputSparseConfig(
        method=DynamicInputMaskMethod.M8_ENERGY, keep_ratio=0.5
    )
    wrap = DynamicInputSparseMLPReference(mlp, cfg)
    _ = wrap(torch.randn(2, 128))
    for n, p in mlp.named_parameters():
        assert torch.equal(before[n], p.detach())
