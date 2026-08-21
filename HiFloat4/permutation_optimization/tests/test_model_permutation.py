"""Tests for model_permutation."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from permutation_optimization.model_permutation import (
    apply_mlp_permutation_,
    discover_swiglu_mlps,
    get_mlp_modules,
    validate_permutation,
)


class TinySwiGLU(nn.Module):
    def __init__(self, d_model: int = 32, d_ff: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([self._layer(d_model, d_ff), self._layer(d_model, d_ff)])

    @staticmethod
    def _layer(d_model, d_ff):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.gate_proj = nn.Linear(d_model, d_ff, bias=True)
                self.mlp.up_proj = nn.Linear(d_model, d_ff, bias=True)
                self.mlp.down_proj = nn.Linear(d_ff, d_model, bias=False)

            def forward(self, x):
                a = torch.nn.functional.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
                return self.mlp.down_proj(a)

        return Block()

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def test_discover_order_and_shapes():
    model = TinySwiGLU()
    specs = discover_swiglu_mlps(model)
    assert len(specs) == 2
    assert specs[0].layer_index == 0
    assert specs[1].layer_index == 1
    assert specs[0].intermediate_size == 128


def test_validate_permutation_errors():
    with pytest.raises(ValueError):
        validate_permutation(torch.tensor([0, 1, 1]), 3)
    with pytest.raises(ValueError):
        validate_permutation(torch.tensor([0, 1]), 3)


def test_apply_preserves_fp_output_and_param_id():
    torch.manual_seed(0)
    model = TinySwiGLU(d_model=32, d_ff=128)
    specs = discover_swiglu_mlps(model)
    gate, up, down = get_mlp_modules(model, specs[0])
    x = torch.randn(4, 8, 32)
    y0 = model(x).clone()
    wid = id(gate.weight)
    perm = torch.randperm(128)
    apply_mlp_permutation_(gate, up, down, perm)
    y1 = model(x)
    assert id(gate.weight) == wid
    assert torch.allclose(y0, y1, rtol=1e-6, atol=1e-6)


def test_bf16_equivalence():
    torch.manual_seed(1)
    model = TinySwiGLU(d_model=32, d_ff=64).to(dtype=torch.bfloat16)
    specs = discover_swiglu_mlps(model)
    gate, up, down = get_mlp_modules(model, specs[0])
    x = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    y0 = model(x).clone()
    apply_mlp_permutation_(gate, up, down, torch.randperm(64))
    y1 = model(x)
    assert torch.allclose(y0, y1, rtol=1e-3, atol=1e-3)


def test_incomplete_mlp_raises():
    class Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].mlp = nn.Module()
            self.layers[0].mlp.gate_proj = nn.Linear(16, 64)
            # missing up/down

    with pytest.raises(ValueError, match="Incomplete"):
        discover_swiglu_mlps(Bad())
