"""Tests for activation collector."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from permutation_optimization.activation_collector import collect_down_inputs
from permutation_optimization.model_permutation import discover_swiglu_mlps


class TinyLM(nn.Module):
    """Minimal model with two SwiGLU MLPs accepting input_ids-like batches via embedding."""

    def __init__(self, vocab=32, d_model=16, d_ff=64, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.Module()
            block.mlp = nn.Module()
            block.mlp.gate_proj = nn.Linear(d_model, d_ff, bias=False)
            block.mlp.up_proj = nn.Linear(d_model, d_ff, bias=False)
            block.mlp.down_proj = nn.Linear(d_ff, d_model, bias=False)
            self.layers.append(block)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        for layer in self.layers:
            a = torch.nn.functional.silu(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x)
            x = x + layer.mlp.down_proj(a)
        return x


def test_collect_same_rows_per_layer():
    torch.manual_seed(0)
    model = TinyLM()
    specs = discover_swiglu_mlps(model)
    batches = [
        {
            "input_ids": torch.randint(0, 32, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }
    ]
    cache = collect_down_inputs(
        model, specs, batches, torch.device("cpu"), max_rows=10, seed=0
    )
    assert len(cache.by_layer) == 2
    shapes = {k: v.shape for k, v in cache.by_layer.items()}
    rows = {v.shape[0] for v in cache.by_layer.values()}
    assert len(rows) == 1
    assert list(rows)[0] <= 10


def test_collect_reproducible():
    model = TinyLM()
    specs = discover_swiglu_mlps(model)
    batches = [{"input_ids": torch.arange(16).reshape(2, 8) % 32}]
    c1 = collect_down_inputs(model, specs, batches, torch.device("cpu"), 8, seed=7)
    c2 = collect_down_inputs(model, specs, batches, torch.device("cpu"), 8, seed=7)
    for k in c1.by_layer:
        assert torch.equal(c1.by_layer[k], c2.by_layer[k])


def test_mask_zero_excluded():
    torch.manual_seed(0)
    model = TinyLM()
    specs = discover_swiglu_mlps(model)
    ids = torch.randint(0, 32, (1, 8))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
    # Collect with mask — only first 4 tokens valid.
    cache = collect_down_inputs(
        model,
        specs,
        [{"input_ids": ids, "attention_mask": mask}],
        torch.device("cpu"),
        max_rows=4,
        seed=0,
    )
    # Manually compute down input for valid tokens only.
    with torch.no_grad():
        x = model.embed(ids)
        layer = model.layers[0]
        a = torch.nn.functional.silu(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x)
        manual = a[0, :4].to(torch.float16)
    got = cache.by_layer[specs[0].name]
    # All collected rows must be among the 4 valid token activations.
    # (order may differ due to sampling)
    for row in got:
        dists = ((manual.float() - row.float().unsqueeze(0)) ** 2).sum(dim=1)
        assert float(dists.min()) < 1e-4


def test_hooks_cleaned_on_error():
    model = TinyLM()
    specs = discover_swiglu_mlps(model)

    class Boom:
        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        collect_down_inputs(model, specs, Boom(), torch.device("cpu"), 4, seed=0)
    # No leftover hooks: forward should work and not append to removed buffers.
    _ = model(input_ids=torch.zeros(1, 4, dtype=torch.long))
