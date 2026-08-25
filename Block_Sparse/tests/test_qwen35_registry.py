from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.mlp_registry import collect_mlp_linears


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(5120, 17408, bias=False)
        self.mlp.up_proj = nn.Linear(5120, 17408, bias=False)
        self.mlp.down_proj = nn.Linear(17408, 5120, bias=False)


class _FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])


def test_registry_accepts_qwen35_mlp_shapes():
    model = _FakeQwen()
    targets = collect_mlp_linears(model, block_height=128, block_width=128)
    assert len(targets) == 6
    names = [t.module_name for t in targets]
    assert names[0] == "model.layers.0.mlp.gate_proj"
    assert all(t.module.weight.shape[0] % 128 == 0 for t in targets)
    assert all(t.module.weight.shape[1] % 128 == 0 for t in targets)


def test_registry_accepts_rect_block_on_qwen35_dims():
    model = _FakeQwen()
    targets = collect_mlp_linears(model, block_height=64, block_width=128)
    assert len(targets) == 6
