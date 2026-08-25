from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.model_loader import resolve_model_input_device


def test_resolve_model_input_device_uses_embedding():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.other = nn.Linear(8, 8)

        def get_input_embeddings(self):
            return self.embed

    model = Tiny()
    if torch.cuda.is_available():
        model.embed.to("cuda:0")
        model.other.to("cpu")
        assert resolve_model_input_device(model).type == "cuda"
    else:
        assert resolve_model_input_device(model).type == "cpu"


def test_resolve_model_input_device_hf_device_map_fallback():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(2))

    model = Tiny()
    model.hf_device_map = {"layers.0": 0, "layers.1": 1}
    dev = resolve_model_input_device(model)
    assert str(dev) in {"cuda:0", "cpu"} or dev.type in {"cuda", "cpu"}
