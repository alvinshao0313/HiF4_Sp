from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.mask_apply import apply_mlp_block_masks, verify_masks_and_weights
from block_pruning.mlp_registry import collect_mlp_linears


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(128, 128, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(128, 256, bias=False)
        self.mlp.up_proj = nn.Linear(128, 256, bias=False)
        self.mlp.down_proj = nn.Linear(256, 128, bias=False)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 128)
        self.layers = nn.ModuleList([_TinyLayer()])
        self.lm_head = nn.Linear(128, 32, bias=False)


def test_apply_mask_zeros_pruned_blocks_only():
    model = _TinyModel()
    h, w = 128, 128
    targets = collect_mlp_linears(model, h, w)
    assert len(targets) == 3

    attn_before = model.layers[0].self_attn.weight.detach().clone()
    embed_before = model.embed.weight.detach().clone()
    lm_before = model.lm_head.weight.detach().clone()

    up = model.layers[0].mlp.up_proj
    up_before = up.weight.detach().clone()
    masks = {
        t.module_name: torch.ones(
            t.module.weight.shape[0] // h,
            t.module.weight.shape[1] // w,
            dtype=torch.bool,
        )
        for t in targets
    }
    up_name = [t.module_name for t in targets if t.projection_type == "up_proj"][0]
    masks[up_name][0, 0] = False

    apply_mlp_block_masks(model, masks, h, w, targets=targets)
    verify_masks_and_weights(masks, targets, h, w)

    assert torch.equal(model.layers[0].self_attn.weight, attn_before)
    assert torch.equal(model.embed.weight, embed_before)
    assert torch.equal(model.lm_head.weight, lm_before)

    pruned = up.weight[:h, :w]
    assert int(torch.count_nonzero(pruned).item()) == 0
    kept = up.weight[h:, :w]
    kept_before = up_before[h:, :w]
    assert torch.equal(kept, kept_before)


def test_apply_mask_rect_blocks():
    model = _TinyModel()
    h, w = 64, 128
    targets = collect_mlp_linears(model, h, w)
    up_name = [t.module_name for t in targets if t.projection_type == "up_proj"][0]
    masks = {
        t.module_name: torch.ones(
            t.module.weight.shape[0] // h,
            t.module.weight.shape[1] // w,
            dtype=torch.bool,
        )
        for t in targets
    }
    masks[up_name][0, 0] = False
    apply_mlp_block_masks(model, masks, h, w, targets=targets)
    verify_masks_and_weights(masks, targets, h, w)
    block = model.layers[0].mlp.up_proj.weight[:h, :w]
    assert int(torch.count_nonzero(block).item()) == 0
