from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.config import (
    GradientBlockPruningConfig,
    parse_projection_prune_shares,
)
from block_pruning.gradient_scorer import BlockScoreRecord
from block_pruning.mask_allocator import allocate_block_masks


def _record(
    name: str,
    projection_type: str,
    score: torch.Tensor,
    layer_index: int = 0,
) -> BlockScoreRecord:
    h, w = score.shape
    return BlockScoreRecord(
        module_name=name,
        layer_index=layer_index,
        projection_type=projection_type,
        weight_shape=(h * 64, w * 64),
        block_size="64",
        block_height=64,
        block_width=64,
        fisher=score.double(),
        abs_taylor=torch.zeros_like(score, dtype=torch.float64),
        signed_mean=torch.zeros_like(score, dtype=torch.float64),
        current_mask=torch.ones_like(score, dtype=torch.bool),
    )


def _three_proj_setup(rows: int = 8, cols: int = 8):
    """One layer with gate/up/down; scores increase with index so lowest prune first."""
    base = torch.arange(rows * cols, dtype=torch.float64).reshape(rows, cols)
    records = {
        "model.layers.0.mlp.gate_proj": _record(
            "model.layers.0.mlp.gate_proj", "gate_proj", base.clone()
        ),
        "model.layers.0.mlp.up_proj": _record(
            "model.layers.0.mlp.up_proj", "up_proj", base.clone() + 1000
        ),
        "model.layers.0.mlp.down_proj": _record(
            "model.layers.0.mlp.down_proj", "down_proj", base.clone() + 2000
        ),
    }
    masks = {k: torch.ones(rows, cols, dtype=torch.bool) for k in records}
    return records, masks, rows * cols * 3


def test_parse_projection_prune_shares():
    shares = parse_projection_prune_shares("gate_proj=1,up_proj=1,down_proj=2")
    assert shares == {"gate_proj": 1.0, "up_proj": 1.0, "down_proj": 2.0}
    with pytest.raises(ValueError):
        parse_projection_prune_shares("gate_proj=1,up_proj=1")
    with pytest.raises(ValueError):
        parse_projection_prune_shares("gate_proj=1,up_proj=1,down_proj=0")


def test_validate_rejects_unequal_shares_with_share_up_gate():
    cfg = GradientBlockPruningConfig(
        target_block_sparsity=0.2,
        share_up_gate_mask=True,
        projection_prune_shares={
            "gate_proj": 1.0,
            "up_proj": 2.0,
            "down_proj": 1.0,
        },
    )
    with pytest.raises(ValueError, match="share_up_gate_mask"):
        cfg.validate()


def test_no_shares_matches_legacy_global():
    records, masks, total = _three_proj_setup()
    sparsity = 0.25
    cfg_legacy = GradientBlockPruningConfig(
        target_block_sparsity=sparsity,
        max_prune_ratio_per_matrix=1.0,
        min_keep_blocks_per_matrix=1,
        score_type="fisher",
        projection_prune_shares=None,
    )
    cfg_legacy.validate()
    out_legacy = allocate_block_masks(records, cfg_legacy, masks)

    # Same call again — deterministic
    out2 = allocate_block_masks(records, cfg_legacy, masks)
    for k in masks:
        assert torch.equal(out_legacy.masks[k], out2.masks[k])
    assert out_legacy.num_pruned_blocks == int(total * sparsity)


def test_equal_shares_balanced_prune_counts():
    records, masks, total = _three_proj_setup()
    sparsity = 0.25
    cfg = GradientBlockPruningConfig(
        target_block_sparsity=sparsity,
        max_prune_ratio_per_matrix=1.0,
        min_keep_blocks_per_matrix=1,
        score_type="fisher",
        projection_prune_shares={
            "gate_proj": 1.0,
            "up_proj": 1.0,
            "down_proj": 1.0,
        },
    )
    cfg.validate()
    out = allocate_block_masks(records, cfg, masks)
    assert out.num_pruned_blocks == int(total * sparsity)
    assert abs(out.actual_block_sparsity - sparsity) < 1e-12

    counts = {
        "gate_proj": int((~out.masks["model.layers.0.mlp.gate_proj"]).sum().item()),
        "up_proj": int((~out.masks["model.layers.0.mlp.up_proj"]).sum().item()),
        "down_proj": int((~out.masks["model.layers.0.mlp.down_proj"]).sum().item()),
    }
    vals = list(counts.values())
    assert max(vals) - min(vals) <= 1


def test_unequal_shares_down_gets_half_budget():
    records, masks, total = _three_proj_setup()
    sparsity = 0.25
    target = int(total * sparsity)
    cfg = GradientBlockPruningConfig(
        target_block_sparsity=sparsity,
        max_prune_ratio_per_matrix=1.0,
        min_keep_blocks_per_matrix=1,
        score_type="fisher",
        projection_prune_shares={
            "gate_proj": 1.0,
            "up_proj": 1.0,
            "down_proj": 2.0,
        },
    )
    cfg.validate()
    out = allocate_block_masks(records, cfg, masks)
    assert out.num_pruned_blocks == target

    g = int((~out.masks["model.layers.0.mlp.gate_proj"]).sum().item())
    u = int((~out.masks["model.layers.0.mlp.up_proj"]).sum().item())
    d = int((~out.masks["model.layers.0.mlp.down_proj"]).sum().item())
    assert g + u + d == target
    # down share = 2/4 => about half; allow ±1 for integer remainder
    assert abs(d - target // 2) <= 1
    assert abs(g - target // 4) <= 1
    assert abs(u - target // 4) <= 1
