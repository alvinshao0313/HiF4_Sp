from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.config import GradientBlockPruningConfig
from block_pruning.gradient_scorer import BlockScoreRecord
from block_pruning.mask_allocator import allocate_block_masks


def _make_record(name: str, fisher: torch.Tensor) -> BlockScoreRecord:
    return BlockScoreRecord(
        module_name=name,
        layer_index=0,
        projection_type="up_proj",
        weight_shape=(fisher.shape[0] * 128, fisher.shape[1] * 128),
        block_size="128",
        block_height=128,
        block_width=128,
        fisher=fisher,
        abs_taylor=torch.zeros_like(fisher),
        signed_mean=torch.zeros_like(fisher),
        current_mask=torch.ones_like(fisher, dtype=torch.bool),
    )


def test_allocator_exact_prune_count():
    fisher = torch.arange(100, dtype=torch.float64).reshape(10, 10)
    records = {"m.mlp.up_proj": _make_record("m.mlp.up_proj", fisher)}
    masks = {"m.mlp.up_proj": torch.ones(10, 10, dtype=torch.bool)}
    cfg = GradientBlockPruningConfig(
        target_block_sparsity=0.30,
        max_prune_ratio_per_matrix=1.0,
        min_keep_blocks_per_matrix=1,
        score_type="fisher",
    )
    result = allocate_block_masks(records, cfg, masks)
    assert result.num_pruned_blocks == 30
    assert result.num_total_blocks == 100
    assert abs(result.actual_block_sparsity - 0.30) < 1e-12
    flat_scores = fisher.reshape(-1)
    del flat_scores
    pruned_idx = (~result.masks["m.mlp.up_proj"]).reshape(-1).nonzero(as_tuple=False).view(-1)
    assert set(pruned_idx.tolist()) == set(range(30))


def test_allocator_raises_when_unreachable():
    fisher = torch.ones(2, 2, dtype=torch.float64)
    records = {"m.mlp.up_proj": _make_record("m.mlp.up_proj", fisher)}
    masks = {"m.mlp.up_proj": torch.ones(2, 2, dtype=torch.bool)}
    cfg = GradientBlockPruningConfig(
        target_block_sparsity=0.75,
        max_prune_ratio_per_matrix=0.25,
        min_keep_blocks_per_matrix=1,
        score_type="fisher",
    )
    try:
        allocate_block_masks(records, cfg, masks)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "Cannot reach target sparsity" in str(e)
