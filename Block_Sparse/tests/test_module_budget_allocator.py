from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.config import GradientBlockPruningConfig
from block_pruning.gradient_scorer import BlockScoreRecord
from block_pruning.mask_allocator import (
    allocate_block_masks,
    allocate_masks_by_module_budget,
    extract_module_prune_budgets,
)
from block_pruning.mlp_registry import MLPLinearTarget
from block_pruning.serialization import (
    save_hybrid_round_artifacts,
    save_module_prune_budget_report,
)


def _wanda_record(
    name: str,
    wanda: torch.Tensor,
    projection_type: str = "up_proj",
    layer_index: int = 0,
) -> BlockScoreRecord:
    zeros = torch.zeros_like(wanda)
    return BlockScoreRecord(
        module_name=name,
        layer_index=layer_index,
        projection_type=projection_type,
        weight_shape=(wanda.shape[0] * 2, wanda.shape[1] * 2),
        block_size="2",
        block_height=2,
        block_width=2,
        fisher=zeros,
        abs_taylor=zeros.clone(),
        signed_mean=zeros.clone(),
        current_mask=torch.ones_like(wanda, dtype=torch.bool),
        wanda=wanda,
    )


def _fisher_record(name: str, fisher: torch.Tensor) -> BlockScoreRecord:
    return BlockScoreRecord(
        module_name=name,
        layer_index=0,
        projection_type="up_proj",
        weight_shape=(fisher.shape[0] * 2, fisher.shape[1] * 2),
        block_size="2",
        block_height=2,
        block_width=2,
        fisher=fisher,
        abs_taylor=torch.zeros_like(fisher),
        signed_mean=torch.zeros_like(fisher),
        current_mask=torch.ones_like(fisher, dtype=torch.bool),
    )


def test_extract_module_prune_budgets_exact():
    masks = {
        "layers.0.mlp.gate_proj": torch.tensor([[True, False], [True, True]]),
        "layers.0.mlp.up_proj": torch.tensor([[False, False], [True, True]]),
        "layers.0.mlp.down_proj": torch.tensor([[True, True], [True, False]]),
    }
    budgets = extract_module_prune_budgets(masks)
    assert budgets == {
        "layers.0.mlp.gate_proj": 1,
        "layers.0.mlp.up_proj": 2,
        "layers.0.mlp.down_proj": 1,
    }
    assert sum(budgets.values()) == sum(int((~m).sum()) for m in masks.values())


def test_extract_module_prune_budgets_validation():
    try:
        extract_module_prune_budgets({})
        assert False, "expected empty error"
    except ValueError:
        pass
    try:
        extract_module_prune_budgets({"m": torch.ones(2, 2)})
        assert False, "expected bool dtype error"
    except ValueError as e:
        assert "bool" in str(e)
    try:
        extract_module_prune_budgets({"m": torch.ones(4, dtype=torch.bool)})
        assert False, "expected rank error"
    except ValueError as e:
        assert "rank 2" in str(e)


def test_allocate_by_module_budget_no_global_ranking():
    # Module A has globally lower scores but smaller budget.
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    b = torch.tensor([[10.0, 20.0, 5.0], [30.0, 40.0, 6.0]], dtype=torch.float64)
    records = {
        "a": _wanda_record("a", a),
        "b": _wanda_record("b", b),
    }
    masks = {
        "a": torch.ones_like(a, dtype=torch.bool),
        "b": torch.ones_like(b, dtype=torch.bool),
    }
    budgets = {"a": 2, "b": 3}
    cfg = GradientBlockPruningConfig(
        block_size="2",
        max_prune_ratio_per_matrix=1.0,
        min_keep_blocks_per_matrix=1,
        score_type="fisher_budget_wanda",
    )
    result = allocate_masks_by_module_budget(
        records, budgets, cfg, masks, ranking_score_type="wanda"
    )
    assert int((~result.masks["a"]).sum().item()) == 2
    assert int((~result.masks["b"]).sum().item()) == 3
    # A should prune its two lowest: (0,0)=1 and (0,1)=2
    assert result.masks["a"][0, 0].item() is False
    assert result.masks["a"][0, 1].item() is False
    assert result.masks["a"][1, 0].item() is True
    # B should prune lowest three: 5,6,10 at (0,2),(1,2),(0,0)
    assert result.masks["b"][0, 2].item() is False
    assert result.masks["b"][1, 2].item() is False
    assert result.masks["b"][0, 0].item() is False


def test_allocate_by_module_budget_stable_tie():
    score = torch.ones(2, 2, dtype=torch.float64)
    records = {"m": _wanda_record("m", score)}
    masks = {"m": torch.ones(2, 2, dtype=torch.bool)}
    cfg = GradientBlockPruningConfig(
        block_size="2",
        max_prune_ratio_per_matrix=1.0,
        min_keep_blocks_per_matrix=1,
        score_type="fisher_budget_wanda",
    )
    result = allocate_masks_by_module_budget(
        records, {"m": 2}, cfg, masks, ranking_score_type="wanda"
    )
    # Lower (out, in) first: (0,0), (0,1)
    assert result.masks["m"][0, 0].item() is False
    assert result.masks["m"][0, 1].item() is False
    assert result.masks["m"][1, 0].item() is True


def test_allocate_by_module_budget_multiround_monotonic():
    score = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    records = {"m": _wanda_record("m", score)}
    masks = {"m": torch.ones(4, 4, dtype=torch.bool)}
    cfg = GradientBlockPruningConfig(
        block_size="2",
        max_prune_ratio_per_matrix=1.0,
        min_keep_blocks_per_matrix=1,
        score_type="fisher_budget_wanda",
    )
    r1 = allocate_masks_by_module_budget(
        records, {"m": 3}, cfg, masks, ranking_score_type="wanda"
    )
    r2 = allocate_masks_by_module_budget(
        records, {"m": 5}, cfg, r1.masks, ranking_score_type="wanda"
    )
    # Old zeros remain zero
    assert torch.all((~r1.masks["m"]) <= (~r2.masks["m"]))
    assert int((~r2.masks["m"]).sum().item()) == 5
    assert r2.newly_pruned == 2


def test_allocate_by_module_budget_unreachable():
    score = torch.arange(4, dtype=torch.float64).reshape(2, 2)
    records = {"m": _wanda_record("m", score)}
    masks = {"m": torch.ones(2, 2, dtype=torch.bool)}
    masks["m"][0, 0] = False
    cfg = GradientBlockPruningConfig(
        block_size="2",
        max_prune_ratio_per_matrix=0.5,
        min_keep_blocks_per_matrix=1,
        score_type="fisher_budget_wanda",
    )
    # target below already pruned
    try:
        allocate_masks_by_module_budget(
            records, {"m": 0}, cfg, masks, ranking_score_type="wanda"
        )
        assert False
    except RuntimeError as e:
        assert "m" in str(e)

    # target above max_prunable (max_ratio 0.5 -> max 2, but already 1; keep floor also)
    try:
        allocate_masks_by_module_budget(
            records, {"m": 3}, cfg, masks, ranking_score_type="wanda"
        )
        assert False
    except RuntimeError as e:
        assert "m" in str(e)

    try:
        allocate_masks_by_module_budget(
            records, {"other": 1}, cfg, masks, ranking_score_type="wanda"
        )
        assert False
    except ValueError as e:
        assert "mismatch" in str(e)


def test_allocate_shared_up_gate_joint_score():
    # Independent minima differ; joint sum decides.
    up = torch.tensor([[1.0, 100.0], [100.0, 100.0]], dtype=torch.float64)
    gate = torch.tensor([[100.0, 1.0], [100.0, 100.0]], dtype=torch.float64)
    down = torch.tensor([[50.0, 50.0], [50.0, 1.0]], dtype=torch.float64)
    records = {
        "layers.0.mlp.up_proj": _wanda_record(
            "layers.0.mlp.up_proj", up, "up_proj"
        ),
        "layers.0.mlp.gate_proj": _wanda_record(
            "layers.0.mlp.gate_proj", gate, "gate_proj"
        ),
        "layers.0.mlp.down_proj": _wanda_record(
            "layers.0.mlp.down_proj", down, "down_proj"
        ),
    }
    masks = {k: torch.ones_like(v.wanda, dtype=torch.bool) for k, v in records.items()}
    # Joint at (0,0)=101, (0,1)=101, (1,0)=200, (1,1)=200 — tie (0,0) then (0,1)
    budgets = {
        "layers.0.mlp.up_proj": 1,
        "layers.0.mlp.gate_proj": 1,
        "layers.0.mlp.down_proj": 1,
    }
    cfg = GradientBlockPruningConfig(
        block_size="2",
        max_prune_ratio_per_matrix=1.0,
        min_keep_blocks_per_matrix=1,
        share_up_gate_mask=True,
        score_type="fisher_budget_wanda",
    )
    result = allocate_masks_by_module_budget(
        records, budgets, cfg, masks, ranking_score_type="wanda"
    )
    assert torch.equal(
        result.masks["layers.0.mlp.up_proj"], result.masks["layers.0.mlp.gate_proj"]
    )
    assert result.masks["layers.0.mlp.up_proj"][0, 0].item() is False
    assert result.num_pruned_blocks == 3  # pair coord=2 + down=1
    assert result.newly_pruned == 3


def test_shared_pair_physical_block_accounting():
    up = torch.arange(4, dtype=torch.float64).reshape(2, 2).float()
    gate = up.clone() + 1
    records = {
        "layers.0.mlp.up_proj": _wanda_record(
            "layers.0.mlp.up_proj", up.double(), "up_proj"
        ),
        "layers.0.mlp.gate_proj": _wanda_record(
            "layers.0.mlp.gate_proj", gate.double(), "gate_proj"
        ),
        "layers.0.mlp.down_proj": _wanda_record(
            "layers.0.mlp.down_proj",
            torch.ones(2, 2, dtype=torch.float64),
            "down_proj",
        ),
    }
    masks = {k: torch.ones(2, 2, dtype=torch.bool) for k in records}
    budgets = {
        "layers.0.mlp.up_proj": 2,
        "layers.0.mlp.gate_proj": 2,
        "layers.0.mlp.down_proj": 0,
    }
    cfg = GradientBlockPruningConfig(
        block_size="2",
        max_prune_ratio_per_matrix=1.0,
        min_keep_blocks_per_matrix=1,
        share_up_gate_mask=True,
        score_type="fisher_budget_wanda",
    )
    result = allocate_masks_by_module_budget(
        records, budgets, cfg, masks, ranking_score_type="wanda"
    )
    # Pair target of 2 coordinates -> 4 physical blocks
    assert result.num_pruned_blocks == 4
    assert result.newly_pruned == 4


def test_ranking_score_type_override_on_global_allocator():
    fisher = torch.arange(4, dtype=torch.float64).reshape(2, 2)
    # Wanda would prefer opposite order if used; put high values where fisher is low.
    wanda = 100.0 - fisher
    rec = BlockScoreRecord(
        module_name="m",
        layer_index=0,
        projection_type="up_proj",
        weight_shape=(4, 4),
        block_size="2",
        block_height=2,
        block_width=2,
        fisher=fisher,
        abs_taylor=torch.zeros_like(fisher),
        signed_mean=torch.zeros_like(fisher),
        current_mask=torch.ones_like(fisher, dtype=torch.bool),
        wanda=wanda,
    )
    masks = {"m": torch.ones(2, 2, dtype=torch.bool)}
    cfg = GradientBlockPruningConfig(
        block_size="2",
        target_block_sparsity=0.25,  # prune 1 of 4
        max_prune_ratio_per_matrix=1.0,
        min_keep_blocks_per_matrix=1,
        score_type="fisher_budget_wanda",
    )
    result = allocate_block_masks(
        {"m": rec}, cfg, masks, ranking_score_type="fisher"
    )
    assert result.num_pruned_blocks == 1
    assert result.masks["m"][0, 0].item() is False  # lowest fisher


def test_config_accepts_fisher_budget_wanda():
    cfg = GradientBlockPruningConfig(score_type="fisher_budget_wanda")
    cfg.validate()
    assert cfg.requires_calibration()
    assert cfg.requires_gradient_checkpointing()


def test_hybrid_serialization(tmp_path: Path):
    fisher = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    wanda = torch.tensor([[4.0, 3.0], [2.0, 1.0]], dtype=torch.float64)
    fisher_rec = _fisher_record("layers.0.mlp.up_proj", fisher)
    wanda_rec = _wanda_record("layers.0.mlp.up_proj", wanda, "up_proj")
    before = {"layers.0.mlp.up_proj": torch.ones(2, 2, dtype=torch.bool)}
    fisher_masks = {"layers.0.mlp.up_proj": torch.tensor([[False, True], [True, True]])}
    final_masks = {"layers.0.mlp.up_proj": torch.tensor([[True, True], [True, False]])}
    from block_pruning.mask_allocator import MaskAllocationResult

    fisher_ref = MaskAllocationResult(
        masks=fisher_masks,
        num_total_blocks=4,
        num_pruned_blocks=1,
        actual_block_sparsity=0.25,
        newly_pruned=1,
        target_pruned=1,
    )
    final = MaskAllocationResult(
        masks=final_masks,
        num_total_blocks=4,
        num_pruned_blocks=1,
        actual_block_sparsity=0.25,
        newly_pruned=1,
        target_pruned=1,
    )
    targets = [
        MLPLinearTarget(
            "layers.0.mlp.up_proj",
            torch.nn.Linear(4, 4, bias=False),
            0,
            "up_proj",
        )
    ]
    cfg = GradientBlockPruningConfig(score_type="fisher_budget_wanda", block_size="2")
    out = tmp_path / "arts"
    save_hybrid_round_artifacts(
        output_dir=out,
        fisher_records={"layers.0.mlp.up_proj": fisher_rec},
        wanda_records={"layers.0.mlp.up_proj": wanda_rec},
        current_masks_before=before,
        fisher_reference_allocation=fisher_ref,
        final_allocation=final,
        targets=targets,
        config=cfg,
        round_idx=None,
    )
    assert (out / "fisher_block_scores.pt").exists()
    assert (out / "wanda_block_scores.pt").exists()
    assert (out / "fisher_reference_masks.pt").exists()
    assert (out / "block_masks.pt").exists()
    assert (out / "module_prune_budget.csv").exists()
    assert (out / "hybrid_per_matrix_report.csv").exists()
    assert (out / "pruning_summary.json").exists()

    save_module_prune_budget_report(
        path=out / "budget2.csv",
        targets=targets,
        current_masks_before=before,
        fisher_reference_masks=fisher_masks,
        final_masks=final_masks,
    )
    text = (out / "budget2.csv").read_text()
    assert "fisher_target_pruned_blocks" in text
    assert "final_pruned_blocks" in text
