#!/usr/bin/env python3
"""Score MLP blocks and export a pruned HF checkpoint for vLLM."""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

BLOCK_SPARSE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BLOCK_SPARSE_ROOT.parent
if str(BLOCK_SPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(BLOCK_SPARSE_ROOT))

from block_pruning.calibration import build_calibration_batches  # noqa: E402
from block_pruning.config import CALIBRATION_DATASETS, GradientBlockPruningConfig  # noqa: E402
from block_pruning.gradient_scorer import (  # noqa: E402
    collect_magnitude_block_scores,
    collect_mlp_block_scores,
    collect_random_block_scores,
)
from block_pruning.mask_allocator import (  # noqa: E402
    allocate_block_masks,
    allocate_masks_by_module_budget,
    extract_module_prune_budgets,
)
from block_pruning.mask_apply import apply_mlp_block_masks, verify_masks_and_weights  # noqa: E402
from block_pruning.mlp_registry import collect_mlp_linears, initialize_all_one_masks  # noqa: E402
from block_pruning.model_loader import load_model_and_tokenizer  # noqa: E402
from block_pruning.serialization import (  # noqa: E402
    save_hybrid_round_artifacts,
    save_pruned_model,
    save_round_artifacts,
)
from block_pruning.wanda_scorer import collect_wanda_block_scores  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-27B")
    p.add_argument(
        "--output_dir",
        type=str,
        default="Block_Sparse/outputs/qwen35_27b_fisher_s0.3",
    )
    p.add_argument(
        "--score_type",
        type=str,
        default="fisher",
        choices=["fisher", "magnitude", "random", "fisher_budget_wanda"],
        help="fisher | magnitude | random | fisher_budget_wanda "
        "(Fisher budgets + Block-Wanda positions)",
    )
    p.add_argument("--target_block_sparsity", type=float, default=0.30)
    p.add_argument(
        "--block_size",
        type=str,
        default="128",
        help="Block size: '128' (square) or 'HxW' e.g. '64x128'. H=d_out, W=d_in.",
    )
    p.add_argument(
        "--calibration_dataset",
        type=str,
        default="wikitext2",
        choices=sorted(CALIBRATION_DATASETS),
        help="Calibration data: wikitext2/c4/ptb (fixed windows) or s1k "
        "(full s1K-1.1_tokenized text samples). Required for fisher and "
        "fisher_budget_wanda.",
    )
    p.add_argument("--calibration_samples", type=int, default=128)
    p.add_argument(
        "--sequence_length",
        type=int,
        default=2048,
        help="Window length for wikitext2/c4/ptb. For s1k: 0 = no truncate "
        "(use full sample length); >0 hard-fails if any sample is longer.",
    )
    p.add_argument("--score_batch_size", type=int, default=1)
    p.add_argument("--max_prune_ratio_per_matrix", type=float, default=0.60)
    p.add_argument("--min_keep_blocks_per_matrix", type=int, default=1)
    p.add_argument("--share_up_gate_mask", action="store_true")
    p.add_argument("--pruning_rounds", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    return p.parse_args()


def args_to_config(args: argparse.Namespace) -> GradientBlockPruningConfig:
    cfg = GradientBlockPruningConfig(
        model_path=args.model_path,
        calibration_dataset=args.calibration_dataset,
        output_dir=args.output_dir,
        block_size=str(args.block_size),
        target_block_sparsity=args.target_block_sparsity,
        calibration_samples=args.calibration_samples,
        sequence_length=args.sequence_length,
        score_batch_size=args.score_batch_size,
        score_type=args.score_type,
        max_prune_ratio_per_matrix=args.max_prune_ratio_per_matrix,
        min_keep_blocks_per_matrix=args.min_keep_blocks_per_matrix,
        share_up_gate_mask=bool(args.share_up_gate_mask),
        pruning_rounds=args.pruning_rounds,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        trust_remote_code=args.trust_remote_code,
    )
    cfg.validate()
    return cfg


def score_blocks(model, batches, targets, config, current_masks):
    if config.score_type == "fisher":
        if batches is None:
            raise ValueError("fisher scoring requires calibration batches")
        return collect_mlp_block_scores(
            model=model,
            batches=batches,
            targets=targets,
            config=config,
            current_masks=current_masks,
        )
    if config.score_type == "magnitude":
        return collect_magnitude_block_scores(
            targets=targets,
            config=config,
            current_masks=current_masks,
        )
    if config.score_type == "random":
        return collect_random_block_scores(
            targets=targets,
            config=config,
            current_masks=current_masks,
        )
    raise ValueError(f"Unknown score_type: {config.score_type}")


def allocate_hybrid_round(
    model,
    batches,
    targets,
    config,
    current_masks,
    cumulative_target_sparsity,
):
    """Two-stage Fisher-budget + Wanda-position allocation for one round.

    Returns:
        fisher_records, wanda_records, module_budgets,
        fisher_reference_allocation, final_allocation
    """
    if batches is None:
        raise ValueError("fisher_budget_wanda requires calibration batches")

    fisher_records = collect_mlp_block_scores(
        model=model,
        batches=batches,
        targets=targets,
        config=config,
        current_masks=current_masks,
    )
    fisher_reference_allocation = allocate_block_masks(
        score_records=fisher_records,
        config=config,
        current_masks=current_masks,
        cumulative_target_sparsity=cumulative_target_sparsity,
        ranking_score_type="fisher",
    )
    module_budgets = extract_module_prune_budgets(
        fisher_reference_allocation.masks
    )
    if sum(module_budgets.values()) != fisher_reference_allocation.num_pruned_blocks:
        raise RuntimeError(
            "Budget sum mismatch vs Fisher reference pruned count: "
            f"{sum(module_budgets.values())} vs "
            f"{fisher_reference_allocation.num_pruned_blocks}"
        )

    wanda_records = collect_wanda_block_scores(
        model=model,
        batches=batches,
        targets=targets,
        config=config,
        current_masks=current_masks,
    )
    final_allocation = allocate_masks_by_module_budget(
        score_records=wanda_records,
        target_pruned_per_module=module_budgets,
        config=config,
        current_masks=current_masks,
        ranking_score_type="wanda",
    )
    if (
        final_allocation.num_pruned_blocks
        != fisher_reference_allocation.num_pruned_blocks
    ):
        raise RuntimeError(
            "Final pruned count != Fisher reference pruned count: "
            f"{final_allocation.num_pruned_blocks} vs "
            f"{fisher_reference_allocation.num_pruned_blocks}"
        )
    for name, budget in module_budgets.items():
        actual = int((~final_allocation.masks[name]).sum().item())
        if actual != budget:
            raise RuntimeError(
                f"Per-module budget mismatch for {name}: "
                f"expected {budget}, got {actual}"
            )

    return (
        fisher_records,
        wanda_records,
        module_budgets,
        fisher_reference_allocation,
        final_allocation,
    )


def main() -> None:
    args = parse_args()
    config = args_to_config(args)
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    config.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[prune] loading model: {config.model_path} "
        f"block_size={config.block_size} ({config.block_height}x{config.block_width})",
        flush=True,
    )
    model, tokenizer = load_model_and_tokenizer(config)

    print("[prune] collecting MLP targets", flush=True)
    targets = collect_mlp_linears(model, config.block_height, config.block_width)
    print(f"[prune] found {len(targets)} MLP Linear modules", flush=True)

    batches = None
    if config.requires_calibration():
        print(
            f"[prune] building calibration: {config.calibration_dataset} "
            f"n={config.calibration_samples} seq={config.sequence_length}",
            flush=True,
        )
        batches = build_calibration_batches(
            model_path=config.model_path,
            dataset_name=config.calibration_dataset,
            num_samples=config.calibration_samples,
            sequence_length=config.sequence_length,
            seed=config.seed,
        )

    current_masks = initialize_all_one_masks(
        targets, config.block_height, config.block_width
    )
    allocation = None
    score_records = None
    hybrid_state = None
    artifacts_dir = output_dir / "pruning_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for round_idx in range(config.pruning_rounds):
        cumulative_target = (
            config.target_block_sparsity * (round_idx + 1) / config.pruning_rounds
        )
        print(
            f"[prune] round {round_idx + 1}/{config.pruning_rounds} "
            f"target_sparsity={cumulative_target:.4f} score={config.score_type}",
            flush=True,
        )
        masks_before = {k: v.clone() for k, v in current_masks.items()}
        round_suffix = round_idx if config.pruning_rounds > 1 else None

        if config.score_type == "fisher_budget_wanda":
            (
                fisher_records,
                wanda_records,
                _module_budgets,
                fisher_ref,
                allocation,
            ) = allocate_hybrid_round(
                model=model,
                batches=batches,
                targets=targets,
                config=config,
                current_masks=current_masks,
                cumulative_target_sparsity=cumulative_target,
            )
            hybrid_state = (
                fisher_records,
                wanda_records,
                masks_before,
                fisher_ref,
                allocation,
            )
            current_masks = allocation.masks
            apply_mlp_block_masks(
                model=model,
                masks=current_masks,
                block_height=config.block_height,
                block_width=config.block_width,
                targets=targets,
            )
            save_hybrid_round_artifacts(
                output_dir=artifacts_dir,
                fisher_records=fisher_records,
                wanda_records=wanda_records,
                current_masks_before=masks_before,
                fisher_reference_allocation=fisher_ref,
                final_allocation=allocation,
                targets=targets,
                config=config,
                round_idx=round_suffix,
            )
        else:
            score_records = score_blocks(
                model=model,
                batches=batches,
                targets=targets,
                config=config,
                current_masks=current_masks,
            )
            allocation = allocate_block_masks(
                score_records=score_records,
                config=config,
                current_masks=current_masks,
                cumulative_target_sparsity=cumulative_target,
            )
            current_masks = allocation.masks
            apply_mlp_block_masks(
                model=model,
                masks=current_masks,
                block_height=config.block_height,
                block_width=config.block_width,
                targets=targets,
            )
            save_round_artifacts(
                output_dir=artifacts_dir,
                score_records=score_records,
                masks=current_masks,
                allocation=allocation,
                config=config,
                round_idx=round_suffix,
            )

        print(
            f"[prune] round done: pruned={allocation.num_pruned_blocks}/"
            f"{allocation.num_total_blocks} "
            f"sparsity={allocation.actual_block_sparsity:.4f}",
            flush=True,
        )

    assert allocation is not None
    verify_masks_and_weights(
        current_masks, targets, config.block_height, config.block_width
    )

    if config.score_type == "fisher_budget_wanda":
        assert hybrid_state is not None
        (
            fisher_records,
            wanda_records,
            masks_before,
            fisher_ref,
            final_alloc,
        ) = hybrid_state
        save_hybrid_round_artifacts(
            output_dir=artifacts_dir,
            fisher_records=fisher_records,
            wanda_records=wanda_records,
            current_masks_before=masks_before,
            fisher_reference_allocation=fisher_ref,
            final_allocation=final_alloc,
            targets=targets,
            config=config,
            round_idx=None,
        )
    else:
        assert score_records is not None
        save_round_artifacts(
            output_dir=artifacts_dir,
            score_records=score_records,
            masks=current_masks,
            allocation=allocation,
            config=config,
            round_idx=None,
        )

    print(f"[prune] saving HF model to {output_dir}", flush=True)
    save_pruned_model(model, tokenizer, output_dir)
    print(
        f"[prune] done. actual_block_sparsity={allocation.actual_block_sparsity:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
