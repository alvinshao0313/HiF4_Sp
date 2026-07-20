from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from block_pruning.config import GradientBlockPruningConfig
from block_pruning.gradient_scorer import BlockScoreRecord
from block_pruning.mask_allocator import MaskAllocationResult
from block_pruning.mlp_registry import MLPLinearTarget


def save_score_records(
    score_records: dict[str, BlockScoreRecord],
    path: str | Path,
) -> None:
    payload = {}
    for name, rec in score_records.items():
        payload[name] = {
            "layer_index": rec.layer_index,
            "projection_type": rec.projection_type,
            "weight_shape": tuple(rec.weight_shape),
            "block_size": rec.block_size,
            "block_height": rec.block_height,
            "block_width": rec.block_width,
            "fisher": rec.fisher.cpu(),
            "abs_taylor": rec.abs_taylor.cpu(),
            "signed_mean": rec.signed_mean.cpu(),
            "wanda": None if rec.wanda is None else rec.wanda.cpu(),
        }
    torch.save(payload, path)


def save_masks(masks: dict[str, torch.Tensor], path: str | Path) -> None:
    torch.save({k: v.cpu().clone() for k, v in masks.items()}, path)


def load_masks(path: str | Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_pruning_summary(
    config: GradientBlockPruningConfig,
    allocation: MaskAllocationResult,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "model_path": config.model_path,
        "block_size": str(config.block_size),
        "block_height": config.block_height,
        "block_width": config.block_width,
        "target_block_sparsity": config.target_block_sparsity,
        "actual_block_sparsity": allocation.actual_block_sparsity,
        "num_total_blocks": allocation.num_total_blocks,
        "num_pruned_blocks": allocation.num_pruned_blocks,
        "num_pruning_rounds": config.pruning_rounds,
        "score_type": config.score_type,
        "selection_mode": config.selection_mode,
        "share_up_gate_mask": config.share_up_gate_mask,
        "max_prune_ratio_per_matrix": config.max_prune_ratio_per_matrix,
        "min_keep_blocks_per_matrix": config.min_keep_blocks_per_matrix,
        "calibration_dataset": config.calibration_dataset,
        "calibration_samples": config.calibration_samples,
        "sequence_length": config.sequence_length,
        "seed": config.seed,
    }
    if extra:
        summary.update(extra)
    return summary


def save_pruning_summary(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def save_per_matrix_report(
    score_records: dict[str, BlockScoreRecord],
    masks: dict[str, torch.Tensor],
    path: str | Path,
    score_type: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "module_name",
        "layer_index",
        "projection_type",
        "weight_shape",
        "num_blocks",
        "num_pruned_blocks",
        "block_sparsity",
        "score_min",
        "score_median",
        "score_mean",
        "score_max",
    ]
    rows = []
    for name, rec in sorted(
        score_records.items(),
        key=lambda kv: (kv[1].layer_index, kv[1].projection_type, kv[0]),
    ):
        mask = masks[name]
        score = rec.primary_score(score_type)
        kept_scores = score[mask]
        num_blocks = int(mask.numel())
        num_pruned = int((~mask).sum().item())
        if kept_scores.numel() == 0:
            s_min = s_med = s_mean = s_max = float("nan")
        else:
            s_min = float(kept_scores.min().item())
            s_med = float(kept_scores.median().item())
            s_mean = float(kept_scores.mean().item())
            s_max = float(kept_scores.max().item())
        rows.append(
            {
                "module_name": name,
                "layer_index": rec.layer_index,
                "projection_type": rec.projection_type,
                "weight_shape": str(tuple(rec.weight_shape)),
                "num_blocks": num_blocks,
                "num_pruned_blocks": num_pruned,
                "block_sparsity": num_pruned / num_blocks if num_blocks else 0.0,
                "score_min": s_min,
                "score_median": s_med,
                "score_mean": s_mean,
                "score_max": s_max,
            }
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_pruned_model(
    model: nn.Module,
    tokenizer: Any,
    output_dir: str | Path,
) -> Path:
    """Save dense HF checkpoint with zeroed blocks for vLLM/lighteval."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Qwen3.5 text export must declare CausalLM for vLLM (not ConditionalGeneration).
    if hasattr(model, "config") and getattr(model.config, "model_type", None) == "qwen3_5_text":
        model.config.architectures = ["Qwen3_5ForCausalLM"]
        model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    return out


def save_round_artifacts(
    output_dir: str | Path,
    score_records: dict[str, BlockScoreRecord],
    masks: dict[str, torch.Tensor],
    allocation: MaskAllocationResult,
    config: GradientBlockPruningConfig,
    round_idx: int | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if round_idx is None else f"_round{round_idx}"
    save_score_records(score_records, output_dir / f"block_scores{suffix}.pt")
    save_masks(masks, output_dir / f"block_masks{suffix}.pt")
    summary = build_pruning_summary(
        config,
        allocation,
        extra={"round_idx": round_idx} if round_idx is not None else None,
    )
    save_pruning_summary(summary, output_dir / f"pruning_summary{suffix}.json")
    save_per_matrix_report(
        score_records,
        masks,
        output_dir / f"per_matrix_report{suffix}.csv",
        score_type=config.score_type,
    )


def save_module_prune_budget_report(
    path: str | Path,
    targets: list[MLPLinearTarget],
    current_masks_before: dict[str, torch.Tensor],
    fisher_reference_masks: dict[str, torch.Tensor],
    final_masks: dict[str, torch.Tensor],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "module_name",
        "layer_index",
        "projection_type",
        "num_total_blocks",
        "current_pruned_blocks",
        "fisher_target_pruned_blocks",
        "newly_pruned_blocks",
        "final_pruned_blocks",
        "final_block_sparsity",
    ]
    target_by_name = {t.module_name: t for t in targets}
    rows = []
    for name in sorted(
        final_masks,
        key=lambda n: (
            target_by_name[n].layer_index if n in target_by_name else -1,
            target_by_name[n].projection_type if n in target_by_name else n,
            n,
        ),
    ):
        before = current_masks_before[name]
        fisher_mask = fisher_reference_masks[name]
        final = final_masks[name]
        num_total = int(final.numel())
        current_pruned = int((~before).sum().item())
        fisher_target = int((~fisher_mask).sum().item())
        final_pruned = int((~final).sum().item())
        if fisher_target != final_pruned:
            raise RuntimeError(
                f"fisher_target_pruned_blocks != final_pruned_blocks for {name}: "
                f"{fisher_target} vs {final_pruned}"
            )
        t = target_by_name[name]
        rows.append(
            {
                "module_name": name,
                "layer_index": t.layer_index,
                "projection_type": t.projection_type,
                "num_total_blocks": num_total,
                "current_pruned_blocks": current_pruned,
                "fisher_target_pruned_blocks": fisher_target,
                "newly_pruned_blocks": final_pruned - current_pruned,
                "final_pruned_blocks": final_pruned,
                "final_block_sparsity": final_pruned / num_total if num_total else 0.0,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _score_stats(score: torch.Tensor) -> tuple[float, float, float, float]:
    flat = score.reshape(-1).double()
    return (
        float(flat.min().item()),
        float(flat.median().item()),
        float(flat.mean().item()),
        float(flat.max().item()),
    )


def save_hybrid_per_matrix_report(
    path: str | Path,
    fisher_records: dict[str, BlockScoreRecord],
    wanda_records: dict[str, BlockScoreRecord],
    fisher_reference_masks: dict[str, torch.Tensor],
    final_masks: dict[str, torch.Tensor],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "module_name",
        "layer_index",
        "projection_type",
        "num_blocks",
        "fisher_target_pruned_blocks",
        "final_pruned_blocks",
        "fisher_score_min",
        "fisher_score_median",
        "fisher_score_mean",
        "fisher_score_max",
        "wanda_score_min",
        "wanda_score_median",
        "wanda_score_mean",
        "wanda_score_max",
        "fisher_wanda_mask_overlap_blocks",
        "fisher_wanda_mask_union_blocks",
        "fisher_wanda_mask_iou",
    ]
    rows = []
    for name, frec in sorted(
        fisher_records.items(),
        key=lambda kv: (kv[1].layer_index, kv[1].projection_type, kv[0]),
    ):
        wrec = wanda_records[name]
        fisher_mask = fisher_reference_masks[name]
        final_mask = final_masks[name]
        fisher_pruned = ~fisher_mask
        final_pruned = ~final_mask
        overlap = int((fisher_pruned & final_pruned).sum().item())
        union = int((fisher_pruned | final_pruned).sum().item())
        iou = 1.0 if union == 0 else overlap / union
        f_min, f_med, f_mean, f_max = _score_stats(frec.primary_score("fisher"))
        w_min, w_med, w_mean, w_max = _score_stats(wrec.primary_score("wanda"))
        rows.append(
            {
                "module_name": name,
                "layer_index": frec.layer_index,
                "projection_type": frec.projection_type,
                "num_blocks": int(final_mask.numel()),
                "fisher_target_pruned_blocks": int(fisher_pruned.sum().item()),
                "final_pruned_blocks": int(final_pruned.sum().item()),
                "fisher_score_min": f_min,
                "fisher_score_median": f_med,
                "fisher_score_mean": f_mean,
                "fisher_score_max": f_max,
                "wanda_score_min": w_min,
                "wanda_score_median": w_med,
                "wanda_score_mean": w_mean,
                "wanda_score_max": w_max,
                "fisher_wanda_mask_overlap_blocks": overlap,
                "fisher_wanda_mask_union_blocks": union,
                "fisher_wanda_mask_iou": iou,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_hybrid_round_artifacts(
    output_dir: str | Path,
    fisher_records: dict[str, BlockScoreRecord],
    wanda_records: dict[str, BlockScoreRecord],
    current_masks_before: dict[str, torch.Tensor],
    fisher_reference_allocation: MaskAllocationResult,
    final_allocation: MaskAllocationResult,
    targets: list[MLPLinearTarget],
    config: GradientBlockPruningConfig,
    round_idx: int | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if round_idx is None else f"_round{round_idx}"

    save_score_records(
        fisher_records, output_dir / f"fisher_block_scores{suffix}.pt"
    )
    save_score_records(
        wanda_records, output_dir / f"wanda_block_scores{suffix}.pt"
    )
    save_masks(
        fisher_reference_allocation.masks,
        output_dir / f"fisher_reference_masks{suffix}.pt",
    )
    save_masks(final_allocation.masks, output_dir / f"block_masks{suffix}.pt")
    save_module_prune_budget_report(
        path=output_dir / f"module_prune_budget{suffix}.csv",
        targets=targets,
        current_masks_before=current_masks_before,
        fisher_reference_masks=fisher_reference_allocation.masks,
        final_masks=final_allocation.masks,
    )
    save_hybrid_per_matrix_report(
        path=output_dir / f"hybrid_per_matrix_report{suffix}.csv",
        fisher_records=fisher_records,
        wanda_records=wanda_records,
        fisher_reference_masks=fisher_reference_allocation.masks,
        final_masks=final_allocation.masks,
    )
    extra: dict[str, Any] = {
        "budget_score": "fisher",
        "selection_score": "wanda",
        "budget_granularity": "mlp_linear_module",
    }
    if round_idx is not None:
        extra["round_idx"] = round_idx
    summary = build_pruning_summary(config, final_allocation, extra=extra)
    save_pruning_summary(summary, output_dir / f"pruning_summary{suffix}.json")
