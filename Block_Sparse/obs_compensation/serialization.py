from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from block_pruning.block_utils import expand_block_mask
from block_pruning.mlp_registry import MLPLinearTarget
from obs_compensation.artifacts import SourceArtifacts
from obs_compensation.config import OBSCompensationConfig
from obs_compensation.layerwise import LayerwiseOBSResult
from obs_compensation.solver import ResolvedOBSOrderPolicy


def verify_fixed_masks_and_weights(
    masks: dict[str, torch.Tensor],
    targets: list[MLPLinearTarget],
    block_height: int,
    block_width: int,
) -> None:
    expected = {t.module_name for t in targets}
    if set(masks) != expected:
        raise ValueError(
            f"mask key mismatch during verification: "
            f"missing={sorted(expected - set(masks))} "
            f"extra={sorted(set(masks) - expected)}"
        )
    for target in targets:
        mask = masks[target.module_name]
        weight = target.module.weight.detach()
        if not torch.isfinite(weight).all():
            raise RuntimeError(f"{target.module_name}: non-finite weight after OBS")
        element_mask = expand_block_mask(mask, block_height, block_width).to(
            device=weight.device
        )
        if tuple(element_mask.shape) != tuple(weight.shape):
            raise ValueError(
                f"{target.module_name}: expanded mask shape mismatch"
            )
        pruned = weight[~element_mask]
        if torch.count_nonzero(pruned) != 0:
            raise RuntimeError(
                f"{target.module_name}: pruned positions are not exactly zero"
            )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def save_obs_model_and_tokenizer(
    model: nn.Module,
    tokenizer: Any,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if getattr(model.config, "model_type", None) == "qwen3_5_text":
        model.config.architectures = ["Qwen3_5ForCausalLM"]
    model.config.use_cache = True
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


def save_obs_artifacts(
    output_dir: Path,
    config: OBSCompensationConfig,
    artifacts: SourceArtifacts,
    order_policy: ResolvedOBSOrderPolicy,
    layerwise_result: LayerwiseOBSResult,
) -> Path:
    art_dir = output_dir / "obs_artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    # Copy source summary as JSON text (byte-stable rewrite of parsed object).
    _atomic_write_json(art_dir / "source_pruning_summary.json", artifacts.raw_summary)

    masks_cpu = {
        name: mask.detach().cpu().clone() for name, mask in artifacts.masks.items()
    }
    torch.save(masks_cpu, art_dir / "block_masks.pt")

    if artifacts.permutation_payload is not None:
        perm_cpu: dict[str, dict[str, Any]] = {}
        for key, record in artifacts.permutation_payload.items():
            cloned: dict[str, Any] = {}
            for field, value in record.items():
                if isinstance(value, torch.Tensor):
                    cloned[field] = value.detach().cpu().clone()
                else:
                    cloned[field] = value
            perm_cpu[key] = cloned
        torch.save(perm_cpu, art_dir / "mlp_permutations.pt")

    cfg_payload = {
        "model_path": config.model_path,
        "source_artifacts_dir": str(config.source_artifacts_dir),
        "output_dir": str(config.output_dir),
        "calibration_dataset": config.calibration_dataset,
        "calibration_samples": config.calibration_samples,
        "sequence_length": config.sequence_length,
        "obs_percdamp": config.obs_percdamp,
        "solver_block_size": config.solver_block_size,
        "obs_order_policy": config.obs_order_policy,
        "dtype": config.dtype,
        "device": config.device,
        "seed": config.seed,
        "trust_remote_code": config.trust_remote_code,
    }
    _atomic_write_json(art_dir / "obs_config.json", cfg_payload)

    layer_rel = [r.output_relative_mse for r in layerwise_result.layer_reports]
    down_left_total = sum(r.down_pruned_blocks_left for r in layerwise_result.layer_reports)
    down_right_total = sum(
        r.down_pruned_blocks_right for r in layerwise_result.layer_reports
    )
    total_pruned_blocks = sum(r.num_pruned_blocks for r in layerwise_result.module_reports)
    summary = {
        "model_path": artifacts.metadata.model_path,
        "source_artifacts_dir": str(artifacts.root),
        "block_size": artifacts.metadata.block_size,
        "block_height": artifacts.metadata.block_height,
        "block_width": artifacts.metadata.block_width,
        "target_block_sparsity": artifacts.metadata.target_block_sparsity,
        "actual_block_sparsity": artifacts.metadata.actual_block_sparsity,
        "score_type": artifacts.metadata.score_type,
        "mlp_permutation": artifacts.metadata.mlp_permutation,
        "requested_obs_order_policy": order_policy.requested_policy,
        "resolved_obs_order_policy": order_policy.resolved_policy,
        "gate_up_direction": order_policy.gate_up_direction,
        "down_direction": order_policy.down_direction,
        "down_pruned_blocks_left_total": down_left_total,
        "down_pruned_blocks_right_total": down_right_total,
        "calibration_dataset": config.calibration_dataset,
        "calibration_samples": config.calibration_samples,
        "sequence_length": config.sequence_length,
        "obs_percdamp": config.obs_percdamp,
        "solver_block_size": config.solver_block_size,
        "num_modules": len(layerwise_result.module_reports),
        "num_layers": len(layerwise_result.layer_reports),
        "max_layer_relative_mse": max(layer_rel) if layer_rel else 0.0,
        "mean_layer_relative_mse": (
            sum(layer_rel) / len(layer_rel) if layer_rel else 0.0
        ),
        "total_pruned_blocks": total_pruned_blocks,
        "num_solver_applied_modules": sum(
            int(report.solver_applied) for report in layerwise_result.module_reports
        ),
        "num_solver_skipped_modules": sum(
            int(not report.solver_applied) for report in layerwise_result.module_reports
        ),
        "total_fully_pruned_block_rows": sum(
            report.num_fully_pruned_block_rows
            for report in layerwise_result.module_reports
        ),
        "total_fully_pruned_output_rows": sum(
            report.num_fully_pruned_output_rows
            for report in layerwise_result.module_reports
        ),
        "source_dense_model_reloaded": True,
        "fixed_mask": True,
        "pruned_weights_exact_zero": True,
        "per_layer_down_pruned_diagnostics": [
            {
                "layer_index": r.layer_index,
                "down_pruned_blocks_left": r.down_pruned_blocks_left,
                "down_pruned_blocks_right": r.down_pruned_blocks_right,
            }
            for r in layerwise_result.layer_reports
        ],
    }
    _atomic_write_json(art_dir / "obs_summary.json", summary)

    module_fields = list(asdict(layerwise_result.module_reports[0]).keys()) if layerwise_result.module_reports else [
        "module_name",
        "layer_index",
        "projection_type",
        "solver_direction",
        "solver_applied",
        "skip_reason",
        "num_total_blocks",
        "num_pruned_blocks",
        "block_sparsity",
        "num_fully_pruned_block_rows",
        "num_fully_pruned_output_rows",
        "num_hessian_tokens",
        "hessian_diagonal_mean",
        "hessian_damp_value",
        "hessian_dead_columns",
        "kept_delta_l2",
        "kept_delta_max_abs",
        "original_pruned_l2",
        "original_max_abs",
        "compensated_max_abs",
    ]
    _atomic_write_csv(
        art_dir / "per_module_obs.csv",
        module_fields,
        [asdict(r) for r in layerwise_result.module_reports],
    )
    layer_fields = list(asdict(layerwise_result.layer_reports[0]).keys()) if layerwise_result.layer_reports else [
        "layer_index",
        "num_output_elements",
        "output_mse",
        "output_relative_mse",
        "output_max_abs_error",
        "down_pruned_blocks_left",
        "down_pruned_blocks_right",
    ]
    _atomic_write_csv(
        art_dir / "per_layer_reconstruction.csv",
        layer_fields,
        [asdict(r) for r in layerwise_result.layer_reports],
    )
    return art_dir


def get_incomplete_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    return output_dir.with_name(f".{output_dir.name}.incomplete")


def validate_atomic_output_paths(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    staging = get_incomplete_output_dir(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output_dir exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(f"output_dir exists and is non-empty: {output_dir}")
    if staging.exists():
        raise ValueError(
            f"incomplete output directory already exists: {staging}; "
            "inspect or remove it before retrying"
        )
    return staging


def save_obs_package_atomically(
    model: nn.Module,
    tokenizer: Any,
    config: OBSCompensationConfig,
    artifacts: SourceArtifacts,
    order_policy: ResolvedOBSOrderPolicy,
    layerwise_result: LayerwiseOBSResult,
) -> Path:
    final_output = Path(config.output_dir)
    staging = validate_atomic_output_paths(final_output)
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()

    save_obs_model_and_tokenizer(model, tokenizer, staging)
    save_obs_artifacts(
        output_dir=staging,
        config=config,
        artifacts=artifacts,
        order_policy=order_policy,
        layerwise_result=layerwise_result,
    )

    if final_output.exists():
        final_output.rmdir()
    staging.replace(final_output)
    return final_output
