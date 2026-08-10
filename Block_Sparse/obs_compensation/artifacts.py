from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from block_pruning.mlp_registry import MLPLinearTarget


_REQUIRED_SUMMARY_KEYS = (
    "model_path",
    "block_size",
    "block_height",
    "block_width",
    "target_block_sparsity",
    "actual_block_sparsity",
    "score_type",
    "mlp_permutation",
    "residual_permutation",
    "num_pruning_rounds",
)

_SUPPORTED_MLP_PERMUTATIONS = frozenset({"none", "wanda_shared"})


@dataclass(frozen=True)
class SourcePruningMetadata:
    model_path: str
    block_size: str
    block_height: int
    block_width: int
    target_block_sparsity: float
    actual_block_sparsity: float
    score_type: str
    mlp_permutation: str
    residual_permutation: str
    num_pruning_rounds: int


@dataclass(frozen=True)
class SourceArtifacts:
    root: Path
    metadata: SourcePruningMetadata
    masks: dict[str, torch.Tensor]
    permutation_payload: dict[str, dict[str, Any]] | None
    raw_summary: dict[str, Any]


def _compute_mask_sparsity(masks: dict[str, torch.Tensor]) -> float:
    total = sum(int(mask.numel()) for mask in masks.values())
    if total == 0:
        raise ValueError("mask payload is empty")
    pruned = sum(int((~mask).sum().item()) for mask in masks.values())
    return pruned / total


def _validate_permutation_payload(
    permutation_payload: dict[str, dict[str, Any]],
) -> None:
    required = {
        "layer_index",
        "gate_module_name",
        "up_module_name",
        "down_module_name",
        "intermediate_size",
        "combined_score",
        "permutation",
        "inverse_permutation",
    }
    if not isinstance(permutation_payload, dict) or not permutation_payload:
        raise ValueError("mlp_permutations.pt payload must be a non-empty dict")
    for layer_key, record in permutation_payload.items():
        if not isinstance(record, dict):
            raise TypeError(f"layer {layer_key}: permutation record must be a dict")
        missing = required - set(record)
        if missing:
            raise ValueError(
                f"layer {layer_key}: missing permutation fields {sorted(missing)}"
            )
        size = int(record["intermediate_size"])
        score = record["combined_score"]
        perm = record["permutation"]
        inverse = record["inverse_permutation"]
        if not isinstance(score, torch.Tensor) or not isinstance(perm, torch.Tensor):
            raise TypeError(f"layer {layer_key}: score/permutation must be tensors")
        if not isinstance(inverse, torch.Tensor):
            raise TypeError(f"layer {layer_key}: inverse_permutation must be a tensor")
        if score.ndim != 1 or score.numel() != size or not torch.isfinite(score).all():
            raise ValueError(f"layer {layer_key}: invalid combined_score")
        if perm.dtype != torch.int64 or inverse.dtype != torch.int64:
            raise TypeError(f"layer {layer_key}: permutation tensors must be int64")
        if (
            perm.ndim != 1
            or inverse.ndim != 1
            or perm.numel() != size
            or inverse.numel() != size
        ):
            raise ValueError(f"layer {layer_key}: permutation length mismatch")
        expected = torch.arange(size, dtype=torch.int64)
        if not torch.equal(torch.sort(perm.cpu()).values, expected):
            raise ValueError(f"layer {layer_key}: permutation is not bijective")
        if not torch.equal(inverse.cpu().index_select(0, perm.cpu()), expected):
            raise ValueError(f"layer {layer_key}: inverse_permutation mismatch")
        ordered_score = score.detach().double().cpu().index_select(0, perm.cpu())
        if torch.any(ordered_score[1:] > ordered_score[:-1] + 1e-12):
            raise ValueError(
                f"layer {layer_key}: permutation importance is not descending"
            )


def load_source_artifacts(root: str | Path) -> SourceArtifacts:
    root = Path(root)
    summary_path = root / "pruning_summary.json"
    mask_path = root / "block_masks.pt"
    permutation_path = root / "mlp_permutations.pt"

    if not summary_path.is_file():
        raise FileNotFoundError(f"missing pruning_summary.json under {root}")
    if not mask_path.is_file():
        raise FileNotFoundError(
            f"missing final unsuffixed block_masks.pt under {root}"
        )

    raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(raw_summary, dict):
        raise TypeError("pruning_summary.json must contain a JSON object")
    missing = [k for k in _REQUIRED_SUMMARY_KEYS if k not in raw_summary]
    if missing:
        raise ValueError(f"pruning_summary.json missing keys: {missing}")

    num_rounds = int(raw_summary["num_pruning_rounds"])
    if num_rounds != 1:
        raise ValueError(
            f"OBS first version requires num_pruning_rounds==1, got {num_rounds}"
        )
    residual = str(raw_summary["residual_permutation"])
    if residual != "none":
        raise ValueError(
            f"OBS first version requires residual_permutation=='none', got {residual!r}"
        )
    mlp_perm = str(raw_summary["mlp_permutation"])
    if mlp_perm not in _SUPPORTED_MLP_PERMUTATIONS:
        raise ValueError(
            f"Unsupported mlp_permutation={mlp_perm!r}; "
            f"expected one of {sorted(_SUPPORTED_MLP_PERMUTATIONS)}"
        )

    masks_raw = torch.load(mask_path, map_location="cpu", weights_only=False)
    if not isinstance(masks_raw, dict):
        raise TypeError("block_masks.pt must contain a dict[str, Tensor]")
    if not masks_raw:
        raise ValueError("block_masks.pt is empty")

    masks: dict[str, torch.Tensor] = {}
    for name, mask in masks_raw.items():
        if not isinstance(name, str):
            raise TypeError(f"mask key must be str, got {type(name).__name__}")
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"mask for {name} must be a Tensor")
        if mask.dtype != torch.bool:
            raise TypeError(
                f"mask for {name} must be bool, got {mask.dtype}; "
                "never silently cast non-bool masks"
            )
        masks[name] = mask.detach().cpu().clone()

    if all(bool(mask.all().item()) for mask in masks.values()):
        # all-True is allowed only if sparsity metadata says 0; still require keep
        pass
    if not any(bool(mask.any().item()) for mask in masks.values()):
        raise ValueError("mask payload contains zero kept blocks")
    for name, mask in masks.items():
        if not bool(mask.any().item()):
            raise ValueError(f"mask {name} retains zero blocks")
        if mask.ndim != 2:
            raise ValueError(f"mask {name} must be rank 2, got {tuple(mask.shape)}")

    actual = _compute_mask_sparsity(masks)
    reported = float(raw_summary["actual_block_sparsity"])
    if abs(actual - reported) > 1e-8:
        raise ValueError(
            f"summary actual_block_sparsity={reported} does not match "
            f"mask sparsity={actual} within 1e-8"
        )

    permutation_payload: dict[str, dict[str, Any]] | None = None
    if mlp_perm == "wanda_shared":
        if not permutation_path.is_file():
            raise FileNotFoundError(
                f"mlp_permutation=wanda_shared requires {permutation_path}"
            )
        loaded = torch.load(permutation_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise TypeError("mlp_permutations.pt must contain a dict")
        permutation_payload = loaded
        _validate_permutation_payload(permutation_payload)
    elif permutation_path.exists():
        raise ValueError(
            "mlp_permutation=none but mlp_permutations.pt is present; "
            "do not synthesize or ignore unexpected permutation artifacts"
        )

    metadata = SourcePruningMetadata(
        model_path=str(raw_summary["model_path"]),
        block_size=str(raw_summary["block_size"]),
        block_height=int(raw_summary["block_height"]),
        block_width=int(raw_summary["block_width"]),
        target_block_sparsity=float(raw_summary["target_block_sparsity"]),
        actual_block_sparsity=reported,
        score_type=str(raw_summary["score_type"]),
        mlp_permutation=mlp_perm,
        residual_permutation=residual,
        num_pruning_rounds=num_rounds,
    )
    return SourceArtifacts(
        root=root,
        metadata=metadata,
        masks=masks,
        permutation_payload=permutation_payload,
        raw_summary=raw_summary,
    )


def validate_source_artifacts_against_targets(
    artifacts: SourceArtifacts,
    targets: list[MLPLinearTarget],
) -> None:
    metadata = artifacts.metadata
    expected_names = {t.module_name for t in targets}
    mask_names = set(artifacts.masks)
    if mask_names != expected_names:
        missing = sorted(expected_names - mask_names)
        extra = sorted(mask_names - expected_names)
        raise ValueError(
            f"mask key mismatch: missing={missing} extra={extra}"
        )

    for target in targets:
        weight = target.module.weight
        if (
            weight.shape[0] % metadata.block_height != 0
            or weight.shape[1] % metadata.block_width != 0
        ):
            raise ValueError(
                f"{target.module_name}: weight shape {tuple(weight.shape)} "
                f"not divisible by {metadata.block_height}x{metadata.block_width}"
            )
        expected_shape = (
            weight.shape[0] // metadata.block_height,
            weight.shape[1] // metadata.block_width,
        )
        mask = artifacts.masks[target.module_name]
        if mask.dtype != torch.bool:
            raise TypeError(f"{target.module_name}: mask dtype must be bool")
        if tuple(mask.shape) != expected_shape:
            raise ValueError(
                f"{target.module_name}: mask shape {tuple(mask.shape)} != "
                f"expected {expected_shape}"
            )
        if not bool(mask.any().item()):
            raise ValueError(f"{target.module_name}: retains zero blocks")

    actual = _compute_mask_sparsity(artifacts.masks)
    if abs(actual - metadata.actual_block_sparsity) > 1e-8:
        raise ValueError(
            f"recomputed mask sparsity {actual} != metadata "
            f"{metadata.actual_block_sparsity}"
        )
