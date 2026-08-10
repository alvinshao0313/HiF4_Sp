from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from block_pruning.mlp_registry import MLPLinearTarget


@dataclass(frozen=True)
class MLPProjectionTriplet:
    layer_index: int
    gate: MLPLinearTarget
    up: MLPLinearTarget
    down: MLPLinearTarget
    intermediate_size: int


def group_mlp_projection_triplets(
    targets: list[MLPLinearTarget],
) -> list[MLPProjectionTriplet]:
    by_layer: dict[int, dict[str, MLPLinearTarget]] = {}
    for target in targets:
        bucket = by_layer.setdefault(target.layer_index, {})
        if target.projection_type in bucket:
            raise ValueError(
                f"Duplicate {target.projection_type} for layer "
                f"{target.layer_index}: {bucket[target.projection_type].module_name} "
                f"and {target.module_name}"
            )
        bucket[target.projection_type] = target

    required = ("gate_proj", "up_proj", "down_proj")
    triplets: list[MLPProjectionTriplet] = []
    for layer_index in sorted(by_layer):
        bucket = by_layer[layer_index]
        missing = [p for p in required if p not in bucket]
        if missing:
            raise ValueError(
                f"Layer {layer_index} missing projections: {missing}. "
                f"Found: {sorted(bucket)}"
            )
        gate = bucket["gate_proj"]
        up = bucket["up_proj"]
        down = bucket["down_proj"]
        if gate.module.weight.shape != up.module.weight.shape:
            raise ValueError(
                f"Layer {layer_index}: gate/up weight shapes differ: "
                f"{tuple(gate.module.weight.shape)} vs {tuple(up.module.weight.shape)}"
            )
        d_ff, d_model = gate.module.weight.shape
        if down.module.weight.shape != (d_model, d_ff):
            raise ValueError(
                f"Layer {layer_index}: down shape {tuple(down.module.weight.shape)} "
                f"incompatible with gate/up {[d_ff, d_model]}"
            )
        triplets.append(
            MLPProjectionTriplet(
                layer_index=layer_index,
                gate=gate,
                up=up,
                down=down,
                intermediate_size=d_ff,
            )
        )

    unused = len(targets) - 3 * len(triplets)
    if unused != 0:
        raise RuntimeError(
            f"Internal error: {unused} targets not covered by triplets"
        )
    return triplets


def _validate_permutation(perm: torch.Tensor, size: int, context: str) -> None:
    if not isinstance(perm, torch.Tensor):
        raise TypeError(f"{context}: permutation must be a Tensor")
    if perm.dtype != torch.int64:
        raise TypeError(f"{context}: permutation dtype must be int64, got {perm.dtype}")
    if perm.ndim != 1 or perm.numel() != size:
        raise ValueError(
            f"{context}: permutation length mismatch, expected {size}, got {tuple(perm.shape)}"
        )
    expected = torch.arange(size, dtype=torch.int64)
    if not torch.equal(torch.sort(perm.cpu()).values, expected):
        raise ValueError(f"{context}: permutation is not bijective")


def _validate_descending_importance_layout(
    combined_score: torch.Tensor,
    permutation: torch.Tensor,
    context: str,
) -> None:
    if not isinstance(combined_score, torch.Tensor):
        raise TypeError(f"{context}: combined_score must be a Tensor")
    if combined_score.ndim != 1 or combined_score.numel() != permutation.numel():
        raise ValueError(f"{context}: invalid combined_score shape")
    if not torch.isfinite(combined_score).all():
        raise ValueError(f"{context}: combined_score contains non-finite values")
    ordered_score = combined_score.detach().double().cpu().index_select(
        0, permutation.cpu()
    )
    if torch.any(ordered_score[1:] > ordered_score[:-1] + 1e-12):
        raise ValueError(
            f"{context}: saved permutation does not place importance in descending order"
        )


def _index_select_parameter_inplace(
    parameter: nn.Parameter,
    dim: int,
    index: torch.Tensor,
    context: str,
) -> None:
    if not isinstance(parameter, nn.Parameter):
        raise TypeError(f"{context}: expected nn.Parameter")
    if dim not in (0, 1):
        raise ValueError(f"{context}: dim must be 0 or 1, got {dim}")
    param_id = id(parameter)
    dtype = parameter.dtype
    device = parameter.device
    index_local = index.to(device=device)
    with torch.no_grad():
        selected = parameter.detach().index_select(dim, index_local)
        parameter.copy_(selected)
    if id(parameter) != param_id:
        raise RuntimeError(f"{context}: Parameter identity changed")
    if parameter.dtype != dtype or parameter.device != device:
        raise RuntimeError(f"{context}: dtype/device changed unexpectedly")


def apply_saved_mlp_permutations(
    triplets: list[MLPProjectionTriplet],
    permutation_payload: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(permutation_payload, dict) or not permutation_payload:
        raise ValueError("permutation_payload must be a non-empty dict")

    by_layer: dict[int, dict[str, Any]] = {}
    for key, record in permutation_payload.items():
        if not isinstance(record, dict):
            raise TypeError(f"layer {key}: record must be a dict")
        layer_index = int(record["layer_index"])
        by_layer[layer_index] = record

    triplet_layers = {t.layer_index for t in triplets}
    if set(by_layer) != triplet_layers:
        raise ValueError(
            f"permutation layers {sorted(by_layer)} != triplet layers {sorted(triplet_layers)}"
        )

    for triplet in triplets:
        record = by_layer[triplet.layer_index]
        context = f"layer {triplet.layer_index}"
        if int(record["intermediate_size"]) != triplet.intermediate_size:
            raise ValueError(
                f"{context}: intermediate_size mismatch "
                f"record={record['intermediate_size']} triplet={triplet.intermediate_size}"
            )
        if record["gate_module_name"] != triplet.gate.module_name:
            raise ValueError(
                f"{context}: gate_module_name mismatch "
                f"{record['gate_module_name']!r} vs {triplet.gate.module_name!r}"
            )
        if record["up_module_name"] != triplet.up.module_name:
            raise ValueError(
                f"{context}: up_module_name mismatch "
                f"{record['up_module_name']!r} vs {triplet.up.module_name!r}"
            )
        if record["down_module_name"] != triplet.down.module_name:
            raise ValueError(
                f"{context}: down_module_name mismatch "
                f"{record['down_module_name']!r} vs {triplet.down.module_name!r}"
            )

        perm = record["permutation"]
        _validate_permutation(perm, triplet.intermediate_size, context)
        if "combined_score" not in record:
            raise ValueError(f"{context}: missing combined_score")
        _validate_descending_importance_layout(
            record["combined_score"], perm, context
        )
        inverse = record["inverse_permutation"]
        _validate_permutation(inverse, triplet.intermediate_size, f"{context} inverse")
        expected = torch.arange(triplet.intermediate_size, dtype=torch.int64)
        if not torch.equal(inverse.cpu().index_select(0, perm.cpu()), expected):
            raise ValueError(f"{context}: inverse_permutation mismatch")

        _index_select_parameter_inplace(
            triplet.gate.module.weight,
            dim=0,
            index=perm,
            context=f"{context} gate_proj",
        )
        _index_select_parameter_inplace(
            triplet.up.module.weight,
            dim=0,
            index=perm,
            context=f"{context} up_proj",
        )
        _index_select_parameter_inplace(
            triplet.down.module.weight,
            dim=1,
            index=perm,
            context=f"{context} down_proj",
        )
        if triplet.gate.module.bias is not None:
            _index_select_parameter_inplace(
                triplet.gate.module.bias,
                dim=0,
                index=perm,
                context=f"{context} gate_proj bias",
            )
        if triplet.up.module.bias is not None:
            _index_select_parameter_inplace(
                triplet.up.module.bias,
                dim=0,
                index=perm,
                context=f"{context} up_proj bias",
            )
