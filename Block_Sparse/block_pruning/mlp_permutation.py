"""One-time shared Wanda permutation over MLP intermediate dimensions."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from block_pruning.config import GradientBlockPruningConfig
from block_pruning.mlp_registry import MLPLinearTarget
from block_pruning.wanda_scorer import InputRMSRecord, collect_mlp_input_rms


@dataclass(frozen=True)
class MLPProjectionTriplet:
    layer_index: int
    gate: MLPLinearTarget
    up: MLPLinearTarget
    down: MLPLinearTarget
    intermediate_size: int


@dataclass
class MLPIntermediatePermutationRecord:
    layer_index: int
    gate_module_name: str
    up_module_name: str
    down_module_name: str
    intermediate_size: int
    gate_score: torch.Tensor
    up_score: torch.Tensor
    down_score: torch.Tensor
    normalized_gate_score: torch.Tensor
    normalized_up_score: torch.Tensor
    normalized_down_score: torch.Tensor
    combined_score: torch.Tensor
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor


def group_mlp_projection_triplets(
    targets: list[MLPLinearTarget],
) -> list[MLPProjectionTriplet]:
    """Group gate/up/down targets into per-layer triplets with shape checks."""
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


def compute_up_or_gate_neuron_score(
    weight: torch.Tensor,
    input_rms: torch.Tensor,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank 2, got {tuple(weight.shape)}")
    if input_rms.ndim != 1:
        raise ValueError(f"input_rms must be rank 1, got {tuple(input_rms.shape)}")
    if input_rms.shape[0] != weight.shape[1]:
        raise ValueError(
            f"input_rms length {input_rms.shape[0]} != weight d_in {weight.shape[1]}"
        )
    w = weight.detach().float()
    rms = input_rms.detach().float().to(device=w.device)
    score = w.abs().matmul(rms)
    if not torch.isfinite(score).all():
        raise ValueError("up/gate neuron score contains non-finite values")
    if (score < 0).any():
        raise ValueError("up/gate neuron score contains negative values")
    return score.detach().cpu().double()


def compute_down_neuron_score(
    weight: torch.Tensor,
    input_rms: torch.Tensor,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank 2, got {tuple(weight.shape)}")
    if input_rms.ndim != 1:
        raise ValueError(f"input_rms must be rank 1, got {tuple(input_rms.shape)}")
    if input_rms.shape[0] != weight.shape[1]:
        raise ValueError(
            f"input_rms length {input_rms.shape[0]} != weight d_in {weight.shape[1]}"
        )
    w = weight.detach().float()
    rms = input_rms.detach().float().to(device=w.device)
    score = w.abs().sum(dim=0) * rms
    if not torch.isfinite(score).all():
        raise ValueError("down neuron score contains non-finite values")
    if (score < 0).any():
        raise ValueError("down neuron score contains negative values")
    return score.detach().cpu().double()


def normalize_projection_score(
    score: torch.Tensor,
    layer_index: int,
    projection_type: str,
) -> torch.Tensor:
    if score.ndim != 1:
        raise ValueError(
            f"layer {layer_index} {projection_type}: score must be rank 1, "
            f"got {tuple(score.shape)}"
        )
    if not torch.isfinite(score).all():
        raise ValueError(
            f"layer {layer_index} {projection_type}: non-finite score values"
        )
    if (score < 0).any():
        raise ValueError(
            f"layer {layer_index} {projection_type}: negative score values"
        )
    total = float(score.sum().item())
    if not (total > 0.0) or not (total == total):
        raise ValueError(
            f"layer {layer_index} {projection_type}: normalization total must be "
            f"positive and finite, got {total}"
        )
    return score / total


def _require_rms(
    input_rms_records: dict[str, InputRMSRecord],
    module_name: str,
    expected_length: int,
) -> torch.Tensor:
    if module_name not in input_rms_records:
        raise KeyError(f"Missing InputRMSRecord for module: {module_name}")
    rms = input_rms_records[module_name].input_rms
    if rms.shape[0] != expected_length:
        raise ValueError(
            f"RMS length {rms.shape[0]} != expected {expected_length} "
            f"for {module_name}"
        )
    return rms


def compute_mlp_shared_wanda_permutations(
    triplets: list[MLPProjectionTriplet],
    input_rms_records: dict[str, InputRMSRecord],
) -> dict[int, MLPIntermediatePermutationRecord]:
    records: dict[int, MLPIntermediatePermutationRecord] = {}
    for triplet in triplets:
        gate_rms = _require_rms(
            input_rms_records,
            triplet.gate.module_name,
            triplet.gate.module.weight.shape[1],
        )
        up_rms = _require_rms(
            input_rms_records,
            triplet.up.module_name,
            triplet.up.module.weight.shape[1],
        )
        down_rms = _require_rms(
            input_rms_records,
            triplet.down.module_name,
            triplet.down.module.weight.shape[1],
        )

        gate_score = compute_up_or_gate_neuron_score(
            triplet.gate.module.weight, gate_rms
        )
        up_score = compute_up_or_gate_neuron_score(
            triplet.up.module.weight, up_rms
        )
        down_score = compute_down_neuron_score(
            triplet.down.module.weight, down_rms
        )
        if not (
            gate_score.shape[0]
            == up_score.shape[0]
            == down_score.shape[0]
            == triplet.intermediate_size
        ):
            raise ValueError(
                f"layer {triplet.layer_index}: neuron score lengths mismatch "
                f"gate={gate_score.shape[0]} up={up_score.shape[0]} "
                f"down={down_score.shape[0]} intermediate={triplet.intermediate_size}"
            )

        norm_gate = normalize_projection_score(
            gate_score, triplet.layer_index, "gate_proj"
        )
        norm_up = normalize_projection_score(
            up_score, triplet.layer_index, "up_proj"
        )
        norm_down = normalize_projection_score(
            down_score, triplet.layer_index, "down_proj"
        )
        combined = norm_gate + norm_up + norm_down
        permutation = torch.argsort(combined, descending=True, stable=True).to(
            dtype=torch.int64
        )
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(
            triplet.intermediate_size, dtype=torch.int64
        )

        records[triplet.layer_index] = MLPIntermediatePermutationRecord(
            layer_index=triplet.layer_index,
            gate_module_name=triplet.gate.module_name,
            up_module_name=triplet.up.module_name,
            down_module_name=triplet.down.module_name,
            intermediate_size=triplet.intermediate_size,
            gate_score=gate_score,
            up_score=up_score,
            down_score=down_score,
            normalized_gate_score=norm_gate,
            normalized_up_score=norm_up,
            normalized_down_score=norm_down,
            combined_score=combined,
            permutation=permutation,
            inverse_permutation=inverse,
        )
    return records


def _validate_permutation(perm: torch.Tensor, intermediate_size: int, ctx: str) -> None:
    if perm.dtype != torch.int64:
        raise TypeError(f"{ctx}: permutation dtype must be int64, got {perm.dtype}")
    if perm.ndim != 1 or perm.numel() != intermediate_size:
        raise ValueError(
            f"{ctx}: permutation length {perm.numel()} != intermediate {intermediate_size}"
        )
    if perm.min().item() < 0 or perm.max().item() >= intermediate_size:
        raise ValueError(f"{ctx}: permutation values out of range")
    if int(torch.unique(perm).numel()) != intermediate_size:
        raise ValueError(f"{ctx}: permutation is not a bijection")


def _index_select_inplace(param: nn.Parameter, dim: int, index: torch.Tensor) -> None:
    with torch.no_grad():
        selected = param.detach().index_select(dim, index)
        param.copy_(selected)


def _permute_linear_along_intermediate(
    linear: nn.Linear,
    perm: torch.Tensor,
    *,
    weight_dim: int,
    permute_bias: bool,
    ctx: str,
) -> None:
    _validate_permutation(perm, linear.weight.shape[weight_dim], ctx)
    index = perm.to(device=linear.weight.device)
    weight_id = id(linear.weight)
    _index_select_inplace(linear.weight, weight_dim, index)
    if id(linear.weight) != weight_id:
        raise RuntimeError(f"{ctx}: Parameter identity changed for weight")
    if permute_bias:
        if linear.bias is None:
            return
        bias_id = id(linear.bias)
        bias_index = perm.to(device=linear.bias.device)
        _index_select_inplace(linear.bias, 0, bias_index)
        if id(linear.bias) != bias_id:
            raise RuntimeError(f"{ctx}: Parameter identity changed for bias")
    elif linear.bias is not None and weight_dim == 1:
        # down_proj bias lives on output (hidden) dim; leave untouched.
        return


def apply_mlp_intermediate_permutations(
    triplets: list[MLPProjectionTriplet],
    records: dict[int, MLPIntermediatePermutationRecord],
) -> None:
    if set(records) != {t.layer_index for t in triplets}:
        raise ValueError(
            f"record layers {sorted(records)} != triplet layers "
            f"{sorted(t.layer_index for t in triplets)}"
        )
    for triplet in triplets:
        record = records[triplet.layer_index]
        if record.intermediate_size != triplet.intermediate_size:
            raise ValueError(
                f"layer {triplet.layer_index}: intermediate size mismatch "
                f"record={record.intermediate_size} triplet={triplet.intermediate_size}"
            )
        perm = record.permutation
        _permute_linear_along_intermediate(
            triplet.gate.module,
            perm,
            weight_dim=0,
            permute_bias=True,
            ctx=f"layer {triplet.layer_index} gate_proj",
        )
        _permute_linear_along_intermediate(
            triplet.up.module,
            perm,
            weight_dim=0,
            permute_bias=True,
            ctx=f"layer {triplet.layer_index} up_proj",
        )
        _permute_linear_along_intermediate(
            triplet.down.module,
            perm,
            weight_dim=1,
            permute_bias=False,
            ctx=f"layer {triplet.layer_index} down_proj",
        )


def undo_mlp_intermediate_permutations(
    triplets: list[MLPProjectionTriplet],
    records: dict[int, MLPIntermediatePermutationRecord],
) -> None:
    """Invert a previously applied shared permutation. Tests only."""
    if set(records) != {t.layer_index for t in triplets}:
        raise ValueError(
            f"record layers {sorted(records)} != triplet layers "
            f"{sorted(t.layer_index for t in triplets)}"
        )
    for triplet in triplets:
        record = records[triplet.layer_index]
        inv = record.inverse_permutation
        _permute_linear_along_intermediate(
            triplet.gate.module,
            inv,
            weight_dim=0,
            permute_bias=True,
            ctx=f"undo layer {triplet.layer_index} gate_proj",
        )
        _permute_linear_along_intermediate(
            triplet.up.module,
            inv,
            weight_dim=0,
            permute_bias=True,
            ctx=f"undo layer {triplet.layer_index} up_proj",
        )
        _permute_linear_along_intermediate(
            triplet.down.module,
            inv,
            weight_dim=1,
            permute_bias=False,
            ctx=f"undo layer {triplet.layer_index} down_proj",
        )


def prepare_and_apply_mlp_permutations(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]] | None,
    targets: list[MLPLinearTarget],
    config: GradientBlockPruningConfig,
) -> dict[int, MLPIntermediatePermutationRecord]:
    """Collect dense RMS once, compute shared perms, apply once. No mask work."""
    if config.mlp_permutation != "wanda_shared":
        raise ValueError(
            f"prepare_and_apply_mlp_permutations requires mlp_permutation="
            f"'wanda_shared', got {config.mlp_permutation!r}"
        )
    if batches is None:
        raise ValueError(
            "wanda_shared permutation requires calibration batches; got None"
        )
    triplets = group_mlp_projection_triplets(targets)
    input_rms_records = collect_mlp_input_rms(
        model,
        batches,
        targets,
        progress_desc="[prune] perm rms",
    )
    records = compute_mlp_shared_wanda_permutations(triplets, input_rms_records)
    apply_mlp_intermediate_permutations(triplets, records)
    return records
