from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn

from block_pruning.mlp_registry import MLPLinearTarget
from obs_compensation.config import OBSCompensationConfig
from obs_compensation.hessian import HessianAccumulator, HessianSnapshot, OBSSystem, build_obs_system
from obs_compensation.model_adapter import (
    CapturedLayerInputs,
    get_decoder_layers,
    run_decoder_layer,
)
from obs_compensation.permutation import MLPProjectionTriplet
from obs_compensation.solver import (
    OBSSolveResult,
    ResolvedOBSOrderPolicy,
    build_directional_column_order,
    solve_fixed_mask_obs,
)


@dataclass(frozen=True)
class ModuleOBSReport:
    module_name: str
    layer_index: int
    projection_type: str
    solver_direction: str
    solver_applied: bool
    skip_reason: str
    num_total_blocks: int
    num_pruned_blocks: int
    block_sparsity: float
    num_fully_pruned_block_rows: int
    num_fully_pruned_output_rows: int
    num_hessian_tokens: int
    hessian_diagonal_mean: float
    hessian_damp_value: float
    hessian_dead_columns: int
    kept_delta_l2: float
    kept_delta_max_abs: float
    original_pruned_l2: float
    original_max_abs: float
    compensated_max_abs: float


@dataclass(frozen=True)
class LayerOBSReport:
    layer_index: int
    num_output_elements: int
    output_mse: float
    output_relative_mse: float
    output_max_abs_error: float
    down_pruned_blocks_left: int
    down_pruned_blocks_right: int


@dataclass(frozen=True)
class LayerwiseOBSResult:
    module_reports: list[ModuleOBSReport]
    layer_reports: list[LayerOBSReport]


def _mask_has_pruned_blocks(block_mask: torch.Tensor) -> bool:
    if block_mask.dtype != torch.bool or block_mask.ndim != 2:
        raise ValueError("block_mask must be a rank-2 bool tensor")
    return bool((~block_mask).any().item())


def _fully_pruned_block_rows(block_mask: torch.Tensor) -> int:
    return int((~block_mask).all(dim=1).sum().item())


def _make_module_report(
    target: MLPLinearTarget,
    block_mask: torch.Tensor,
    hessian: HessianSnapshot,
    system: OBSSystem,
    solve_result: OBSSolveResult,
    solver_direction: str,
    block_height: int,
) -> ModuleOBSReport:
    num_total_blocks = int(block_mask.numel())
    num_pruned_blocks = int((~block_mask).sum().item())
    fully_pruned_block_rows = _fully_pruned_block_rows(block_mask)
    return ModuleOBSReport(
        module_name=target.module_name,
        layer_index=target.layer_index,
        projection_type=target.projection_type,
        solver_direction=solver_direction,
        solver_applied=True,
        skip_reason="",
        num_total_blocks=num_total_blocks,
        num_pruned_blocks=num_pruned_blocks,
        block_sparsity=num_pruned_blocks / num_total_blocks,
        num_fully_pruned_block_rows=fully_pruned_block_rows,
        num_fully_pruned_output_rows=fully_pruned_block_rows * int(block_height),
        num_hessian_tokens=hessian.num_tokens,
        hessian_diagonal_mean=system.diagonal_mean,
        hessian_damp_value=system.damp_value,
        hessian_dead_columns=int(system.dead_columns.sum().item()),
        kept_delta_l2=solve_result.kept_delta_l2,
        kept_delta_max_abs=solve_result.kept_delta_max_abs,
        original_pruned_l2=solve_result.original_pruned_l2,
        original_max_abs=solve_result.original_max_abs,
        compensated_max_abs=solve_result.compensated_max_abs,
    )


def _make_skipped_module_report(
    target: MLPLinearTarget,
    block_mask: torch.Tensor,
    solver_direction: str,
    block_height: int,
) -> ModuleOBSReport:
    if _mask_has_pruned_blocks(block_mask):
        raise ValueError(
            f"{target.module_name}: skipped report requires an all-kept mask"
        )
    weight = target.module.weight.detach().float()
    max_abs = float(weight.abs().max().item())
    return ModuleOBSReport(
        module_name=target.module_name,
        layer_index=target.layer_index,
        projection_type=target.projection_type,
        solver_direction=solver_direction,
        solver_applied=False,
        skip_reason="mask_all_kept",
        num_total_blocks=int(block_mask.numel()),
        num_pruned_blocks=0,
        block_sparsity=0.0,
        num_fully_pruned_block_rows=0,
        num_fully_pruned_output_rows=0,
        num_hessian_tokens=0,
        hessian_diagonal_mean=0.0,
        hessian_damp_value=0.0,
        hessian_dead_columns=0,
        kept_delta_l2=0.0,
        kept_delta_max_abs=0.0,
        original_pruned_l2=0.0,
        original_max_abs=max_abs,
        compensated_max_abs=max_abs,
    )


@contextmanager
def _accumulate_linear_inputs(
    module: nn.Linear,
    accumulator: HessianAccumulator,
) -> Iterator[None]:
    called = {"value": False}

    def hook(_mod, inputs):
        if not inputs:
            raise RuntimeError(f"{accumulator.context}: linear hook received empty inputs")
        x = inputs[0]
        if int(x.shape[-1]) != accumulator.dimension:
            raise ValueError(
                f"{accumulator.context}: input last dim {int(x.shape[-1])} != "
                f"{accumulator.dimension}"
            )
        accumulator.add_batch(x)
        called["value"] = True

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()
    if not called["value"]:
        raise RuntimeError(f"{accumulator.context}: module was never called")


def _validate_layerwise_setup(
    model: nn.Module,
    triplets: list[MLPProjectionTriplet],
    masks: dict[str, torch.Tensor],
    order_policy: ResolvedOBSOrderPolicy,
) -> nn.ModuleList:
    layers = get_decoder_layers(model)
    if len(triplets) != len(layers):
        raise ValueError(
            f"triplet count {len(triplets)} != decoder layer count {len(layers)}"
        )
    if order_policy.gate_up_direction != "left_to_right":
        raise ValueError(
            f"order_policy.gate_up_direction must be left_to_right, "
            f"got {order_policy.gate_up_direction!r}"
        )
    if order_policy.down_direction not in {"left_to_right", "right_to_left"}:
        raise ValueError(
            f"invalid down_direction={order_policy.down_direction!r}"
        )
    if order_policy.resolved_policy == "permutation_aware":
        if order_policy.down_direction != "right_to_left":
            raise ValueError(
                "permutation_aware requires down_direction=right_to_left"
            )
    if order_policy.resolved_policy == "standard":
        if order_policy.down_direction != "left_to_right":
            raise ValueError("standard requires down_direction=left_to_right")

    seen_ids: set[int] = set()
    for idx, triplet in enumerate(triplets):
        if triplet.layer_index != idx:
            raise ValueError(
                f"triplet layer_index {triplet.layer_index} != decoder index {idx}"
            )
        layer = layers[idx]
        layer_modules = {id(m) for m in layer.modules()}
        for target in (triplet.gate, triplet.up, triplet.down):
            if id(target.module) not in layer_modules:
                raise ValueError(
                    f"{target.module_name} is not contained in decoder layer {idx}"
                )
            if id(target.module) in seen_ids:
                raise ValueError(
                    f"duplicate module object across triplets: {target.module_name}"
                )
            seen_ids.add(id(target.module))
            if target.module_name not in masks:
                raise KeyError(f"missing mask for {target.module_name}")
    return layers


def _write_compensated_weight(module: nn.Linear, compensated: torch.Tensor) -> None:
    with torch.no_grad():
        module.weight.copy_(compensated.to(dtype=module.weight.dtype))


@torch.no_grad()
def run_layerwise_mlp_obs(
    model: nn.Module,
    captured: CapturedLayerInputs,
    triplets: list[MLPProjectionTriplet],
    masks: dict[str, torch.Tensor],
    config: OBSCompensationConfig,
    order_policy: ResolvedOBSOrderPolicy,
    block_height: int,
    block_width: int,
) -> LayerwiseOBSResult:
    layers = _validate_layerwise_setup(model, triplets, masks, order_policy)
    if len(captured.hidden_states) != len(captured.layer_kwargs):
        raise ValueError("captured hidden_states/layer_kwargs length mismatch")
    if not captured.hidden_states:
        raise ValueError("no captured layer inputs")

    current_hidden = [h.clone() for h in captured.hidden_states]
    module_reports: list[ModuleOBSReport] = []
    layer_reports: list[LayerOBSReport] = []

    assert order_policy.gate_up_direction == "left_to_right"

    for triplet in triplets:
        layer = layers[triplet.layer_index]
        layer_device = triplet.gate.module.weight.device
        gate_mask = masks[triplet.gate.module_name]
        up_mask = masks[triplet.up.module_name]
        down_mask = masks[triplet.down.module_name]
        gate_needs_obs = _mask_has_pruned_blocks(gate_mask)
        up_needs_obs = _mask_has_pruned_blocks(up_mask)
        down_needs_obs = _mask_has_pruned_blocks(down_mask)
        layer_needs_obs = gate_needs_obs or up_needs_obs or down_needs_obs

        if not layer_needs_obs:
            next_hidden: list[torch.Tensor] = []
            num_output_elements = 0
            for sample_index, hidden in enumerate(current_hidden):
                hidden_on_device = hidden.to(device=layer_device)
                output = run_decoder_layer(
                    model=model,
                    layer=layer,
                    hidden_states=hidden_on_device,
                    base_layer_kwargs=captured.layer_kwargs[sample_index],
                )
                output_cpu = output.detach().to(device="cpu").clone()
                next_hidden.append(output_cpu)
                num_output_elements += int(output_cpu.numel())

            module_reports.extend(
                [
                    _make_skipped_module_report(
                        triplet.gate, gate_mask, order_policy.gate_up_direction, block_height
                    ),
                    _make_skipped_module_report(
                        triplet.up, up_mask, order_policy.gate_up_direction, block_height
                    ),
                    _make_skipped_module_report(
                        triplet.down, down_mask, order_policy.down_direction, block_height
                    ),
                ]
            )
            layer_reports.append(
                LayerOBSReport(
                    layer_index=triplet.layer_index,
                    num_output_elements=num_output_elements,
                    output_mse=0.0,
                    output_relative_mse=0.0,
                    output_max_abs_error=0.0,
                    down_pruned_blocks_left=0,
                    down_pruned_blocks_right=0,
                )
            )
            current_hidden = next_hidden
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # Pass 1: reference outputs + optional shared gate/up Hessian
        gate_up_needs_obs = gate_needs_obs or up_needs_obs
        reference_outputs: list[torch.Tensor] = []

        if gate_up_needs_obs:
            gate_acc = HessianAccumulator(
                dimension=int(triplet.gate.module.weight.shape[1]),
                device=layer_device,
                context=f"layer{triplet.layer_index}.gate_up",
            )
            with _accumulate_linear_inputs(triplet.gate.module, gate_acc):
                for sample_index, hidden in enumerate(current_hidden):
                    hidden_on_device = hidden.to(device=layer_device)
                    reference = run_decoder_layer(
                        model=model,
                        layer=layer,
                        hidden_states=hidden_on_device,
                        base_layer_kwargs=captured.layer_kwargs[sample_index],
                    )
                    reference_outputs.append(
                        reference.detach().to(device="cpu").clone()
                    )

            gate_up_hessian = gate_acc.finalize()
            gate_up_order = build_directional_column_order(
                triplet.gate.module.weight.shape[1],
                order_policy.gate_up_direction,
            )
            gate_up_system = build_obs_system(
                gate_up_hessian,
                gate_up_order,
                config.obs_percdamp,
                f"layer{triplet.layer_index}.gate_up",
            )

            if gate_needs_obs:
                gate_result = solve_fixed_mask_obs(
                    weight=triplet.gate.module.weight,
                    block_keep_mask=gate_mask,
                    block_height=block_height,
                    block_width=block_width,
                    system=gate_up_system,
                    solver_block_size=config.solver_block_size,
                    context=f"layer{triplet.layer_index}.gate_proj",
                )
                _write_compensated_weight(
                    triplet.gate.module,
                    gate_result.compensated_weight,
                )
                module_reports.append(
                    _make_module_report(
                        triplet.gate,
                        gate_mask,
                        gate_up_hessian,
                        gate_up_system,
                        gate_result,
                        order_policy.gate_up_direction,
                        block_height,
                    )
                )
                del gate_result
            else:
                module_reports.append(
                    _make_skipped_module_report(
                        triplet.gate,
                        gate_mask,
                        order_policy.gate_up_direction,
                        block_height,
                    )
                )

            if up_needs_obs:
                up_result = solve_fixed_mask_obs(
                    weight=triplet.up.module.weight,
                    block_keep_mask=up_mask,
                    block_height=block_height,
                    block_width=block_width,
                    system=gate_up_system,
                    solver_block_size=config.solver_block_size,
                    context=f"layer{triplet.layer_index}.up_proj",
                )
                _write_compensated_weight(
                    triplet.up.module,
                    up_result.compensated_weight,
                )
                module_reports.append(
                    _make_module_report(
                        triplet.up,
                        up_mask,
                        gate_up_hessian,
                        gate_up_system,
                        up_result,
                        order_policy.gate_up_direction,
                        block_height,
                    )
                )
                del up_result
            else:
                module_reports.append(
                    _make_skipped_module_report(
                        triplet.up,
                        up_mask,
                        order_policy.gate_up_direction,
                        block_height,
                    )
                )

            del gate_acc, gate_up_hessian, gate_up_system
        else:
            for sample_index, hidden in enumerate(current_hidden):
                hidden_on_device = hidden.to(device=layer_device)
                reference = run_decoder_layer(
                    model=model,
                    layer=layer,
                    hidden_states=hidden_on_device,
                    base_layer_kwargs=captured.layer_kwargs[sample_index],
                )
                reference_outputs.append(reference.detach().to(device="cpu").clone())

            module_reports.append(
                _make_skipped_module_report(
                    triplet.gate,
                    gate_mask,
                    order_policy.gate_up_direction,
                    block_height,
                )
            )
            module_reports.append(
                _make_skipped_module_report(
                    triplet.up,
                    up_mask,
                    order_policy.gate_up_direction,
                    block_height,
                )
            )

        # Pass 2: down Hessian after compensated gate/up
        if down_needs_obs:
            down_acc = HessianAccumulator(
                dimension=int(triplet.down.module.weight.shape[1]),
                device=layer_device,
                context=f"layer{triplet.layer_index}.down",
            )
            with _accumulate_linear_inputs(triplet.down.module, down_acc):
                for sample_index, hidden in enumerate(current_hidden):
                    hidden_on_device = hidden.to(device=layer_device)
                    _ = run_decoder_layer(
                        model=model,
                        layer=layer,
                        hidden_states=hidden_on_device,
                        base_layer_kwargs=captured.layer_kwargs[sample_index],
                    )
            down_hessian = down_acc.finalize()
            down_order = build_directional_column_order(
                triplet.down.module.weight.shape[1], order_policy.down_direction
            )
            down_system = build_obs_system(
                down_hessian,
                down_order,
                config.obs_percdamp,
                f"layer{triplet.layer_index}.down",
            )
            down_result = solve_fixed_mask_obs(
                weight=triplet.down.module.weight,
                block_keep_mask=down_mask,
                block_height=block_height,
                block_width=block_width,
                system=down_system,
                solver_block_size=config.solver_block_size,
                context=f"layer{triplet.layer_index}.down_proj",
            )
            _write_compensated_weight(triplet.down.module, down_result.compensated_weight)
            module_reports.append(
                _make_module_report(
                    triplet.down,
                    down_mask,
                    down_hessian,
                    down_system,
                    down_result,
                    order_policy.down_direction,
                    block_height,
                )
            )
            del down_acc, down_hessian, down_system, down_result
        else:
            module_reports.append(
                _make_skipped_module_report(
                    triplet.down,
                    down_mask,
                    order_policy.down_direction,
                    block_height,
                )
            )

        # Pass 3: compensated outputs + metrics + propagate
        squared_error_sum = 0.0
        reference_squared_sum = 0.0
        num_output_elements = 0
        max_abs_error = 0.0
        next_hidden = []
        for sample_index, hidden in enumerate(current_hidden):
            hidden_on_device = hidden.to(device=layer_device)
            final_output = run_decoder_layer(
                model=model,
                layer=layer,
                hidden_states=hidden_on_device,
                base_layer_kwargs=captured.layer_kwargs[sample_index],
            )
            reference = reference_outputs[sample_index].to(final_output.device)
            error = final_output.float() - reference.float()
            err_cpu = error.detach().double().cpu()
            ref_cpu = reference.detach().double().cpu()
            squared_error_sum += float(err_cpu.square().sum().item())
            reference_squared_sum += float(ref_cpu.square().sum().item())
            num_output_elements += int(err_cpu.numel())
            max_abs_error = max(max_abs_error, float(err_cpu.abs().max().item()))
            next_hidden.append(final_output.detach().to(device="cpu").clone())

        output_mse = squared_error_sum / max(num_output_elements, 1)
        output_relative_mse = squared_error_sum / max(reference_squared_sum, 1e-30)
        num_block_columns = down_mask.shape[1]
        split = num_block_columns // 2
        pruned_left = int((~down_mask[:, :split]).sum().item())
        pruned_right = int((~down_mask[:, split:]).sum().item())
        print(
            f"[obs] layer={triplet.layer_index} "
            f"output_mse={output_mse:.6e} relative_mse={output_relative_mse:.6e} "
            f"down_pruned_left={pruned_left} down_pruned_right={pruned_right} "
            f"down_direction={order_policy.down_direction}",
            flush=True,
        )
        layer_reports.append(
            LayerOBSReport(
                layer_index=triplet.layer_index,
                num_output_elements=num_output_elements,
                output_mse=output_mse,
                output_relative_mse=output_relative_mse,
                output_max_abs_error=max_abs_error,
                down_pruned_blocks_left=pruned_left,
                down_pruned_blocks_right=pruned_right,
            )
        )
        current_hidden = next_hidden
        del reference_outputs, next_hidden
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return LayerwiseOBSResult(
        module_reports=module_reports,
        layer_reports=layer_reports,
    )
