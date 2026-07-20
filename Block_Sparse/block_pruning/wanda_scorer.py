from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from block_pruning.block_utils import reduce_weight_wanda_to_blocks
from block_pruning.calibration import move_batch_to_device
from block_pruning.config import GradientBlockPruningConfig
from block_pruning.gradient_scorer import BlockScoreRecord, _make_record
from block_pruning.mlp_registry import MLPLinearTarget
from block_pruning.model_loader import resolve_model_input_device


@dataclass
class InputRMSRecord:
    module_name: str
    layer_index: int
    projection_type: str
    num_tokens: int
    channel_square_sum: torch.Tensor
    input_rms: torch.Tensor


def _make_rms_hook(
    module_name: str,
    expected_d_in: int,
    accumulators: dict[str, dict],
):
    def hook(_module: nn.Module, inputs: tuple) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError(
                f"Hook input missing a tensor for module: {module_name}"
            )
        x = inputs[0]
        if x.shape[-1] != expected_d_in:
            raise ValueError(
                f"Module {module_name} input last dim {x.shape[-1]} "
                f"!= weight d_in {expected_d_in}"
            )
        x2d = x.detach().float().reshape(-1, x.shape[-1])
        accumulators[module_name]["square_sum"] += (
            x2d.square().sum(dim=0).double().cpu()
        )
        accumulators[module_name]["num_tokens"] += int(x2d.shape[0])
        accumulators[module_name]["num_calls"] += 1

    return hook


def collect_mlp_input_rms(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    targets: list[MLPLinearTarget],
) -> dict[str, InputRMSRecord]:
    """Collect per-module input-channel RMS via forward_pre_hook."""
    if not batches:
        raise ValueError("calibration batches is empty")
    if not targets:
        raise ValueError("targets is empty")

    device = resolve_model_input_device(model)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    accumulators: dict[str, dict] = {}
    handles = []
    for target in targets:
        d_in = int(target.module.weight.shape[1])
        accumulators[target.module_name] = {
            "square_sum": torch.zeros(d_in, dtype=torch.float64),
            "num_tokens": 0,
            "num_calls": 0,
            "layer_index": target.layer_index,
            "projection_type": target.projection_type,
        }
        handles.append(
            target.module.register_forward_pre_hook(
                _make_rms_hook(target.module_name, d_in, accumulators)
            )
        )

    try:
        with torch.no_grad():
            for batch in batches:
                batch_dev = move_batch_to_device(batch, device)
                model(
                    input_ids=batch_dev["input_ids"],
                    attention_mask=batch_dev["attention_mask"],
                    use_cache=False,
                )
    finally:
        for handle in handles:
            handle.remove()

    records: dict[str, InputRMSRecord] = {}
    for target in targets:
        name = target.module_name
        acc = accumulators[name]
        if acc["num_calls"] == 0 or acc["num_tokens"] <= 0:
            raise RuntimeError(
                f"Target module never invoked or zero tokens: {name}"
            )
        square_sum = acc["square_sum"]
        num_tokens = int(acc["num_tokens"])
        input_rms = torch.sqrt(square_sum / num_tokens)
        records[name] = InputRMSRecord(
            module_name=name,
            layer_index=acc["layer_index"],
            projection_type=acc["projection_type"],
            num_tokens=num_tokens,
            channel_square_sum=square_sum,
            input_rms=input_rms,
        )
    return records


def collect_wanda_block_scores(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    targets: list[MLPLinearTarget],
    config: GradientBlockPruningConfig,
    current_masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, BlockScoreRecord]:
    """Collect Block-Wanda scores for each MLP Linear target."""
    h, w = config.block_height, config.block_width
    if current_masks is None:
        from block_pruning.mlp_registry import initialize_all_one_masks

        current_masks = initialize_all_one_masks(targets, h, w)

    rms_records = collect_mlp_input_rms(model, batches, targets)
    records: dict[str, BlockScoreRecord] = {}
    for target in targets:
        name = target.module_name
        rms = rms_records[name].input_rms
        wanda = (
            reduce_weight_wanda_to_blocks(
                target.module.weight.detach(),
                rms,
                h,
                w,
            )
            .double()
            .cpu()
        )
        zeros = torch.zeros_like(wanda)
        mask = current_masks[name]
        records[name] = _make_record(
            target=target,
            config=config,
            fisher=zeros,
            abs_taylor=zeros.clone(),
            signed_mean=zeros.clone(),
            current_mask=mask.clone(),
            wanda=wanda,
        )
    return records
