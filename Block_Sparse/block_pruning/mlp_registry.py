from __future__ import annotations

import re
from dataclasses import dataclass

import torch.nn as nn

from block_pruning.block_utils import validate_weight_divisible
from block_pruning.config import PROJECTION_TYPES


_LAYER_INDEX_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.")


@dataclass(frozen=True)
class MLPLinearTarget:
    module_name: str
    module: nn.Linear
    layer_index: int
    projection_type: str  # gate_proj | up_proj | down_proj


def _parse_layer_index(module_name: str) -> int:
    match = _LAYER_INDEX_RE.search(module_name)
    if match is None:
        raise ValueError(f"Cannot parse layer index from module name: {module_name}")
    return int(match.group(1))


def _is_mlp_projection(module_name: str) -> str | None:
    """Return projection type if this is a dense MLP linear, else None.

    Matches *.mlp.{gate,up,down}_proj and rejects MoE expert paths.
    """
    if ".experts." in module_name:
        return None
    for proj in PROJECTION_TYPES:
        if module_name.endswith(f".mlp.{proj}") or module_name.endswith(f".{proj}"):
            parts = module_name.split(".")
            if len(parts) >= 2 and parts[-2] == "mlp" and parts[-1] == proj:
                return proj
    return None


def collect_mlp_linears(
    model: nn.Module,
    block_height: int,
    block_width: int,
) -> list[MLPLinearTarget]:
    targets: list[MLPLinearTarget] = []
    for name, module in model.named_modules():
        proj = _is_mlp_projection(name)
        if proj is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"Target module {name} is {type(module).__name__}, expected nn.Linear"
            )
        validate_weight_divisible(module.weight, block_height, block_width)
        targets.append(
            MLPLinearTarget(
                module_name=name,
                module=module,
                layer_index=_parse_layer_index(name),
                projection_type=proj,
            )
        )

    if not targets:
        raise RuntimeError(
            "No MLP Linear targets found. Expected modules matching "
            "*.mlp.{gate_proj,up_proj,down_proj}."
        )

    targets.sort(
        key=lambda t: (t.layer_index, PROJECTION_TYPES.index(t.projection_type), t.module_name)
    )
    return targets


def initialize_all_one_masks(
    targets: list[MLPLinearTarget],
    block_height: int,
    block_width: int,
) -> dict[str, "torch.Tensor"]:
    import torch

    masks: dict[str, torch.Tensor] = {}
    for target in targets:
        d_out, d_in = target.module.weight.shape
        masks[target.module_name] = torch.ones(
            d_out // block_height,
            d_in // block_width,
            dtype=torch.bool,
        )
    return masks
