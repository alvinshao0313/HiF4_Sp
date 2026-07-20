from __future__ import annotations

import torch
import torch.nn as nn

from block_pruning.block_utils import expand_block_mask
from block_pruning.mlp_registry import MLPLinearTarget


def apply_mlp_block_masks(
    model: nn.Module,
    masks: dict[str, torch.Tensor],
    block_height: int,
    block_width: int,
    targets: list[MLPLinearTarget] | None = None,
) -> None:
    """Physically zero pruned HxW blocks in-place. Does not replace nn.Linear."""
    del model  # masks are applied via target modules / named lookup
    name_to_module: dict[str, nn.Linear] = {}
    if targets is not None:
        name_to_module = {t.module_name: t.module for t in targets}

    for module_name, block_mask in masks.items():
        if name_to_module:
            module = name_to_module[module_name]
        else:
            raise ValueError(
                "targets must be provided to resolve modules for mask application"
            )
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{module_name} is not nn.Linear")

        expected = (
            module.weight.shape[0] // block_height,
            module.weight.shape[1] // block_width,
        )
        if tuple(block_mask.shape) != expected:
            raise ValueError(
                f"Mask shape {tuple(block_mask.shape)} != expected {expected} "
                f"for {module_name}"
            )

        element_mask = expand_block_mask(block_mask, block_height, block_width)
        with torch.no_grad():
            module.weight.mul_(
                element_mask.to(device=module.weight.device, dtype=module.weight.dtype)
            )


def verify_masks_and_weights(
    masks: dict[str, torch.Tensor],
    targets: list[MLPLinearTarget],
    block_height: int,
    block_width: int,
) -> None:
    """Assert pruned blocks are exactly zero."""
    name_to_target = {t.module_name: t for t in targets}
    for module_name, block_mask in masks.items():
        target = name_to_target[module_name]
        weight = target.module.weight.detach()
        d_out, d_in = weight.shape
        for out_b in range(d_out // block_height):
            for in_b in range(d_in // block_width):
                block = weight[
                    out_b * block_height : (out_b + 1) * block_height,
                    in_b * block_width : (in_b + 1) * block_width,
                ]
                if not bool(block_mask[out_b, in_b].item()):
                    nnz = int(torch.count_nonzero(block).item())
                    if nnz != 0:
                        raise AssertionError(
                            f"Pruned block not zero: {module_name} "
                            f"({out_b},{in_b}) nnz={nnz}"
                        )
