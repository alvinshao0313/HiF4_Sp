from __future__ import annotations

import torch
import torch.nn.functional as F

from Block_Sparse.dynamic_input_sparse.common import (
    expand_k_block_mask,
    flatten_tokens,
    restore_tokens,
)


def apply_input_block_mask(
    x: torch.Tensor,
    mx: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """Zero K-blocks in x according to per-token MX. Does not modify weights."""
    if int(block_size) != 64:
        raise ValueError(f"block_size must be 64, got {block_size}")
    x_flat, leading = flatten_tokens(x)
    t, d_in = int(x_flat.shape[0]), int(x_flat.shape[1])
    if mx.ndim != 2 or int(mx.shape[0]) != t:
        raise ValueError(
            f"mx shape {tuple(mx.shape)} incompatible with flattened x tokens={t}"
        )
    kb = int(mx.shape[1])
    if kb * block_size != d_in:
        raise ValueError(
            f"mx Kb={kb} * block={block_size} != D_in={d_in}"
        )
    element = expand_k_block_mask(mx, block_size=block_size)
    masked = x_flat * element.to(dtype=x_flat.dtype)
    return restore_tokens(masked, leading)


def masked_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mx: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    x_masked = apply_input_block_mask(x, mx, block_size=block_size)
    return F.linear(x_masked, weight, bias)
