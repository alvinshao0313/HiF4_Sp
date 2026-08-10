from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import torch

from Block_Sparse.input_mask_proxy_study.block_layout import stable_topk_mask


def ratio_to_keep_count(ratio: float, total: int) -> int:
    if not (0.0 < float(ratio) <= 1.0):
        raise ValueError(f"ratio must satisfy 0 < ratio <= 1, got {ratio}")
    if int(total) < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    product = Decimal(str(ratio)) * Decimal(int(total))
    keep = int(product.to_integral_value(rounding=ROUND_HALF_UP))
    return max(1, keep)


def check_divisible(dim: int, block: int, name: str = "dim") -> int:
    dim_i = int(dim)
    block_i = int(block)
    if dim_i % block_i != 0:
        raise ValueError(f"{name}={dim_i} not divisible by block={block_i}")
    return dim_i // block_i


def flatten_tokens(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if x.ndim < 1:
        raise ValueError(f"x must have at least 1 dim, got {tuple(x.shape)}")
    leading = tuple(int(s) for s in x.shape[:-1])
    d_in = int(x.shape[-1])
    flat = x.reshape(-1, d_in)
    return flat, leading


def restore_tokens(flat: torch.Tensor, leading: tuple[int, ...]) -> torch.Tensor:
    if flat.ndim != 2:
        raise ValueError(f"flat must be 2D, got {tuple(flat.shape)}")
    return flat.reshape(*leading, flat.shape[-1])


def expand_k_block_mask(mx: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    if mx.ndim != 2:
        raise ValueError(f"mx must be 2D [T,Kb], got {tuple(mx.shape)}")
    if int(block_size) != 64:
        raise ValueError(f"block_size must be 64, got {block_size}")
    return mx.repeat_interleave(int(block_size), dim=-1)


def stable_topk_input_mask(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    return stable_topk_mask(scores, keep_count)


def is_target_mlp_prefix(prefix: str) -> bool:
    """True for dense Qwen3.5 MLP linears that may receive dynamic input masks."""
    if not prefix:
        return False
    if ".experts." in prefix or "expert" in prefix.split("."):
        return False
    return prefix.endswith(".mlp.gate_up_proj") or prefix.endswith(".mlp.down_proj")


def classify_mlp_prefix(prefix: str) -> str | None:
    """Return 'gate_up', 'down', or None."""
    if not is_target_mlp_prefix(prefix):
        return None
    if prefix.endswith(".mlp.gate_up_proj"):
        return "gate_up"
    if prefix.endswith(".mlp.down_proj"):
        return "down"
    return None


def all_ones_mask(num_tokens: int, kb: int, device: torch.device) -> torch.Tensor:
    return torch.ones(num_tokens, kb, dtype=torch.bool, device=device)
