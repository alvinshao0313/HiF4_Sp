from __future__ import annotations

import torch


def split_activation_blocks(
    x: torch.Tensor,
    block_rows: int = 32,
    block_k: int = 64,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"activation must be 2D [T,K], got shape {tuple(x.shape)}")
    t, k = int(x.shape[0]), int(x.shape[1])
    if t % block_rows != 0:
        raise ValueError(f"T={t} not divisible by block_rows={block_rows}")
    if k % block_k != 0:
        raise ValueError(f"K={k} not divisible by block_k={block_k}")
    a = t // block_rows
    kb = k // block_k
    return (
        x.reshape(a, block_rows, kb, block_k)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def split_weight_blocks(
    w: torch.Tensor,
    block_out: int = 32,
    block_k: int = 64,
) -> torch.Tensor:
    if w.ndim != 2:
        raise ValueError(f"weight must be 2D [N,K], got shape {tuple(w.shape)}")
    n, k = int(w.shape[0]), int(w.shape[1])
    if n % block_out != 0:
        raise ValueError(f"N={n} not divisible by block_out={block_out}")
    if k % block_k != 0:
        raise ValueError(f"K={k} not divisible by block_k={block_k}")
    jb = n // block_out
    kb = k // block_k
    return (
        w.reshape(jb, block_out, kb, block_k)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def output_block_scores(
    y: torch.Tensor,
    block_rows: int = 32,
    block_out: int = 32,
) -> torch.Tensor:
    if y.ndim != 2:
        raise ValueError(f"output must be 2D [T,N], got shape {tuple(y.shape)}")
    t, n = int(y.shape[0]), int(y.shape[1])
    if t % block_rows != 0:
        raise ValueError(f"T={t} not divisible by block_rows={block_rows}")
    if n % block_out != 0:
        raise ValueError(f"N={n} not divisible by block_out={block_out}")
    a = t // block_rows
    jb = n // block_out
    blocks = (
        y.reshape(a, block_rows, jb, block_out)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return blocks.square().mean(dim=(-1, -2))


def stable_topk_mask(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got shape {tuple(scores.shape)}")
    if keep_count < 1 or keep_count > scores.shape[-1]:
        raise ValueError(
            f"keep_count must be in [1, {scores.shape[-1]}], got {keep_count}"
        )
    # Stable descending sort: equal scores keep original order -> smaller index first.
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    top = order[:, :keep_count]
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, top, True)
    return mask
