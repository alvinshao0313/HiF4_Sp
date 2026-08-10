from __future__ import annotations

import torch

from Block_Sparse.dynamic_input_sparse.common import (
    all_ones_mask,
    check_divisible,
    flatten_tokens,
    ratio_to_keep_count,
    stable_topk_input_mask,
)
from Block_Sparse.input_mask_proxy_study.block_layout import split_weight_blocks
from Block_Sparse.input_mask_proxy_study.hif4_proxy import build_hif4_ternary_proxy


def compute_all_output_weight_energy(
    weight: torch.Tensor,
    k_block_size: int = 64,
    output_block_size: int = 32,
) -> torch.Tensor:
    """Static G_W[k] = sum_j mean(W_block[j,k]^2), shape [Kb], FP32."""
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [D_out,D_in], got {tuple(weight.shape)}")
    if int(k_block_size) != 64 or int(output_block_size) != 32:
        raise ValueError("only k_block_size=64 and output_block_size=32 are supported")
    w = weight.detach().to(torch.float32)
    if not bool(torch.isfinite(w).all().item()):
        raise ValueError("weight contains NaN/Inf")
    w_blocks = split_weight_blocks(w, block_out=output_block_size, block_k=k_block_size)
    energy = w_blocks.square().mean(dim=(-1, -2))  # [Jb,Kb]
    g_w = energy.sum(dim=0).contiguous()
    if not bool(torch.isfinite(g_w).all().item()):
        raise ValueError("computed G_W contains NaN/Inf")
    return g_w


def predict_m8_input_mask(
    x: torch.Tensor,
    all_output_weight_energy: torch.Tensor,
    keep_ratio: float,
    k_block_size: int = 64,
) -> torch.Tensor:
    """Predict per-token input K-block mask via M8 energy score. No MY."""
    if int(k_block_size) != 64:
        raise ValueError(f"k_block_size must be 64, got {k_block_size}")
    x_flat, _ = flatten_tokens(x)
    t, d_in = int(x_flat.shape[0]), int(x_flat.shape[1])
    kb = check_divisible(d_in, k_block_size, "D_in")
    if all_output_weight_energy.ndim != 1:
        raise ValueError(
            f"all_output_weight_energy must be 1D [Kb], got {tuple(all_output_weight_energy.shape)}"
        )
    if int(all_output_weight_energy.shape[0]) != kb:
        raise ValueError(
            f"Kb mismatch: D_in/Kb={kb}, G_W={int(all_output_weight_energy.shape[0])}"
        )
    if not bool(torch.isfinite(x_flat).all().item()):
        raise ValueError("activation contains NaN/Inf")
    g_w = all_output_weight_energy.to(device=x_flat.device, dtype=torch.float32)
    if not bool(torch.isfinite(g_w).all().item()):
        raise ValueError("G_W contains NaN/Inf")

    if float(keep_ratio) == 1.0:
        return all_ones_mask(t, kb, x_flat.device)

    keep_count = ratio_to_keep_count(keep_ratio, kb)
    proxy = build_hif4_ternary_proxy(x_flat).proxy  # FP32 [T,D_in]
    xp_blocks = proxy.reshape(t, kb, k_block_size)
    e_x = xp_blocks.square().mean(dim=-1)  # [T,Kb]
    scores = e_x * g_w.unsqueeze(0)
    return stable_topk_input_mask(scores, keep_count)
