from __future__ import annotations

from dataclasses import dataclass

import torch

from Block_Sparse.input_mask_proxy_study.block_layout import stable_topk_mask


@dataclass(frozen=True)
class S0MeanEnergyRecoveryResult:
    masks_by_keep: dict[int, torch.Tensor]
    ranking: torch.Tensor
    scores: torch.Tensor
    activation_statistic: torch.Tensor


def recover_input_masks_s0mean_energy(
    activation_s0: torch.Tensor,
    activation_block_rows: int,
    weight_energy: torch.Tensor,
    output_mask: torch.Tensor,
    keep_counts: tuple[int, ...],
) -> S0MeanEnergyRecoveryResult:
    if activation_s0.ndim != 2:
        raise ValueError("activation_s0 must be 2D [T,Kb]")
    if weight_energy.ndim != 2:
        raise ValueError("weight_energy must be 2D [Jb,Kb]")
    if output_mask.ndim != 2 or output_mask.dtype != torch.bool:
        raise ValueError("output_mask must be bool[A,Jb]")
    if int(activation_block_rows) < 1:
        raise ValueError(
            f"activation_block_rows must be >= 1, got {activation_block_rows}"
        )
    t, kb = activation_s0.shape
    if t % int(activation_block_rows) != 0:
        raise ValueError(
            f"T={t} not divisible by activation_block_rows={activation_block_rows}"
        )
    a = t // int(activation_block_rows)
    jb, kb_w = weight_energy.shape
    if kb != kb_w:
        raise ValueError("activation_s0/weight_energy Kb mismatch")
    if tuple(output_mask.shape) != (a, jb):
        raise ValueError(
            f"output_mask shape {tuple(output_mask.shape)} != expected {(a, jb)}"
        )
    if not torch.isfinite(activation_s0).all():
        raise ValueError("activation_s0 contains NaN/Inf")
    if not torch.isfinite(weight_energy).all():
        raise ValueError("weight_energy contains NaN/Inf")
    if not keep_counts:
        raise ValueError("keep_counts must be non-empty")
    keep_counts = tuple(sorted(set(int(k) for k in keep_counts)))
    for kc in keep_counts:
        if kc < 1 or kc > kb:
            raise ValueError(f"keep count {kc} out of range for Kb={kb}")

    activation_s0 = activation_s0.to(torch.float32)
    weight_energy = weight_energy.to(torch.float32)
    s0_mean = activation_s0.reshape(a, int(activation_block_rows), kb).mean(dim=1)
    weight_sum = output_mask.to(weight_energy.dtype) @ weight_energy
    scores = s0_mean * weight_sum
    ranking = torch.argsort(scores, dim=-1, descending=True, stable=True)

    masks_by_keep: dict[int, torch.Tensor] = {}
    for kc in keep_counts:
        masks_by_keep[kc] = stable_topk_mask(scores, kc)

    return S0MeanEnergyRecoveryResult(
        masks_by_keep=masks_by_keep,
        ranking=ranking,
        scores=scores,
        activation_statistic=s0_mean,
    )
