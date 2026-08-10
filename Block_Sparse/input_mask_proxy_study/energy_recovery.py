from __future__ import annotations

from dataclasses import dataclass

import torch

from Block_Sparse.input_mask_proxy_study.block_layout import stable_topk_mask


@dataclass(frozen=True)
class EnergyRecoveryResult:
    masks_by_keep: dict[int, torch.Tensor]
    ranking: torch.Tensor
    scores: torch.Tensor


def recover_input_masks_energy(
    activation_blocks: torch.Tensor,
    weight_energy: torch.Tensor,
    output_mask: torch.Tensor,
    keep_counts: tuple[int, ...],
) -> EnergyRecoveryResult:
    if activation_blocks.ndim != 4:
        raise ValueError("activation_blocks must be 4D [A,Kb,32,64]")
    if weight_energy.ndim != 2:
        raise ValueError("weight_energy must be 2D [Jb,Kb]")
    if output_mask.ndim != 2 or output_mask.dtype != torch.bool:
        raise ValueError("output_mask must be bool[A,Jb]")
    a, kb, _, _ = activation_blocks.shape
    jb, kb_w = weight_energy.shape
    if kb != kb_w:
        raise ValueError("activation/weight Kb mismatch")
    if tuple(output_mask.shape) != (a, jb):
        raise ValueError(
            f"output_mask shape {tuple(output_mask.shape)} != expected {(a, jb)}"
        )
    if not torch.isfinite(activation_blocks).all():
        raise ValueError("activation_blocks contains NaN/Inf")
    if not torch.isfinite(weight_energy).all():
        raise ValueError("weight_energy contains NaN/Inf")
    if not keep_counts:
        raise ValueError("keep_counts must be non-empty")
    keep_counts = tuple(sorted(set(int(k) for k in keep_counts)))
    for kc in keep_counts:
        if kc < 1 or kc > kb:
            raise ValueError(f"keep count {kc} out of range for Kb={kb}")

    activation_blocks = activation_blocks.to(torch.float32)
    weight_energy = weight_energy.to(torch.float32)
    x_energy = activation_blocks.square().mean(dim=(-1, -2))  # [A,Kb]
    weight_sum = output_mask.to(weight_energy.dtype) @ weight_energy  # [A,Kb]
    scores = x_energy * weight_sum

    ranking = torch.argsort(scores, dim=-1, descending=True, stable=True)

    masks_by_keep: dict[int, torch.Tensor] = {}
    for kc in keep_counts:
        masks_by_keep[kc] = stable_topk_mask(scores, kc)

    return EnergyRecoveryResult(
        masks_by_keep=masks_by_keep,
        ranking=ranking,
        scores=scores,
    )


def recover_input_masks_energy_unconditioned(
    activation_blocks: torch.Tensor,
    all_output_weight_energy: torch.Tensor,
    keep_counts: tuple[int, ...],
) -> EnergyRecoveryResult:
    if activation_blocks.ndim != 4:
        raise ValueError("activation_blocks must be 4D [A,Kb,32,64]")
    if all_output_weight_energy.ndim != 1:
        raise ValueError("all_output_weight_energy must be 1D [Kb]")
    a, kb, _, _ = activation_blocks.shape
    if int(all_output_weight_energy.shape[0]) != kb:
        raise ValueError(
            f"Kb mismatch: activation Kb={kb}, "
            f"all_output_weight_energy={int(all_output_weight_energy.shape[0])}"
        )
    if not torch.isfinite(activation_blocks).all():
        raise ValueError("activation_blocks contains NaN/Inf")
    if not torch.isfinite(all_output_weight_energy).all():
        raise ValueError("all_output_weight_energy contains NaN/Inf")
    if not keep_counts:
        raise ValueError("keep_counts must be non-empty")
    keep_counts = tuple(sorted(set(int(k) for k in keep_counts)))
    for kc in keep_counts:
        if kc < 1 or kc > kb:
            raise ValueError(f"keep count {kc} out of range for Kb={kb}")

    activation_blocks = activation_blocks.to(torch.float32)
    all_w = all_output_weight_energy.to(torch.float32)
    x_energy = activation_blocks.square().mean(dim=(-1, -2))  # [A,Kb]
    scores = x_energy * all_w.unsqueeze(0)
    ranking = torch.argsort(scores, dim=-1, descending=True, stable=True)

    masks_by_keep: dict[int, torch.Tensor] = {}
    for kc in keep_counts:
        masks_by_keep[kc] = stable_topk_mask(scores, kc)

    return EnergyRecoveryResult(
        masks_by_keep=masks_by_keep,
        ranking=ranking,
        scores=scores,
    )
