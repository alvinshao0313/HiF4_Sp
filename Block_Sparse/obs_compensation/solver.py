from __future__ import annotations

from dataclasses import dataclass

import torch

from block_pruning.block_utils import expand_block_mask
from obs_compensation.hessian import OBSSystem


@dataclass(frozen=True)
class ResolvedOBSOrderPolicy:
    requested_policy: str
    resolved_policy: str
    gate_up_direction: str
    down_direction: str


@dataclass(frozen=True)
class OBSSolveResult:
    compensated_weight: torch.Tensor
    kept_delta_l2: float
    kept_delta_max_abs: float
    original_pruned_l2: float
    original_max_abs: float
    compensated_max_abs: float


def resolve_obs_order_policy(
    requested_policy: str,
    mlp_permutation: str,
) -> ResolvedOBSOrderPolicy:
    if requested_policy not in {"auto", "standard", "permutation_aware"}:
        raise ValueError(
            f"Unsupported obs_order_policy={requested_policy!r}; "
            "choose from ['auto', 'standard', 'permutation_aware']"
        )
    if mlp_permutation not in {"none", "wanda_shared"}:
        raise ValueError(
            f"Unsupported source mlp_permutation={mlp_permutation!r}; "
            "expected 'none' or 'wanda_shared'"
        )
    if requested_policy == "auto":
        resolved = "standard" if mlp_permutation == "none" else "permutation_aware"
    else:
        resolved = requested_policy
    if resolved == "permutation_aware" and mlp_permutation != "wanda_shared":
        raise ValueError(
            "permutation_aware requires mlp_permutation=wanda_shared; "
            f"requested_policy={requested_policy!r}, "
            f"mlp_permutation={mlp_permutation!r}"
        )
    down_direction = (
        "right_to_left" if resolved == "permutation_aware" else "left_to_right"
    )
    return ResolvedOBSOrderPolicy(
        requested_policy=requested_policy,
        resolved_policy=resolved,
        gate_up_direction="left_to_right",
        down_direction=down_direction,
    )


def build_directional_column_order(
    num_columns: int,
    direction: str,
) -> torch.Tensor:
    if int(num_columns) < 1:
        raise ValueError(f"num_columns must be >= 1, got {num_columns}")
    if direction == "left_to_right":
        return torch.arange(num_columns, dtype=torch.int64)
    if direction == "right_to_left":
        return torch.arange(num_columns - 1, -1, -1, dtype=torch.int64)
    raise ValueError(
        f"Unsupported solver direction={direction!r}; "
        "expected 'left_to_right' or 'right_to_left'"
    )


def solve_fixed_mask_obs(
    weight: torch.Tensor,
    block_keep_mask: torch.Tensor,
    block_height: int,
    block_width: int,
    system: OBSSystem,
    solver_block_size: int,
    context: str,
) -> OBSSolveResult:
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError(f"{context}: weight must be rank-2 Tensor")
    if not isinstance(block_keep_mask, torch.Tensor) or block_keep_mask.ndim != 2:
        raise ValueError(f"{context}: block_keep_mask must be rank-2 Tensor")
    if block_keep_mask.dtype != torch.bool:
        raise TypeError(f"{context}: block_keep_mask dtype must be bool")
    if int(solver_block_size) <= 0:
        raise ValueError(
            f"{context}: solver_block_size must be > 0, got {solver_block_size}"
        )
    if not torch.isfinite(weight).all():
        raise ValueError(f"{context}: weight contains non-finite values")
    d_out, d_in = weight.shape
    if d_out % block_height != 0 or d_in % block_width != 0:
        raise ValueError(
            f"{context}: weight shape {tuple(weight.shape)} not divisible by "
            f"{block_height}x{block_width}"
        )
    expected_mask = (d_out // block_height, d_in // block_width)
    if tuple(block_keep_mask.shape) != expected_mask:
        raise ValueError(
            f"{context}: mask shape {tuple(block_keep_mask.shape)} != {expected_mask}"
        )
    if system.upper_inverse_cholesky.shape != (d_in, d_in):
        raise ValueError(
            f"{context}: factor shape {tuple(system.upper_inverse_cholesky.shape)} "
            f"!= ({d_in}, {d_in})"
        )
    if system.column_order.numel() != d_in:
        raise ValueError(f"{context}: column_order length mismatch")

    order = system.column_order.to(device=weight.device)
    W = weight.detach().float().index_select(1, order).clone()
    element_mask = expand_block_mask(
        block_keep_mask,
        block_height,
        block_width,
    ).to(device=W.device)
    element_mask = element_mask.index_select(1, order)
    R = system.upper_inverse_cholesky.to(device=W.device, dtype=torch.float32)
    if not torch.isfinite(R).all():
        raise RuntimeError(f"{context}: inverse-Cholesky factor is non-finite")
    diag_r = torch.diag(R)
    if bool((diag_r <= 0).any().item()) or not torch.isfinite(diag_r).all():
        raise RuntimeError(
            f"{context}: inverse-Cholesky diagonal has non-positive or non-finite entries"
        )

    columns = d_in
    Q = torch.zeros_like(W)
    for i1 in range(0, columns, solver_block_size):
        i2 = min(i1 + solver_block_size, columns)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        R1 = R[i1:i2, i1:i2]

        for i in range(count):
            global_i = i1 + i
            w = W1[:, i]
            keep = element_mask[:, global_i]
            q = torch.where(keep, w, torch.zeros_like(w))
            d = R1[i, i]
            if (not torch.isfinite(d)) or float(d.item()) <= 0.0:
                raise RuntimeError(
                    f"{context}: invalid R diagonal at column {global_i}: {float(d.item())}"
                )
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1).matmul(R1[i, i:].unsqueeze(0))
            Q1[:, i] = q
            Err1[:, i] = err

        if not torch.isfinite(Q1).all() or not torch.isfinite(Err1).all():
            raise RuntimeError(f"{context}: non-finite values generated during OBS update")
        Q[:, i1:i2] = Q1
        if i2 < columns:
            W[:, i2:] -= Err1.matmul(R[i1:i2, i2:])
            if not torch.isfinite(W[:, i2:]).all():
                raise RuntimeError(
                    f"{context}: non-finite values generated during OBS block update"
                )

    Q_original = torch.empty_like(Q)
    Q_original[:, order] = Q
    original_element_mask = expand_block_mask(
        block_keep_mask, block_height, block_width
    ).to(device=Q_original.device)
    Q_original.masked_fill_(~original_element_mask, 0.0)
    if torch.count_nonzero(Q_original[~original_element_mask]) != 0:
        raise RuntimeError(f"{context}: pruned positions are not exactly zero")
    if not torch.isfinite(Q_original).all():
        raise RuntimeError(f"{context}: compensated weight is non-finite")

    original = weight.detach().float()
    kept = original_element_mask
    pruned = ~kept
    delta = Q_original - original
    kept_delta = delta[kept]
    kept_delta_l2 = float(torch.linalg.vector_norm(kept_delta).item()) if kept.any() else 0.0
    kept_delta_max_abs = float(kept_delta.abs().max().item()) if kept.any() else 0.0
    original_pruned_l2 = (
        float(torch.linalg.vector_norm(original[pruned]).item()) if pruned.any() else 0.0
    )
    return OBSSolveResult(
        compensated_weight=Q_original,
        kept_delta_l2=kept_delta_l2,
        kept_delta_max_abs=kept_delta_max_abs,
        original_pruned_l2=original_pruned_l2,
        original_max_abs=float(original.abs().max().item()),
        compensated_max_abs=float(Q_original.abs().max().item()),
    )
