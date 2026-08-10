from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from Block_Sparse.input_mask_proxy_study.block_layout import (
    output_block_scores,
    split_activation_blocks,
    split_weight_blocks,
    stable_topk_mask,
)
from Block_Sparse.input_mask_proxy_study.config import ExperimentConfig, MethodId, ratio_to_keep_count
from Block_Sparse.input_mask_proxy_study.energy_recovery import (
    recover_input_masks_energy,
    recover_input_masks_energy_unconditioned,
)
from Block_Sparse.input_mask_proxy_study.exact_recovery import recover_input_masks_exact
from Block_Sparse.input_mask_proxy_study.hif4_proxy import build_hif4_ternary_proxy
from Block_Sparse.input_mask_proxy_study.s0mean_recovery import recover_input_masks_s0mean_energy


@dataclass(frozen=True)
class PreparedOperands:
    x: torch.Tensor
    w: torch.Tensor
    xp: torch.Tensor
    wp: torch.Tensor
    xp_s0: torch.Tensor
    w_energy: torch.Tensor
    all_output_weight_energy: torch.Tensor
    x_blocks: torch.Tensor
    w_blocks: torch.Tensor
    xp_blocks: torch.Tensor
    wp_blocks: torch.Tensor
    y_ref: torch.Tensor
    y_xp: torch.Tensor
    y_xpwp: torch.Tensor
    my_ref_by_ratio: dict[float, torch.Tensor]
    my_xp_by_ratio: dict[float, torch.Tensor]
    my_xpwp_by_ratio: dict[float, torch.Tensor]


@dataclass(frozen=True)
class MethodSpec:
    method_id: MethodId
    output_source: Literal["ref", "xp", "xpwp"]
    contribution_source: Literal["full", "xp_fullw", "xp_wp"]
    recovery_kind: Literal["exact", "energy", "s0mean_energy", "energy_unconditioned"]


METHOD_SPECS: dict[MethodId, MethodSpec] = {
    MethodId.FULL_EXACT_REF: MethodSpec(
        MethodId.FULL_EXACT_REF, "ref", "full", "exact"
    ),
    MethodId.XPROXY_EXACT_OWN_OUTPUT: MethodSpec(
        MethodId.XPROXY_EXACT_OWN_OUTPUT, "xp", "xp_fullw", "exact"
    ),
    MethodId.XPROXY_ENERGY_OWN_OUTPUT: MethodSpec(
        MethodId.XPROXY_ENERGY_OWN_OUTPUT, "xp", "xp_fullw", "energy"
    ),
    MethodId.FULL_ENERGY_REF_OUTPUT: MethodSpec(
        MethodId.FULL_ENERGY_REF_OUTPUT, "ref", "full", "energy"
    ),
    MethodId.XWPROXY_EXACT_REF_OUTPUT: MethodSpec(
        MethodId.XWPROXY_EXACT_REF_OUTPUT, "ref", "xp_wp", "exact"
    ),
    MethodId.XWPROXY_EXACT_OWN_OUTPUT: MethodSpec(
        MethodId.XWPROXY_EXACT_OWN_OUTPUT, "xpwp", "xp_wp", "exact"
    ),
    MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT: MethodSpec(
        MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT, "xp", "xp_fullw", "s0mean_energy"
    ),
    MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT: MethodSpec(
        MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT,
        "xp",
        "xp_fullw",
        "energy_unconditioned",
    ),
}


@dataclass(frozen=True)
class MethodResult:
    method_id: MethodId
    output_masks_by_ratio: dict[float, torch.Tensor]
    input_masks_by_ratio: dict[tuple[float, float], torch.Tensor]
    compute_masks_by_ratio: dict[tuple[float, float], torch.Tensor]
    removal_order_by_output_ratio: dict[float, torch.Tensor] | None
    ranking_by_output_ratio: dict[float, torch.Tensor]
    recovery_mse_by_ratio: dict[tuple[float, float], torch.Tensor] | None


@dataclass(frozen=True)
class ConditionalOracleResult:
    mx_cond_xp: dict[tuple[float, float], torch.Tensor]
    mx_cond_xpwp: dict[tuple[float, float], torch.Tensor]


def _masks_from_y(
    y: torch.Tensor,
    ratios: tuple[float, ...],
    block_rows: int,
    block_out: int,
) -> dict[float, torch.Tensor]:
    scores = output_block_scores(y, block_rows=block_rows, block_out=block_out)
    jb = scores.shape[-1]
    out: dict[float, torch.Tensor] = {}
    for r in ratios:
        keep = ratio_to_keep_count(r, jb)
        out[float(r)] = stable_topk_mask(scores, keep)
    return out


def prepare_operands(
    x: torch.Tensor,
    w: torch.Tensor,
    config: ExperimentConfig,
) -> PreparedOperands:
    x = x.to(torch.float32)
    w = w.to(torch.float32)
    xp_result = build_hif4_ternary_proxy(x)
    xp = xp_result.proxy
    xp_s0 = xp_result.s0
    wp_result = build_hif4_ternary_proxy(w)
    wp = wp_result.proxy
    x_blocks = split_activation_blocks(
        x, config.activation_block_rows, config.k_block_size
    )
    w_blocks = split_weight_blocks(w, config.output_block_cols, config.k_block_size)
    xp_blocks = split_activation_blocks(
        xp, config.activation_block_rows, config.k_block_size
    )
    wp_blocks = split_weight_blocks(wp, config.output_block_cols, config.k_block_size)
    w_energy = w_blocks.square().mean(dim=(-1, -2))
    all_output_weight_energy = w_energy.sum(dim=0)

    y_ref = x @ w.T
    y_xp = xp @ w.T
    y_xpwp = xp @ wp.T

    my_ref = _masks_from_y(
        y_ref, config.output_keep_ratios, config.activation_block_rows, config.output_block_cols
    )
    my_xp = _masks_from_y(
        y_xp, config.output_keep_ratios, config.activation_block_rows, config.output_block_cols
    )
    my_xpwp = _masks_from_y(
        y_xpwp, config.output_keep_ratios, config.activation_block_rows, config.output_block_cols
    )
    return PreparedOperands(
        x=x,
        w=w,
        xp=xp,
        wp=wp,
        xp_s0=xp_s0,
        w_energy=w_energy,
        all_output_weight_energy=all_output_weight_energy,
        x_blocks=x_blocks,
        w_blocks=w_blocks,
        xp_blocks=xp_blocks,
        wp_blocks=wp_blocks,
        y_ref=y_ref,
        y_xp=y_xp,
        y_xpwp=y_xpwp,
        my_ref_by_ratio=my_ref,
        my_xp_by_ratio=my_xp,
        my_xpwp_by_ratio=my_xpwp,
    )


def _select_output_masks(
    prepared: PreparedOperands,
    source: str,
) -> dict[float, torch.Tensor]:
    if source == "ref":
        return prepared.my_ref_by_ratio
    if source == "xp":
        return prepared.my_xp_by_ratio
    if source == "xpwp":
        return prepared.my_xpwp_by_ratio
    raise ValueError(f"unknown output_source={source!r}")


def _select_contribution_blocks(
    prepared: PreparedOperands,
    source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source == "full":
        return prepared.x_blocks, prepared.w_blocks
    if source == "xp_fullw":
        return prepared.xp_blocks, prepared.w_blocks
    if source == "xp_wp":
        return prepared.xp_blocks, prepared.wp_blocks
    raise ValueError(f"unknown contribution_source={source!r}")


def _removal_order_to_ranking(removal_order: torch.Tensor) -> torch.Tensor:
    """Later removal => higher importance. ranking value = position from end."""
    a, kb = removal_order.shape
    ranking = torch.empty_like(removal_order)
    step_importance = torch.arange(1, kb + 1, device=removal_order.device, dtype=torch.int64)
    for i in range(a):
        ranking[i, removal_order[i]] = step_importance
    return ranking


def _is_fast_recovery(kind: str) -> bool:
    return kind in ("energy", "s0mean_energy", "energy_unconditioned")


def run_method(
    method_id: MethodId,
    prepared: PreparedOperands,
    config: ExperimentConfig,
) -> MethodResult:
    if not isinstance(method_id, MethodId):
        raise ValueError(f"unknown method_id={method_id!r}")
    if method_id not in METHOD_SPECS:
        raise ValueError(f"unknown method_id={method_id!r}")
    spec = METHOD_SPECS[method_id]

    output_masks = _select_output_masks(prepared, spec.output_source)
    kb = int(prepared.x_blocks.shape[1])
    input_keep_counts = tuple(
        ratio_to_keep_count(r, kb) for r in config.input_keep_ratios
    )
    count_to_ratios: dict[int, list[float]] = {}
    for r, kc in zip(config.input_keep_ratios, input_keep_counts):
        count_to_ratios.setdefault(kc, []).append(float(r))

    input_masks: dict[tuple[float, float], torch.Tensor] = {}
    compute_masks: dict[tuple[float, float], torch.Tensor] = {}
    removal_order_by_out: dict[float, torch.Tensor] | None
    ranking_by_out: dict[float, torch.Tensor] = {}
    mse_by_ratio: dict[tuple[float, float], torch.Tensor] | None

    if spec.recovery_kind == "exact":
        x_b, w_b = _select_contribution_blocks(prepared, spec.contribution_source)
        removal_order_by_out = {}
        mse_by_ratio = {}
        for out_r, my in output_masks.items():
            result = recover_input_masks_exact(x_b, w_b, my, input_keep_counts)
            removal_order_by_out[out_r] = result.removal_order
            ranking_by_out[out_r] = _removal_order_to_ranking(result.removal_order)
            for kc, ratios in count_to_ratios.items():
                for in_r in ratios:
                    mx = result.masks_by_keep[kc]
                    input_masks[(out_r, in_r)] = mx
                    compute_masks[(out_r, in_r)] = my[:, :, None] & mx[:, None, :]
                    mse_by_ratio[(out_r, in_r)] = result.mse_by_keep[kc]
    elif spec.recovery_kind == "energy":
        x_b, _ = _select_contribution_blocks(prepared, spec.contribution_source)
        removal_order_by_out = None
        mse_by_ratio = None
        for out_r, my in output_masks.items():
            result = recover_input_masks_energy(
                x_b, prepared.w_energy, my, input_keep_counts
            )
            ranking_by_out[out_r] = result.ranking
            for kc, ratios in count_to_ratios.items():
                for in_r in ratios:
                    mx = result.masks_by_keep[kc]
                    input_masks[(out_r, in_r)] = mx
                    compute_masks[(out_r, in_r)] = my[:, :, None] & mx[:, None, :]
    elif spec.recovery_kind == "s0mean_energy":
        removal_order_by_out = None
        mse_by_ratio = None
        for out_r, my in output_masks.items():
            result = recover_input_masks_s0mean_energy(
                prepared.xp_s0,
                config.activation_block_rows,
                prepared.w_energy,
                my,
                input_keep_counts,
            )
            ranking_by_out[out_r] = result.ranking
            for kc, ratios in count_to_ratios.items():
                for in_r in ratios:
                    mx = result.masks_by_keep[kc]
                    input_masks[(out_r, in_r)] = mx
                    compute_masks[(out_r, in_r)] = my[:, :, None] & mx[:, None, :]
    elif spec.recovery_kind == "energy_unconditioned":
        removal_order_by_out = None
        mse_by_ratio = None
        # Ranking is independent of MY / output keep ratio; compute once.
        result = recover_input_masks_energy_unconditioned(
            prepared.xp_blocks,
            prepared.all_output_weight_energy,
            input_keep_counts,
        )
        for out_r, my in output_masks.items():
            ranking_by_out[out_r] = result.ranking
            for kc, ratios in count_to_ratios.items():
                for in_r in ratios:
                    mx = result.masks_by_keep[kc]
                    input_masks[(out_r, in_r)] = mx
                    compute_masks[(out_r, in_r)] = my[:, :, None] & mx[:, None, :]
    else:
        raise ValueError(f"unknown recovery_kind={spec.recovery_kind!r}")

    return MethodResult(
        method_id=method_id,
        output_masks_by_ratio=output_masks,
        input_masks_by_ratio=input_masks,
        compute_masks_by_ratio=compute_masks,
        removal_order_by_output_ratio=removal_order_by_out,
        ranking_by_output_ratio=ranking_by_out,
        recovery_mse_by_ratio=mse_by_ratio,
    )


def build_conditional_oracles(
    prepared: PreparedOperands,
    config: ExperimentConfig,
) -> ConditionalOracleResult:
    kb = int(prepared.x_blocks.shape[1])
    input_keep_counts = tuple(
        ratio_to_keep_count(r, kb) for r in config.input_keep_ratios
    )
    count_to_ratios: dict[int, list[float]] = {}
    for r, kc in zip(config.input_keep_ratios, input_keep_counts):
        count_to_ratios.setdefault(kc, []).append(float(r))

    mx_cond_xp: dict[tuple[float, float], torch.Tensor] = {}
    mx_cond_xpwp: dict[tuple[float, float], torch.Tensor] = {}

    for out_r, my in prepared.my_xp_by_ratio.items():
        result = recover_input_masks_exact(
            prepared.x_blocks, prepared.w_blocks, my, input_keep_counts
        )
        for kc, ratios in count_to_ratios.items():
            for in_r in ratios:
                mx_cond_xp[(out_r, in_r)] = result.masks_by_keep[kc]

    for out_r, my in prepared.my_xpwp_by_ratio.items():
        result = recover_input_masks_exact(
            prepared.x_blocks, prepared.w_blocks, my, input_keep_counts
        )
        for kc, ratios in count_to_ratios.items():
            for in_r in ratios:
                mx_cond_xpwp[(out_r, in_r)] = result.masks_by_keep[kc]

    return ConditionalOracleResult(mx_cond_xp=mx_cond_xp, mx_cond_xpwp=mx_cond_xpwp)
