from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from Block_Sparse.input_mask_proxy_study.artifacts import write_run_artifacts
from Block_Sparse.input_mask_proxy_study.benchmark import benchmark_cuda
from Block_Sparse.input_mask_proxy_study.block_layout import (
    output_block_scores,
    split_activation_blocks,
    split_weight_blocks,
    stable_topk_mask,
)
from Block_Sparse.input_mask_proxy_study.capture import (
    capture_manifest_fields,
    capture_or_load,
)
from Block_Sparse.input_mask_proxy_study.config import (
    ExperimentConfig,
    MethodId,
    load_config,
    ratio_to_keep_count,
)
from Block_Sparse.input_mask_proxy_study.energy_recovery import (
    recover_input_masks_energy,
    recover_input_masks_energy_unconditioned,
)
from Block_Sparse.input_mask_proxy_study.exact_recovery import recover_input_masks_exact
from Block_Sparse.input_mask_proxy_study.hif4_proxy import build_hif4_ternary_proxy
from Block_Sparse.input_mask_proxy_study.methods import (
    METHOD_SPECS,
    PreparedOperands,
    build_conditional_oracles,
    prepare_operands,
    run_method,
)
from Block_Sparse.input_mask_proxy_study.metrics import (
    kendall_tau_b,
    mask_metrics,
    mse,
    nrmse,
    pearson_corr,
    reconstruct_joint_sparse_output,
    reconstruct_real_output,
    spearman_rank,
)
from Block_Sparse.input_mask_proxy_study.parallel_timing import time_methods_multi_gpu
from Block_Sparse.input_mask_proxy_study.report import build_aggregate_summary, render_report
from Block_Sparse.input_mask_proxy_study.s0mean_recovery import recover_input_masks_s0mean_energy


def _configure_deterministic() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    t = torch.tensor(xs, dtype=torch.float64)
    return {
        "mean": float(t.mean().item()),
        "median": float(t.median().item()),
        "p10": float(torch.quantile(t, 0.10).item()),
        "p90": float(torch.quantile(t, 0.90).item()),
    }


def _conditional_ref_for_method(
    method_id: MethodId,
    out_r: float,
    in_r: float,
    m1_mx: torch.Tensor,
    oracles,
) -> torch.Tensor:
    if method_id in (
        MethodId.XPROXY_EXACT_OWN_OUTPUT,
        MethodId.XPROXY_ENERGY_OWN_OUTPUT,
        MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT,
        MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT,
    ):
        return oracles.mx_cond_xp[(out_r, in_r)]
    if method_id == MethodId.XWPROXY_EXACT_OWN_OUTPUT:
        return oracles.mx_cond_xpwp[(out_r, in_r)]
    return m1_mx


def _is_fast_recovery(kind: str) -> bool:
    return kind in ("energy", "s0mean_energy", "energy_unconditioned")


def _latency_median(
    latency_rows: list[dict[str, Any]],
    method_id: str,
    scope: str,
) -> float:
    vals = [
        float(r["median_ms"])
        for r in latency_rows
        if r["method_id"] == method_id
        and r["timing_scope"] == scope
        and r["output_keep_ratio"] != ""
        and r["input_keep_ratio"] != ""
    ]
    if not vals:
        return float("nan")
    return float(torch.tensor(vals, dtype=torch.float64).median().item())


def compute_m7_vs_m3(
    *,
    xp_blocks: torch.Tensor,
    xp_s0: torch.Tensor,
    results: dict[MethodId, Any],
    condition_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    config: ExperimentConfig,
) -> dict[str, Any]:
    a = xp_blocks.shape[0]
    kb = xp_blocks.shape[1]
    xp_energy = xp_blocks.square().mean(dim=(-1, -2))
    s0_mean = xp_s0.reshape(a, config.activation_block_rows, kb).mean(dim=1)

    pearsons: list[float] = []
    spearmans: list[float] = []
    for i in range(a):
        pearsons.append(pearson_corr(s0_mean[i], xp_energy[i]))
        spearmans.append(spearman_rank(s0_mean[i], xp_energy[i]))

    overlaps: list[float] = []
    ious: list[float] = []
    m3 = results[MethodId.XPROXY_ENERGY_OWN_OUTPUT]
    m7 = results[MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT]
    for out_r in config.output_keep_ratios:
        for in_r in config.input_keep_ratios:
            mx3 = m3.input_masks_by_ratio[(out_r, in_r)]
            mx7 = m7.input_masks_by_ratio[(out_r, in_r)]
            for i in range(a):
                mm = mask_metrics(mx7[i : i + 1], mx3[i : i + 1])
                overlaps.append(mm["overlap"])
                ious.append(mm["iou"])

    def _cond_mean(mid: str, key: str) -> float:
        vals = [float(r[key]) for r in condition_rows if r["method_id"] == mid]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    nrmse_m3 = _cond_mean(MethodId.XPROXY_ENERGY_OWN_OUTPUT.value, "real_output_nrmse_mean")
    nrmse_m7 = _cond_mean(
        MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT.value, "real_output_nrmse_mean"
    )
    t_m3_rec = _latency_median(
        latency_rows, MethodId.XPROXY_ENERGY_OWN_OUTPUT.value, "input_recovery_ms"
    )
    t_m7_rec = _latency_median(
        latency_rows,
        MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT.value,
        "input_recovery_ms",
    )
    t_m3_stat = _latency_median(
        latency_rows, MethodId.XPROXY_ENERGY_OWN_OUTPUT.value, "activation_statistic_ms"
    )
    t_m7_stat = _latency_median(
        latency_rows,
        MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT.value,
        "activation_statistic_ms",
    )
    t_m3_online = _latency_median(
        latency_rows, MethodId.XPROXY_ENERGY_OWN_OUTPUT.value, "online_total_ms"
    )
    t_m7_online = _latency_median(
        latency_rows, MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT.value, "online_total_ms"
    )

    def _speedup(a: float, b: float) -> float:
        if not (math.isfinite(a) and math.isfinite(b)) or b <= 0:
            return float("nan")
        return float(a / b)

    return {
        "input_mask_overlap_mean": float(sum(overlaps) / len(overlaps)),
        "input_mask_overlap_median": float(
            torch.tensor(overlaps, dtype=torch.float64).median().item()
        ),
        "input_mask_iou_mean": float(sum(ious) / len(ious)),
        "real_output_nrmse_delta_mean": float(nrmse_m7 - nrmse_m3),
        "input_recovery_speedup": _speedup(t_m3_rec, t_m7_rec),
        "activation_statistic_speedup": _speedup(t_m3_stat, t_m7_stat),
        "online_total_speedup": _speedup(t_m3_online, t_m7_online),
        "s0mean_vs_xp_energy_spearman": float(sum(spearmans) / len(spearmans)),
        "s0mean_vs_xp_energy_pearson": float(sum(pearsons) / len(pearsons)),
        "s0mean_vs_xp_energy_spearman_stats": _stats(spearmans),
        "s0mean_vs_xp_energy_pearson_stats": _stats(pearsons),
    }


def compute_m8_vs_m3(
    *,
    results: dict[MethodId, Any],
    condition_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    config: ExperimentConfig,
) -> dict[str, Any]:
    m3 = results[MethodId.XPROXY_ENERGY_OWN_OUTPUT]
    m8 = results[MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT]
    overlaps: list[float] = []
    ious: list[float] = []
    a = next(iter(m3.input_masks_by_ratio.values())).shape[0]
    for out_r in config.output_keep_ratios:
        for in_r in config.input_keep_ratios:
            mx3 = m3.input_masks_by_ratio[(out_r, in_r)]
            mx8 = m8.input_masks_by_ratio[(out_r, in_r)]
            for i in range(a):
                mm = mask_metrics(mx8[i : i + 1], mx3[i : i + 1])
                overlaps.append(mm["overlap"])
                ious.append(mm["iou"])

    def _cond_mean(mid: str, key: str) -> float:
        vals = [float(r[key]) for r in condition_rows if r["method_id"] == mid]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def _group_mean(mid: str, key: str, field: str, value: float) -> float:
        vals = [
            float(r[key])
            for r in condition_rows
            if r["method_id"] == mid and float(r[field]) == float(value)
        ]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    m3_id = MethodId.XPROXY_ENERGY_OWN_OUTPUT.value
    m8_id = MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT.value
    ov_m1_m3 = _cond_mean(m3_id, "input_overlap_to_m1_median")
    ov_m1_m8 = _cond_mean(m8_id, "input_overlap_to_m1_median")
    ov_cond_m3 = _cond_mean(m3_id, "input_overlap_to_conditional_oracle_median")
    ov_cond_m8 = _cond_mean(m8_id, "input_overlap_to_conditional_oracle_median")
    nrmse_m3 = _cond_mean(m3_id, "real_output_nrmse_mean")
    nrmse_m8 = _cond_mean(m8_id, "real_output_nrmse_mean")
    regret_m3 = _cond_mean(m3_id, "nrmse_regret_vs_m1_mean")
    regret_m8 = _cond_mean(m8_id, "nrmse_regret_vs_m1_mean")

    t_m3_rec = _latency_median(latency_rows, m3_id, "input_recovery_ms")
    t_m8_rec = _latency_median(latency_rows, m8_id, "input_recovery_ms")
    t_m3_online = _latency_median(latency_rows, m3_id, "online_total_ms")
    t_m8_online = _latency_median(latency_rows, m8_id, "online_total_ms")

    def _speedup(a: float, b: float) -> float:
        if not (math.isfinite(a) and math.isfinite(b)) or b <= 0:
            return float("nan")
        return float(a / b)

    by_out: dict[str, dict[str, float]] = {}
    for out_r in config.output_keep_ratios:
        ov_pairs: list[float] = []
        for in_r in config.input_keep_ratios:
            mx3 = m3.input_masks_by_ratio[(out_r, in_r)]
            mx8 = m8.input_masks_by_ratio[(out_r, in_r)]
            for i in range(a):
                ov_pairs.append(mask_metrics(mx8[i : i + 1], mx3[i : i + 1])["overlap"])
        by_out[str(out_r)] = {
            "m8_m3_overlap_mean": float(sum(ov_pairs) / len(ov_pairs)),
            "m3_nrmse_mean": _group_mean(m3_id, "real_output_nrmse_mean", "output_keep_ratio", out_r),
            "m8_nrmse_mean": _group_mean(m8_id, "real_output_nrmse_mean", "output_keep_ratio", out_r),
            "nrmse_delta_mean": (
                _group_mean(m8_id, "real_output_nrmse_mean", "output_keep_ratio", out_r)
                - _group_mean(m3_id, "real_output_nrmse_mean", "output_keep_ratio", out_r)
            ),
            "m3_overlap_to_m1_mean": _group_mean(
                m3_id, "input_overlap_to_m1_mean", "output_keep_ratio", out_r
            ),
            "m8_overlap_to_m1_mean": _group_mean(
                m8_id, "input_overlap_to_m1_mean", "output_keep_ratio", out_r
            ),
        }

    by_in: dict[str, dict[str, float]] = {}
    for in_r in config.input_keep_ratios:
        ov_pairs = []
        for out_r in config.output_keep_ratios:
            mx3 = m3.input_masks_by_ratio[(out_r, in_r)]
            mx8 = m8.input_masks_by_ratio[(out_r, in_r)]
            for i in range(a):
                ov_pairs.append(mask_metrics(mx8[i : i + 1], mx3[i : i + 1])["overlap"])
        by_in[str(in_r)] = {
            "m8_m3_overlap_mean": float(sum(ov_pairs) / len(ov_pairs)),
            "m3_nrmse_mean": _group_mean(m3_id, "real_output_nrmse_mean", "input_keep_ratio", in_r),
            "m8_nrmse_mean": _group_mean(m8_id, "real_output_nrmse_mean", "input_keep_ratio", in_r),
            "nrmse_delta_mean": (
                _group_mean(m8_id, "real_output_nrmse_mean", "input_keep_ratio", in_r)
                - _group_mean(m3_id, "real_output_nrmse_mean", "input_keep_ratio", in_r)
            ),
            "m3_overlap_to_m1_mean": _group_mean(
                m3_id, "input_overlap_to_m1_mean", "input_keep_ratio", in_r
            ),
            "m8_overlap_to_m1_mean": _group_mean(
                m8_id, "input_overlap_to_m1_mean", "input_keep_ratio", in_r
            ),
        }

    return {
        "input_mask_overlap_mean": float(sum(overlaps) / len(overlaps)),
        "input_mask_overlap_median": float(
            torch.tensor(overlaps, dtype=torch.float64).median().item()
        ),
        "input_mask_overlap_p10": float(
            torch.quantile(torch.tensor(overlaps, dtype=torch.float64), 0.10).item()
        ),
        "input_mask_overlap_p90": float(
            torch.quantile(torch.tensor(overlaps, dtype=torch.float64), 0.90).item()
        ),
        "input_mask_iou_mean": float(sum(ious) / len(ious)),
        "overlap_to_m1_m8_mean": ov_m1_m8,
        "overlap_to_m1_m3_mean": ov_m1_m3,
        "overlap_to_m1_delta_mean": float(ov_m1_m8 - ov_m1_m3),
        "conditional_overlap_m8_mean": ov_cond_m8,
        "conditional_overlap_m3_mean": ov_cond_m3,
        "conditional_overlap_delta_mean": float(ov_cond_m8 - ov_cond_m3),
        "real_output_nrmse_delta_mean": float(nrmse_m8 - nrmse_m3),
        "nrmse_regret_delta_mean": float(regret_m8 - regret_m3),
        "input_recovery_speedup": _speedup(t_m3_rec, t_m8_rec),
        "online_total_speedup": _speedup(t_m3_online, t_m8_online),
        "by_output_keep_ratio": by_out,
        "by_input_keep_ratio": by_in,
    }


def _run_correctness_gates(
    *,
    results: dict[MethodId, Any],
    my_ref_by_ratio: dict[float, torch.Tensor],
    my_xp_by_ratio: dict[float, torch.Tensor],
    my_xpwp_by_ratio: dict[float, torch.Tensor],
    kb: int,
    jb: int,
    config: ExperimentConfig,
    condition_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    xp_s0_shape: tuple[int, ...],
    w_energy_shape: tuple[int, ...],
    all_output_weight_energy_shape: tuple[int, ...],
) -> list[str]:
    gates: list[str] = []
    if len(MethodId) != 8:
        gates.append(f"method_count={len(MethodId)} expected=8")
    if set(results) != set(MethodId):
        gates.append("missing_methods")

    for mid in (
        MethodId.FULL_EXACT_REF,
        MethodId.FULL_ENERGY_REF_OUTPUT,
        MethodId.XWPROXY_EXACT_REF_OUTPUT,
    ):
        for r, my in results[mid].output_masks_by_ratio.items():
            if not torch.equal(my.cpu(), my_ref_by_ratio[r].cpu()):
                gates.append(f"{mid.value}_output_mask_ne_my_ref@{r}")
    for mid in (
        MethodId.XPROXY_EXACT_OWN_OUTPUT,
        MethodId.XPROXY_ENERGY_OWN_OUTPUT,
        MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT,
        MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT,
    ):
        for r, my in results[mid].output_masks_by_ratio.items():
            if not torch.equal(my.cpu(), my_xp_by_ratio[r].cpu()):
                gates.append(f"{mid.value}_output_mask_ne_my_xp@{r}")
    for r, my in results[MethodId.XWPROXY_EXACT_OWN_OUTPUT].output_masks_by_ratio.items():
        if not torch.equal(my.cpu(), my_xpwp_by_ratio[r].cpu()):
            gates.append(f"m6_output_mask_ne_my_xpwp@{r}")

    for r in config.output_keep_ratios:
        m2 = results[MethodId.XPROXY_EXACT_OWN_OUTPUT].output_masks_by_ratio[r].cpu()
        m3 = results[MethodId.XPROXY_ENERGY_OWN_OUTPUT].output_masks_by_ratio[r].cpu()
        m7 = results[MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT].output_masks_by_ratio[r].cpu()
        m8 = results[MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT].output_masks_by_ratio[r].cpu()
        if not torch.equal(m2, m3):
            gates.append(f"m2_m3_output_mismatch@{r}")
        if not torch.equal(m3, m7):
            gates.append(f"m3_m7_output_mismatch@{r}")
        if not torch.equal(m3, m8):
            gates.append(f"m3_m8_output_mismatch@{r}")
        if not torch.equal(
            results[MethodId.FULL_EXACT_REF].output_masks_by_ratio[r].cpu(),
            results[MethodId.XWPROXY_EXACT_REF_OUTPUT].output_masks_by_ratio[r].cpu(),
        ):
            gates.append(f"m1_m5_output_mismatch@{r}")

    # M8 MX must be independent of output keep ratio.
    m8_res = results[MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT]
    for in_r in config.input_keep_ratios:
        out_ratios = list(config.output_keep_ratios)
        base = m8_res.input_masks_by_ratio[(out_ratios[0], in_r)].cpu()
        for out_r in out_ratios[1:]:
            if not torch.equal(base, m8_res.input_masks_by_ratio[(out_r, in_r)].cpu()):
                gates.append(f"m8_mx_depends_on_output_ratio@in={in_r},out={out_r}")

    expected_t = int(config.num_samples) * int(config.max_seq_len)
    expected_kb = kb
    if tuple(xp_s0_shape) != (expected_t, expected_kb):
        gates.append(f"xp_s0_shape={xp_s0_shape} expected={(expected_t, expected_kb)}")
    if tuple(w_energy_shape) != (jb, kb):
        gates.append(f"w_energy_shape={w_energy_shape} expected={(jb, kb)}")
    if tuple(all_output_weight_energy_shape) != (kb,):
        gates.append(
            f"all_output_weight_energy_shape={all_output_weight_energy_shape} "
            f"expected={(kb,)}"
        )

    m7_res = results[MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT]
    # nested masks across keep ratios for M7
    for out_r in config.output_keep_ratios:
        sorted_in = sorted(config.input_keep_ratios)
        for i in range(len(sorted_in) - 1):
            lo = m7_res.input_masks_by_ratio[(out_r, sorted_in[i])]
            hi = m7_res.input_masks_by_ratio[(out_r, sorted_in[i + 1])]
            if not torch.all(lo <= hi):
                gates.append(f"m7_nested_mask_fail@{out_r},{sorted_in[i]}->{sorted_in[i+1]}")

    for mid, res in results.items():
        for out_r, my in res.output_masks_by_ratio.items():
            expect_y = ratio_to_keep_count(out_r, jb)
            if not torch.all(my.sum(-1) == expect_y):
                gates.append(f"{mid.value}_bad_output_keep@{out_r}")
        for (out_r, in_r), mx in res.input_masks_by_ratio.items():
            expect_x = ratio_to_keep_count(in_r, kb)
            if not torch.all(mx.sum(-1) == expect_x):
                gates.append(f"{mid.value}_bad_input_keep@{out_r},{in_r}")
            my = res.output_masks_by_ratio[out_r]
            comp = res.compute_masks_by_ratio[(out_r, in_r)]
            if not torch.equal(comp, my[:, :, None] & mx[:, None, :]):
                gates.append(f"{mid.value}_compute_mask_mismatch@{out_r},{in_r}")
            if mid == MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT:
                if not torch.isfinite(mx.float()).all():
                    gates.append(f"m7_nonfinite_mask@{out_r},{in_r}")

    expected_conditions = (
        len(MethodId) * len(config.output_keep_ratios) * len(config.input_keep_ratios)
    )
    if len(condition_rows) != expected_conditions:
        gates.append(
            f"condition_rows={len(condition_rows)} expected={expected_conditions}"
        )

    online_scopes = {
        "activation_proxy_build_ms",
        "output_generation_ms",
        "input_recovery_ms",
        "online_total_ms",
    }
    online = [
        r
        for r in latency_rows
        if r["timing_scope"] in online_scopes
        and r["output_keep_ratio"] != ""
        and r["input_keep_ratio"] != ""
    ]
    act_stat = [
        r
        for r in latency_rows
        if r["timing_scope"] == "activation_statistic_ms"
        and r["output_keep_ratio"] != ""
        and r["input_keep_ratio"] != ""
    ]
    offline = [
        r
        for r in latency_rows
        if r["timing_scope"] in ("weight_proxy_offline_ms", "weight_energy_offline_ms")
    ]
    n_cond = len(config.output_keep_ratios) * len(config.input_keep_ratios)
    # 8 methods × n_cond × 4 online scopes
    # activation_statistic for M3/M4/M7/M8 = 4 × n_cond
    # offline: weight_proxy×2 + weight_energy×3 = 5
    expected_online = len(MethodId) * n_cond * 4
    expected_act_stat = 4 * n_cond
    expected_offline = 5
    expected_total = expected_online + expected_act_stat + expected_offline
    if (
        len(online) != expected_online
        or len(act_stat) != expected_act_stat
        or len(offline) != expected_offline
        or len(latency_rows) != expected_total
    ):
        gates.append(
            f"latency_rows online={len(online)} act_stat={len(act_stat)} "
            f"offline={len(offline)} total={len(latency_rows)} "
            f"expected {expected_online}+{expected_act_stat}+{expected_offline}={expected_total}"
        )

    for r in offline:
        if r["output_keep_ratio"] != "" or r["input_keep_ratio"] != "":
            gates.append("offline_ratio_fields_must_be_empty")
    for r in online + act_stat:
        if r["output_keep_ratio"] == "" or r["input_keep_ratio"] == "":
            gates.append("online_ratio_fields_must_be_nonempty")

    for mid in (MethodId.FULL_EXACT_REF, MethodId.FULL_ENERGY_REF_OUTPUT):
        rows = [
            r
            for r in latency_rows
            if r["method_id"] == mid.value
            and r["timing_scope"] == "activation_proxy_build_ms"
        ]
        for r in rows:
            if float(r["median_ms"]) != 0.0:
                gates.append(f"{mid.value}_activation_proxy_not_zero")

    for r in latency_rows:
        if r["timing_scope"] == "activation_proxy_build_ms" and r["method_id"] in {
            MethodId.FULL_EXACT_REF.value,
            MethodId.FULL_ENERGY_REF_OUTPUT.value,
        }:
            continue
        if r["timing_scope"] in online_scopes | {
            "weight_proxy_offline_ms",
            "weight_energy_offline_ms",
            "activation_statistic_ms",
        }:
            if float(r["median_ms"]) <= 0.0 and not (
                r["timing_scope"] == "activation_proxy_build_ms"
                and r["method_id"]
                in {
                    MethodId.FULL_EXACT_REF.value,
                    MethodId.FULL_ENERGY_REF_OUTPUT.value,
                }
            ):
                if r["timing_scope"].endswith("_offline_ms") or r[
                    "timing_scope"
                ] in (online_scopes | {"activation_statistic_ms"}) - {
                    "activation_proxy_build_ms"
                }:
                    if float(r["median_ms"]) <= 0.0:
                        gates.append(
                            f"nonpositive_latency:{r['method_id']}:{r['timing_scope']}"
                        )

    return gates


def _time_methods(
    *,
    prepared: PreparedOperands,
    config: ExperimentConfig,
    x: torch.Tensor,
    w: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_row(method_id: str, out_r, in_r, scope: str, stats, repeats: int):
        rows.append(
            {
                "method_id": method_id,
                "output_keep_ratio": "" if out_r is None else float(out_r),
                "input_keep_ratio": "" if in_r is None else float(in_r),
                "timing_scope": scope,
                "median_ms": stats.median_ms,
                "p10_ms": stats.p10_ms,
                "p90_ms": stats.p90_ms,
                "repeats": repeats,
                "peak_memory_bytes": stats.peak_memory_bytes,
            }
        )

    def build_wp():
        _ = build_hif4_ternary_proxy(w).proxy

    st = benchmark_cuda(build_wp, config.warmup, config.fast_repeats)
    for mid in (
        MethodId.XWPROXY_EXACT_REF_OUTPUT.value,
        MethodId.XWPROXY_EXACT_OWN_OUTPUT.value,
    ):
        add_row(mid, None, None, "weight_proxy_offline_ms", st, config.fast_repeats)

    w_blocks = prepared.w_blocks

    def build_we():
        _ = w_blocks.square().mean(dim=(-1, -2))

    st = benchmark_cuda(build_we, config.warmup, config.fast_repeats)
    for mid in (
        MethodId.XPROXY_ENERGY_OWN_OUTPUT.value,
        MethodId.FULL_ENERGY_REF_OUTPUT.value,
        MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT.value,
    ):
        add_row(mid, None, None, "weight_energy_offline_ms", st, config.fast_repeats)

    kb = prepared.x_blocks.shape[1]
    jb = prepared.w_blocks.shape[0]

    for method_id in MethodId:
        spec = METHOD_SPECS[method_id]
        repeats = (
            config.fast_repeats
            if _is_fast_recovery(spec.recovery_kind)
            else config.exact_repeats
        )
        needs_act_proxy = method_id not in (
            MethodId.FULL_EXACT_REF,
            MethodId.FULL_ENERGY_REF_OUTPUT,
        )
        records_act_stat = method_id in (
            MethodId.XPROXY_ENERGY_OWN_OUTPUT,
            MethodId.FULL_ENERGY_REF_OUTPUT,
            MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT,
            MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT,
        )
        for out_r in config.output_keep_ratios:
            for in_r in config.input_keep_ratios:
                in_keep = ratio_to_keep_count(in_r, kb)
                out_keep = ratio_to_keep_count(out_r, jb)

                if not needs_act_proxy:
                    class _Z:
                        median_ms = 0.0
                        p10_ms = 0.0
                        p90_ms = 0.0
                        peak_memory_bytes = 0

                    add_row(
                        method_id.value,
                        out_r,
                        in_r,
                        "activation_proxy_build_ms",
                        _Z(),
                        repeats,
                    )
                else:

                    def act_proxy():
                        _ = build_hif4_ternary_proxy(x).proxy

                    st = benchmark_cuda(act_proxy, config.warmup, repeats)
                    add_row(
                        method_id.value,
                        out_r,
                        in_r,
                        "activation_proxy_build_ms",
                        st,
                        repeats,
                    )

                def output_gen():
                    if spec.output_source == "ref":
                        y = x @ w.T
                    elif spec.output_source == "xp":
                        y = prepared.xp @ w.T
                    else:
                        y = prepared.xp @ prepared.wp.T
                    scores = output_block_scores(
                        y,
                        config.activation_block_rows,
                        config.output_block_cols,
                    )
                    _ = stable_topk_mask(scores, out_keep)

                st = benchmark_cuda(output_gen, config.warmup, repeats)
                add_row(
                    method_id.value, out_r, in_r, "output_generation_ms", st, repeats
                )

                my = (
                    prepared.my_ref_by_ratio[out_r]
                    if spec.output_source == "ref"
                    else (
                        prepared.my_xp_by_ratio[out_r]
                        if spec.output_source == "xp"
                        else prepared.my_xpwp_by_ratio[out_r]
                    )
                )

                if records_act_stat:
                    if method_id in (
                        MethodId.XPROXY_ENERGY_OWN_OUTPUT,
                        MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT,
                    ):

                        def act_stat():
                            _ = prepared.xp_blocks.square().mean(dim=(-1, -2))

                    elif method_id == MethodId.FULL_ENERGY_REF_OUTPUT:

                        def act_stat():
                            _ = prepared.x_blocks.square().mean(dim=(-1, -2))

                    else:

                        def act_stat():
                            a = prepared.xp_s0.shape[0] // config.activation_block_rows
                            kb_local = prepared.xp_s0.shape[1]
                            _ = prepared.xp_s0.reshape(
                                a, config.activation_block_rows, kb_local
                            ).mean(dim=1)

                    st = benchmark_cuda(act_stat, config.warmup, repeats)
                    add_row(
                        method_id.value,
                        out_r,
                        in_r,
                        "activation_statistic_ms",
                        st,
                        repeats,
                    )

                ref_mx_before = None

                def input_recovery():
                    nonlocal ref_mx_before
                    if spec.recovery_kind == "exact":
                        if spec.contribution_source == "full":
                            x_b, w_b = prepared.x_blocks, prepared.w_blocks
                        elif spec.contribution_source == "xp_fullw":
                            x_b, w_b = prepared.xp_blocks, prepared.w_blocks
                        else:
                            x_b, w_b = prepared.xp_blocks, prepared.wp_blocks
                        out = recover_input_masks_exact(x_b, w_b, my, (in_keep,))
                        mx = out.masks_by_keep[in_keep]
                    elif spec.recovery_kind == "energy":
                        x_b = (
                            prepared.x_blocks
                            if spec.contribution_source == "full"
                            else prepared.xp_blocks
                        )
                        out = recover_input_masks_energy(
                            x_b, prepared.w_energy, my, (in_keep,)
                        )
                        mx = out.masks_by_keep[in_keep]
                    elif spec.recovery_kind == "s0mean_energy":
                        out = recover_input_masks_s0mean_energy(
                            prepared.xp_s0,
                            config.activation_block_rows,
                            prepared.w_energy,
                            my,
                            (in_keep,),
                        )
                        mx = out.masks_by_keep[in_keep]
                    else:
                        out = recover_input_masks_energy_unconditioned(
                            prepared.xp_blocks,
                            prepared.all_output_weight_energy,
                            (in_keep,),
                        )
                        mx = out.masks_by_keep[in_keep]
                    if ref_mx_before is None:
                        ref_mx_before = mx.detach().clone()
                    elif not torch.equal(mx, ref_mx_before):
                        raise RuntimeError("timing path mask not bitwise stable")

                st = benchmark_cuda(input_recovery, config.warmup, repeats)
                add_row(method_id.value, out_r, in_r, "input_recovery_ms", st, repeats)

                def online_total():
                    if needs_act_proxy:
                        xp_result = build_hif4_ternary_proxy(x)
                        xp_local = xp_result.proxy
                        s0_local = xp_result.s0
                    else:
                        xp_local = x
                        s0_local = None
                    if spec.output_source == "ref":
                        y = x @ w.T
                    elif spec.output_source == "xp":
                        y = xp_local @ w.T
                    else:
                        y = xp_local @ prepared.wp.T
                    scores = output_block_scores(
                        y,
                        config.activation_block_rows,
                        config.output_block_cols,
                    )
                    my_local = stable_topk_mask(scores, out_keep)
                    if spec.recovery_kind == "exact":
                        if spec.contribution_source == "full":
                            xb = split_activation_blocks(
                                x, config.activation_block_rows, config.k_block_size
                            )
                            wb = prepared.w_blocks
                        elif spec.contribution_source == "xp_fullw":
                            xb = split_activation_blocks(
                                xp_local,
                                config.activation_block_rows,
                                config.k_block_size,
                            )
                            wb = prepared.w_blocks
                        else:
                            xb = split_activation_blocks(
                                xp_local,
                                config.activation_block_rows,
                                config.k_block_size,
                            )
                            wb = prepared.wp_blocks
                        _ = recover_input_masks_exact(xb, wb, my_local, (in_keep,))
                    elif spec.recovery_kind == "energy":
                        if spec.contribution_source == "full":
                            xb = split_activation_blocks(
                                x, config.activation_block_rows, config.k_block_size
                            )
                        else:
                            xb = split_activation_blocks(
                                xp_local,
                                config.activation_block_rows,
                                config.k_block_size,
                            )
                        _ = recover_input_masks_energy(
                            xb, prepared.w_energy, my_local, (in_keep,)
                        )
                    elif spec.recovery_kind == "s0mean_energy":
                        assert s0_local is not None
                        _ = recover_input_masks_s0mean_energy(
                            s0_local,
                            config.activation_block_rows,
                            prepared.w_energy,
                            my_local,
                            (in_keep,),
                        )
                    else:
                        xb = split_activation_blocks(
                            xp_local,
                            config.activation_block_rows,
                            config.k_block_size,
                        )
                        _ = recover_input_masks_energy_unconditioned(
                            xb,
                            prepared.all_output_weight_energy,
                            (in_keep,),
                        )

                st = benchmark_cuda(online_total, config.warmup, repeats)
                add_row(method_id.value, out_r, in_r, "online_total_ms", st, repeats)

    return rows


def run(
    config: ExperimentConfig,
    output_dir: Path,
    capture_cache: Path,
    devices: list[int] | None = None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _configure_deterministic()

    if devices is None or len(devices) == 0:
        devices = [0]
    for d in devices:
        if d < 0 or d >= torch.cuda.device_count():
            raise ValueError(
                f"device {d} invalid; visible cuda device count={torch.cuda.device_count()}"
            )
    primary = f"cuda:{devices[0]}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    log = logging.getLogger("input_mask_proxy_study")
    log.info("using devices=%s (primary=%s)", devices, primary)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    expected_meta = {
        "model_path": config.model_path,
        "dataset_hf_id": config.dataset_hf_id,
        "seed": config.seed,
        "num_samples": config.num_samples,
        "max_seq_len": config.max_seq_len,
        "max_activation_blocks": config.max_activation_blocks,
        "layer_index": config.layer_index,
        "projection": config.projection,
        "activation_block_rows": config.activation_block_rows,
    }
    captured = capture_or_load(
        model_path=config.model_path,
        dataset_hf_id=config.dataset_hf_id,
        seed=config.seed,
        num_samples=config.num_samples,
        max_seq_len=config.max_seq_len,
        max_activation_blocks=config.max_activation_blocks,
        layer_index=config.layer_index,
        projection=config.projection,
        activation_block_rows=config.activation_block_rows,
        cache_path=capture_cache,
        expected_meta=expected_meta,
    )

    x = captured.activations_bf16.to(device=primary, dtype=torch.float32)
    w = captured.weight_bf16.to(device=primary, dtype=torch.float32)
    if x.shape[0] % config.activation_block_rows != 0:
        raise RuntimeError("activation rows not divisible")
    if x.shape[1] % config.k_block_size != 0:
        raise RuntimeError("K not divisible by 64")
    if w.shape[0] % config.output_block_cols != 0:
        raise RuntimeError("N not divisible by 32")

    log.info("building proxies / outputs / masks...")
    prepared = prepare_operands(x, w, config)
    if not torch.isfinite(prepared.xp).all() or not torch.isfinite(prepared.wp).all():
        raise RuntimeError("proxy contains NaN/Inf")

    log.info("running eight methods...")
    results = {}
    for mid in MethodId:
        log.info("method %s", mid.value)
        results[mid] = run_method(mid, prepared, config)
    log.info("building conditional oracles...")
    oracles = build_conditional_oracles(prepared, config)
    m1 = results[MethodId.FULL_EXACT_REF]
    log.info("computing metrics...")

    per_block: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []

    y_real = prepared.y_ref
    x_blocks = prepared.x_blocks
    w_blocks = prepared.w_blocks
    my_ref = prepared.my_ref_by_ratio
    xp_s0_shape = tuple(int(v) for v in prepared.xp_s0.shape)
    w_energy_shape = tuple(int(v) for v in prepared.w_energy.shape)
    all_output_weight_energy_shape = tuple(
        int(v) for v in prepared.all_output_weight_energy.shape
    )
    xp_blocks_for_diag = prepared.xp_blocks.detach()
    xp_s0_for_diag = prepared.xp_s0.detach()

    for mid, res in results.items():
        for out_r in config.output_keep_ratios:
            for in_r in config.input_keep_ratios:
                my = res.output_masks_by_ratio[out_r]
                mx = res.input_masks_by_ratio[(out_r, in_r)]
                m1_mx = m1.input_masks_by_ratio[(out_r, in_r)]
                cond = _conditional_ref_for_method(mid, out_r, in_r, m1_mx, oracles)

                out_overlaps = []
                in_overlaps = []
                cond_overlaps = []
                nrmses = []
                regrets = []
                joint_nrmses = []
                spears = []
                kendalls = []

                y_m1 = reconstruct_real_output(x_blocks, w_blocks, m1_mx)
                nrmse_m1_rows = []
                for i in range(x_blocks.shape[0]):
                    # per-block metrics
                    om = mask_metrics(my[i : i + 1], my_ref[out_r][i : i + 1])
                    im = mask_metrics(mx[i : i + 1], m1_mx[i : i + 1])
                    cm = mask_metrics(mx[i : i + 1], cond[i : i + 1])
                    y_hat = reconstruct_real_output(
                        x_blocks[i : i + 1], w_blocks, mx[i : i + 1]
                    )
                    y_true = y_real[
                        i
                        * config.activation_block_rows : (i + 1)
                        * config.activation_block_rows
                    ]
                    y_m1_i = y_m1[
                        i
                        * config.activation_block_rows : (i + 1)
                        * config.activation_block_rows
                    ]
                    nr = nrmse(y_hat, y_true, my_ref[out_r][i : i + 1])
                    nr_m1 = nrmse(y_m1_i, y_true, my_ref[out_r][i : i + 1])
                    y_joint = reconstruct_joint_sparse_output(
                        x_blocks[i : i + 1],
                        w_blocks,
                        my[i : i + 1],
                        mx[i : i + 1],
                    )
                    jn = nrmse(y_joint, y_true, None)
                    # ranking vs M1
                    if (
                        res.ranking_by_output_ratio.get(out_r) is not None
                        and m1.ranking_by_output_ratio.get(out_r) is not None
                    ):
                        sp = spearman_rank(
                            res.ranking_by_output_ratio[out_r][i],
                            m1.ranking_by_output_ratio[out_r][i].float(),
                        )
                        kd = kendall_tau_b(
                            res.ranking_by_output_ratio[out_r][i].float(),
                            m1.ranking_by_output_ratio[out_r][i].float(),
                        )
                    else:
                        sp = float("nan")
                        kd = float("nan")

                    out_overlaps.append(om["overlap"])
                    in_overlaps.append(im["overlap"])
                    cond_overlaps.append(cm["overlap"])
                    nrmses.append(nr)
                    regrets.append(nr - nr_m1)
                    joint_nrmses.append(jn)
                    spears.append(sp)
                    kendalls.append(kd)
                    nrmse_m1_rows.append(nr_m1)

                    per_block.append(
                        {
                            "method_id": mid.value,
                            "output_keep_ratio": out_r,
                            "input_keep_ratio": in_r,
                            "activation_block_index": i,
                            "output_overlap_to_ref": om["overlap"],
                            "input_overlap_to_m1": im["overlap"],
                            "input_overlap_to_conditional_oracle": cm["overlap"],
                            "real_output_nrmse_on_ref_mask": nr,
                            "nrmse_regret_vs_m1": nr - nr_m1,
                            "joint_sparse_output_nrmse": jn,
                            "spearman": sp,
                            "kendall": kd,
                        }
                    )

                comp = res.compute_masks_by_ratio[(out_r, in_r)]
                a, jb, kb = comp.shape
                num_compute = int(comp.sum().item())
                dense = a * jb * kb
                so = _stats(out_overlaps)
                si = _stats(in_overlaps)
                sc = _stats(cond_overlaps)
                sn = _stats(nrmses)
                sr = _stats(regrets)
                sj = _stats(joint_nrmses)
                condition_rows.append(
                    {
                        "method_id": mid.value,
                        "output_keep_ratio": out_r,
                        "input_keep_ratio": in_r,
                        "output_overlap_to_ref_mean": so["mean"],
                        "output_overlap_to_ref_median": so["median"],
                        "output_overlap_to_ref_p10": so["p10"],
                        "output_overlap_to_ref_p90": so["p90"],
                        "input_overlap_to_m1_mean": si["mean"],
                        "input_overlap_to_m1_median": si["median"],
                        "input_overlap_to_m1_p10": si["p10"],
                        "input_overlap_to_m1_p90": si["p90"],
                        "input_overlap_to_conditional_oracle_mean": sc["mean"],
                        "input_overlap_to_conditional_oracle_median": sc["median"],
                        "real_output_nrmse_mean": sn["mean"],
                        "real_output_nrmse_median": sn["median"],
                        "nrmse_regret_vs_m1_mean": sr["mean"],
                        "nrmse_regret_vs_m1_median": sr["median"],
                        "joint_sparse_output_nrmse_mean": sj["mean"],
                        "joint_sparse_output_nrmse_median": sj["median"],
                        "num_output_blocks_kept": int(my.sum().item()),
                        "num_input_blocks_kept": int(mx.sum().item()),
                        "num_compute_blocks": num_compute,
                        "compute_block_ratio": float(num_compute / dense),
                        "spearman_mean": _stats(spears)["mean"],
                        "kendall_mean": _stats(kendalls)["mean"],
                    }
                )

    my_ref_cpu = {k: v.detach().cpu() for k, v in prepared.my_ref_by_ratio.items()}
    my_xp_cpu = {k: v.detach().cpu() for k, v in prepared.my_xp_by_ratio.items()}
    my_xpwp_cpu = {k: v.detach().cpu() for k, v in prepared.my_xpwp_by_ratio.items()}
    kb_dim = int(prepared.x_blocks.shape[1])
    jb_dim = int(prepared.w_blocks.shape[0])

    log.info("timing... (devices=%s)", devices)
    if len(devices) == 1:
        latency_rows = _time_methods(prepared=prepared, config=config, x=x, w=w)
    else:
        # Keep method results for gates/artifacts; free large primary-GPU tensors.
        x_cpu = x.detach().cpu().contiguous()
        w_cpu = w.detach().cpu().contiguous()
        xp_blocks_for_diag = xp_blocks_for_diag.detach().cpu().contiguous()
        xp_s0_for_diag = xp_s0_for_diag.detach().cpu().contiguous()
        del prepared, oracles, m1, x, w, y_real, x_blocks, w_blocks, my_ref
        torch.cuda.empty_cache()
        latency_rows = time_methods_multi_gpu(
            config=config,
            x_cpu=x_cpu,
            w_cpu=w_cpu,
            devices=devices,
        )

    m7_vs_m3 = compute_m7_vs_m3(
        xp_blocks=xp_blocks_for_diag,
        xp_s0=xp_s0_for_diag,
        results=results,
        condition_rows=condition_rows,
        latency_rows=latency_rows,
        config=config,
    )
    m8_vs_m3 = compute_m8_vs_m3(
        results=results,
        condition_rows=condition_rows,
        latency_rows=latency_rows,
        config=config,
    )

    gates = _run_correctness_gates(
        results=results,
        my_ref_by_ratio=my_ref_cpu,
        my_xp_by_ratio=my_xp_cpu,
        my_xpwp_by_ratio=my_xpwp_cpu,
        kb=kb_dim,
        jb=jb_dim,
        config=config,
        condition_rows=condition_rows,
        latency_rows=latency_rows,
        xp_s0_shape=xp_s0_shape,
        w_energy_shape=w_energy_shape,
        all_output_weight_energy_shape=all_output_weight_energy_shape,
    )
    for label, payload in (("m7_vs_m3", m7_vs_m3), ("m8_vs_m3", m8_vs_m3)):
        for k, v in payload.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                gates.append(f"{label}_nan_inf:{k}")
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, float) and (math.isnan(sv) or math.isinf(sv)):
                        gates.append(f"{label}_nan_inf:{k}.{sk}")
                    elif isinstance(sv, dict):
                        for ssk, ssv in sv.items():
                            if isinstance(ssv, float) and (
                                math.isnan(ssv) or math.isinf(ssv)
                            ):
                                gates.append(f"{label}_nan_inf:{k}.{sk}.{ssk}")
    # NaN/Inf check on key metrics
    for row in condition_rows:
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                gates.append(f"nan_inf:{row['method_id']}:{k}")

    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "run_id": output_dir.name,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "model_path": config.model_path,
        "methods": [m.value for m in MethodId],
        "correctness_gates": gates,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        **capture_manifest_fields(captured),
        "block": {
            "activation_block_rows": config.activation_block_rows,
            "k_block_size": config.k_block_size,
            "output_block_cols": config.output_block_cols,
        },
    }

    masks_payload = {
        mid.value: {
            "output_masks_by_ratio": res.output_masks_by_ratio,
            "input_masks_by_ratio": res.input_masks_by_ratio,
            "compute_masks_by_ratio": res.compute_masks_by_ratio,
            "removal_order_by_output_ratio": res.removal_order_by_output_ratio,
            "ranking_by_output_ratio": res.ranking_by_output_ratio,
        }
        for mid, res in results.items()
    }

    if gates:
        write_run_artifacts(
            output_dir,
            manifest=manifest,
            config=dataclasses.asdict(config),
            masks=masks_payload,
            per_block_metrics=per_block,
            condition_summary=condition_rows,
            latency_rows=latency_rows,
            aggregate_summary={"m7_vs_m3": m7_vs_m3, "m8_vs_m3": m8_vs_m3},
            report_md=None,
        )
        raise RuntimeError(f"correctness gates failed: {gates}")

    aggregate = build_aggregate_summary(
        condition_rows, latency_rows, m7_vs_m3=m7_vs_m3, m8_vs_m3=m8_vs_m3
    )
    report_md = render_report(
        aggregate=aggregate, condition_summary=condition_rows, manifest=manifest
    )
    write_run_artifacts(
        output_dir,
        manifest=manifest,
        config=dataclasses.asdict(config),
        masks=masks_payload,
        per_block_metrics=per_block,
        condition_summary=condition_rows,
        latency_rows=latency_rows,
        aggregate_summary=aggregate,
        report_md=report_md,
    )
    log.info("done: %s", output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="HiF4 proxy input-mask recovery ablation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--capture-cache", type=str, required=True)
    parser.add_argument(
        "--devices",
        type=str,
        default="0",
        help="Comma-separated GPU ids for parallel timing, e.g. 0,1,6,7",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    devices = [int(x.strip()) for x in args.devices.split(",") if x.strip() != ""]
    run(cfg, Path(args.output_dir), Path(args.capture_cache), devices=devices)


if __name__ == "__main__":
    main()
