"""Multi-GPU timing: one process per method, bound to a physical GPU."""

from __future__ import annotations

import os
from itertools import cycle
from typing import Any

import torch
import torch.multiprocessing as mp

from Block_Sparse.input_mask_proxy_study.benchmark import LatencyStats, benchmark_cuda
from Block_Sparse.input_mask_proxy_study.block_layout import (
    output_block_scores,
    split_activation_blocks,
    stable_topk_mask,
)
from Block_Sparse.input_mask_proxy_study.config import (
    ExperimentConfig,
    MethodId,
    ratio_to_keep_count,
)
from Block_Sparse.input_mask_proxy_study.energy_recovery import (
    recover_input_masks_energy,
    recover_input_masks_energy_unconditioned,
)
from Block_Sparse.input_mask_proxy_study.exact_recovery import recover_input_masks_exact
from Block_Sparse.input_mask_proxy_study.hif4_proxy import build_hif4_ternary_proxy
from Block_Sparse.input_mask_proxy_study.methods import METHOD_SPECS, prepare_operands
from Block_Sparse.input_mask_proxy_study.s0mean_recovery import recover_input_masks_s0mean_energy


def _configure_deterministic() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def _is_fast_recovery(kind: str) -> bool:
    return kind in ("energy", "s0mean_energy", "energy_unconditioned")


def _add_row(
    rows: list[dict[str, Any]],
    method_id: str,
    out_r,
    in_r,
    scope: str,
    stats: LatencyStats | Any,
    repeats: int,
) -> None:
    rows.append(
        {
            "method_id": method_id,
            "output_keep_ratio": "" if out_r is None else float(out_r),
            "input_keep_ratio": "" if in_r is None else float(in_r),
            "timing_scope": scope,
            "median_ms": float(stats.median_ms),
            "p10_ms": float(stats.p10_ms),
            "p90_ms": float(stats.p90_ms),
            "repeats": repeats,
            "peak_memory_bytes": int(stats.peak_memory_bytes),
        }
    )


def _time_offline_on_device(
    x: torch.Tensor,
    w: torch.Tensor,
    config: ExperimentConfig,
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x = x.to(device=device, dtype=torch.float32)
    w = w.to(device=device, dtype=torch.float32)
    prepared = prepare_operands(x, w, config)

    def build_wp():
        _ = build_hif4_ternary_proxy(w).proxy

    st = benchmark_cuda(build_wp, config.warmup, config.fast_repeats)
    for mid in (
        MethodId.XWPROXY_EXACT_REF_OUTPUT.value,
        MethodId.XWPROXY_EXACT_OWN_OUTPUT.value,
    ):
        _add_row(rows, mid, None, None, "weight_proxy_offline_ms", st, config.fast_repeats)

    w_blocks = prepared.w_blocks

    def build_we():
        _ = w_blocks.square().mean(dim=(-1, -2))

    st = benchmark_cuda(build_we, config.warmup, config.fast_repeats)
    for mid in (
        MethodId.XPROXY_ENERGY_OWN_OUTPUT.value,
        MethodId.FULL_ENERGY_REF_OUTPUT.value,
        MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT.value,
    ):
        _add_row(rows, mid, None, None, "weight_energy_offline_ms", st, config.fast_repeats)
    return rows


def _time_one_method_on_device(
    method_id: MethodId,
    x: torch.Tensor,
    w: torch.Tensor,
    config: ExperimentConfig,
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x = x.to(device=device, dtype=torch.float32)
    w = w.to(device=device, dtype=torch.float32)
    prepared = prepare_operands(x, w, config)
    spec = METHOD_SPECS[method_id]
    repeats = (
        config.fast_repeats if _is_fast_recovery(spec.recovery_kind) else config.exact_repeats
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
    kb = prepared.x_blocks.shape[1]
    jb = prepared.w_blocks.shape[0]

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

                _add_row(
                    rows,
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
                _add_row(
                    rows,
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
                    y, config.activation_block_rows, config.output_block_cols
                )
                _ = stable_topk_mask(scores, out_keep)

            st = benchmark_cuda(output_gen, config.warmup, repeats)
            _add_row(
                rows, method_id.value, out_r, in_r, "output_generation_ms", st, repeats
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
                _add_row(
                    rows,
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
                    if spec.contribution_source == "full":
                        x_b = prepared.x_blocks
                    else:
                        x_b = prepared.xp_blocks
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
            _add_row(
                rows, method_id.value, out_r, in_r, "input_recovery_ms", st, repeats
            )

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
                    y, config.activation_block_rows, config.output_block_cols
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
                            xp_local, config.activation_block_rows, config.k_block_size
                        )
                        wb = prepared.w_blocks
                    else:
                        xb = split_activation_blocks(
                            xp_local, config.activation_block_rows, config.k_block_size
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
                            xp_local, config.activation_block_rows, config.k_block_size
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
                        xp_local, config.activation_block_rows, config.k_block_size
                    )
                    _ = recover_input_masks_energy_unconditioned(
                        xb,
                        prepared.all_output_weight_energy,
                        (in_keep,),
                    )

            st = benchmark_cuda(online_total, config.warmup, repeats)
            _add_row(rows, method_id.value, out_r, in_r, "online_total_ms", st, repeats)

    return rows


def _worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Isolate this process to one physical GPU before any CUDA context is created.
    device_index = int(payload["device_index"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
    torch.cuda.set_device(0)
    device = "cuda:0"
    _configure_deterministic()
    torch.manual_seed(int(payload["seed"]))
    torch.cuda.manual_seed_all(int(payload["seed"]))

    config: ExperimentConfig = payload["config"]
    x: torch.Tensor = payload["x_cpu"]
    w: torch.Tensor = payload["w_cpu"]
    kind = payload["kind"]
    if kind == "offline":
        return _time_offline_on_device(x, w, config, device)
    method_id = MethodId(payload["method_id"])
    return _time_one_method_on_device(method_id, x, w, config, device)


def time_methods_multi_gpu(
    *,
    config: ExperimentConfig,
    x_cpu: torch.Tensor,
    w_cpu: torch.Tensor,
    devices: list[int],
) -> list[dict[str, Any]]:
    if not devices:
        raise ValueError("devices must be non-empty")
    for d in devices:
        if d < 0 or d >= torch.cuda.device_count():
            raise ValueError(f"invalid device {d}, visible count={torch.cuda.device_count()}")

    ctx = mp.get_context("spawn")
    x_cpu = x_cpu.detach().cpu().contiguous()
    w_cpu = w_cpu.detach().cpu().contiguous()

    # Offline once on first device.
    offline_payload = {
        "kind": "offline",
        "device_index": devices[0],
        "config": config,
        "x_cpu": x_cpu,
        "w_cpu": w_cpu,
        "seed": config.seed,
    }
    with ctx.Pool(processes=1) as pool:
        offline_rows = pool.apply(_worker, (offline_payload,))

    method_list = list(MethodId)
    assignments = list(zip(method_list, cycle(devices)))
    payloads = [
        {
            "kind": "method",
            "method_id": mid.value,
            "device_index": int(dev),
            "config": config,
            "x_cpu": x_cpu,
            "w_cpu": w_cpu,
            "seed": config.seed,
        }
        for mid, dev in assignments
    ]
    # One process per method; up to len(devices) run concurrently.
    nproc = min(len(payloads), len(devices))
    with ctx.Pool(processes=nproc) as pool:
        method_rows_list = pool.map(_worker, payloads)

    rows = list(offline_rows)
    for part in method_rows_list:
        rows.extend(part)
    return rows
