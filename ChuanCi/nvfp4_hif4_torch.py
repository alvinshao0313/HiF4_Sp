#!/usr/bin/env python3
"""纯 PyTorch NVFP4→HiF4 原生转换与真实权重评测脚本（绿地实现）。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import torch

# =============================================================================
# 数据结构：HiF4 / 实验 / 量化结果 / 误差累计
# =============================================================================


@dataclass(frozen=True)
class HiF4Config:
    """HiF4 量化配置。"""

    group_size: int = 64
    group_dim: int = -1
    scale_mode: str = "hardware"
    compute_dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class ExperimentConfig:
    """合成实验全局配置。"""

    seed: int = 20260723
    samples_per_repeat: int = 320_000
    repeats: int = 10
    phase_points: int = 257
    phase_seed: int = 7


@dataclass
class NVFP4SimulationResult:
    """NVFP4 合成伪量化结果（仅用于受控实验）。"""

    values: torch.Tensor
    global_scale: torch.Tensor
    block_scales: torch.Tensor
    payload: torch.Tensor


@dataclass
class HiF4Result:
    """HiF4 量化输出及元数据。"""

    values: torch.Tensor
    top_scale: torch.Tensor
    e1_per_8: torch.Tensor
    e1_per_4: torch.Tensor
    payload_magnitude: torch.Tensor
    local_scale: torch.Tensor


@dataclass
class ErrorSums:
    """FP64 累计的误差原始量，用于全局 merge。"""

    numel: int = 0
    reference_energy: float = 0.0
    approximation_energy: float = 0.0
    error_energy: float = 0.0
    dot: float = 0.0
    absolute_error_sum: float = 0.0
    max_absolute_error: float = 0.0


# =============================================================================
# 常量与模块级码本缓存
# =============================================================================

SCHEMA_VERSION = 1
IMPLEMENTATION = "greenfield_torch"

VALID_SCALE_MODES = frozenset({"continuous", "bf16_math", "e6m2_only", "hardware"})

SKIP_REASONS = (
    "not_requested",
    "include_regex_miss",
    "exclude_regex_match",
    "non_floating",
    "ndim_lt_2",
    "invalid_group_dim",
    "not_group_divisible",
    "max_tensors_reached",
)

TENSOR_CATEGORIES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "embed_tokens",
    "lm_head",
    "other",
)

DISTRIBUTION_NAMES = (
    "gaussian",
    "laplace",
    "student_t3",
    "outlier_0p1pct_20x",
)

DEFAULT_EXPERIMENT_SAMPLES = 320_000
DEFAULT_EXPERIMENT_REPEATS = 10
DEFAULT_EXPERIMENT_PHASE_POINTS = 257
QUICK_SAMPLES = 6_400
QUICK_REPEATS = 1
QUICK_PHASE_POINTS = 17

T_CI_975_9 = 2.262


# =============================================================================
# E4M3FN / E6M2 / E2M1 码本构造
# =============================================================================


def build_e4m3fn_codebook() -> tuple[torch.Tensor, torch.Tensor]:
    """构造非负 E4M3FN 码本（codes 0..126，共 127 个值，max=448）。"""
    values: list[float] = []
    codes: list[int] = []
    for code in range(127):
        exponent = (code >> 3) & 0x0F
        mantissa = code & 0x07
        if exponent == 0:
            value = mantissa * (2.0**-9)
        else:
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        values.append(value)
        codes.append(code)
    value_tensor = torch.tensor(values, dtype=torch.float32)
    code_tensor = torch.tensor(codes, dtype=torch.int16)
    return value_tensor, code_tensor


def build_e6m2_codebook() -> tuple[torch.Tensor, torch.Tensor]:
    """构造无符号 E6M2 scale 码本（codes 0..254，共 255 个值）。"""
    values: list[float] = []
    codes: list[int] = []
    for code in range(255):
        exponent = (code >> 2) & 0x3F
        mantissa = code & 0x03
        value = (1.0 + mantissa / 4.0) * (2.0 ** (exponent - 48))
        values.append(value)
        codes.append(code)
    value_tensor = torch.tensor(values, dtype=torch.float32)
    code_tensor = torch.tensor(codes, dtype=torch.int16)
    return value_tensor, code_tensor


E4M3FN_VALUES, E4M3FN_CODES = build_e4m3fn_codebook()
E6M2_VALUES, E6M2_CODES = build_e6m2_codebook()
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
E2M1_CODES = torch.arange(8, dtype=torch.int16)


# =============================================================================
# 通用 RNE 舍入、BF16 carrier、E2M1 magnitude
# =============================================================================


def round_positive_to_codebook(
    values: torch.Tensor,
    codebook_values: torch.Tensor,
    codebook_codes: torch.Tensor,
) -> torch.Tensor:
    """非负输入 RNE 到码本；负数 clamp 为 0；等距时选偶数 code。"""
    if not values.is_floating_point():
        raise TypeError("values must be floating point")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    x = values.to(torch.float32).clamp_min(0.0)
    book = codebook_values.to(device=x.device, dtype=torch.float32)
    codes = codebook_codes.to(device=x.device)
    hi = torch.searchsorted(book, x)
    hi = hi.clamp(0, book.numel() - 1)
    lo = (hi - 1).clamp(0, book.numel() - 1)
    d_lo = x - book[lo]
    d_hi = book[hi] - x
    choose_hi = d_hi < d_lo
    tie = d_hi == d_lo
    choose_hi = choose_hi | (tie & ((codes[hi] & 1) == 0))
    return torch.where(choose_hi, book[hi], book[lo])


def round_bfloat16(values: torch.Tensor) -> torch.Tensor:
    """显式 BF16 carrier：FP32 → BF16 → FP32。"""
    if not values.is_floating_point():
        raise TypeError("values must be floating point")
    return values.to(torch.bfloat16).to(torch.float32)


def quantize_e2m1_magnitude(x: torch.Tensor) -> torch.Tensor:
    """E2M1 正幅值量化。"""
    return round_positive_to_codebook(x, E2M1_VALUES, E2M1_CODES)


def round_e4m3fn(x: torch.Tensor) -> torch.Tensor:
    """E4M3FN 正 scale 舍入。"""
    return round_positive_to_codebook(x, E4M3FN_VALUES, E4M3FN_CODES)


def round_e6m2(x: torch.Tensor) -> torch.Tensor:
    """E6M2 scale 舍入。"""
    return round_positive_to_codebook(x, E6M2_VALUES, E6M2_CODES)


# =============================================================================
# HiF4 配置校验与分组工具
# =============================================================================


def _validate_hif4_inputs(values: torch.Tensor, config: HiF4Config) -> None:
    """校验 HiF4 输入 tensor 与配置。"""
    if not values.is_floating_point():
        raise TypeError("values must be floating point")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    if config.group_size < 8 or config.group_size % 8 != 0:
        raise ValueError("group_size must be >= 8 and divisible by 8")
    if config.compute_dtype != torch.float32:
        raise ValueError("compute_dtype must be torch.float32")
    if config.scale_mode not in VALID_SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {sorted(VALID_SCALE_MODES)}")
    # 先按原始 ndim 判定越界，再做负索引归一化，避免 group_dim=3 对 2D 被 % 吃掉。
    if config.group_dim < -values.ndim or config.group_dim >= values.ndim:
        raise ValueError("group_dim out of range")
    normalized_dim = config.group_dim % values.ndim
    if values.shape[normalized_dim] % config.group_size != 0:
        raise ValueError("grouped dimension length must be divisible by group_size")


def _move_groups_to_last(values: torch.Tensor, group_dim: int) -> tuple[torch.Tensor, int, tuple[int, ...]]:
    """将分组维移到最后一维，返回 (moved, normalized_dim, moved_shape)。"""
    normalized_dim = group_dim % values.ndim
    moved = values.movedim(normalized_dim, -1).contiguous()
    return moved, normalized_dim, tuple(moved.shape)


def _restore_from_last(moved: torch.Tensor, normalized_dim: int, original_ndim: int) -> torch.Tensor:
    """从最后一维分组布局恢复原始维度顺序。"""
    return moved.movedim(-1, normalized_dim)


def _compute_top_scale(amax64: torch.Tensor, scale_mode: str) -> torch.Tensor:
    """按 scale_mode 计算每组顶层 S0。"""
    if scale_mode == "continuous":
        s0 = amax64 / 7.0
        return s0
    if scale_mode == "bf16_math":
        return round_bfloat16(amax64 * round_bfloat16(torch.tensor(1.0 / 7.0, device=amax64.device)))
    if scale_mode == "e6m2_only":
        return round_e6m2(amax64 / 7.0)
    if scale_mode == "hardware":
        bf16_ratio = round_bfloat16(amax64 * round_bfloat16(torch.tensor(1.0 / 7.0, device=amax64.device)))
        return round_e6m2(bf16_ratio)
    raise ValueError(f"unsupported scale_mode: {scale_mode}")


def _compute_reciprocal_scale(s0: torch.Tensor, scale_mode: str) -> torch.Tensor:
    """计算 1/S0，hardware/bf16_math 模式在 BF16 中计算。"""
    if scale_mode in {"bf16_math", "hardware"}:
        return round_bfloat16(1.0 / s0)
    return 1.0 / s0


def quantize_hif4(
    values: torch.Tensor,
    *,
    config: HiF4Config = HiF4Config(),
) -> HiF4Result:
    """HiF4 分层量化：64 元素标准 group，不跨行分组。"""
    _validate_hif4_inputs(values, config)
    x = values.to(torch.float32)
    moved, normalized_dim, moved_shape = _move_groups_to_last(x, config.group_dim)
    group_size = config.group_size
    groups_per_row = moved_shape[-1] // group_size
    num_groups = moved.numel() // group_size
    groups = moved.reshape(-1, group_size)

    abs_g = groups.abs()
    amax64 = abs_g.amax(dim=-1)
    nonzero = amax64 > 0

    # 顶层 S0：按 scale_mode 走 continuous / BF16 / E6M2 / hardware。
    s0 = _compute_top_scale(amax64, config.scale_mode)
    safe_s0 = torch.where(nonzero, s0, torch.ones_like(s0))
    # hardware/bf16_math 用 BF16 reciprocal，对齐现有 quant_hifx 路径。
    reciprocal = _compute_reciprocal_scale(safe_s0, config.scale_mode)

    blocks_per_group = group_size // 8
    abs_8 = abs_g.reshape(num_groups, blocks_per_group, 8)
    amax8 = abs_8.amax(dim=-1)
    abs_4 = abs_g.reshape(num_groups, group_size // 4, 4)
    amax4 = abs_4.amax(dim=-1)

    e8 = (amax8 * reciprocal.unsqueeze(-1) >= 4.0).to(torch.float32)
    e8_per4 = e8.repeat_interleave(2, dim=-1)
    e4 = (amax4 * reciprocal.unsqueeze(-1) / (2.0**e8_per4) >= 2.0).to(torch.float32)

    e8_elem = e8.repeat_interleave(8, dim=-1)
    e4_elem = e4.repeat_interleave(4, dim=-1)
    local_scale = safe_s0.unsqueeze(-1) * (2.0 ** (e8_elem + e4_elem))

    # S1P2 payload：round(4*|x|/S_i)/4，并饱和到 1.75。
    ratio = torch.floor(4.0 * abs_g * (reciprocal.unsqueeze(-1) / (2.0 ** (e8_elem + e4_elem))) + 0.5) / 4.0
    payload = torch.minimum(ratio, torch.full_like(ratio, 1.75))
    recon = groups.sign() * local_scale * payload
    recon = torch.where(nonzero.unsqueeze(-1), recon, torch.zeros_like(recon))

    recon_moved = recon.reshape(moved_shape)
    values_out = _restore_from_last(recon_moved, normalized_dim, values.ndim)

    leading = moved_shape[:-1]
    meta_prefix = leading + (groups_per_row,)
    top_scale = safe_s0.reshape(meta_prefix)
    e1_per_8 = e8.reshape(meta_prefix + (blocks_per_group,))
    e1_per_4 = e4.reshape(meta_prefix + (group_size // 4,))

    payload_moved = payload.reshape(moved_shape)
    local_moved = local_scale.reshape(moved_shape)

    return HiF4Result(
        values=values_out,
        top_scale=top_scale,
        e1_per_8=e1_per_8,
        e1_per_4=e1_per_4,
        payload_magnitude=_restore_from_last(payload_moved, normalized_dim, values.ndim),
        local_scale=_restore_from_last(local_moved, normalized_dim, values.ndim),
    )


# =============================================================================
# NVFP4 合成伪量化（仅用于受控实验，禁止在 evaluate 路径调用）
# =============================================================================


def simulate_nvfp4(
    values: torch.Tensor,
    *,
    block_dim: int = -1,
) -> NVFP4SimulationResult:
    """合成 NVFP4 fake-quant 参考值。

    Only generates synthetic NVFP4 references for controlled experiments.
    Never call this function inside evaluate_nvfp4_fake_weight or
    evaluate_checkpoint(input_kind="nvfp4_fake").
    """
    if not values.is_floating_point():
        raise TypeError("values must be floating point")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")

    normalized_dim = block_dim % values.ndim
    moved = values.movedim(normalized_dim, -1).contiguous()
    moved_shape = moved.shape
    block_size = 16
    if moved_shape[-1] % block_size != 0:
        raise ValueError("block dimension length must be divisible by 16")

    x = moved.to(torch.float32)
    tensor_amax = x.abs().amax()
    if tensor_amax == 0:
        # 全零约定：s_T=1，block scale / payload / values 全零。
        # block_scales 保持 moved 布局（与 HiF4 metadata 一致），不回迁到原始 block_dim。
        global_scale = torch.tensor(1.0, dtype=torch.float32, device=values.device)
        block_scales = torch.zeros(
            moved_shape[:-1] + (moved_shape[-1] // block_size,),
            dtype=torch.float32,
            device=values.device,
        )
        payload = torch.zeros_like(x)
        result_values = torch.zeros_like(x)
        restored_values = result_values.movedim(-1, normalized_dim)
        restored_payload = payload.movedim(-1, normalized_dim)
        return NVFP4SimulationResult(
            values=restored_values,
            global_scale=global_scale,
            block_scales=block_scales,
            payload=restored_payload,
        )

    global_scale = (tensor_amax / (448.0 * 6.0)).to(torch.float32)
    if global_scale == 0:
        global_scale = torch.nextafter(
            torch.tensor(0.0, dtype=torch.float32, device=values.device),
            torch.tensor(1.0, dtype=torch.float32, device=values.device),
        )

    num_blocks = moved_shape[-1] // block_size
    blocked = x.reshape(*moved_shape[:-1], num_blocks, block_size)
    block_amax = blocked.abs().amax(dim=-1)
    raw_block_scale = block_amax / (6.0 * global_scale)
    block_scale = round_e4m3fn(raw_block_scale)
    effective = global_scale * block_scale
    effective_expanded = effective.unsqueeze(-1).expand_as(blocked)
    magnitude = quantize_e2m1_magnitude((blocked.abs() / effective_expanded).reshape(-1)).reshape_as(blocked)
    payload_blocked = blocked.sign() * magnitude
    result_blocked = effective_expanded * payload_blocked
    result_values = result_blocked.reshape(moved_shape)
    payload = payload_blocked.reshape(moved_shape)

    restored_values = result_values.movedim(-1, normalized_dim)
    restored_payload = payload.movedim(-1, normalized_dim)
    # block_scales 保持 moved 布局：leading dims + num_blocks（例如 [32,3]/dim0 → (3,2)）。
    block_scales = block_scale.reshape(moved_shape[:-1] + (num_blocks,))

    return NVFP4SimulationResult(
        values=restored_values,
        global_scale=global_scale,
        block_scales=block_scales,
        payload=restored_payload,
    )


# =============================================================================
# FP64 误差累计与 JSON-safe 指标派生
# =============================================================================


def compute_error_sums(reference: torch.Tensor, approximation: torch.Tensor) -> ErrorSums:
    """在 FP64 中累计 reference 与 approximation 的误差原始量。"""
    if reference.shape != approximation.shape:
        raise ValueError("reference and approximation must have the same shape")
    if not reference.is_floating_point() or not approximation.is_floating_point():
        raise TypeError("reference and approximation must be floating point")
    reference64 = reference.detach().to(torch.float64)
    approximation64 = approximation.detach().to(torch.float64)
    if not torch.isfinite(reference64).all() or not torch.isfinite(approximation64).all():
        raise ValueError("reference and approximation must be finite")
    difference64 = approximation64 - reference64
    abs_diff = difference64.abs()
    return ErrorSums(
        numel=int(reference.numel()),
        reference_energy=float((reference64 * reference64).sum().item()),
        approximation_energy=float((approximation64 * approximation64).sum().item()),
        error_energy=float((difference64 * difference64).sum().item()),
        dot=float((reference64 * approximation64).sum().item()),
        absolute_error_sum=float(abs_diff.sum().item()),
        max_absolute_error=float(abs_diff.max().item()) if abs_diff.numel() > 0 else 0.0,
    )


def merge_error_sums(destination: ErrorSums, source: ErrorSums) -> ErrorSums:
    """合并两个 ErrorSums（用于 chunk / repeat / 全局聚合）。"""
    if source.numel == 0:
        return destination
    if destination.numel == 0:
        return ErrorSums(
            numel=source.numel,
            reference_energy=source.reference_energy,
            approximation_energy=source.approximation_energy,
            error_energy=source.error_energy,
            dot=source.dot,
            absolute_error_sum=source.absolute_error_sum,
            max_absolute_error=source.max_absolute_error,
        )
    return ErrorSums(
        numel=destination.numel + source.numel,
        reference_energy=destination.reference_energy + source.reference_energy,
        approximation_energy=destination.approximation_energy + source.approximation_energy,
        error_energy=destination.error_energy + source.error_energy,
        dot=destination.dot + source.dot,
        absolute_error_sum=destination.absolute_error_sum + source.absolute_error_sum,
        max_absolute_error=max(destination.max_absolute_error, source.max_absolute_error),
    )


def finalize_error_metrics(sums: ErrorSums) -> dict[str, float | int | str | None]:
    """从 ErrorSums 派生 NMSE/NRMSE/Cosine/SQNR 等 JSON-safe 指标。"""
    numel = sums.numel
    ref_energy = sums.reference_energy
    approx_energy = sums.approximation_energy
    err_energy = sums.error_energy
    dot = sums.dot
    mae = sums.absolute_error_sum / numel if numel > 0 else 0.0

    ref_zero = ref_energy == 0.0
    approx_zero = approx_energy == 0.0
    err_zero = err_energy == 0.0

    if ref_zero and approx_zero:
        nmse: float | str = 0.0
        nrmse: float | str = 0.0
        cosine: float | str = 1.0
        sqnr_db: float | str | None = "inf"
    elif ref_zero and not approx_zero:
        nmse = "inf"
        nrmse = "inf"
        cosine = 0.0
        sqnr_db = "-inf" if err_energy > 0 else "inf"
    else:
        nmse = err_energy / ref_energy
        nrmse = math.sqrt(nmse)
        ref_norm = math.sqrt(ref_energy)
        approx_norm = math.sqrt(approx_energy)
        if ref_norm == 0.0 or approx_norm == 0.0:
            cosine = 0.0
        else:
            cosine = dot / (ref_norm * approx_norm)
        if err_zero:
            sqnr_db = "inf"
        else:
            sqnr_db = 10.0 * math.log10(ref_energy / err_energy)

    return {
        "numel": numel,
        "reference_energy": ref_energy,
        "approximation_energy": approx_energy,
        "error_energy": err_energy,
        "dot": dot,
        "absolute_error_sum": sums.absolute_error_sum,
        "max_absolute_error": sums.max_absolute_error,
        "nmse": nmse,
        "nrmse": nrmse,
        "cosine": cosine,
        "sqnr_db": sqnr_db,
        "mae": mae,
    }


def _metrics_from_pair(reference: torch.Tensor, approximation: torch.Tensor) -> tuple[ErrorSums, dict[str, Any]]:
    sums = compute_error_sums(reference, approximation)
    return sums, finalize_error_metrics(sums)


# =============================================================================
# PTS scale 校验与 NV/BF16 tensor 评测 API
# =============================================================================


def _normalize_pts_scale(
    pts_scale: torch.Tensor | float | None,
    *,
    device: torch.device,
    weight: torch.Tensor,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """校验并广播 PTS scale；缺失时返回 None。"""
    meta: dict[str, Any] = {"pts_status": "not_provided"}
    if pts_scale is None:
        return None, meta

    if isinstance(pts_scale, float):
        if not math.isfinite(pts_scale) or pts_scale <= 0:
            raise ValueError("pts_scale must be a positive finite scalar")
        scale = torch.tensor(pts_scale, dtype=torch.float32, device=device)
        meta = {
            "pts_status": "provided",
            "pts_scale_value": float(scale.item()),
            "pts_scale_dtype": str(scale.dtype),
            "pts_scale_shape": list(scale.shape),
        }
        return scale, meta

    if isinstance(pts_scale, torch.Tensor):
        if pts_scale.is_complex() or not pts_scale.is_floating_point():
            raise TypeError("pts_scale tensor must be floating point")
        scale = pts_scale.detach().to(device=device, dtype=torch.float32)
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("pts_scale must be positive and finite")
        if scale.numel() == 1:
            scale = scale.reshape(())
            meta = {
                "pts_status": "provided",
                "pts_scale_value": float(scale.item()),
                "pts_scale_dtype": str(scale.dtype),
                "pts_scale_shape": list(scale.shape),
            }
            return scale, meta
        # 允许可广播到 weight，或元素个数一致后 reshape。
        try:
            torch.broadcast_shapes(tuple(scale.shape), tuple(weight.shape))
        except RuntimeError:
            if scale.numel() == weight.numel():
                scale = scale.reshape(weight.shape)
            else:
                raise ValueError("pts_scale cannot broadcast to weight shape") from None
        meta = {
            "pts_status": "provided",
            "pts_scale_value": None,
            "pts_scale_dtype": str(scale.dtype),
            "pts_scale_shape": list(scale.shape),
        }
        return scale, meta

    raise TypeError("pts_scale must be float, tensor, or None")


def evaluate_nvfp4_fake_weight(
    weight: torch.Tensor,
    *,
    pts_scale: torch.Tensor | float | None = None,
    hif4_config: HiF4Config = HiF4Config(),
    return_reconstructions: bool = False,
) -> dict[str, Any]:
    """评测 NVFP4 fake weight 原生转 HiF4（direct / PTS-FP32 / PTS-BF16）。"""
    reference_fp32 = weight.detach().to(torch.float32)
    if not reference_fp32.is_floating_point() or not torch.isfinite(reference_fp32).all():
        raise ValueError("weight must be finite floating point")

    direct_recon = quantize_hif4(reference_fp32, config=hif4_config).values
    direct_sums, direct_metrics = _metrics_from_pair(reference_fp32, direct_recon)

    paths: dict[str, Any] = {
        "direct": {
            "sums": asdict(direct_sums),
            "metrics": direct_metrics,
        }
    }
    if return_reconstructions:
        paths["direct"]["reconstruction"] = direct_recon

    scale, pts_meta = _normalize_pts_scale(pts_scale, device=reference_fp32.device, weight=weight)
    result: dict[str, Any] = {
        "input_kind": "nvfp4_fake",
        "reference": reference_fp32,
        "paths": paths,
        "pts_status": pts_meta["pts_status"],
        "hif4_config": {
            "group_size": hif4_config.group_size,
            "group_dim": hif4_config.group_dim,
            "scale_mode": hif4_config.scale_mode,
        },
    }

    if scale is None:
        result["paths"]["pts_fp32"] = None
        result["paths"]["pts_bf16"] = None
        return result

    normalized_fp32 = reference_fp32 / scale
    pts_fp32_recon = quantize_hif4(normalized_fp32, config=hif4_config).values * scale
    pts_fp32_sums, pts_fp32_metrics = _metrics_from_pair(reference_fp32, pts_fp32_recon)
    paths["pts_fp32"] = {"sums": asdict(pts_fp32_sums), "metrics": pts_fp32_metrics}
    if return_reconstructions:
        paths["pts_fp32"]["reconstruction"] = pts_fp32_recon

    normalized_bf16 = round_bfloat16(normalized_fp32)
    inner_sums, inner_metrics = _metrics_from_pair(normalized_fp32, normalized_bf16)
    pts_bf16_recon = quantize_hif4(normalized_bf16, config=hif4_config).values * scale
    pts_bf16_sums, pts_bf16_metrics = _metrics_from_pair(reference_fp32, pts_bf16_recon)
    paths["pts_bf16"] = {"sums": asdict(pts_bf16_sums), "metrics": pts_bf16_metrics}
    if return_reconstructions:
        paths["pts_bf16"]["reconstruction"] = pts_bf16_recon

    result.update(pts_meta)
    result["inner_bf16_projection"] = {"sums": asdict(inner_sums), "metrics": inner_metrics}

    direct_nmse = direct_metrics["nmse"]
    pts_bf16_nmse = pts_bf16_metrics["nmse"]
    if isinstance(direct_nmse, (int, float)) and isinstance(pts_bf16_nmse, (int, float)):
        delta = pts_bf16_nmse - direct_nmse
        result["pts_delta_nmse"] = delta
        if direct_nmse == 0.0:
            result["pts_relative_change"] = None if delta == 0.0 else "inf"
        else:
            result["pts_relative_change"] = delta / direct_nmse
    else:
        result["pts_delta_nmse"] = None
        result["pts_relative_change"] = None

    equal_fraction = float(torch.equal(pts_fp32_recon, pts_bf16_recon))
    result["pts_fp32_vs_pts_bf16_value_equal_fraction"] = equal_fraction
    return result


def evaluate_bf16_weight(
    weight: torch.Tensor,
    *,
    hif4_config: HiF4Config = HiF4Config(),
    return_reconstruction: bool = False,
) -> dict[str, Any]:
    """评测 BF16 原生转 HiF4；reference 为显式 BF16 投影后的值。"""
    reference = round_bfloat16(weight.detach())
    if not reference.is_floating_point() or not torch.isfinite(reference).all():
        raise ValueError("weight must be finite floating point")
    reconstruction = quantize_hif4(reference, config=hif4_config).values
    sums, metrics = _metrics_from_pair(reference, reconstruction)
    result: dict[str, Any] = {
        "input_kind": "bf16",
        "reference": reference,
        "reconstruction": reconstruction if return_reconstruction else None,
        "sums": asdict(sums),
        "metrics": metrics,
        "hif4_config": {
            "group_size": hif4_config.group_size,
            "group_dim": hif4_config.group_dim,
            "scale_mode": hif4_config.scale_mode,
        },
    }
    return result


def evaluate_output_error(
    activations: torch.Tensor,
    reference_weight: torch.Tensor,
    approximation_weight: torch.Tensor,
    *,
    token_batch_size: int = 256,
) -> dict[str, float | int | str | None]:
    """计算 activations @ weight.T 的输出误差（按 token batch 分块，FP64 累计）。"""
    if activations.ndim != 2 or reference_weight.ndim != 2 or approximation_weight.ndim != 2:
        raise ValueError("activations and weights must be 2-D tensors")
    if reference_weight.shape != approximation_weight.shape:
        raise ValueError("reference_weight and approximation_weight must have the same shape")
    if activations.shape[1] != reference_weight.shape[1]:
        raise ValueError("activations in_features must match weight in_features")
    if token_batch_size <= 0:
        raise ValueError("token_batch_size must be positive")
    if not activations.is_floating_point() or not torch.isfinite(activations).all():
        raise ValueError("activations must be finite floating point")

    merged = ErrorSums()
    num_tokens = activations.shape[0]
    ref_w = reference_weight.detach().float()
    approx_w = approximation_weight.detach().float()
    for start in range(0, num_tokens, token_batch_size):
        end = min(start + token_batch_size, num_tokens)
        x_batch = activations[start:end].detach().float()
        ref_out = x_batch @ ref_w.T
        approx_out = x_batch @ approx_w.T
        merged = merge_error_sums(merged, compute_error_sums(ref_out, approx_out))
    return finalize_error_metrics(merged)

# =============================================================================
# Checkpoint 读取、PTS 映射、tensor 分类与筛选
# =============================================================================


def _torch_load(path: Path) -> Any:
    """安全加载 .pt/.pth；不支持 weights_only 时抛出明确 TypeError。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise TypeError(
            f"PyTorch version must support weights_only=True for loading {path}"
        ) from exc


def _extract_state_dict(obj: Any, source: Path) -> dict[str, torch.Tensor]:
    """从 checkpoint 对象提取 state_dict；仅支持明确 wrapper。"""
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            obj = obj["model_state_dict"]
        if obj and all(isinstance(k, str) for k in obj.keys()) and all(
            isinstance(v, torch.Tensor) for v in obj.values()
        ):
            return {k: v for k, v in obj.items()}
        raise ValueError(f"Cannot infer tensor mapping from checkpoint file: {source}")
    raise ValueError(f"Unsupported checkpoint content in file: {source}")


def _iter_tensors_from_state(state: dict[str, torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
    for name in sorted(state.keys()):
        tensor = state[name]
        if isinstance(tensor, torch.Tensor):
            yield name, tensor


def _load_safetensors_file(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "Reading .safetensors requires safetensors; install with: pip install safetensors"
        ) from exc
    return load_file(str(path), device="cpu")


def _resolve_checkpoint_files(checkpoint_path: Path) -> list[tuple[str, Path | None, dict[str, str] | None]]:
    """解析 checkpoint 路径，返回 (kind, file_or_none, weight_map) 列表。"""
    if checkpoint_path.is_file():
        if checkpoint_path.suffix == ".safetensors":
            return [("safetensors", checkpoint_path, None)]
        return [("file", checkpoint_path, None)]

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    safetensors_index = checkpoint_path / "model.safetensors.index.json"
    if safetensors_index.exists():
        index = json.loads(safetensors_index.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = index["weight_map"]
        return [("index", checkpoint_path, weight_map)]

    bin_index = checkpoint_path / "pytorch_model.bin.index.json"
    if bin_index.exists():
        index = json.loads(bin_index.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        return [("index", checkpoint_path, weight_map)]

    single_st = checkpoint_path / "model.safetensors"
    if single_st.exists():
        return [("safetensors", single_st, None)]

    candidates: list[Path] = []
    for pattern in ("*.pt", "*.pth", "*.bin"):
        candidates.extend(checkpoint_path.glob(pattern))
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return [("file", candidates[0], None)]

    st_files = sorted(checkpoint_path.glob("*.safetensors"))
    if st_files:
        return [("safetensors_list", checkpoint_path, None)] + [
            ("safetensors", p, None) for p in st_files
        ]

    if candidates:
        return [("file", p, None) for p in candidates]

    raise FileNotFoundError(f"No supported checkpoint files found in directory: {checkpoint_path}")


def decode_nvfp4_packed_weight(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    *,
    group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """解码 compressed-tensors NVFP4 权重并返回 FP32 PTS scale。"""
    if weight_packed.dtype != torch.uint8:
        raise TypeError(f"weight_packed must be torch.uint8, got {weight_packed.dtype}")
    if weight_packed.ndim != 2:
        raise ValueError(f"weight_packed must be 2D, got {weight_packed.ndim}D")
    if weight_scale.ndim != 2:
        raise ValueError(f"weight_scale must be 2D, got {weight_scale.ndim}D")
    if weight_global_scale.numel() != 1:
        raise ValueError("weight_global_scale must contain exactly one value")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    out_features, packed_in_features = weight_packed.shape
    in_features = packed_in_features * 2
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} must be divisible by group_size={group_size}"
        )
    expected_scale_shape = (out_features, in_features // group_size)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"weight_scale shape must be {expected_scale_shape}, "
            f"got {tuple(weight_scale.shape)}"
        )

    global_scale = weight_global_scale.reshape(()).to(torch.float32)
    if not torch.isfinite(global_scale).item() or global_scale.item() <= 0:
        raise ValueError("weight_global_scale must be positive and finite")
    scale = weight_scale.to(torch.float32)
    if not torch.isfinite(scale).all().item():
        raise ValueError("weight_scale must be finite")

    packed = weight_packed.contiguous()
    low = packed & 15
    high = (packed & 240) >> 4
    codes = torch.stack((low, high), dim=-1).reshape(out_features, in_features)
    sign = torch.where(
        (codes & 8).bool(),
        torch.tensor(-1.0, device=codes.device),
        torch.tensor(1.0, device=codes.device),
    )
    values = E2M1_VALUES.to(codes.device)[(codes & 7).long()] * sign
    normalized = values.reshape(
        out_features, in_features // group_size, group_size
    ) * scale.unsqueeze(-1)
    pts_scale = torch.reciprocal(global_scale)
    decoded = normalized.reshape(out_features, in_features) * pts_scale
    return decoded.to(torch.float32), pts_scale


def iter_nvfp4_packed_weights(
    checkpoint_path: Path,
    *,
    tensor_names: tuple[str, ...] = (),
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    max_tensors: int | None = None,
) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
    """按需解码单文件 safetensors 中的 packed NVFP4 Linear 权重。"""
    checkpoint_path = Path(checkpoint_path)
    resolved = _resolve_checkpoint_files(checkpoint_path)
    if len(resolved) != 1 or resolved[0][0] != "safetensors":
        raise ValueError(
            "nvfp4_packed requires one .safetensors file "
            "or a directory containing model.safetensors"
        )
    safetensors_path = resolved[0][1]
    assert safetensors_path is not None
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "Reading packed NVFP4 requires safetensors; install with: pip install safetensors"
        ) from exc

    if max_tensors is not None and max_tensors < 0:
        raise ValueError("max_tensors must be non-negative")

    with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        bases = sorted(
            name[: -len("weight_packed")]
            for name in keys
            if name.endswith("weight_packed")
        )
        available_names = {base + "weight" for base in bases}
        missing = sorted(set(tensor_names) - available_names)
        if missing:
            raise KeyError(
                "Requested packed NVFP4 tensors were not found: " + ", ".join(missing)
            )

        accepted = 0
        for base in bases:
            name = base + "weight"
            if tensor_names and name not in tensor_names:
                continue
            if include_regex is not None and re.search(include_regex, name) is None:
                continue
            if exclude_regex is not None and re.search(exclude_regex, name) is not None:
                continue
            if max_tensors is not None and accepted >= max_tensors:
                break

            packed_name = base + "weight_packed"
            scale_name = base + "weight_scale"
            global_name = base + "weight_global_scale"
            missing_components = [
                component
                for component in (scale_name, global_name)
                if component not in keys
            ]
            if missing_components:
                raise KeyError(
                    f"Incomplete packed NVFP4 weight {name}: "
                    + ", ".join(missing_components)
                )

            decoded, pts_scale = decode_nvfp4_packed_weight(
                handle.get_tensor(packed_name),
                handle.get_tensor(scale_name),
                handle.get_tensor(global_name),
            )
            yield name, decoded, pts_scale
            accepted += 1


def iter_checkpoint_tensors(checkpoint_path: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """按参数名顺序迭代 checkpoint 中的 tensor（CPU）。"""
    checkpoint_path = Path(checkpoint_path)
    resolved = _resolve_checkpoint_files(checkpoint_path)

    if len(resolved) == 1 and resolved[0][0] == "safetensors":
        assert resolved[0][1] is not None
        state = _load_safetensors_file(resolved[0][1])
        yield from _iter_tensors_from_state(state)
        return

    if len(resolved) == 1 and resolved[0][0] == "file":
        obj = _torch_load(resolved[0][1])
        state = _extract_state_dict(obj, resolved[0][1])
        yield from _iter_tensors_from_state(state)
        return

    if resolved[0][0] == "index":
        _, base_dir, weight_map = resolved[0]
        assert weight_map is not None
        shard_to_names: dict[str, list[str]] = defaultdict(list)
        for name, shard in weight_map.items():
            shard_to_names[shard].append(name)
        loaded_shards: dict[str, dict[str, torch.Tensor]] = {}
        for name in sorted(weight_map.keys()):
            shard = weight_map[name]
            if shard not in loaded_shards:
                shard_path = base_dir / shard
                if shard_path.suffix == ".safetensors":
                    loaded_shards[shard] = _load_safetensors_file(shard_path)
                else:
                    obj = _torch_load(shard_path)
                    state = _extract_state_dict(obj, shard_path)
                    loaded_shards[shard] = state
            tensor = loaded_shards[shard][name]
            yield name, tensor
        return

    if resolved[0][0] == "safetensors_list":
        for kind, path, _ in resolved[1:]:
            assert path is not None
            state = _load_safetensors_file(path)
            yield from _iter_tensors_from_state(state)
        return

    for kind, path, _ in resolved:
        if path is None:
            continue
        if kind == "safetensors":
            state = _load_safetensors_file(path)
            yield from _iter_tensors_from_state(state)
        elif kind == "file":
            obj = _torch_load(path)
            state = _extract_state_dict(obj, path)
            yield from _iter_tensors_from_state(state)


def load_pts_scales(path: Path | None) -> dict[str, torch.Tensor | float]:
    """加载 per-tensor PTS scale 映射（JSON 或 .pt/.pth dict）。"""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PTS scales file not found: {path}")

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("PTS JSON must be an object")
        result: dict[str, torch.Tensor | float] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key:
                raise ValueError("PTS JSON keys must be non-empty strings")
            if isinstance(value, (int, float)):
                if not math.isfinite(float(value)) or float(value) <= 0:
                    raise ValueError(f"PTS scale for {key} must be positive and finite")
                result[key] = float(value)
            else:
                raise ValueError(f"PTS scale for {key} must be a numeric scalar")
        return result

    obj = _torch_load(path)
    if not isinstance(obj, dict):
        raise ValueError("PTS .pt/.pth file must contain a dict")
    result = {}
    for key, value in obj.items():
        if not isinstance(key, str) or not key:
            raise ValueError("PTS mapping keys must be non-empty strings")
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"PTS tensor for {key} must be scalar")
            scalar = value.detach().cpu().to(torch.float32).reshape(()).item()
            if not math.isfinite(scalar) or scalar <= 0:
                raise ValueError(f"PTS scale for {key} must be positive and finite")
            result[key] = torch.tensor(scalar, dtype=torch.float32)
        elif isinstance(value, (int, float)):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"PTS scale for {key} must be positive and finite")
            result[key] = float(value)
        else:
            raise ValueError(f"PTS value for {key} must be float or scalar tensor")
    return result


def classify_tensor_category(name: str) -> str:
    """按参数名 suffix/substring 分类 tensor。"""
    ordered = (
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
        ("o_proj", "o_proj"),
        ("gate_proj", "gate_proj"),
        ("up_proj", "up_proj"),
        ("down_proj", "down_proj"),
        ("embed_tokens", "embed_tokens"),
        ("lm_head", "lm_head"),
    )
    for token, category in ordered:
        if token in name:
            return category
    return "other"


def _should_evaluate_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    tensor_names: tuple[str, ...],
    include_regex: str | None,
    exclude_regex: str | None,
    hif4_config: HiF4Config,
    max_tensors: int | None = None,
    accepted_count: int = 0,
) -> str | None:
    """按固定顺序判断是否评测 tensor；返回 skip_reason，或 None 表示应评测。"""
    if tensor_names and name not in tensor_names:
        return "not_requested"
    if include_regex is not None and re.search(include_regex, name) is None:
        return "include_regex_miss"
    if exclude_regex is not None and re.search(exclude_regex, name) is not None:
        return "exclude_regex_match"
    if not tensor.is_floating_point():
        return "non_floating"
    if tensor.ndim < 2:
        return "ndim_lt_2"
    if hif4_config.group_dim < -tensor.ndim or hif4_config.group_dim >= tensor.ndim:
        return "invalid_group_dim"
    normalized_dim = hif4_config.group_dim % tensor.ndim
    if tensor.shape[normalized_dim] % hif4_config.group_size != 0:
        return "not_group_divisible"
    if max_tensors is not None and accepted_count >= max_tensors:
        return "max_tensors_reached"
    return None


# =============================================================================
# group-aligned chunk 迭代与 group 级误差摘要
# =============================================================================


def iter_group_aligned_chunks(
    weight: torch.Tensor,
    *,
    group_size: int,
    group_dim: int,
    chunk_groups: int,
) -> Iterator[torch.Tensor]:
    """按完整 HiF4 group 切分，每个 chunk 形状为 [n_groups, group_size]。"""
    if chunk_groups <= 0:
        raise ValueError("chunk_groups must be positive")
    moved, _, _ = _move_groups_to_last(weight, group_dim)
    grouped = moved.reshape(-1, group_size)
    total = grouped.shape[0]
    for start in range(0, total, chunk_groups):
        end = min(start + chunk_groups, total)
        yield grouped[start:end]


def _compute_group_nmse_summary(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    group_size: int,
    group_dim: int,
) -> dict[str, Any]:
    """按 group 统计 NMSE 分位数与 top-20（仅摘要，不落盘全量 group 列表）。"""
    moved_ref, _, _ = _move_groups_to_last(reference, group_dim)
    moved_approx, _, _ = _move_groups_to_last(approximation, group_dim)
    ref_groups = moved_ref.reshape(-1, group_size).to(torch.float64)
    approx_groups = moved_approx.reshape(-1, group_size).to(torch.float64)
    ref_energy = (ref_groups * ref_groups).sum(dim=-1)
    err = approx_groups - ref_groups
    err_energy = (err * err).sum(dim=-1)
    zero_ref = ref_energy == 0
    group_nmse = torch.where(
        ref_energy > 0,
        err_energy / ref_energy,
        torch.full_like(err_energy, float("nan")),
    )
    valid = group_nmse[torch.isfinite(group_nmse)]
    valid_list = sorted(float(v) for v in valid.cpu().tolist())

    def quantile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        idx = int(round(q * (len(values) - 1)))
        idx = max(0, min(idx, len(values) - 1))
        return values[idx]

    top_pairs = sorted(
        ((int(i), float(v)) for i, v in enumerate(group_nmse.cpu().tolist()) if math.isfinite(v)),
        key=lambda item: item[1],
        reverse=True,
    )[:20]

    return {
        "group_count": int(ref_groups.shape[0]),
        "zero_reference_group_count": int(zero_ref.sum().item()),
        "group_nmse_p50": quantile(valid_list, 0.50),
        "group_nmse_p90": quantile(valid_list, 0.90),
        "group_nmse_p95": quantile(valid_list, 0.95),
        "group_nmse_p99": quantile(valid_list, 0.99),
        "group_nmse_max": max(valid_list) if valid_list else None,
        "top_groups": [{"group_index": i, "group_nmse": v} for i, v in top_pairs],
    }


def _chunked_evaluate_tensor(
    weight: torch.Tensor,
    *,
    input_kind: Literal["nvfp4_fake", "bf16"],
    pts_scale: torch.Tensor | float | None,
    hif4_config: HiF4Config,
    chunk_groups: int,
    return_reconstructions: bool = True,
) -> dict[str, Any]:
    """对单个 tensor 做 group-aligned chunk 评测，合并 ErrorSums，并可拼回 reconstruction。"""
    if chunk_groups <= 0:
        raise ValueError("chunk_groups must be positive")

    # BF16 路径：先整体投影，保证 reference 语义与分 chunk 无关。
    if input_kind == "bf16":
        working = round_bfloat16(weight.detach())
    else:
        working = weight.detach().to(torch.float32)

    moved, normalized_dim, moved_shape = _move_groups_to_last(working, hif4_config.group_dim)
    group_size = hif4_config.group_size
    grouped = moved.reshape(-1, group_size)
    total_groups = grouped.shape[0]
    chunk_cfg = replace(hif4_config, group_dim=-1)

    merged: dict[str, ErrorSums] = {}
    recon_parts: list[torch.Tensor] = []

    for start in range(0, total_groups, chunk_groups):
        end = min(start + chunk_groups, total_groups)
        chunk = grouped[start:end]
        if input_kind == "nvfp4_fake":
            ev = evaluate_nvfp4_fake_weight(
                chunk,
                pts_scale=pts_scale,
                hif4_config=chunk_cfg,
                return_reconstructions=return_reconstructions,
            )
            path_names = ("direct", "pts_fp32", "pts_bf16")
        else:
            bf = evaluate_bf16_weight(chunk, hif4_config=chunk_cfg, return_reconstruction=return_reconstructions)
            ev = {
                "paths": {
                    "native": {
                        "sums": bf["sums"],
                        "metrics": bf["metrics"],
                        "reconstruction": bf.get("reconstruction"),
                    }
                }
            }
            path_names = ("native",)

        for name in path_names:
            path = ev["paths"].get(name)
            if path is None:
                continue
            sums = ErrorSums(**path["sums"])
            merged[name] = merge_error_sums(merged.get(name, ErrorSums()), sums)

        if return_reconstructions:
            if input_kind == "nvfp4_fake":
                recon_parts.append(ev["paths"]["direct"]["reconstruction"])
            else:
                recon_parts.append(ev["paths"]["native"]["reconstruction"])

    result_paths: dict[str, Any] = {}
    if input_kind == "nvfp4_fake":
        for name in ("direct", "pts_fp32", "pts_bf16"):
            if name in merged:
                result_paths[name] = {
                    "sums": asdict(merged[name]),
                    "metrics": finalize_error_metrics(merged[name]),
                }
            else:
                result_paths[name] = None
        pts_status = "provided" if pts_scale is not None else "not_provided"
    else:
        result_paths["native"] = {
            "sums": asdict(merged["native"]),
            "metrics": finalize_error_metrics(merged["native"]),
        }
        pts_status = "not_applicable"

    if return_reconstructions and recon_parts:
        recon_groups = torch.cat(recon_parts, dim=0).reshape(moved_shape)
        recon = _restore_from_last(recon_groups, normalized_dim, working.ndim)
        primary = "direct" if input_kind == "nvfp4_fake" else "native"
        if result_paths.get(primary) is not None:
            result_paths[primary]["reconstruction"] = recon
        # PTS 路径也给出完整 reconstruction，便于对照实验。
        if input_kind == "nvfp4_fake" and pts_scale is not None:
            full = evaluate_nvfp4_fake_weight(
                working,
                pts_scale=pts_scale,
                hif4_config=hif4_config,
                return_reconstructions=True,
            )
            for name in ("pts_fp32", "pts_bf16"):
                if result_paths.get(name) is not None:
                    result_paths[name]["reconstruction"] = full["paths"][name]["reconstruction"]

    return {
        "input_kind": input_kind,
        "reference": working,
        "paths": result_paths,
        "pts_status": pts_status,
    }


# =============================================================================
# Checkpoint 评测：筛选、分类、能量聚合
# =============================================================================


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    input_kind: Literal["nvfp4_fake", "nvfp4_packed", "bf16"],
    pts_scales_path: Path | None,
    include_regex: str | None,
    exclude_regex: str | None,
    device: torch.device,
    hif4_config: HiF4Config,
    chunk_groups: int = 16_384,
    require_pts: bool = False,
    tensor_names: tuple[str, ...] = (),
    max_tensors: int | None = None,
    activations_path: Path | None = None,
    token_batch_size: int = 256,
    compute_group_summary: bool = True,
) -> dict[str, Any]:
    """评测真实 checkpoint：逐 tensor / 类别 / 全局能量加权指标。"""
    if input_kind not in {"nvfp4_fake", "nvfp4_packed", "bf16"}:
        raise ValueError(
            "input_kind must be 'nvfp4_fake', 'nvfp4_packed', or 'bf16'"
        )
    if input_kind in {"bf16", "nvfp4_packed"} and pts_scales_path is not None:
        raise ValueError(
            f"pts_scales_path is incompatible with input_kind={input_kind}"
        )
    if require_pts and input_kind == "bf16":
        raise ValueError("require_pts is only valid for NVFP4 inputs")

    pts_map = (
        load_pts_scales(pts_scales_path)
        if input_kind == "nvfp4_fake"
        else {}
    )
    if require_pts and input_kind == "nvfp4_fake" and not pts_map:
        raise ValueError("require_pts=True but no PTS scales were provided")

    activations: dict[str, torch.Tensor] | None = None
    if activations_path is not None:
        activations = _load_activation_map(Path(activations_path))

    global_sums: dict[str, ErrorSums] = {}
    category_sums: dict[str, dict[str, ErrorSums]] = defaultdict(dict)
    tensor_results: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    accepted = 0

    if input_kind == "nvfp4_packed":
        tensor_items = iter_nvfp4_packed_weights(
            Path(checkpoint_path),
            tensor_names=tensor_names,
            include_regex=include_regex,
            exclude_regex=exclude_regex,
            max_tensors=max_tensors,
        )
    else:
        tensor_items = (
            (name, tensor, None)
            for name, tensor in iter_checkpoint_tensors(Path(checkpoint_path))
        )

    for name, tensor, embedded_pts_scale in tensor_items:
        decision = _should_evaluate_tensor(
            name,
            tensor,
            tensor_names=tensor_names,
            include_regex=include_regex,
            exclude_regex=exclude_regex,
            hif4_config=hif4_config,
            max_tensors=max_tensors,
            accepted_count=accepted,
        )
        if decision is not None:
            skipped[name] = decision
            continue

        if input_kind == "nvfp4_fake" and require_pts and name not in pts_map:
            raise ValueError(f"require_pts=True but missing PTS for {name}")

        if input_kind == "nvfp4_packed":
            pts_scale = embedded_pts_scale
            eval_input_kind = "nvfp4_fake"
        else:
            pts_scale = pts_map.get(name) if input_kind == "nvfp4_fake" else None
            eval_input_kind = input_kind

        weight = tensor.to(device=device)
        ev = _chunked_evaluate_tensor(
            weight,
            input_kind=eval_input_kind,
            pts_scale=pts_scale,
            hif4_config=hif4_config,
            chunk_groups=chunk_groups,
            return_reconstructions=compute_group_summary or activations is not None,
        )
        if input_kind == "nvfp4_packed":
            ev["pts_status"] = "checkpoint_global_scale"

        category = classify_tensor_category(name)
        path_names = ("direct", "pts_fp32", "pts_bf16") if input_kind != "bf16" else ("native",)
        for path_name in path_names:
            path = ev["paths"].get(path_name)
            if path is None:
                continue
            sums = ErrorSums(**path["sums"])
            global_sums[path_name] = merge_error_sums(global_sums.get(path_name, ErrorSums()), sums)
            cat_bucket = category_sums[category]
            cat_bucket[path_name] = merge_error_sums(cat_bucket.get(path_name, ErrorSums()), sums)

        primary = "direct" if input_kind != "bf16" else "native"
        primary_path = ev["paths"].get(primary)
        group_summary = None
        if primary_path is not None and "reconstruction" in primary_path:
            group_summary = _compute_group_nmse_summary(
                ev["reference"],
                primary_path["reconstruction"],
                hif4_config.group_size,
                hif4_config.group_dim,
            )

        entry: dict[str, Any] = {
            "shape": list(tensor.shape),
            "storage_dtype": (
                "nvfp4-pack-quantized"
                if input_kind == "nvfp4_packed"
                else str(tensor.dtype)
            ),
            "decoded_dtype": str(tensor.dtype),
            "category": category,
            "paths": ev["paths"],
            "pts_status": ev["pts_status"],
            "status": "evaluated",
            "group_summary": group_summary,
        }

        if activations is not None and primary_path is not None and "reconstruction" in primary_path:
            act = _lookup_activation(activations, name)
            if act is not None and tensor.ndim == 2:
                out_metrics = evaluate_output_error(
                    act.to(device=device),
                    ev["reference"],
                    primary_path["reconstruction"],
                    token_batch_size=token_batch_size,
                )
                entry["output_error"] = out_metrics
                entry["activation_status"] = "matched"
            else:
                entry["activation_status"] = "missing"
        elif activations is not None:
            entry["activation_status"] = "skipped"

        tensor_results[name] = entry
        accepted += 1

    def _pack_sums(bucket: dict[str, ErrorSums]) -> dict[str, Any]:
        packed: dict[str, Any] = {}
        expected = ("direct", "pts_fp32", "pts_bf16") if input_kind != "bf16" else ("native",)
        for name in expected:
            if name in bucket:
                packed[name] = {
                    "sums": asdict(bucket[name]),
                    "metrics": finalize_error_metrics(bucket[name]),
                }
            else:
                packed[name] = None
        return packed

    categories_out = {cat: _pack_sums(paths) for cat, paths in category_sums.items()}
    global_out = _pack_sums(global_sums)

    pts_tensor_coverage = 0
    pts_numel_coverage = 0
    pts_ref_energy_coverage = 0.0
    if input_kind != "bf16":
        for name, entry in tensor_results.items():
            if entry["paths"].get("pts_bf16") is not None:
                pts_tensor_coverage += 1
                pts_numel_coverage += int(entry["paths"]["direct"]["sums"]["numel"])
                pts_ref_energy_coverage += float(entry["paths"]["direct"]["sums"]["reference_energy"])

    return {
        "schema_version": SCHEMA_VERSION,
        "implementation": IMPLEMENTATION,
        "run_kind": "checkpoint",
        "checkpoint": str(checkpoint_path),
        "input_kind": input_kind,
        "config": {
            "group_size": hif4_config.group_size,
            "group_dim": hif4_config.group_dim,
            "scale_mode": hif4_config.scale_mode,
            "chunk_groups": chunk_groups,
            "device": str(device),
            "include_regex": include_regex,
            "exclude_regex": exclude_regex,
            "tensor_names": list(tensor_names),
            "max_tensors": max_tensors,
            "require_pts": require_pts,
            "compute_group_summary": compute_group_summary,
        },
        "global": global_out,
        "categories": categories_out,
        "tensors": tensor_results,
        "skipped": skipped,
        "pts_tensor_coverage": pts_tensor_coverage,
        "pts_numel_coverage": pts_numel_coverage,
        "pts_reference_energy_coverage": pts_ref_energy_coverage,
    }


def _load_activation_map(path: Path) -> dict[str, torch.Tensor]:
    """读取 activation 文件：dict[str, Tensor]。"""
    obj = _torch_load(path)
    if not isinstance(obj, dict):
        raise ValueError(f"activation file must be a dict: {path}")
    result: dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if isinstance(value, torch.Tensor):
            result[str(key)] = value
    return result


def _lookup_activation(activations: dict[str, torch.Tensor], weight_name: str) -> torch.Tensor | None:
    """按完整 weight 名或去掉 .weight 的 module 名查找 activation。"""
    candidates = [weight_name]
    if weight_name.endswith(".weight"):
        candidates.append(weight_name[: -len(".weight")])
    found: list[torch.Tensor] = []
    for key in candidates:
        if key in activations:
            found.append(activations[key])
    if not found:
        return None
    if len(found) == 2 and not torch.equal(found[0], found[1]):
        raise ValueError(f"conflicting activations for {weight_name}")
    return found[0]


# =============================================================================
# 合成实验：分布采样、repeat 汇总、E1–E7
# =============================================================================


def make_distribution(name: str, n: int, seed: int) -> torch.Tensor:
    """在 CPU 上确定性生成合成分布样本。"""
    if n <= 0:
        raise ValueError("n must be positive")
    if n % 64 != 0:
        raise ValueError("sample count must be divisible by 64")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if name == "gaussian":
        return torch.randn(n, generator=generator, dtype=torch.float32)
    if name == "laplace":
        u = torch.rand(n, generator=generator, dtype=torch.float32) - 0.5
        return -torch.sign(u) * torch.log1p(-2.0 * u.abs()) / math.sqrt(2.0)
    if name == "student_t3":
        z0 = torch.randn(n, generator=generator, dtype=torch.float32)
        z1 = torch.randn(n, generator=generator, dtype=torch.float32)
        z2 = torch.randn(n, generator=generator, dtype=torch.float32)
        z3 = torch.randn(n, generator=generator, dtype=torch.float32)
        return z0 / torch.sqrt(z1 * z1 + z2 * z2 + z3 * z3)
    if name == "outlier_0p1pct_20x":
        base = torch.randn(n, generator=generator, dtype=torch.float32)
        index_generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
        count = int(round(0.001 * n))
        indices = torch.randperm(n, generator=index_generator)[:count]
        out = base.clone()
        out[indices] = out[indices] * 20.0
        return out
    raise ValueError(f"unknown distribution: {name}")


def summarize_repeats(
    repeat_sums: list[ErrorSums],
    repeat_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """合并 repeat 能量，并计算 mean/std/CI95。"""
    merged = ErrorSums()
    for sums in repeat_sums:
        merged = merge_error_sums(merged, sums)
    nmse_values = [float(item["nmse"]) for item in repeat_metrics]
    mean_nmse = statistics.fmean(nmse_values) if nmse_values else 0.0
    std_nmse = statistics.stdev(nmse_values) if len(nmse_values) > 1 else 0.0
    if len(nmse_values) == 10:
        half_width = T_CI_975_9 * std_nmse / math.sqrt(10)
        ci95: list[float] | None = [mean_nmse - half_width, mean_nmse + half_width]
    else:
        ci95 = None
    return {
        "energy_weighted": finalize_error_metrics(merged),
        "repeat_nmse": nmse_values,
        "repeat_count": len(nmse_values),
        "mean_nmse": mean_nmse,
        "std_nmse": std_nmse,
        "ci95": ci95,
    }


def _path_bundle_from_nv(result: dict[str, Any], path_name: str) -> dict[str, Any]:
    path = result["paths"][path_name]
    return {"sums": path["sums"], "metrics": path["metrics"]}


def run_simulation(
    config: ExperimentConfig,
    *,
    device: torch.device,
    quick: bool = False,
) -> dict[str, Any]:
    """运行 E1–E7 合成实验并返回结构化结果。"""
    if quick:
        config = ExperimentConfig(
            seed=config.seed,
            samples_per_repeat=QUICK_SAMPLES if config.samples_per_repeat == DEFAULT_EXPERIMENT_SAMPLES else config.samples_per_repeat,
            repeats=QUICK_REPEATS if config.repeats == DEFAULT_EXPERIMENT_REPEATS else config.repeats,
            phase_points=QUICK_PHASE_POINTS if config.phase_points == DEFAULT_EXPERIMENT_PHASE_POINTS else config.phase_points,
            phase_seed=config.phase_seed,
        )

    if config.samples_per_repeat % 64 != 0:
        raise ValueError("samples_per_repeat must be divisible by 64")
    if config.repeats <= 0 or config.phase_points <= 0:
        raise ValueError("repeats and phase_points must be positive")

    hif4 = HiF4Config()
    e1: dict[str, Any] = {}
    e2: dict[str, Any] = {}
    e5: dict[str, Any] = {"nv_direct": {}, "bf16_native": {}, "nv_pts_bf16": {}}
    e6: dict[str, Any] = {"nv_direct": {}, "bf16_native": {}}
    e7_parts: dict[str, list] = {
        "storage_projection": [],
        "fp32_container_conversion": [],
        "bf16_container_conversion": [],
        "storage_exact": [],
        "recon_equal": [],
        "e8_equal": [],
        "e4_equal": [],
    }

    for dist_name in DISTRIBUTION_NAMES:
        e1[dist_name] = {
            "nv_direct": {"repeats": [], "sums": [], "metrics": []},
            "nv_pts_fp32": {"repeats": [], "sums": [], "metrics": []},
            "nv_pts_bf16": {"repeats": [], "sums": [], "metrics": []},
            "bf16_native": {"repeats": [], "sums": [], "metrics": []},
        }
        e2[dist_name] = {}
        for source in e5:
            e5[source][dist_name] = {}
        for source in e6:
            e6[source][dist_name] = {}

        for repeat_index in range(config.repeats):
            seed = config.seed + repeat_index
            base = make_distribution(dist_name, config.samples_per_repeat, seed=seed)
            # 合成样本先在 CPU 生成，再搬到目标 device，保证 CPU/CUDA 输入一致。
            base_dev = base.to(device)
            nv = simulate_nvfp4(base_dev)
            bf16_reference = round_bfloat16(base_dev)

            nv_result = evaluate_nvfp4_fake_weight(
                nv.values,
                pts_scale=nv.global_scale,
                hif4_config=hif4,
            )
            bf16_result = evaluate_bf16_weight(bf16_reference, hif4_config=hif4)

            for path_name, key in (
                ("direct", "nv_direct"),
                ("pts_fp32", "nv_pts_fp32"),
                ("pts_bf16", "nv_pts_bf16"),
            ):
                bundle = _path_bundle_from_nv(nv_result, path_name)
                e1[dist_name][key]["repeats"].append(
                    {
                        "distribution": dist_name,
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "base_numel": int(base.numel()),
                        "nv_global_scale": float(nv.global_scale.item()),
                        **bundle,
                    }
                )
                e1[dist_name][key]["sums"].append(ErrorSums(**bundle["sums"]))
                e1[dist_name][key]["metrics"].append(bundle["metrics"])

            bf_bundle = {"sums": bf16_result["sums"], "metrics": bf16_result["metrics"]}
            e1[dist_name]["bf16_native"]["repeats"].append(
                {
                    "distribution": dist_name,
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "base_numel": int(base.numel()),
                    **bf_bundle,
                }
            )
            e1[dist_name]["bf16_native"]["sums"].append(ErrorSums(**bf_bundle["sums"]))
            e1[dist_name]["bf16_native"]["metrics"].append(bf_bundle["metrics"])

            # E2：同 repeat 内配对 PTS-BF16 vs direct。
            d_nmse = float(nv_result["paths"]["direct"]["metrics"]["nmse"])
            p_nmse = float(nv_result["paths"]["pts_bf16"]["metrics"]["nmse"])
            delta = p_nmse - d_nmse
            rel = None if d_nmse == 0 else delta / d_nmse
            e2[dist_name][f"repeat_{repeat_index}"] = {
                "nv_direct": _path_bundle_from_nv(nv_result, "direct"),
                "nv_pts_fp32": _path_bundle_from_nv(nv_result, "pts_fp32"),
                "nv_pts_bf16": _path_bundle_from_nv(nv_result, "pts_bf16"),
                "delta_nmse": delta,
                "relative_change": rel,
                "paired_delta_mean": delta,
                "paired_delta_std": 0.0,
                "paired_delta_ci95_low": None,
                "paired_delta_ci95_high": None,
                "pts_bf16_win_count": int(delta < 0),
            }

            # E5：四种 scale_mode 分解。
            for mode in sorted(VALID_SCALE_MODES):
                mode_cfg = HiF4Config(scale_mode=mode)
                nv_mode = evaluate_nvfp4_fake_weight(nv.values, pts_scale=nv.global_scale, hif4_config=mode_cfg)
                bf_mode = evaluate_bf16_weight(bf16_reference, hif4_config=mode_cfg)
                e5["nv_direct"][dist_name][mode] = _path_bundle_from_nv(nv_mode, "direct")
                e5["nv_pts_bf16"][dist_name][mode] = _path_bundle_from_nv(nv_mode, "pts_bf16")
                e5["bf16_native"][dist_name][mode] = {
                    "sums": bf_mode["sums"],
                    "metrics": bf_mode["metrics"],
                }

            # E6：group size 消融。
            for group_size in (16, 32, 64):
                for mode in ("continuous", "hardware"):
                    gcfg = HiF4Config(group_size=group_size, scale_mode=mode)
                    nv_g = evaluate_nvfp4_fake_weight(nv.values, pts_scale=nv.global_scale, hif4_config=gcfg)
                    bf_g = evaluate_bf16_weight(bf16_reference, hif4_config=gcfg)
                    key = f"g{group_size}_{mode}"
                    e6["nv_direct"][dist_name][key] = {
                        "group_size": group_size,
                        "is_standard_hif4": group_size == 64,
                        "scale_mode": mode,
                        **_path_bundle_from_nv(nv_g, "direct"),
                    }
                    e6["bf16_native"][dist_name][key] = {
                        "group_size": group_size,
                        "is_standard_hif4": group_size == 64,
                        "scale_mode": mode,
                        "sums": bf_g["sums"],
                        "metrics": bf_g["metrics"],
                    }

            # E7：同一 NV fake 的 FP32 / BF16 容器。
            nv_fp32 = nv.values
            nv_bf16 = nv_fp32.to(torch.bfloat16)
            storage_ref = nv_bf16.to(torch.float32)
            storage_sums, storage_metrics = _metrics_from_pair(nv_fp32, storage_ref)
            fp32_eval = evaluate_nvfp4_fake_weight(nv_fp32, pts_scale=nv.global_scale, hif4_config=hif4, return_reconstructions=True)
            bf16_eval = evaluate_nvfp4_fake_weight(nv_bf16, pts_scale=nv.global_scale, hif4_config=hif4, return_reconstructions=True)
            e7_parts["storage_projection"].append({"sums": asdict(storage_sums), "metrics": storage_metrics})
            e7_parts["fp32_container_conversion"].append(_path_bundle_from_nv(fp32_eval, "direct"))
            e7_parts["bf16_container_conversion"].append(_path_bundle_from_nv(bf16_eval, "direct"))
            e7_parts["storage_exact"].append(float(torch.equal(nv_fp32, storage_ref)))
            e7_parts["recon_equal"].append(
                float(torch.equal(fp32_eval["paths"]["direct"]["reconstruction"], bf16_eval["paths"]["direct"]["reconstruction"]))
            )
            q_fp = quantize_hif4(nv_fp32, config=hif4)
            q_bf = quantize_hif4(storage_ref, config=hif4)
            e7_parts["e8_equal"].append(float(torch.equal(q_fp.e1_per_8, q_bf.e1_per_8)))
            e7_parts["e4_equal"].append(float(torch.equal(q_fp.e1_per_4, q_bf.e1_per_4)))

        # 汇总 E1 / E2
        for key in ("nv_direct", "nv_pts_fp32", "nv_pts_bf16", "bf16_native"):
            summary = summarize_repeats(e1[dist_name][key]["sums"], e1[dist_name][key]["metrics"])
            e1[dist_name][key] = {
                "summary": summary,
                "repeats": e1[dist_name][key]["repeats"],
            }

        deltas = [e2[dist_name][f"repeat_{i}"]["delta_nmse"] for i in range(config.repeats)]
        mean_delta = statistics.fmean(deltas)
        std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        if len(deltas) == 10:
            half = T_CI_975_9 * std_delta / math.sqrt(10)
            ci_low, ci_high = mean_delta - half, mean_delta + half
        else:
            ci_low = ci_high = None
        for i in range(config.repeats):
            e2[dist_name][f"repeat_{i}"]["paired_delta_mean"] = mean_delta
            e2[dist_name][f"repeat_{i}"]["paired_delta_std"] = std_delta
            e2[dist_name][f"repeat_{i}"]["paired_delta_ci95_low"] = ci_low
            e2[dist_name][f"repeat_{i}"]["paired_delta_ci95_high"] = ci_high
            e2[dist_name][f"repeat_{i}"]["pts_bf16_win_count"] = int(sum(d < 0 for d in deltas))
        e2[dist_name]["summary"] = {
            "paired_delta_mean": mean_delta,
            "paired_delta_std": std_delta,
            "paired_delta_ci95_low": ci_low,
            "paired_delta_ci95_high": ci_high,
            "pts_bf16_win_count": int(sum(d < 0 for d in deltas)),
        }

    # E3：码本穷举 + 合成归一化 carrier。
    e4_vals, _ = build_e4m3fn_codebook()
    e2_vals = E2M1_VALUES
    products = (e4_vals[:, None] * e2_vals[None, :]).reshape(-1)
    carrier = products.to(torch.bfloat16).to(torch.float32)
    exact = products == carrier
    e3_legal = {
        "total_pairs": int(products.numel()),
        "exact_count": int(exact.sum().item()),
        "exact_fraction": float(exact.float().mean().item()),
        "max_absolute_error": float((products - carrier).abs().max().item()),
        "projection_nmse": finalize_error_metrics(compute_error_sums(products, carrier))["nmse"],
    }
    # 用最后一次 gaussian repeat 的归一化值做 synthetic carrier 分析。
    base = make_distribution("gaussian", config.samples_per_repeat, seed=config.seed).to(device)
    nv = simulate_nvfp4(base)
    normalized_fp32 = nv.values.float() / nv.global_scale.float()
    normalized_bf16 = round_bfloat16(normalized_fp32)
    synth_sums, synth_metrics = _metrics_from_pair(normalized_fp32, normalized_bf16)
    pts_compare = evaluate_nvfp4_fake_weight(nv.values, pts_scale=nv.global_scale, hif4_config=hif4, return_reconstructions=True)
    equal_frac = float(
        torch.eq(
            pts_compare["paths"]["pts_fp32"]["reconstruction"],
            pts_compare["paths"]["pts_bf16"]["reconstruction"],
        ).float().mean().item()
    )
    e3 = {
        "legal_codebook_products": e3_legal,
        "synthetic_normalized_values": {
            "sums": asdict(synth_sums),
            "metrics": synth_metrics,
            "exact_fraction": float(torch.eq(normalized_fp32, normalized_bf16).float().mean().item()),
        },
        "final_hif4_impact": {
            "pts_fp32_vs_pts_bf16_equal_fraction": equal_frac,
            "nmse_delta": float(pts_compare["paths"]["pts_bf16"]["metrics"]["nmse"])
            - float(pts_compare["paths"]["pts_fp32"]["metrics"]["nmse"]),
        },
    }

    # E4：phase sweep。
    phase_base = make_distribution("gaussian", config.samples_per_repeat, seed=config.phase_seed).to(device)
    nv_phase = simulate_nvfp4(phase_base)
    expanded_block_scales = (
        nv_phase.block_scales.unsqueeze(-1)
        .expand(*nv_phase.block_scales.shape, 16)
        .reshape_as(nv_phase.payload)
    )
    normalized_legal = expanded_block_scales * nv_phase.payload
    k = torch.arange(config.phase_points, dtype=torch.float64)
    phase = torch.pow(2.0, k / config.phase_points).to(torch.float32)
    points = []
    for g in phase.tolist():
        g_t = torch.tensor(g, dtype=torch.float32, device=device)
        reference = g_t * normalized_legal
        direct = quantize_hif4(reference, config=hif4)
        pts_fp32 = g_t * quantize_hif4(normalized_legal, config=hif4).values
        pts_bf16 = g_t * quantize_hif4(round_bfloat16(normalized_legal), config=hif4).values
        d_sums, d_metrics = _metrics_from_pair(reference, direct.values)
        p32_sums, p32_metrics = _metrics_from_pair(reference, pts_fp32)
        p16_sums, p16_metrics = _metrics_from_pair(reference, pts_bf16)
        point = {
            "phase": g,
            "paths": {
                "direct": {
                    "sums": asdict(d_sums),
                    "metrics": d_metrics,
                    "top_scale_mean": float(direct.top_scale.mean().item()),
                },
                "pts_fp32": {
                    "sums": asdict(p32_sums),
                    "metrics": p32_metrics,
                },
                "pts_bf16": {
                    "sums": asdict(p16_sums),
                    "metrics": p16_metrics,
                },
            },
            "delta_nmse_pts_bf16_minus_direct": float(p16_metrics["nmse"]) - float(d_metrics["nmse"]),
        }
        # 仅 phase=1（首点）保留 reconstruction，供正确性断言且避免全量 phase 爆内存。
        if abs(g - 1.0) < 1e-12:
            point["paths"]["direct"]["reconstruction"] = direct.values.detach().cpu()
            point["paths"]["pts_fp32"]["reconstruction"] = pts_fp32.detach().cpu()
            point["paths"]["pts_bf16"]["reconstruction"] = pts_bf16.detach().cpu()
            point["direct_equals_pts_fp32"] = bool(torch.equal(direct.values, pts_fp32))
        points.append(point)
    direct_nmse = [p["paths"]["direct"]["metrics"]["nmse"] for p in points]
    pts32_nmse = [p["paths"]["pts_fp32"]["metrics"]["nmse"] for p in points]
    pts16_nmse = [p["paths"]["pts_bf16"]["metrics"]["nmse"] for p in points]
    e4 = {
        "phase": phase.tolist(),
        "points": points,
        "argmin_phase": phase[int(min(range(len(direct_nmse)), key=lambda i: direct_nmse[i]))].item(),
        "argmax_phase": phase[int(max(range(len(direct_nmse)), key=lambda i: direct_nmse[i]))].item(),
        "pts_favorable_fraction": float(sum(1 for p in points if p["delta_nmse_pts_bf16_minus_direct"] < 0) / len(points)),
        "direct_nmse_peak_to_peak": float(max(direct_nmse) - min(direct_nmse)),
        "pts_fp32_nmse_peak_to_peak": float(max(pts32_nmse) - min(pts32_nmse)),
        "pts_bf16_nmse_peak_to_peak": float(max(pts16_nmse) - min(pts16_nmse)),
    }

    e7 = {
        "storage_projection": e7_parts["storage_projection"][-1],
        "fp32_container_conversion": e7_parts["fp32_container_conversion"][-1],
        "bf16_container_conversion": e7_parts["bf16_container_conversion"][-1],
        "storage_exact_fraction": statistics.fmean(e7_parts["storage_exact"]),
        "hif4_reconstruction_equal_fraction": statistics.fmean(e7_parts["recon_equal"]),
        "e1_per_8_equal_fraction": statistics.fmean(e7_parts["e8_equal"]),
        "e1_per_4_equal_fraction": statistics.fmean(e7_parts["e4_equal"]),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "implementation": IMPLEMENTATION,
        "run_kind": "simulation",
        "config": asdict(config) | {"device": str(device), "quick": quick},
        "conventions": {
            "nvfp4_reference": "decoded_fake_quantized_value",
            "bf16_reference": "value_after_bfloat16_projection",
            "direct_path": "Q_hif4(W_nv)",
            "pts_fp32_path": "s_T * Q_hif4(W_nv / s_T)",
            "pts_bf16_path": "s_T * Q_hif4(BF16(W_nv / s_T))",
            "hif4_standard_group_size": 64,
            "group_dim": -1,
        },
        "experiments": {
            "e0_correctness": {"status": "covered_by_unit_tests"},
            "e1_native_source": e1,
            "e2_pts_pairing": e2,
            "e3_bf16_carrier": e3,
            "e4_phase_sweep": e4,
            "e5_scale_mode_decomposition": e5,
            "e6_group_size_ablation": e6,
            "e7_storage_dtype": e7,
        },
    }


# =============================================================================
# 结果写出：JSON / CSV / Markdown（原子替换）
# =============================================================================


def _json_sanitize(obj: Any) -> Any:
    """把结果树变成 JSON 可序列化对象；大 tensor 只保留摘要。"""
    if isinstance(obj, torch.Tensor):
        if obj.numel() <= 64:
            return obj.detach().cpu().tolist()
        return {
            "_tensor_omitted": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as handle:
        handle.write(text)
        tmp = Path(handle.name)
    tmp.replace(path)


def write_results(result: dict[str, Any], output_dir: Path) -> None:
    """写出 results.json / results.csv / report.md。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized = _json_sanitize(result)
    _atomic_write_text(
        output_dir / "results.json",
        json.dumps(sanitized, indent=2, ensure_ascii=False, allow_nan=False),
    )

    rows: list[dict[str, Any]] = []

    def add_row(**kwargs: Any) -> None:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_kind": result.get("run_kind"),
                **kwargs,
            }
        )

    if result.get("run_kind") == "simulation":
        e1 = result["experiments"]["e1_native_source"]
        for dist, paths in e1.items():
            for path_name, payload in paths.items():
                metrics = payload["summary"]["energy_weighted"]
                for metric, value in metrics.items():
                    add_row(
                        scope="e1",
                        tensor_name="",
                        category="",
                        distribution=dist,
                        repeat="",
                        path=path_name,
                        metric=metric,
                        value=value,
                        unit="ratio" if metric == "nmse" else "",
                    )
    elif result.get("run_kind") == "checkpoint":
        for path_name, payload in (result.get("global") or {}).items():
            if payload is None:
                continue
            for metric, value in payload["metrics"].items():
                add_row(
                    scope="global",
                    tensor_name="",
                    category="",
                    distribution="",
                    repeat="",
                    path=path_name,
                    metric=metric,
                    value=value,
                    unit="ratio" if metric == "nmse" else "",
                )
        for name, entry in (result.get("tensors") or {}).items():
            for path_name, payload in entry.get("paths", {}).items():
                if payload is None:
                    continue
                for metric, value in payload["metrics"].items():
                    add_row(
                        scope="tensor",
                        tensor_name=name,
                        category=entry.get("category", ""),
                        distribution="",
                        repeat="",
                        path=path_name,
                        metric=metric,
                        value=value,
                        unit="ratio" if metric == "nmse" else "",
                    )

    csv_path = output_dir / "results.csv"
    fieldnames = [
        "schema_version",
        "run_kind",
        "scope",
        "tensor_name",
        "category",
        "distribution",
        "repeat",
        "path",
        "metric",
        "value",
        "unit",
    ]
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(output_dir), encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        tmp_csv = Path(handle.name)
    tmp_csv.replace(csv_path)

    report = _build_markdown_report(result)
    _atomic_write_text(output_dir / "report.md", report)


def _build_markdown_report(result: dict[str, Any]) -> str:
    """生成简短 Markdown 报告。"""
    lines = [
        f"# NVFP4→HiF4 Report ({result.get('run_kind')})",
        "",
        f"- implementation: `{result.get('implementation')}`",
        f"- schema_version: {result.get('schema_version')}",
    ]
    if result.get("run_kind") == "simulation":
        lines.append("- experiments: E1–E7")
        e1 = result["experiments"]["e1_native_source"]
        lines.extend(["", "## E1 native source (energy-weighted NMSE)", ""])
        for dist, paths in e1.items():
            lines.append(f"### {dist}")
            for path_name, payload in paths.items():
                nmse = payload["summary"]["energy_weighted"]["nmse"]
                lines.append(f"- `{path_name}` NMSE = {nmse}")
            lines.append("")
        e2 = result["experiments"]["e2_pts_pairing"]
        lines.extend(["## E2 PTS pairing", ""])
        for dist, payload in e2.items():
            summary = payload.get("summary", {})
            lines.append(
                f"- `{dist}` paired_delta_mean={summary.get('paired_delta_mean')} "
                f"win_count={summary.get('pts_bf16_win_count')}"
            )
        lines.extend(["", "## E3/E4/E5/E6/E7", "详见 results.json。", ""])
    else:
        lines.append(f"- checkpoint: `{result.get('checkpoint')}`")
        lines.append(f"- input_kind: `{result.get('input_kind')}`")
        lines.extend(["", "## Global metrics", ""])
        for path_name, payload in (result.get("global") or {}).items():
            if payload is None:
                lines.append(f"- `{path_name}`: null")
            else:
                lines.append(f"- `{path_name}` NMSE = {payload['metrics']['nmse']}")
        lines.append("")
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================


def _parse_device(text: str) -> torch.device:
    if text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available in current hif4 environment")
    return torch.device(text)


def main(argv: list[str] | None = None) -> int:
    """命令行入口：simulate / evaluate-tensor-file / evaluate-checkpoint。"""
    parser = argparse.ArgumentParser(description="NVFP4→HiF4 greenfield torch experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="Run E1–E7 simulation experiments")
    sim.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sim.add_argument("--seed", type=int, default=20260723)
    sim.add_argument("--samples-per-repeat", type=int, default=DEFAULT_EXPERIMENT_SAMPLES)
    sim.add_argument("--repeats", type=int, default=DEFAULT_EXPERIMENT_REPEATS)
    sim.add_argument("--phase-points", type=int, default=DEFAULT_EXPERIMENT_PHASE_POINTS)
    sim.add_argument("--quick", action="store_true")
    sim.add_argument("--output-dir", type=Path, required=True)

    ten = sub.add_parser("evaluate-tensor-file", help="Evaluate a single tensor file")
    ten.add_argument("--tensor", type=Path, required=True)
    ten.add_argument("--input-kind", choices=["nvfp4_fake", "bf16"], required=True)
    ten.add_argument("--pts-scale", type=float, default=None)
    ten.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ten.add_argument("--group-size", type=int, default=64)
    ten.add_argument("--group-dim", type=int, default=-1)
    ten.add_argument("--scale-mode", default="hardware")
    ten.add_argument("--output-dir", type=Path, required=True)

    ckpt = sub.add_parser("evaluate-checkpoint", help="Evaluate a real checkpoint")
    ckpt.add_argument("--checkpoint", type=Path, required=True)
    ckpt.add_argument(
        "--input-kind",
        choices=["nvfp4_fake", "nvfp4_packed", "bf16"],
        required=True,
    )
    ckpt.add_argument("--pts-scales", type=Path, default=None)
    ckpt.add_argument("--include-regex", type=str, default=None)
    ckpt.add_argument("--exclude-regex", type=str, default=None)
    ckpt.add_argument("--tensor-name", action="append", default=[])
    ckpt.add_argument("--max-tensors", type=int, default=None)
    ckpt.add_argument("--require-pts", action="store_true")
    ckpt.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ckpt.add_argument("--group-size", type=int, default=64)
    ckpt.add_argument("--group-dim", type=int, default=-1)
    ckpt.add_argument("--scale-mode", default="hardware")
    ckpt.add_argument("--chunk-groups", type=int, default=16_384)
    ckpt.add_argument("--skip-group-summary", action="store_true")
    ckpt.add_argument("--activations", type=Path, default=None)
    ckpt.add_argument("--token-batch-size", type=int, default=256)
    ckpt.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    device = _parse_device(args.device)

    if args.command == "simulate":
        cfg = ExperimentConfig(
            seed=args.seed,
            samples_per_repeat=args.samples_per_repeat,
            repeats=args.repeats,
            phase_points=args.phase_points,
        )
        # --quick 仅在用户未显式覆盖默认值时改写采样规模。
        result = run_simulation(cfg, device=device, quick=args.quick)
        write_results(result, args.output_dir)
        return 0

    if args.command == "evaluate-tensor-file":
        if args.input_kind == "bf16" and args.pts_scale is not None:
            raise ValueError("pts-scale is incompatible with input-kind=bf16")
        obj = _torch_load(args.tensor)
        if not isinstance(obj, torch.Tensor):
            raise ValueError("tensor file must contain a single torch.Tensor")
        hif4 = HiF4Config(
            group_size=args.group_size,
            group_dim=args.group_dim,
            scale_mode=args.scale_mode,
        )
        weight = obj.to(device=device)
        if args.input_kind == "nvfp4_fake":
            result = evaluate_nvfp4_fake_weight(weight, pts_scale=args.pts_scale, hif4_config=hif4, return_reconstructions=False)
        else:
            result = evaluate_bf16_weight(weight, hif4_config=hif4)
        result = {
            "schema_version": SCHEMA_VERSION,
            "implementation": IMPLEMENTATION,
            "run_kind": "tensor",
            "input_kind": args.input_kind,
            "result": result,
        }
        write_results(result, args.output_dir)
        return 0

    if args.command == "evaluate-checkpoint":
        if args.group_size < 8 or args.group_size % 8 != 0:
            raise ValueError("group_size must be a positive multiple of 8")
        if args.chunk_groups <= 0 or args.token_batch_size <= 0:
            raise ValueError("chunk_groups and token_batch_size must be positive")
        hif4 = HiF4Config(
            group_size=args.group_size,
            group_dim=args.group_dim,
            scale_mode=args.scale_mode,
        )
        result = evaluate_checkpoint(
            args.checkpoint,
            input_kind=args.input_kind,
            pts_scales_path=args.pts_scales,
            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
            device=device,
            hif4_config=hif4,
            chunk_groups=args.chunk_groups,
            require_pts=args.require_pts,
            tensor_names=tuple(args.tensor_name),
            max_tensors=args.max_tensors,
            compute_group_summary=not args.skip_group_summary,
            activations_path=args.activations,
            token_batch_size=args.token_batch_size,
        )
        write_results(result, args.output_dir)
        return 0

    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
