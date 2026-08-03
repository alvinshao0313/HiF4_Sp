"""Configurable HiF4 reference quantizer (S0 / e8 / e4 / S1P2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .formats import quantize_s1p2_magnitude, round_bfloat16, round_e6m2

S0Mode = Literal["continuous", "bf16_math", "e6m2", "hardware"]
VALID_S0_MODES = frozenset({"continuous", "bf16_math", "e6m2", "hardware"})


@dataclass(frozen=True)
class HiF4QuantConfig:
    group_size: int = 64
    group_dim: int = -1
    s0_divisor: float = 7.0
    e8_threshold: float = 4.0
    e4_threshold: float = 2.0
    s0_mode: S0Mode = "hardware"
    enable_exp8: bool = True
    enable_exp4: bool = True

    def __post_init__(self) -> None:
        if self.group_size < 8 or self.group_size % 8 != 0:
            raise ValueError("group_size must be >= 8 and divisible by 8")
        if self.s0_divisor <= 0:
            raise ValueError("s0_divisor must be positive")
        if self.e8_threshold <= 0 or self.e4_threshold <= 0:
            raise ValueError("thresholds must be positive")
        if self.s0_mode not in VALID_S0_MODES:
            raise ValueError(f"s0_mode must be one of {sorted(VALID_S0_MODES)}")


@dataclass
class HiF4QuantResult:
    reconstruction: torch.Tensor
    s0: torch.Tensor
    e8: torch.Tensor
    e4: torch.Tensor
    payload: torch.Tensor
    normalized: torch.Tensor
    local_scale: torch.Tensor


def _move_groups_to_last(
    values: torch.Tensor, group_dim: int
) -> tuple[torch.Tensor, int, tuple[int, ...]]:
    normalized_dim = group_dim % values.ndim
    moved = values.movedim(normalized_dim, -1).contiguous()
    return moved, normalized_dim, tuple(moved.shape)


def _restore_from_last(moved: torch.Tensor, normalized_dim: int) -> torch.Tensor:
    return moved.movedim(-1, normalized_dim)


def compute_s0(amax64: torch.Tensor, divisor: float, s0_mode: S0Mode) -> torch.Tensor:
    """Compute per-group top-level S0."""
    if s0_mode == "continuous":
        return amax64 / divisor
    if s0_mode == "bf16_math":
        reciprocal_divisor = round_bfloat16(
            torch.tensor(1.0 / divisor, device=amax64.device, dtype=torch.float32)
        )
        return round_bfloat16(amax64 * reciprocal_divisor)
    if s0_mode == "e6m2":
        return round_e6m2(amax64 / divisor)
    if s0_mode == "hardware":
        reciprocal_divisor = round_bfloat16(
            torch.tensor(1.0 / divisor, device=amax64.device, dtype=torch.float32)
        )
        bf16_ratio = round_bfloat16(amax64 * reciprocal_divisor)
        return round_e6m2(bf16_ratio)
    raise ValueError(f"unsupported s0_mode: {s0_mode}")


def compute_reciprocal_s0(s0: torch.Tensor, s0_mode: S0Mode) -> torch.Tensor:
    if s0_mode in {"bf16_math", "hardware"}:
        return round_bfloat16(1.0 / s0)
    return 1.0 / s0


def quantize_hif4(
    values: torch.Tensor,
    *,
    config: HiF4QuantConfig | None = None,
) -> HiF4QuantResult:
    """Full HiF4 path: S0 -> e8 -> e4 -> S1P2 -> reconstruction."""
    cfg = config or HiF4QuantConfig()
    if not values.is_floating_point():
        raise TypeError("values must be floating point")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    if cfg.group_dim < -values.ndim or cfg.group_dim >= values.ndim:
        raise ValueError("group_dim out of range")
    normalized_dim = cfg.group_dim % values.ndim
    if values.shape[normalized_dim] % cfg.group_size != 0:
        raise ValueError("grouped dimension length must be divisible by group_size")

    x = values.to(torch.float32)
    moved, normalized_dim, moved_shape = _move_groups_to_last(x, cfg.group_dim)
    group_size = cfg.group_size
    groups_per_row = moved_shape[-1] // group_size
    num_groups = moved.numel() // group_size
    groups = moved.reshape(-1, group_size)

    abs_g = groups.abs()
    amax64 = abs_g.amax(dim=-1)
    nonzero = amax64 > 0

    s0 = compute_s0(amax64, cfg.s0_divisor, cfg.s0_mode)
    safe_s0 = torch.where(nonzero, s0, torch.ones_like(s0))
    reciprocal = compute_reciprocal_s0(safe_s0, cfg.s0_mode)

    blocks_per_group = group_size // 8
    abs_8 = abs_g.reshape(num_groups, blocks_per_group, 8)
    amax8 = abs_8.amax(dim=-1)
    abs_4 = abs_g.reshape(num_groups, group_size // 4, 4)
    amax4 = abs_4.amax(dim=-1)

    if cfg.enable_exp8:
        e8 = (amax8 * reciprocal.unsqueeze(-1) >= cfg.e8_threshold).to(torch.float32)
    else:
        e8 = torch.zeros_like(amax8)
    e8_per4 = e8.repeat_interleave(2, dim=-1)
    if cfg.enable_exp4:
        e4 = (
            amax4 * reciprocal.unsqueeze(-1) / (2.0**e8_per4) >= cfg.e4_threshold
        ).to(torch.float32)
    else:
        e4 = torch.zeros_like(amax4)

    e8_elem = e8.repeat_interleave(8, dim=-1)
    e4_elem = e4.repeat_interleave(4, dim=-1)
    local_scale = safe_s0.unsqueeze(-1) * (2.0 ** (e8_elem + e4_elem))

    normalized = abs_g * (reciprocal.unsqueeze(-1) / (2.0 ** (e8_elem + e4_elem)))
    payload = quantize_s1p2_magnitude(normalized)
    recon = groups.sign() * local_scale * payload
    recon = torch.where(nonzero.unsqueeze(-1), recon, torch.zeros_like(recon))

    leading = moved_shape[:-1]
    meta_prefix = leading + (groups_per_row,)

    recon_out = _restore_from_last(recon.reshape(moved_shape), normalized_dim)
    payload_out = _restore_from_last(payload.reshape(moved_shape), normalized_dim)
    local_out = _restore_from_last(local_scale.reshape(moved_shape), normalized_dim)
    norm_out = _restore_from_last(normalized.reshape(moved_shape), normalized_dim)

    return HiF4QuantResult(
        reconstruction=recon_out.to(dtype=values.dtype),
        s0=safe_s0.reshape(meta_prefix),
        e8=e8.reshape(meta_prefix + (blocks_per_group,)),
        e4=e4.reshape(meta_prefix + (group_size // 4,)),
        payload=payload_out,
        normalized=norm_out,
        local_scale=local_out,
    )


def quantize_hif4_with_fixed_s0(
    values: torch.Tensor,
    s0: torch.Tensor,
    *,
    config: HiF4QuantConfig | None = None,
) -> HiF4QuantResult:
    """Quantize using externally supplied per-group S0 (same layout as quantize_hif4.s0)."""
    cfg = config or HiF4QuantConfig()
    x = values.to(torch.float32)
    moved, normalized_dim, moved_shape = _move_groups_to_last(x, cfg.group_dim)
    group_size = cfg.group_size
    groups_per_row = moved_shape[-1] // group_size
    num_groups = moved.numel() // group_size
    groups = moved.reshape(-1, group_size)

    abs_g = groups.abs()
    amax64 = abs_g.amax(dim=-1)
    nonzero = amax64 > 0

    s0_flat = s0.reshape(-1).to(dtype=torch.float32, device=groups.device)
    if s0_flat.numel() != num_groups:
        raise ValueError(f"s0 has {s0_flat.numel()} groups, expected {num_groups}")
    safe_s0 = torch.where(nonzero, s0_flat, torch.ones_like(s0_flat))
    reciprocal = compute_reciprocal_s0(safe_s0, cfg.s0_mode)

    blocks_per_group = group_size // 8
    abs_8 = abs_g.reshape(num_groups, blocks_per_group, 8)
    amax8 = abs_8.amax(dim=-1)
    abs_4 = abs_g.reshape(num_groups, group_size // 4, 4)
    amax4 = abs_4.amax(dim=-1)

    if cfg.enable_exp8:
        e8 = (amax8 * reciprocal.unsqueeze(-1) >= cfg.e8_threshold).to(torch.float32)
    else:
        e8 = torch.zeros_like(amax8)
    e8_per4 = e8.repeat_interleave(2, dim=-1)
    if cfg.enable_exp4:
        e4 = (
            amax4 * reciprocal.unsqueeze(-1) / (2.0**e8_per4) >= cfg.e4_threshold
        ).to(torch.float32)
    else:
        e4 = torch.zeros_like(amax4)

    e8_elem = e8.repeat_interleave(8, dim=-1)
    e4_elem = e4.repeat_interleave(4, dim=-1)
    local_scale = safe_s0.unsqueeze(-1) * (2.0 ** (e8_elem + e4_elem))
    normalized = abs_g * (reciprocal.unsqueeze(-1) / (2.0 ** (e8_elem + e4_elem)))
    payload = quantize_s1p2_magnitude(normalized)
    recon = groups.sign() * local_scale * payload
    recon = torch.where(nonzero.unsqueeze(-1), recon, torch.zeros_like(recon))

    leading = moved_shape[:-1]
    meta_prefix = leading + (groups_per_row,)
    return HiF4QuantResult(
        reconstruction=_restore_from_last(recon.reshape(moved_shape), normalized_dim).to(
            dtype=values.dtype
        ),
        s0=safe_s0.reshape(meta_prefix),
        e8=e8.reshape(meta_prefix + (blocks_per_group,)),
        e4=e4.reshape(meta_prefix + (group_size // 4,)),
        payload=_restore_from_last(payload.reshape(moved_shape), normalized_dim),
        normalized=_restore_from_last(normalized.reshape(moved_shape), normalized_dim),
        local_scale=_restore_from_last(local_scale.reshape(moved_shape), normalized_dim),
    )
