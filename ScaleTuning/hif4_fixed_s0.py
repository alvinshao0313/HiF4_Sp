"""给定 S0 的 HiF4 重建，以及连续 S0 的 E6M2 STE。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHUANCI_DIR = _REPO_ROOT / "ChuanCi"
if str(_CHUANCI_DIR) not in sys.path:
    sys.path.insert(0, str(_CHUANCI_DIR))

from nvfp4_hif4_torch import (  # noqa: E402
    HiF4Config,
    HiF4Result,
    VALID_SCALE_MODES,
    _compute_reciprocal_scale,
    _move_groups_to_last,
    _restore_from_last,
    _validate_hif4_inputs,
    quantize_hif4,
    round_bfloat16,
    round_e6m2,
)

__all__ = [
    "apply_e6m2_ste",
    "init_s0_from_weight",
    "quantize_hif4_with_fixed_s0",
]


def apply_e6m2_ste(s0_continuous: torch.Tensor, *, scale_mode: str = "hardware") -> torch.Tensor:
    """连续 S0 → hardware/E6M2 前向，反向梯度走 continuous（STE）。"""
    if scale_mode not in VALID_SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {sorted(VALID_SCALE_MODES)}, got {scale_mode!r}")
    if scale_mode == "continuous":
        return s0_continuous
    if scale_mode == "bf16_math":
        s0_hw = round_bfloat16(s0_continuous)
    elif scale_mode == "e6m2_only":
        s0_hw = round_e6m2(s0_continuous)
    elif scale_mode == "hardware":
        s0_hw = round_e6m2(round_bfloat16(s0_continuous))
    else:
        raise ValueError(f"unsupported scale_mode: {scale_mode}")
    return s0_continuous + (s0_hw - s0_continuous).detach()


def init_s0_from_weight(
    weight: torch.Tensor,
    *,
    config: HiF4Config = HiF4Config(),
) -> torch.Tensor:
    """从权重做一次 hardware HiF4，取出 top_scale 作为 S0 初值。"""
    with torch.no_grad():
        result = quantize_hif4(weight.detach().to(torch.float32), config=config)
    return result.top_scale.detach().clone()


def quantize_hif4_with_fixed_s0(
    values: torch.Tensor,
    s0: torch.Tensor,
    *,
    config: HiF4Config = HiF4Config(),
    apply_ste: bool = True,
) -> HiF4Result:
    """用注入的 S0 做 HiF4 重建（e8/e4/payload 仍由当前权重与 S0 重算）。

    Args:
        values: 原始浮点权重。
        s0: 连续或已 STE 的顶层 scale，shape 与 quantize_hif4(...).top_scale 一致。
        config: HiF4 配置；scale_mode 决定 reciprocal / STE 语义。
        apply_ste: True 时对 s0 做 E6M2 STE 再进入量化。
    """
    _validate_hif4_inputs(values, config)
    x = values.to(torch.float32)
    moved, normalized_dim, moved_shape = _move_groups_to_last(x, config.group_dim)
    group_size = config.group_size
    groups_per_row = moved_shape[-1] // group_size
    num_groups = moved.numel() // group_size
    groups = moved.reshape(-1, group_size)

    leading = moved_shape[:-1]
    meta_prefix = leading + (groups_per_row,)
    expected_s0_shape = meta_prefix
    if tuple(s0.shape) != expected_s0_shape:
        raise ValueError(
            f"s0 shape {tuple(s0.shape)} != expected top_scale shape {expected_s0_shape} "
            f"for values shape {tuple(values.shape)} group_dim={config.group_dim} group_size={group_size}"
        )

    s0_in = s0.to(device=x.device, dtype=torch.float32)
    if apply_ste:
        s0_in = apply_e6m2_ste(s0_in, scale_mode=config.scale_mode)

    abs_g = groups.abs()
    amax64 = abs_g.amax(dim=-1)
    nonzero = amax64 > 0

    s0_flat = s0_in.reshape(-1)
    if s0_flat.numel() != num_groups:
        raise ValueError(f"s0 numel {s0_flat.numel()} != num_groups {num_groups}")
    safe_s0 = torch.where(nonzero, s0_flat, torch.ones_like(s0_flat))
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

    ratio = torch.floor(4.0 * abs_g * (reciprocal.unsqueeze(-1) / (2.0 ** (e8_elem + e4_elem))) + 0.5) / 4.0
    payload = torch.minimum(ratio, torch.full_like(ratio, 1.75))
    recon = groups.sign() * local_scale * payload
    recon = torch.where(nonzero.unsqueeze(-1), recon, torch.zeros_like(recon))

    recon_moved = recon.reshape(moved_shape)
    values_out = _restore_from_last(recon_moved, normalized_dim, values.ndim)

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
