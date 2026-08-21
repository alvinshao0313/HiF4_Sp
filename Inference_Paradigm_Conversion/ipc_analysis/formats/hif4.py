"""HiF4 weight/activation adapter: read-only reuse of ChuanCi.quantize_hif4."""

from __future__ import annotations

from typing import Any

import torch

from ChuanCi.nvfp4_hif4_torch import HiF4Config, quantize_hif4  # noqa: E402
from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import QuantizedTensorView

# Formal / counterfactual variants exposed for W3.
# "full" == hardware scale + s1p2 payload + exp8/exp4 (deployment HiF4).
VARIANT_CONFIGS: dict[str, HiF4Config] = {
    "full": HiF4Config(
        group_size=64,
        group_dim=-1,
        scale_mode="hardware",
        payload_format="s1p2",
        enable_exp8=True,
        enable_exp4=True,
    ),
    "continuous_s0": HiF4Config(
        group_size=64,
        group_dim=-1,
        scale_mode="continuous",
        payload_format="s1p2",
        enable_exp8=True,
        enable_exp4=True,
    ),
    "bf16_s0_no_e6m2": HiF4Config(
        group_size=64,
        group_dim=-1,
        scale_mode="bf16_math",
        payload_format="s1p2",
        enable_exp8=True,
        enable_exp4=True,
    ),
    "continuous_payload_clipped": HiF4Config(
        group_size=64,
        group_dim=-1,
        scale_mode="hardware",
        payload_format="bf16_range_matched",
        enable_exp8=True,
        enable_exp4=True,
    ),
    "rounded_payload_no_clip_probe": HiF4Config(
        group_size=64,
        group_dim=-1,
        scale_mode="hardware",
        payload_format="bf16_unclipped",
        enable_exp8=True,
        enable_exp4=True,
    ),
    "no_exp8": HiF4Config(
        group_size=64,
        group_dim=-1,
        scale_mode="hardware",
        payload_format="s1p2",
        enable_exp8=False,
        enable_exp4=True,
    ),
    "no_exp4": HiF4Config(
        group_size=64,
        group_dim=-1,
        scale_mode="hardware",
        payload_format="s1p2",
        enable_exp8=True,
        enable_exp4=False,
    ),
    "no_exp8_exp4": HiF4Config(
        group_size=64,
        group_dim=-1,
        scale_mode="hardware",
        payload_format="s1p2",
        enable_exp8=False,
        enable_exp4=False,
    ),
    "group16_full_hierarchy": HiF4Config(
        group_size=16,
        group_dim=-1,
        scale_mode="hardware",
        payload_format="s1p2",
        enable_exp8=True,
        enable_exp4=True,
    ),
    "group32_full_hierarchy": HiF4Config(
        group_size=32,
        group_dim=-1,
        scale_mode="hardware",
        payload_format="s1p2",
        enable_exp8=True,
        enable_exp4=True,
    ),
    "group64_full_hierarchy": HiF4Config(
        group_size=64,
        group_dim=-1,
        scale_mode="hardware",
        payload_format="s1p2",
        enable_exp8=True,
        enable_exp4=True,
    ),
}


def quantize_hif4_with_divisor(
    x: torch.Tensor,
    *,
    divisor: float,
    group_size: int = 64,
    group_dim: int = -1,
    output_dtype: torch.dtype = torch.bfloat16,
) -> QuantizedTensorView:
    """HiF4 QDQ with explicit S0 divisor (AX1 oracle search)."""
    cfg = HiF4Config(
        group_size=group_size,
        group_dim=group_dim,
        scale_mode="hardware",
        payload_format="s1p2",
        enable_exp8=True,
        enable_exp4=True,
        s0_divisor=divisor,
    )
    if x.shape[group_dim] % cfg.group_size != 0:
        raise ValueError(
            f"HiF4 group dim length {x.shape[group_dim]} must be divisible by {cfg.group_size}"
        )
    result = quantize_hif4(x.to(torch.float32), config=cfg)
    values_fp32 = result.values
    values_out = values_fp32.to(dtype=output_dtype)
    meta: dict[str, Any] = {
        "format": "hif4",
        "variant": "s0_divisor_oracle",
        "group_size": cfg.group_size,
        "group_dim": group_dim,
        "scale_mode": cfg.scale_mode,
        "payload_format": cfg.payload_format,
        "enable_exp8": cfg.enable_exp8,
        "enable_exp4": cfg.enable_exp4,
        "s0_divisor": divisor,
        "hadamard_runtime": "disabled",
        "top_scale": result.top_scale,
        "e1_per_8": result.e1_per_8,
        "e1_per_4": result.e1_per_4,
        "payload_magnitude": result.payload_magnitude,
        "local_scale": result.local_scale,
        "values_fp32": values_fp32,
    }
    return QuantizedTensorView(
        format_name="hif4",
        dequantized=values_out,
        source_shape=tuple(x.shape),
        metadata=meta,
    )


def quantize_hif4_tensor(
    x: torch.Tensor,
    group_dim: int = -1,
    variant: str = "full",
    output_dtype: torch.dtype = torch.bfloat16,
) -> QuantizedTensorView:
    """HiF4 QDQ. Weights/activations group along Linear K dim (last dim by default).

    Input for weights: source checkpoint weight cast to float32.
    Formal `full` uses group size 64. Counterfactual variants may change group size.
    """
    if variant not in VARIANT_CONFIGS:
        raise ValueError(
            f"unsupported HiF4 variant {variant!r}; "
            f"allowed={sorted(VARIANT_CONFIGS)}"
        )
    base = VARIANT_CONFIGS[variant]
    cfg = HiF4Config(
        group_size=base.group_size,
        group_dim=group_dim,
        scale_mode=base.scale_mode,
        payload_format=base.payload_format,
        hierarchy_format=base.hierarchy_format,
        enable_exp8=base.enable_exp8,
        enable_exp4=base.enable_exp4,
    )
    if x.shape[group_dim] % cfg.group_size != 0:
        raise ValueError(
            f"HiF4 group dim length {x.shape[group_dim]} must be divisible by {cfg.group_size}"
        )
    result = quantize_hif4(x.to(torch.float32), config=cfg)
    values_fp32 = result.values
    values_out = values_fp32.to(dtype=output_dtype)
    meta: dict[str, Any] = {
        "format": "hif4",
        "variant": variant,
        "group_size": cfg.group_size,
        "group_dim": group_dim,
        "scale_mode": cfg.scale_mode,
        "payload_format": cfg.payload_format,
        "enable_exp8": cfg.enable_exp8,
        "enable_exp4": cfg.enable_exp4,
        "hadamard_runtime": "disabled",
        "top_scale": result.top_scale,
        "e1_per_8": result.e1_per_8,
        "e1_per_4": result.e1_per_4,
        "payload_magnitude": result.payload_magnitude,
        "local_scale": result.local_scale,
        "values_fp32": values_fp32,
    }
    return QuantizedTensorView(
        format_name="hif4",
        dequantized=values_out,
        source_shape=tuple(x.shape),
        metadata=meta,
    )
