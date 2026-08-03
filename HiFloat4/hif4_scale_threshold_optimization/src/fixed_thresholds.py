"""Fixed threshold baseline presets."""

from __future__ import annotations

from .quantizer import HiF4QuantConfig

FIXED_BASELINES: dict[str, HiF4QuantConfig] = {
    "standard": HiF4QuantConfig(
        s0_divisor=7.0,
        e8_threshold=4.0,
        e4_threshold=2.0,
        s0_mode="hardware",
    ),
    "scalar_mse": HiF4QuantConfig(
        s0_divisor=7.0,
        e8_threshold=3.75,
        e4_threshold=1.875,
        s0_mode="hardware",
    ),
    "no_clip": HiF4QuantConfig(
        s0_divisor=7.0,
        e8_threshold=3.5,
        e4_threshold=1.75,
        s0_mode="hardware",
    ),
}


def get_baseline_config(name: str) -> HiF4QuantConfig:
    if name not in FIXED_BASELINES:
        raise KeyError(f"unknown baseline {name!r}; known={sorted(FIXED_BASELINES)}")
    return FIXED_BASELINES[name]
