"""Inference-only NVFP4 / FP8 emulation; quantized operands remain BF16."""

from .activation_quant import (
    fp8_fake_quant,
    fp8_reference,
    nvfp4_fake_quant,
    nvfp4_reference,
)
from .checkpoint_quant import (
    e4m3_values,
    linear as bf16_linear,
    nvfp4 as dequantize_nvfp4,
)

__version__ = "0.1.0"
__all__ = [
    "nvfp4_fake_quant",
    "fp8_fake_quant",
    "nvfp4_reference",
    "fp8_reference",
    "dequantize_nvfp4",
    "e4m3_values",
    "bf16_linear",
]
