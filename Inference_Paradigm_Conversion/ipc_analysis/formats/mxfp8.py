"""MXFP8 activation Oracle: OCP MXFP8 E4M3 block-32 along last dim."""

from __future__ import annotations

from typing import Any

import torch

from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import QuantizedTensorView

# Reuse the same E4M3 cast as NVFP4 path for consistency.
from NVFP4.torch_fake import FP8_E4M3FN_MAX, cast_to_fp8_e4m3fn  # noqa: E402

MXFP8_BLOCK_SIZE = 32


def _quantize_mxfp8_last_dim(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Software Oracle MXFP8 QDQ along last dimension (block size 32)."""
    if x.shape[-1] % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP8 requires last dim divisible by {MXFP8_BLOCK_SIZE}, got {x.shape[-1]}"
        )
    original_shape = x.shape
    xf = x.to(torch.float32)
    grouped = xf.reshape(*original_shape[:-1], -1, MXFP8_BLOCK_SIZE)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    # E8M0 shared scale: ceil(log2(amax / max_e4m3))
    scale_exp = torch.ceil(torch.log2(amax / FP8_E4M3FN_MAX))
    scale_exp = torch.clamp(scale_exp, min=-127.0, max=127.0)
    scale = torch.where(amax == 0, torch.ones_like(amax), torch.exp2(scale_exp))
    quant = cast_to_fp8_e4m3fn(grouped / scale).to(torch.float32)
    dequant = (quant * scale).reshape(original_shape)
    meta = {
        "block_size": MXFP8_BLOCK_SIZE,
        "e8m0_scale_exp": scale_exp.squeeze(-1),
        "e8m0_scale": scale.squeeze(-1),
        "e4m3_payload": quant,
        "block_amax": amax.squeeze(-1),
    }
    return dequant, meta


def quantize_mxfp8_activation(
    x_bf16: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> QuantizedTensorView:
    """Fair MXFP8 QDQ on the same BF16 pre-quant activation as NVFP4/HiF4."""
    if x_bf16.dtype != torch.bfloat16:
        raise TypeError(f"x_bf16 must be bfloat16, got {x_bf16.dtype}")
    dequant_fp32, meta = _quantize_mxfp8_last_dim(x_bf16)
    dequant = dequant_fp32.to(dtype=output_dtype)
    meta.update(
        {
            "format": "mxfp8",
            "hadamard_runtime": "disabled",
            "elem_format": "fp8_e4m3",
            "shared_scale_format": "e8m0",
        }
    )
    return QuantizedTensorView(
        format_name="mxfp8_activation",
        dequantized=dequant,
        source_shape=tuple(x_bf16.shape),
        metadata=meta,
    )
