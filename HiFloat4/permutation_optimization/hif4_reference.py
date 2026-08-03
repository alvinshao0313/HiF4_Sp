"""HiF4 reference wrappers: S1P2 oracle and real hifx4 fake quant."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure HiFloat4 root is importable when this package is used standalone.
_HIFLOAT4_ROOT = Path(__file__).resolve().parents[1]
if str(_HIFLOAT4_ROOT) not in sys.path:
    sys.path.insert(0, str(_HIFLOAT4_ROOT))

from hif4_gpu.quant_cy import QType, quant_dequant_float  # noqa: E402

_S1P2_MAX = 1.75
_S1P2_STEP = 0.25
_HIFX4 = QType("hifx4").dim(-1)


def s1p2_oracle_quantize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Fake-quantize along the last dim with continuous S1P2 scale.

    Last dimension may be 1..4. Zero rows stay zero (no NaN).
    Internal math is FP32; output dtype matches input.
    """
    if x.ndim < 1:
        raise ValueError(f"x must have at least 1 dim, got shape {tuple(x.shape)}")
    last = x.shape[-1]
    if last < 1 or last > 4:
        raise ValueError(f"last dim must be in 1..4 for S1P2 oracle, got {last}")

    orig_dtype = x.dtype
    xf = x.detach().to(torch.float32)
    abs_x = xf.abs()
    row_max = abs_x.amax(dim=-1, keepdim=True)
    zero_row = row_max <= eps

    scale = row_max / _S1P2_MAX
    # Avoid division by zero on zero rows; result overwritten below.
    safe_scale = torch.where(zero_row, torch.ones_like(scale), scale)
    normalized = xf / safe_scale
    quantized = torch.round(normalized / _S1P2_STEP) * _S1P2_STEP
    quantized = quantized.clamp(-_S1P2_MAX, _S1P2_MAX)
    dequantized = quantized * safe_scale
    dequantized = torch.where(zero_row.expand_as(dequantized), torch.zeros_like(dequantized), dequantized)
    return dequantized.to(dtype=orig_dtype)


def hif4_fake_quantize(x: torch.Tensor) -> torch.Tensor:
    """hifx4 fake quant; last dim must be divisible by 64.

    Fast path: CUDA FP32 kernel (bit-exact vs ``force_py=True, force_fp32=True``).
    CPU fallback uses the Python reference path.
    """
    if x.shape[-1] % 64 != 0:
        raise ValueError(
            f"hif4_fake_quantize requires last dim divisible by 64, got {x.shape[-1]}"
        )
    orig_device = x.device
    orig_dtype = x.dtype
    if torch.cuda.is_available():
        xc = x.detach()
        if not xc.is_cuda:
            xc = xc.to(device="cuda", dtype=torch.float32)
        else:
            xc = xc.to(dtype=torch.float32)
        out = quant_dequant_float(xc.contiguous(), _HIFX4, force_py=False, force_fp32=True)
        return out.to(device=orig_device, dtype=orig_dtype)
    return quant_dequant_float(
        x.detach().to(dtype=torch.float32), _HIFX4, force_py=True, force_fp32=True
    ).to(dtype=orig_dtype)
