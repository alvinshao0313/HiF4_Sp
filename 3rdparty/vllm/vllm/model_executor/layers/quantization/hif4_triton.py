# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton implementation of HiF4 hifx4 quantize-dequantize.

The numerical layout matches ``HiFloat4/hif4_gpu`` and the local
``hif4_fake.hif4_fake_quantize_hifx4`` reference:

* one level-0 scale per 64 contiguous values on the last dimension;
* eight level-1 2x shared exponent multipliers per group;
* sixteen level-2 2x shared exponent multipliers per group;
* 2-bit fractional mantissa (grid step 0.25, saturated at 1.75).

This kernel intentionally returns dequantized values in the input dtype. The
materialized HiF4 checkpoints keep quantized weights in BF16, so this is the
activation-side QDQ used by the emulation runtime.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

_HIF4_GROUP_SIZE = 64


@triton.jit
def _round_nearest_even_positive(x):
    """Round non-negative fp32 values to nearest integer, ties to even."""
    floored = tl.floor(x)
    frac = x - floored
    half = frac == 0.5
    odd = (floored - 2.0 * tl.floor(floored * 0.5)) == 1.0
    return tl.where((frac > 0.5) | (half & odd), floored + 1.0, floored)


@triton.jit
def _hif4_fake_quant_block_tl(
    x,
    valid_mask,
    BLOCK_DIMS: tl.constexpr,
):
    """64-value HiF4 block math shared with the KV-cache Triton path."""
    offsets = tl.arange(0, BLOCK_DIMS)
    abs_x = tl.abs(tl.where(valid_mask, x, 0.0))
    max_lv1 = tl.max(abs_x, axis=0)

    max_lv2 = tl.zeros((BLOCK_DIMS,), tl.float32)
    for group_start in tl.static_range(0, BLOCK_DIMS, 8):
        group_mask = (offsets >= group_start) & (offsets < group_start + 8)
        group_max = tl.max(
            tl.where(group_mask & valid_mask, abs_x, 0.0), axis=0
        )
        max_lv2 = tl.where(group_mask, group_max, max_lv2)

    max_lv3 = tl.zeros((BLOCK_DIMS,), tl.float32)
    for group_start in tl.static_range(0, BLOCK_DIMS, 4):
        group_mask = (offsets >= group_start) & (offsets < group_start + 4)
        group_max = tl.max(
            tl.where(group_mask & valid_mask, abs_x, 0.0), axis=0
        )
        max_lv3 = tl.where(group_mask, group_max, max_lv3)

    div7 = tl.full((), 1.0 / 7.0, tl.float32).to(tl.bfloat16).to(tl.float32)
    scale_factor = (max_lv1 * div7).to(tl.bfloat16).to(tl.float32)
    scale_factor = tl.minimum(
        tl.maximum(scale_factor, 3.552713678800501e-15), 49152.0
    )
    exp_sf = tl.floor(tl.log2(scale_factor))
    mant_sf = scale_factor / tl.exp2(exp_sf) * 128.0
    scale_factor = (
        _round_nearest_even_positive(mant_sf) / 128.0 * tl.exp2(exp_sf)
    )
    exp_sf = tl.floor(tl.log2(scale_factor))
    scale_factor = (
        _round_nearest_even_positive(scale_factor * tl.exp2(2.0 - exp_sf))
        * tl.exp2(exp_sf - 2.0)
    )
    rec_sf = (1.0 / scale_factor).to(tl.bfloat16).to(tl.float32)

    scale_lv2 = tl.exp2(
        tl.floor(
            tl.minimum(tl.maximum(max_lv2 * rec_sf, 0.0), 4.0) / 4.0
        )
    )
    scale_lv3 = tl.exp2(
        tl.floor(
            tl.minimum(
                tl.maximum(max_lv3 * rec_sf / scale_lv2, 0.0), 2.0
            )
            / 2.0
        )
    )
    mant = abs_x / scale_lv2 / scale_lv3 * rec_sf
    mant = tl.floor(mant * 4.0 + 0.5) / 4.0
    mant = tl.minimum(mant, 1.75)
    sign = tl.where(x > 0.0, 1.0, tl.where(x < 0.0, -1.0, 0.0))
    return sign * mant * scale_lv2 * scale_lv3 * scale_factor


@triton.jit
def _hif4_hifx4_qdq_kernel(
    x_ptr,
    out_ptr,
    cols: tl.constexpr,
    groups_per_row: tl.constexpr,
    BLOCK_DIMS: tl.constexpr,
):
    """Quantize one last-dimension 64-value group per Triton program."""
    pid = tl.program_id(axis=0)
    row = pid // groups_per_row
    group = pid - row * groups_per_row
    offsets = tl.arange(0, BLOCK_DIMS)
    col = group * BLOCK_DIMS + offsets
    valid = col < cols
    x = tl.load(x_ptr + row * cols + col, mask=valid, other=0.0).to(
        tl.float32
    )
    out = _hif4_fake_quant_block_tl(x, valid, BLOCK_DIMS)
    tl.store(out_ptr + row * cols + col, out, mask=valid)


def hif4_quantize_hifx4_triton(
    x: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply HiF4 hifx4 QDQ along the last dimension on CUDA.

    ``out`` may alias ``x``. Each Triton program owns one 64-value group, so an
    in-place call is safe and is used for the post-SwiGLU MoE workspace.
    """
    if not x.is_cuda:
        raise ValueError("HiF4 Triton QDQ requires a CUDA tensor")
    if x.ndim == 0:
        raise ValueError("HiF4 QDQ requires at least one dimension")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported HiF4 QDQ dtype: {x.dtype}")

    work = x if x.is_contiguous() else x.contiguous()
    cols = int(work.shape[-1])
    if cols == 0 or work.numel() == 0:
        if out is None:
            return work.clone()
        out.copy_(work)
        return out

    rows = work.numel() // cols
    groups_per_row = (cols + _HIF4_GROUP_SIZE - 1) // _HIF4_GROUP_SIZE

    if out is None:
        result = torch.empty_like(work)
    else:
        if out.shape != work.shape:
            raise ValueError(f"HiF4 QDQ out shape mismatch: {out.shape} != {work.shape}")
        if out.dtype != work.dtype or out.device != work.device:
            raise ValueError("HiF4 QDQ out must match input dtype and device")
        if not out.is_contiguous():
            raise ValueError("HiF4 QDQ out must be contiguous")
        if work.data_ptr() != x.data_ptr() and out.data_ptr() == x.data_ptr():
            raise ValueError("cannot write in-place when input required a contiguous copy")
        result = out

    grid = (rows * groups_per_row,)
    _hif4_hifx4_qdq_kernel[grid](
        work,
        result,
        cols=cols,
        groups_per_row=groups_per_row,
        BLOCK_DIMS=_HIF4_GROUP_SIZE,
        num_warps=4,
    )
    return result
