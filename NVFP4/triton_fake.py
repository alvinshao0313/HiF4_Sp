"""Triton NVFP4 fake-quant kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_DTYPE_TO_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}
_CODE_TO_DTYPE = {value: key for key, value in _DTYPE_TO_CODE.items()}


@triton.jit
def _round_nearest_even(x):
    floored = tl.floor(x)
    frac = x - floored
    half = frac == 0.5
    odd = (floored - 2.0 * tl.floor(floored * 0.5)) == 1.0
    return tl.where((frac > 0.5) | (half & odd), floored + 1.0, floored)


@triton.jit
def _cast_to_fp8_e4m3fn_tl(x):
    sign = tl.where(x < 0.0, -1.0, 1.0)
    abs_x = tl.minimum(tl.abs(x), 448.0)
    safe_abs_x = tl.where(abs_x == 0.0, 1.0, abs_x)
    exponent = tl.floor(tl.log2(safe_abs_x))
    exponent = tl.minimum(tl.maximum(exponent, -6.0), 8.0)
    step = tl.exp2(exponent - 3.0)
    rounded = _round_nearest_even(abs_x / step) * step
    rounded = tl.minimum(rounded, 448.0)
    return rounded * sign


@triton.jit
def _cast_to_fp4_e2m1_tl(x):
    sign = tl.where(x < 0.0, -1.0, 1.0)
    abs_x = tl.abs(x)
    out = tl.full(abs_x.shape, 6.0, tl.float32)
    out = tl.where(abs_x <= 0.25, 0.0, out)
    out = tl.where((abs_x > 0.25) & (abs_x < 0.75), 0.5, out)
    out = tl.where((abs_x >= 0.75) & (abs_x <= 1.25), 1.0, out)
    out = tl.where((abs_x > 1.25) & (abs_x < 1.75), 1.5, out)
    out = tl.where((abs_x >= 1.75) & (abs_x <= 2.5), 2.0, out)
    out = tl.where((abs_x > 2.5) & (abs_x < 3.5), 3.0, out)
    out = tl.where((abs_x >= 3.5) & (abs_x <= 5.0), 4.0, out)
    return out * sign


@triton.jit
def _fake_quant_nvfp4_activation_triton_kernel(
    x_ptr,
    global_scale_ptr,
    out_ptr,
    group_size: tl.constexpr,
    block_size: tl.constexpr,
):
    group_id = tl.program_id(0)
    group_offsets = tl.arange(0, block_size)
    mask = group_offsets < group_size
    offsets = group_id * group_size + group_offsets

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    global_scale = tl.load(global_scale_ptr).to(tl.float32)

    amax = tl.max(tl.abs(x), axis=0)
    block_scale = global_scale * (amax / 6.0)
    block_scale = tl.minimum(tl.maximum(block_scale, -448.0), 448.0)
    block_scale = _cast_to_fp8_e4m3fn_tl(block_scale)

    safe_block_scale = tl.where(block_scale == 0.0, 1.0, block_scale)
    output_scale = tl.where(block_scale == 0.0, 0.0, global_scale / safe_block_scale)
    scaled = x * output_scale
    scaled = tl.minimum(tl.maximum(scaled, -6.0), 6.0)
    x_fp4 = _cast_to_fp4_e2m1_tl(scaled)

    safe_global_scale = tl.where(global_scale == 0.0, 1.0, global_scale)
    dequant_scale = tl.where(global_scale == 0.0, 0.0, block_scale / safe_global_scale)
    x_dequant = x_fp4 * dequant_scale
    tl.store(out_ptr + offsets, x_dequant, mask=mask)


def _dtype_to_code(dtype: torch.dtype) -> int:
    if dtype not in _DTYPE_TO_CODE:
        raise TypeError(f"Unsupported NVFP4 fake activation output dtype: {dtype}")
    return _DTYPE_TO_CODE[dtype]


def _code_to_dtype(dtype_code: int) -> torch.dtype:
    if dtype_code not in _CODE_TO_DTYPE:
        raise TypeError(f"Unsupported NVFP4 fake activation output dtype code: {dtype_code}")
    return _CODE_TO_DTYPE[dtype_code]


@torch.library.custom_op("nvfp4::fake_quant_activation_triton", mutates_args=())
def _fake_quant_nvfp4_activation_triton_op(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int,
    output_dtype_code: int,
) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("fake_quant_nvfp4_activation kernel path requires a CUDA input tensor")
    if global_scale.numel() != 1:
        raise ValueError(
            "Only scalar per-tensor input_global_scale is supported; "
            f"got shape {tuple(global_scale.shape)}"
        )

    output_dtype = _code_to_dtype(output_dtype_code)
    x_contiguous = x.contiguous()
    output = torch.empty(x_contiguous.shape, device=x.device, dtype=output_dtype)
    if x_contiguous.numel() == 0:
        return output

    block_size = triton.next_power_of_2(group_size)
    grid = (triton.cdiv(x_contiguous.numel(), group_size),)
    _fake_quant_nvfp4_activation_triton_kernel[grid](
        x_contiguous,
        global_scale,
        output,
        group_size,
        block_size,
    )
    return output


@_fake_quant_nvfp4_activation_triton_op.register_fake
def _(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int,
    output_dtype_code: int,
) -> torch.Tensor:
    return torch.empty(x.shape, device=x.device, dtype=_code_to_dtype(output_dtype_code))


def fake_quant_nvfp4_activation_triton(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run the Triton NVFP4 activation fake quant-dequant path."""
    return _fake_quant_nvfp4_activation_triton_op(
        x,
        global_scale,
        group_size,
        _dtype_to_code(output_dtype),
    )
