"""NVFP4 fake-quant helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from NVFP4.triton_fake import fake_quant_nvfp4_activation_triton


FP4_E2M1_MAX = 6.0
FP8_E4M3FN_MAX = 448.0
FP8_E4M3FN_MIN_EXP = -6
FP8_E4M3FN_MAX_EXP = 8
FP8_E4M3FN_MANTISSA_BITS = 3
USE_TRITON_NVFP4_KERNEL = True


def cast_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Map values to the NVFP4 E2M1 value grid."""
    sign = torch.sign(x)
    abs_x = torch.abs(x)

    abs_x_f32 = abs_x.to(torch.float32)
    out = torch.where(abs_x_f32 <= 0.25, 0.0, 6.0)
    out = torch.where((abs_x_f32 > 0.25) & (abs_x_f32 < 0.75), 0.5, out)
    out = torch.where((abs_x_f32 >= 0.75) & (abs_x_f32 <= 1.25), 1.0, out)
    out = torch.where((abs_x_f32 > 1.25) & (abs_x_f32 < 1.75), 1.5, out)
    out = torch.where((abs_x_f32 >= 1.75) & (abs_x_f32 <= 2.5), 2.0, out)
    out = torch.where((abs_x_f32 > 2.5) & (abs_x_f32 < 3.5), 3.0, out)
    out = torch.where((abs_x_f32 >= 3.5) & (abs_x_f32 <= 5.0), 4.0, out)
    return (out * sign.to(torch.float32)).to(x.dtype)


def cast_to_fp8_e4m3fn(x: torch.Tensor) -> torch.Tensor:
    """Saturating E4M3FN cast emulation without using torch.float8 dtypes."""
    sign = torch.sign(x)
    abs_x = torch.abs(x).to(torch.float32)
    abs_x = torch.clamp(abs_x, max=FP8_E4M3FN_MAX)

    safe_abs_x = torch.where(abs_x == 0, torch.ones_like(abs_x), abs_x)
    exponent = torch.floor(torch.log2(safe_abs_x))
    exponent = torch.clamp(
        exponent,
        min=FP8_E4M3FN_MIN_EXP,
        max=FP8_E4M3FN_MAX_EXP,
    )
    step = torch.pow(2.0, exponent - FP8_E4M3FN_MANTISSA_BITS)
    rounded = torch.round(abs_x / step) * step
    rounded = torch.clamp(rounded, max=FP8_E4M3FN_MAX)
    return (rounded * sign.to(torch.float32)).to(x.dtype)


def _fake_quant_nvfp4_activation_torch(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    original_shape = x.shape
    hidden_size = original_shape[-1]
    x_2d = x.reshape(-1, hidden_size).to(torch.float32)
    x_grouped = x_2d.reshape(x_2d.shape[0], hidden_size // group_size, group_size)

    amax = x_grouped.abs().amax(dim=-1, keepdim=True)
    block_scale = global_scale * (amax / FP4_E2M1_MAX)
    block_scale = torch.clamp(block_scale, min=-448.0, max=448.0)
    block_scale = cast_to_fp8_e4m3fn(block_scale).to(torch.float32)

    output_scale = torch.where(
        block_scale == 0,
        torch.zeros_like(block_scale),
        global_scale / block_scale,
    )
    scaled = x_grouped * output_scale
    scaled = torch.clamp(scaled, min=-FP4_E2M1_MAX, max=FP4_E2M1_MAX)
    x_fp4 = cast_to_fp4_e2m1(scaled)
    dequant_scale = torch.where(
        global_scale == 0,
        torch.zeros_like(block_scale),
        block_scale / global_scale,
    )
    x_dequant = x_fp4 * dequant_scale
    return x_dequant.reshape(original_shape).to(output_dtype)


def _fake_quant_nvfp4_activation_kernel(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("fake_quant_nvfp4_activation kernel path requires a CUDA input tensor")

    return fake_quant_nvfp4_activation_triton(
        x,
        global_scale,
        group_size=group_size,
        output_dtype=output_dtype,
    )


def fake_quant_nvfp4_activation(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    group_size: int = 16,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fake quant-dequant activations with NVFP4 A4 rules."""
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"Last dimension {x.shape[-1]} must be divisible by group_size={group_size}"
        )
    if input_global_scale.numel() != 1:
        raise ValueError(
            "Only scalar per-tensor input_global_scale is supported; "
            f"got shape {tuple(input_global_scale.shape)}"
        )
    if output_dtype is None:
        output_dtype = x.dtype

    global_scale = input_global_scale.reshape(()).to(device=x.device, dtype=torch.float32)

    if USE_TRITON_NVFP4_KERNEL:
        return _fake_quant_nvfp4_activation_kernel(
            x,
            global_scale,
            group_size=group_size,
            output_dtype=output_dtype,
        )
    return _fake_quant_nvfp4_activation_torch(
        x,
        global_scale,
        group_size=group_size,
        output_dtype=output_dtype,
    )


class NvFp4FakeLinear(torch.nn.Module):
    """Linear layer with NVFP4 activation fake quant-dequant before GEMM."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        input_global_scale: torch.Tensor,
        group_size: int = 16,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got {weight.ndim}D")
        if weight.shape[1] % group_size != 0:
            raise ValueError(
                f"weight input dimension {weight.shape[1]} must be divisible by "
                f"group_size={group_size}"
            )
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
        self.input_global_scale = torch.nn.Parameter(
            input_global_scale.reshape(()).to(torch.float32), requires_grad=False
        )
        self.group_size = group_size
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dequant = fake_quant_nvfp4_activation(
            x,
            self.input_global_scale,
            group_size=self.group_size,
            output_dtype=self.output_dtype or x.dtype,
        )
        return F.linear(x_dequant, self.weight, self.bias)
