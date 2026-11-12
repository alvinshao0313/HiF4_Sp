"""Decode checkpoint values in FP32, round once to BF16, and use BF16 GEMM.

This is weight-only emulation: activation and KV-cache quantization are absent.
The FP4 order is low nibble first. Scale layouts are checkpoint (linear) layouts.
"""
import torch
import torch.nn.functional as F


def e4m3_values(raw: torch.Tensor) -> torch.Tensor:
    if raw.dtype == torch.float8_e4m3fn:
        raw = raw.view(torch.uint8)
    if raw.dtype != torch.uint8:
        raise TypeError(f"Expected E4M3 bytes, got {raw.dtype}")
    bits = raw.to(torch.int32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    value = torch.where(
        exponent == 0,
        mantissa.float() * (2.0 ** -9),
        (1 + mantissa.float() / 8) * torch.exp2(exponent.float() - 7),
    )
    value = torch.where((bits & 128) != 0, -value, value)
    return torch.where((bits & 127) == 127, float("nan"), value)


def nvfp4(weight, scale, global_scale, *, divisor=False, dtype=torch.bfloat16):
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise ValueError("Packed FP4 must be a uint8 matrix")
    rows, half_cols = weight.shape
    if half_cols % 8 or scale.shape != (rows, half_cols // 8):
        raise ValueError(f"Invalid NVFP4 shapes: {weight.shape}, {scale.shape}")
    if global_scale.numel() != 1 or not bool(torch.isfinite(global_scale).all()) or not bool((global_scale > 0).all()):
        raise ValueError("Expected one finite positive global scale")
    if scale.device != weight.device or global_scale.device != weight.device:
        raise ValueError("Weight and scales must share a device")
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).reshape(rows, half_cols * 2)
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=weight.device, dtype=torch.float32)
    values = table[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -values, values)
    scales = e4m3_values(scale)
    g = global_scale.float().reshape(())
    scales = scales / g if divisor else scales * g
    result = (values.reshape(rows, -1, 16) * scales.unsqueeze(-1)).reshape(rows, -1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Nonfinite decoded NVFP4 weights")
    return result.to(dtype)


def mxfp8(weight, scale, *, dtype=torch.bfloat16):
    if weight.ndim != 2 or weight.shape[-1] % 32:
        raise ValueError("MXFP8 requires a matrix with columns divisible by 32")
    if scale.dtype != torch.uint8 or scale.shape != (weight.shape[0], weight.shape[1] // 32):
        raise ValueError("Invalid MXFP8 E8M0 scale layout")
    if bool((scale == 255).any()):
        raise ValueError("E8M0 code 255 denotes NaN")
    values = e4m3_values(weight).reshape(weight.shape[0], -1, 32)
    scales = torch.exp2(scale.float() - 127)
    result = (values * scales.unsqueeze(-1)).reshape(weight.shape)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Nonfinite decoded MXFP8 weights")
    return result.to(dtype)


def linear(inputs, decoded_weight, bias=None):
    if inputs.dtype != torch.bfloat16 or decoded_weight.dtype != torch.bfloat16:
        raise TypeError("Weight-only emulation GEMM expects BF16 inputs and weights")
    return F.linear(inputs, decoded_weight, bias)
