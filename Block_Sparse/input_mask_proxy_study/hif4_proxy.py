from __future__ import annotations

from dataclasses import dataclass

import torch

_E6M2_MIN = 2.0**-48
_E6M2_MAX = 1.5 * (2.0**15)
_GROUP = 64


def bf16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


@dataclass(frozen=True)
class HiF4ProxyResult:
    proxy: torch.Tensor
    hif4_dequant: torch.Tensor
    local_scale: torch.Tensor
    s0: torch.Tensor
    e8: torch.Tensor
    e4: torch.Tensor
    payload: torch.Tensor
    ternary_code: torch.Tensor


def quantize_e6m2_nonnegative(x: torch.Tensor) -> torch.Tensor:
    if not x.is_floating_point():
        raise TypeError(f"x must be floating point, got {x.dtype}")
    xf = x.to(torch.float32)
    if bool((xf < 0).any().item()):
        raise ValueError("quantize_e6m2_nonnegative rejects negative values")
    if not bool(torch.isfinite(xf).all().item()):
        raise ValueError("quantize_e6m2_nonnegative rejects NaN/Inf")

    out = torch.zeros_like(xf)
    nonzero = xf > 0
    if not bool(nonzero.any().item()):
        return out

    z = xf[nonzero].clamp(min=_E6M2_MIN, max=_E6M2_MAX)
    e = torch.floor(torch.log2(z))
    scaled = z * torch.pow(torch.tensor(2.0, device=z.device, dtype=z.dtype), -e + 2.0)
    m = torch.round(scaled)  # ties-to-even
    out[nonzero] = m * torch.pow(torch.tensor(2.0, device=z.device, dtype=z.dtype), e - 2.0)
    return out


def build_hif4_ternary_proxy(x: torch.Tensor) -> HiF4ProxyResult:
    if x.ndim < 1:
        raise ValueError(f"x must have at least 1 dim, got shape {tuple(x.shape)}")
    if int(x.shape[-1]) % _GROUP != 0:
        raise ValueError(
            f"last dim must be divisible by {_GROUP}, got {int(x.shape[-1])}"
        )
    if not x.is_floating_point():
        raise TypeError(f"x must be floating point, got {x.dtype}")

    xf = x.detach().to(torch.float32)
    if not bool(torch.isfinite(xf).all().item()):
        raise ValueError("build_hif4_ternary_proxy rejects NaN/Inf inputs")

    prefix = tuple(xf.shape[:-1])
    k = int(xf.shape[-1])
    groups_per_row = k // _GROUP
    groups = xf.reshape(-1, _GROUP)
    n = groups.shape[0]
    abs_g = groups.abs()
    a64 = abs_g.amax(dim=-1)
    zero = a64 <= 0

    inv7 = bf16_round(
        torch.tensor(1.0 / 7.0, device=xf.device, dtype=torch.float32)
    )
    s0_all = quantize_e6m2_nonnegative(bf16_round(a64 * inv7))
    s0_flat = torch.where(zero, torch.zeros_like(s0_all), s0_all)

    safe_s0 = torch.where(zero, torch.ones_like(s0_flat), s0_flat)
    r0 = bf16_round(1.0 / safe_s0)

    abs_8 = abs_g.reshape(n, 8, 8)
    a8 = abs_8.amax(dim=-1)
    e8 = (a8 * r0.unsqueeze(-1) >= 4.0).to(torch.float32)

    abs_4 = abs_g.reshape(n, 16, 4)
    a4 = abs_4.amax(dim=-1)
    e8_per4 = e8.repeat_interleave(2, dim=-1)
    e4 = (a4 * r0.unsqueeze(-1) * torch.pow(2.0, -e8_per4) >= 2.0).to(torch.float32)

    e8_elem = e8.repeat_interleave(8, dim=-1)
    e4_elem = e4.repeat_interleave(4, dim=-1)
    local_scale = safe_s0.unsqueeze(-1) * torch.pow(2.0, e8_elem + e4_elem)

    # Payload uses BF16 reciprocal r0 (same as hardware HiF4), not 1/S0 in FP32.
    normalized = abs_g * (
        r0.unsqueeze(-1) / torch.pow(2.0, e8_elem + e4_elem)
    )
    payload = torch.minimum(
        torch.full_like(abs_g, 1.75),
        torch.floor(4.0 * normalized + 0.5) / 4.0,
    )

    ternary = torch.zeros_like(groups)
    ternary = torch.where((payload > 0) & (groups > 0), torch.ones_like(ternary), ternary)
    ternary = torch.where((payload > 0) & (groups < 0), -torch.ones_like(ternary), ternary)

    proxy = local_scale * ternary
    sign = torch.sign(groups)
    hif4_dequant = local_scale * sign * payload

    # Zero groups: force exact zeros (no reciprocal / NaN path leakage).
    zero_mask = zero.unsqueeze(-1)
    local_scale = torch.where(zero_mask, torch.zeros_like(local_scale), local_scale)
    payload = torch.where(zero_mask, torch.zeros_like(payload), payload)
    ternary = torch.where(zero_mask, torch.zeros_like(ternary), ternary)
    proxy = torch.where(zero_mask, torch.zeros_like(proxy), proxy)
    hif4_dequant = torch.where(zero_mask, torch.zeros_like(hif4_dequant), hif4_dequant)
    e8 = torch.where(zero.unsqueeze(-1), torch.zeros_like(e8), e8)
    e4 = torch.where(zero.unsqueeze(-1), torch.zeros_like(e4), e4)

    out_shape = prefix + (k,)
    s0_shape = prefix + (groups_per_row,)
    e8_shape = prefix + (groups_per_row, 8)
    e4_shape = prefix + (groups_per_row, 16)

    return HiF4ProxyResult(
        proxy=proxy.reshape(out_shape),
        hif4_dequant=hif4_dequant.reshape(out_shape),
        local_scale=local_scale.reshape(out_shape),
        s0=s0_flat.reshape(s0_shape),
        e8=e8.reshape(e8_shape),
        e4=e4.reshape(e4_shape),
        payload=payload.reshape(out_shape),
        ternary_code=ternary.reshape(out_shape),
    )
