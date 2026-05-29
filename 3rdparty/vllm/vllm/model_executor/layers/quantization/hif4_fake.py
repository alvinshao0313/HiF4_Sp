# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch  # [hif4 quant]
import torch.nn.functional as F  # [hif4 quant]


def hif4_fake_quantize_hifx4(x: torch.Tensor) -> torch.Tensor:  # [hif4 quant]
    """Apply HiF4 hifx4 fake quant-dequant along the last dimension."""  # [hif4 quant]
    orig_dtype = x.dtype  # [hif4 quant]
    orig_cols = x.shape[-1]  # [hif4 quant]
    pad_cols = (64 - orig_cols % 64) % 64  # [hif4 quant]
    work = x if x.is_contiguous() else x.contiguous()  # [hif4 quant]
    if pad_cols > 0:  # [hif4 quant]
        work = F.pad(work, (0, pad_cols), value=0.0)  # [hif4 quant]
    work_fp32 = work.float()  # [hif4 quant]
    grouped = work_fp32.unflatten(-1, (-1, 8, 2, 4))  # [hif4 quant]
    unsigned = torch.abs(grouped)  # [hif4 quant]
    sign = torch.sign(grouped)  # [hif4 quant]
    max_lv3 = torch.max(unsigned, dim=-1, keepdim=True)[0]  # [hif4 quant]
    max_lv2 = torch.max(max_lv3, dim=-2, keepdim=True)[0]  # [hif4 quant]
    max_lv1 = torch.max(max_lv2, dim=-3, keepdim=True)[0]  # [hif4 quant]
    div7 = (torch.ones_like(max_lv1) / 7.0).to(torch.bfloat16).float()  # [hif4 quant]
    scale_factor = (max_lv1 * div7).to(torch.bfloat16).float()  # [hif4 quant]
    scale_factor = scale_factor.clamp(min=2 ** (-48), max=49152)  # [hif4 quant]
    exp_sf = torch.floor(torch.log2(scale_factor))  # [hif4 quant]
    mant_sf = scale_factor / torch.exp2(exp_sf) * 2**7  # [hif4 quant]
    scale_factor = torch.round(mant_sf) / 2**7 * torch.exp2(exp_sf)  # [hif4 quant]
    exp_sf = torch.floor(torch.log2(scale_factor))  # [hif4 quant]
    scale_factor = (  # [hif4 quant]
        torch.round(scale_factor * torch.exp2(2 - exp_sf))  # [hif4 quant]
        * torch.exp2(exp_sf - 2)  # [hif4 quant]
    )  # [hif4 quant]
    rec_sf = (1.0 / scale_factor).to(torch.bfloat16).float()  # [hif4 quant]
    scale_lv2 = torch.exp2(  # [hif4 quant]
        torch.floor((max_lv2 * rec_sf).clamp(0, 4) / 4)  # [hif4 quant]
    )  # [hif4 quant]
    scale_lv3 = torch.exp2(  # [hif4 quant]
        torch.floor(((max_lv3 * rec_sf / scale_lv2).clamp(0, 2)) / 2)  # [hif4 quant]
    )  # [hif4 quant]
    mant = unsigned / scale_lv2 / scale_lv3 * rec_sf  # [hif4 quant]
    mant = torch.floor(mant * 2**2 + 0.5) / 2**2  # [hif4 quant]
    mant = torch.clamp(mant, max=2 - 2**-2)  # [hif4 quant]
    out = sign * mant * scale_lv2 * scale_lv3 * scale_factor  # [hif4 quant]
    out = out.flatten(-4, -1)  # [hif4 quant]
    if pad_cols > 0:  # [hif4 quant]
        out = out[..., :orig_cols]  # [hif4 quant]
    return out.to(orig_dtype)  # [hif4 quant]


def hif4_fake_quantize_hifx4_1(x: torch.Tensor) -> torch.Tensor:
    """Apply hif4-1 fake quant-dequant along the last dimension."""
    orig_dtype = x.dtype
    orig_cols = x.shape[-1]
    pad_cols = (64 - orig_cols % 64) % 64
    work = x if x.is_contiguous() else x.contiguous()
    if pad_cols > 0:
        work = F.pad(work, (0, pad_cols), value=0.0)

    work_fp32 = work.float()
    grouped = work_fp32.unflatten(-1, (-1, 64))
    unsigned = torch.abs(grouped)
    sign = torch.sign(grouped)
    max_abs = torch.max(unsigned, dim=-1, keepdim=True)[0]

    div = (torch.ones_like(max_abs) / 1.75).to(torch.bfloat16).float()
    scale_factor = (max_abs * div).to(torch.bfloat16).float()
    scale_factor = scale_factor.clamp(min=2 ** (-48), max=49152)
    exp_sf = torch.floor(torch.log2(scale_factor))
    scale_factor = (
        torch.round(scale_factor * torch.exp2(2 - exp_sf))
        * torch.exp2(exp_sf - 2)
    )

    rec_sf = (1.0 / scale_factor).to(torch.bfloat16).float()
    mant = unsigned * rec_sf
    mant = torch.floor(mant * 2**2 + 0.5) / 2**2
    mant = torch.clamp(mant, max=1.75)
    out = sign * mant * scale_factor
    out = out.flatten(-2, -1)
    if pad_cols > 0:
        out = out[..., :orig_cols]
    return out.to(orig_dtype)
