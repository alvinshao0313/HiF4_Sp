"""A2: NVFP4 / HiF4 activation internal-step counterfactuals."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ChuanCi.nvfp4_hif4_torch import (  # noqa: E402
    _compute_reciprocal_scale,
    _compute_top_scale,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics
from NVFP4.torch_fake import (  # noqa: E402
    FP4_E2M1_MAX,
    _fake_quant_nvfp4_activation_torch,
    cast_to_fp4_e2m1,
    cast_to_fp8_e4m3fn,
)

HIF4_A2_VARIANTS = [
    "full",
    "continuous_s0",
    "bf16_s0_no_e6m2",
    "oracle_e8",
    "oracle_e4",
    "oracle_e8_e4_joint",
    "continuous_payload_clipped",
    "rounded_payload_no_clip_probe",
]


def _nvfp4_with_options(
    x_bf16: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    e4m3_local: bool = True,
    e2m1_payload: bool = True,
) -> torch.Tensor:
    """NVFP4 QDQ with optional continuous local scale / continuous payload."""
    group_size = 16
    original_shape = x_bf16.shape
    hidden = original_shape[-1]
    x_2d = x_bf16.reshape(-1, hidden).to(torch.float32)
    grouped = x_2d.reshape(x_2d.shape[0], hidden // group_size, group_size)
    g = global_scale.reshape(()).to(device=x_2d.device, dtype=torch.float32)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    raw_block = torch.clamp(g * (amax / FP4_E2M1_MAX), min=-448.0, max=448.0)
    block_scale = cast_to_fp8_e4m3fn(raw_block).to(torch.float32) if e4m3_local else raw_block
    output_scale = torch.where(
        block_scale == 0, torch.zeros_like(block_scale), g / block_scale
    )
    scaled = torch.clamp(grouped * output_scale, min=-FP4_E2M1_MAX, max=FP4_E2M1_MAX)
    if e2m1_payload:
        x_fp4 = cast_to_fp4_e2m1(scaled)
    else:
        x_fp4 = scaled  # continuous payload in legal range
    dequant_scale = torch.where(
        g == 0, torch.zeros_like(block_scale), block_scale / g
    )
    out = (x_fp4 * dequant_scale).reshape(original_shape)
    return out.to(torch.bfloat16)


def search_oracle_global_scale(
    x_bf16: torch.Tensor,
    sidecar_scale: torch.Tensor,
) -> dict[str, Any]:
    """Preregistered 2-stage grid search minimizing NVFP4 reconstruction NMSE."""
    s0 = float(sidecar_scale.reshape(()).item())
    if s0 <= 0:
        raise ValueError("sidecar scale must be positive")
    x_ref = x_bf16.float()
    best_u, best_nmse, best_s = 0.0, float("inf"), s0
    us = [i / 16.0 for i in range(-4 * 16, 4 * 16 + 1)]
    for u in us:
        s = s0 * (2.0**u)
        y = _fake_quant_nvfp4_activation_torch(
            x_bf16,
            torch.tensor(s, dtype=torch.float32, device=x_bf16.device),
            group_size=16,
            output_dtype=torch.float32,
        )
        m = compute_pair_metrics(x_ref, y)
        if m["nmse"] < best_nmse:
            best_nmse, best_u, best_s = m["nmse"], u, s
    boundary = abs(best_u) >= 4.0 - 1e-9
    lo, hi = best_u - 1.0 / 16.0, best_u + 1.0 / 16.0
    for i in range(33):
        u = lo + (hi - lo) * i / 32.0
        s = s0 * (2.0**u)
        y = _fake_quant_nvfp4_activation_torch(
            x_bf16,
            torch.tensor(s, dtype=torch.float32, device=x_bf16.device),
            group_size=16,
            output_dtype=torch.float32,
        )
        m = compute_pair_metrics(x_ref, y)
        if m["nmse"] < best_nmse:
            best_nmse, best_u, best_s = m["nmse"], u, s
    return {
        "oracle_scale": best_s,
        "oracle_u": best_u,
        "oracle_nmse": best_nmse,
        "oracle_scale_search_boundary_hit": boundary,
    }


def _s1p2_payload(normalized: torch.Tensor) -> torch.Tensor:
    ratio = torch.floor(4.0 * normalized + 0.5) / 4.0
    return torch.minimum(ratio, torch.full_like(ratio, 1.75))


def _hif4_recon_from_exponents(
    groups: torch.Tensor,
    *,
    s0: torch.Tensor,
    reciprocal: torch.Tensor,
    e8: torch.Tensor,
    e4: torch.Tensor,
) -> torch.Tensor:
    """groups [N,64], s0/reciprocal [N], e8 [N,8], e4 [N,16] → recon [N,64]."""
    nonzero = s0 > 0  # approximate; zeros handled below via amax
    amax64 = groups.abs().amax(dim=-1)
    nonzero = amax64 > 0
    safe_s0 = torch.where(nonzero, s0, torch.ones_like(s0))
    e8_elem = e8.repeat_interleave(8, dim=-1)
    e4_elem = e4.repeat_interleave(4, dim=-1)
    local_scale = safe_s0.unsqueeze(-1) * (2.0 ** (e8_elem + e4_elem))
    normalized = groups.abs() * (
        reciprocal.unsqueeze(-1) / (2.0 ** (e8_elem + e4_elem))
    )
    payload = _s1p2_payload(normalized)
    recon = groups.sign() * local_scale * payload
    return torch.where(nonzero.unsqueeze(-1), recon, torch.zeros_like(recon))


def _hif4_hardware_base(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pack last-dim into 64-groups and compute hardware S0 / threshold e8/e4."""
    original = x.shape
    hidden = original[-1]
    assert hidden % 64 == 0
    groups = x.reshape(-1, 64).to(torch.float32)
    amax64 = groups.abs().amax(dim=-1)
    nonzero = amax64 > 0
    s0 = _compute_top_scale(amax64, "hardware", 7.0)
    safe_s0 = torch.where(nonzero, s0, torch.ones_like(s0))
    reciprocal = _compute_reciprocal_scale(safe_s0, "hardware")
    abs_g = groups.abs()
    amax8 = abs_g.reshape(-1, 8, 8).amax(dim=-1)
    amax4 = abs_g.reshape(-1, 16, 4).amax(dim=-1)
    e8_hw = (amax8 * reciprocal.unsqueeze(-1) >= 4.0).to(torch.float32)
    e8_per4 = e8_hw.repeat_interleave(2, dim=-1)
    e4_hw = (amax4 * reciprocal.unsqueeze(-1) / (2.0**e8_per4) >= 2.0).to(torch.float32)
    return {
        "groups": groups,
        "s0": s0,
        "safe_s0": safe_s0,
        "reciprocal": reciprocal,
        "e8_hw": e8_hw,
        "e4_hw": e4_hw,
        "amax8": amax8,
        "amax4": amax4,
        "original_shape": original,
    }


def _oracle_e8(x: torch.Tensor) -> torch.Tensor:
    b = _hif4_hardware_base(x)
    groups, s0, recip = b["groups"], b["s0"], b["reciprocal"]
    e8 = b["e8_hw"].clone()
    # for each 8-block independently try e8∈{0,1}; e4 re-thresholded under that e8
    n = groups.shape[0]
    best_e8 = e8.clone()
    for bi in range(8):
        errs = []
        cands = []
        for e8v in (0.0, 1.0):
            e8_try = e8.clone()
            e8_try[:, bi] = e8v
            e8_per4 = e8_try.repeat_interleave(2, dim=-1)
            e4_try = (
                b["amax4"] * recip.unsqueeze(-1) / (2.0**e8_per4) >= 2.0
            ).to(torch.float32)
            recon = _hif4_recon_from_exponents(
                groups, s0=s0, reciprocal=recip, e8=e8_try, e4=e4_try
            )
            # per-group MSE on the 8 elements of block bi
            sl = slice(bi * 8, (bi + 1) * 8)
            err = ((recon[:, sl] - groups[:, sl]) ** 2).sum(dim=-1)
            errs.append(err)
            cands.append(e8v)
        # pick better e8 per group
        pick1 = errs[1] < errs[0]
        best_e8[:, bi] = torch.where(pick1, torch.ones(n, device=groups.device), torch.zeros(n, device=groups.device))
    e8_per4 = best_e8.repeat_interleave(2, dim=-1)
    e4 = (b["amax4"] * recip.unsqueeze(-1) / (2.0**e8_per4) >= 2.0).to(torch.float32)
    recon = _hif4_recon_from_exponents(groups, s0=s0, reciprocal=recip, e8=best_e8, e4=e4)
    return recon.reshape(b["original_shape"])


def _oracle_e4(x: torch.Tensor) -> torch.Tensor:
    b = _hif4_hardware_base(x)
    groups, s0, recip = b["groups"], b["s0"], b["reciprocal"]
    e8 = b["e8_hw"]
    best_e4 = b["e4_hw"].clone()
    n = groups.shape[0]
    for bi in range(16):
        errs = []
        for e4v in (0.0, 1.0):
            e4_try = best_e4.clone()
            e4_try[:, bi] = e4v
            recon = _hif4_recon_from_exponents(
                groups, s0=s0, reciprocal=recip, e8=e8, e4=e4_try
            )
            sl = slice(bi * 4, (bi + 1) * 4)
            err = ((recon[:, sl] - groups[:, sl]) ** 2).sum(dim=-1)
            errs.append(err)
        pick1 = errs[1] < errs[0]
        best_e4[:, bi] = torch.where(
            pick1,
            torch.ones(n, device=groups.device),
            torch.zeros(n, device=groups.device),
        )
    recon = _hif4_recon_from_exponents(groups, s0=s0, reciprocal=recip, e8=e8, e4=best_e4)
    return recon.reshape(b["original_shape"])


def _oracle_e8_e4_joint(x: torch.Tensor) -> torch.Tensor:
    """Per 8-group: enumerate e8∈{0,1} and two child e4∈{0,1}^2 (8 combos)."""
    b = _hif4_hardware_base(x)
    groups, s0, recip = b["groups"], b["s0"], b["reciprocal"]
    e8 = b["e8_hw"].clone()
    e4 = b["e4_hw"].clone()
    n = groups.shape[0]
    for bi in range(8):
        e4_i0, e4_i1 = 2 * bi, 2 * bi + 1
        best_err = torch.full((n,), float("inf"), device=groups.device)
        best_e8v = torch.zeros(n, device=groups.device)
        best_e40 = torch.zeros(n, device=groups.device)
        best_e41 = torch.zeros(n, device=groups.device)
        for e8v in (0.0, 1.0):
            for e40 in (0.0, 1.0):
                for e41 in (0.0, 1.0):
                    e8_try = e8.clone()
                    e4_try = e4.clone()
                    e8_try[:, bi] = e8v
                    e4_try[:, e4_i0] = e40
                    e4_try[:, e4_i1] = e41
                    recon = _hif4_recon_from_exponents(
                        groups, s0=s0, reciprocal=recip, e8=e8_try, e4=e4_try
                    )
                    sl = slice(bi * 8, (bi + 1) * 8)
                    err = ((recon[:, sl] - groups[:, sl]) ** 2).sum(dim=-1)
                    better = err < best_err
                    best_err = torch.where(better, err, best_err)
                    best_e8v = torch.where(better, torch.full_like(best_e8v, e8v), best_e8v)
                    best_e40 = torch.where(better, torch.full_like(best_e40, e40), best_e40)
                    best_e41 = torch.where(better, torch.full_like(best_e41, e41), best_e41)
        e8[:, bi] = best_e8v
        e4[:, e4_i0] = best_e40
        e4[:, e4_i1] = best_e41
    recon = _hif4_recon_from_exponents(groups, s0=s0, reciprocal=recip, e8=e8, e4=e4)
    return recon.reshape(b["original_shape"])


def _row(
    *,
    variant: str,
    x_ref: torch.Tensor,
    y: torch.Tensor,
    e_full: float,
    w_n: torch.Tensor | None,
    e_full_out: float | None,
    legal: bool = True,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    e = float(((y - x_ref) ** 2).sum().item())
    row: dict[str, Any] = {
        "variant": variant,
        "nmse": compute_pair_metrics(x_ref, y)["nmse"],
        "error_energy": e,
        "R_cf": (1.0 - e / e_full) if e_full > 0 else 0.0,
        "legal": legal,
    }
    if w_n is not None and e_full_out is not None:
        y_out = F.linear(y, w_n)
        y_ref = F.linear(x_ref, w_n)
        e_out = float(((y_out - y_ref) ** 2).sum().item())
        # recoverable vs full-quant output error (not vs X)
        row["output_error_energy"] = e_out
        row["R_cf_output"] = (1.0 - e_out / e_full_out) if e_full_out > 0 else 0.0
    if extras:
        row.update(extras)
    return row


def nvfp4_activation_counterfactuals(
    x_bf16: torch.Tensor,
    sidecar_scale: torch.Tensor,
    *,
    w_n: torch.Tensor | None = None,
) -> dict[str, Any]:
    x_ref = x_bf16.float()
    full = quantize_nvfp4_activation(x_bf16, sidecar_scale).dequantized.float()
    e_full = float(((full - x_ref) ** 2).sum().item())
    e_full_out = None
    w32 = None
    if w_n is not None:
        w32 = w_n.float()
        y_full = F.linear(full, w32)
        y_ref = F.linear(x_ref, w32)
        e_full_out = float(((y_full - y_ref) ** 2).sum().item())

    variants = {
        "nv_full": full,
        "nv_continuous_local_scale": _nvfp4_with_options(
            x_bf16, sidecar_scale, e4m3_local=False, e2m1_payload=True
        ).float(),
        "nv_continuous_payload": _nvfp4_with_options(
            x_bf16, sidecar_scale, e4m3_local=True, e2m1_payload=False
        ).float(),
    }
    oracle = search_oracle_global_scale(x_bf16, sidecar_scale)
    variants["nv_oracle_global_scale"] = _fake_quant_nvfp4_activation_torch(
        x_bf16,
        torch.tensor(oracle["oracle_scale"], dtype=torch.float32, device=x_bf16.device),
        group_size=16,
        output_dtype=torch.float32,
    )

    rows = []
    for name, y in variants.items():
        extras = None
        if name == "nv_oracle_global_scale":
            extras = {
                "oracle_u": oracle["oracle_u"],
                "oracle_scale_search_boundary_hit": oracle["oracle_scale_search_boundary_hit"],
            }
        rows.append(
            _row(
                variant=name,
                x_ref=x_ref,
                y=y,
                e_full=e_full,
                w_n=w32,
                e_full_out=e_full_out,
                extras=extras,
            )
        )
    return {
        "e_full": e_full,
        "e_full_out": e_full_out,
        "variants": rows,
        "oracle_search": oracle,
        "path_id": "P2_matched_semantic",
        "format": "nvfp4",
    }


def hif4_activation_counterfactuals(
    x_bf16: torch.Tensor,
    *,
    w_n: torch.Tensor | None = None,
) -> dict[str, Any]:
    x = x_bf16.float()
    full = quantize_hif4_tensor(x, variant="full", output_dtype=torch.float32)
    w_full = full.metadata["values_fp32"].float()
    e_full = float(((w_full - x) ** 2).sum().item())
    e_full_out = None
    w32 = None
    if w_n is not None:
        w32 = w_n.float()
        y_full = F.linear(w_full, w32)
        y_ref = F.linear(x, w32)
        e_full_out = float(((y_full - y_ref) ** 2).sum().item())

    # Config-driven variants
    cfg_variants = [
        "full",
        "continuous_s0",
        "bf16_s0_no_e6m2",
        "continuous_payload_clipped",
        "rounded_payload_no_clip_probe",
    ]
    rows = []
    for variant in cfg_variants:
        view = quantize_hif4_tensor(x, variant=variant, output_dtype=torch.float32)
        y = view.metadata["values_fp32"].float()
        rows.append(
            _row(
                variant=variant,
                x_ref=x,
                y=y,
                e_full=e_full,
                w_n=w32,
                e_full_out=e_full_out,
                legal=variant != "rounded_payload_no_clip_probe",
            )
        )

    # Oracle exponent variants (must not worsen vs hardware full)
    for name, fn in [
        ("oracle_e8", _oracle_e8),
        ("oracle_e4", _oracle_e4),
        ("oracle_e8_e4_joint", _oracle_e8_e4_joint),
    ]:
        y = fn(x)
        rows.append(
            _row(
                variant=name,
                x_ref=x,
                y=y,
                e_full=e_full,
                w_n=w32,
                e_full_out=e_full_out,
                legal=True,
            )
        )
    return {
        "e_full": e_full,
        "e_full_out": e_full_out,
        "variants": rows,
        "path_id": "P2_matched_semantic",
        "format": "hif4",
    }
