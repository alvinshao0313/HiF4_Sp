"""AX4: Scale system vs payload cross-format factorization."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation
from NVFP4.torch_fake import FP4_E2M1_MAX, cast_to_fp4_e2m1

MatchKind = Literal["raw", "range_matched"]

_HIF4_PAYLOAD_MAX = 1.75


def _s1p2_payload(normalized: torch.Tensor) -> torch.Tensor:
    ratio = torch.floor(4.0 * normalized + 0.5) / 4.0
    return torch.minimum(ratio, torch.full_like(ratio, _HIF4_PAYLOAD_MAX))


def _nvfp4_dequant_scale(x_bf16: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Per-element NVFP4 dequant scale = E4M3_local / global (= block_scale / g)."""
    view = quantize_nvfp4_activation(x_bf16, global_scale, collect_metadata=True)
    hidden = x_bf16.shape[-1]
    g = global_scale.reshape(()).to(device=x_bf16.device, dtype=torch.float32)
    e4 = view.metadata["e4m3_local_scale"].float().reshape(-1, hidden // 16, 1)
    dequant = torch.where(e4 == 0, torch.zeros_like(e4), e4 / g)
    return dequant.expand(-1, -1, 16).reshape(x_bf16.shape)


def _hif4_local_scale(x_bf16: torch.Tensor) -> torch.Tensor:
    view = quantize_hif4_tensor(x_bf16.float(), variant="full", output_dtype=torch.float32)
    return view.metadata["local_scale"].float()


def _apply_payload(
    normalized_abs: torch.Tensor,
    *,
    payload: Literal["N", "H"],
    match: MatchKind,
    scale_payload_max: float,
) -> torch.Tensor:
    """Quantize |normalized| onto payload grid.

    raw: use payload's native numeric range on the scale-normalized values.
    range_matched: map to [0,1] by scale system's max, quantize on payload's
    unit grid, then map back to scale system's max.
    """
    if match == "range_matched":
        unit = (normalized_abs / scale_payload_max).clamp(0.0, 1.0)
        if payload == "H":
            # S1P2 on [0,1] via temporary stretch to [0,1.75]
            p_native = _s1p2_payload(unit * _HIF4_PAYLOAD_MAX)
            return (p_native / _HIF4_PAYLOAD_MAX) * scale_payload_max
        p_native = cast_to_fp4_e2m1(unit * FP4_E2M1_MAX)
        return (p_native.abs() / FP4_E2M1_MAX) * scale_payload_max

    # raw
    if payload == "H":
        return _s1p2_payload(normalized_abs.clamp(min=0.0))
    return cast_to_fp4_e2m1(normalized_abs.clamp(max=FP4_E2M1_MAX)).abs()


def _reconstruct_hybrid(
    x_bf16: torch.Tensor,
    *,
    scale: Literal["N", "H"],
    payload: Literal["N", "H"],
    match: MatchKind,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    x = x_bf16.float()
    if scale == "H":
        eff = _hif4_local_scale(x_bf16)
        scale_payload_max = _HIF4_PAYLOAD_MAX
    else:
        eff = _nvfp4_dequant_scale(x_bf16, global_scale)
        scale_payload_max = FP4_E2M1_MAX

    # normalized = |x| / dequant_scale  (payload domain)
    safe_eff = eff.clamp_min(1e-12)
    normalized = x.abs() / safe_eff
    # zero-scale groups stay zero
    normalized = torch.where(eff > 0, normalized, torch.zeros_like(normalized))
    p = _apply_payload(
        normalized,
        payload=payload,
        match=match,
        scale_payload_max=scale_payload_max,
    )
    recon = torch.sign(x) * eff * p
    return recon


@torch.no_grad()
def run_cross_format_factorization(
    x_bf16: torch.Tensor,
    a_n: torch.Tensor,
    w_n: torch.Tensor,
    input_global_scale: torch.Tensor,
) -> list[dict[str, Any]]:
    """NN/HH/HN/NH raw and range-matched hybrids; R_* vs HH→A_N conversion error."""
    a_n_f = a_n.float()
    hh = quantize_hif4_tensor(x_bf16.float(), variant="full", output_dtype=torch.float32)
    a_h = hh.metadata["values_fp32"].float()
    nn = quantize_nvfp4_activation(x_bf16, input_global_scale).dequantized.float()
    w = w_n.float()

    e_full_a = float(((a_h - a_n_f) ** 2).sum().item())
    e_full_y = float((F.linear(a_h - a_n_f, w) ** 2).sum().item())
    if e_full_a <= 0 or e_full_y <= 0:
        raise ValueError("HH vs A_N baseline error must be positive for AX4 ranking")

    combos: list[tuple[str, MatchKind, torch.Tensor, bool]] = [
        ("NN", "raw", nn, True),
        ("HH", "raw", a_h, True),
        (
            "HN",
            "raw",
            _reconstruct_hybrid(
                x_bf16, scale="H", payload="N", match="raw", global_scale=input_global_scale
            ),
            False,
        ),
        (
            "NH",
            "raw",
            _reconstruct_hybrid(
                x_bf16, scale="N", payload="H", match="raw", global_scale=input_global_scale
            ),
            False,
        ),
        (
            "HN",
            "range_matched",
            _reconstruct_hybrid(
                x_bf16,
                scale="H",
                payload="N",
                match="range_matched",
                global_scale=input_global_scale,
            ),
            False,
        ),
        (
            "NH",
            "range_matched",
            _reconstruct_hybrid(
                x_bf16,
                scale="N",
                payload="H",
                match="range_matched",
                global_scale=input_global_scale,
            ),
            False,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, match, a_v, is_hw in combos:
        a_v_f = a_v.float()
        if not torch.isfinite(a_v_f).all():
            raise ValueError(f"non-finite hybrid recon for {name}/{match}")
        e_a = float(((a_v_f - a_n_f) ** 2).sum().item())
        e_y = float((F.linear(a_v_f - a_n_f, w) ** 2).sum().item())
        rows.append(
            {
                "hybrid": name,
                "match_kind": match,
                "is_valid_hardware_format": is_hw,
                "purpose": "reference" if is_hw else "mechanism_probe",
                "R_A": 1.0 - e_a / e_full_a,
                "R_Y": 1.0 - e_y / e_full_y,
                "activation_error_energy": e_a,
                "output_error_energy": e_y,
            }
        )
    return rows
