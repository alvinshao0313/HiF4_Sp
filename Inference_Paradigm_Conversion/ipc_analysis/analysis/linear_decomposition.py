"""L1/L2: exact algebraic decomposition of Linear output error + Shapley."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics
from Inference_Paradigm_Conversion.ipc_analysis.records import LinearDecompositionRecord


def shapley_wa(
    a_n: torch.Tensor,
    a_h: torch.Tensor,
    w_n: torch.Tensor,
    w_h: torch.Tensor,
) -> dict[str, Any]:
    """Order-independent two-factor Shapley for W/A format shift via F.linear."""
    a_n, a_h = a_n.float(), a_h.float()
    w_n, w_h = w_n.float(), w_h.float()
    y_nn = F.linear(a_n, w_n)
    y_hn = F.linear(a_n, w_h)
    y_nh = F.linear(a_h, w_n)
    y_hh = F.linear(a_h, w_h)
    phi_w = 0.5 * ((y_hn - y_nn) + (y_hh - y_nh))
    phi_a = 0.5 * ((y_nh - y_nn) + (y_hh - y_hn))
    delta = y_hh - y_nn
    resid = delta - (phi_w + phi_a)
    de = float((delta * delta).sum().item())
    re = float((resid * resid).sum().item())
    return {
        "energy_phi_w": float((phi_w * phi_w).sum().item()),
        "energy_phi_a": float((phi_a * phi_a).sum().item()),
        "cross_phi_w_phi_a": float((phi_w * phi_a).sum().item()),
        "energy_delta_y": de,
        "shapley_residual_rel": re / de if de > 0 else re,
    }


def fp64_decomposition_audit(
    a_n: torch.Tensor,
    a_h: torch.Tensor,
    w_n: torch.Tensor,
    w_h: torch.Tensor,
    *,
    max_tokens: int = 8,
    max_out_channels: int = 8,
) -> dict[str, Any]:
    """FP64 identity audit with γ_n error bound (not fixed atol)."""
    a_n64 = a_n.double().reshape(-1, a_n.shape[-1])
    a_h64 = a_h.double().reshape(-1, a_h.shape[-1])
    w_n64 = w_n.double()
    w_h64 = w_h.double()
    t = min(a_n64.shape[0], max_tokens)
    c = min(w_n64.shape[0], max_out_channels)
    a_n64, a_h64 = a_n64[:t], a_h64[:t]
    w_n64, w_h64 = w_n64[:c], w_h64[:c]
    d_a = a_h64 - a_n64
    d_w = w_h64 - w_n64
    y_nn = a_n64 @ w_n64.T
    y_hh = a_h64 @ w_h64.T
    e_w = a_n64 @ d_w.T
    e_a = d_a @ w_n64.T
    e_wa = d_a @ d_w.T
    recon = y_nn + e_w + e_a + e_wa
    resid = (y_hh - recon).abs()
    # γ_n bound for K-term dot products
    k = a_n64.shape[-1]
    u = torch.finfo(torch.float64).eps
    gamma = (k * u) / (1.0 - k * u) if k * u < 1 else float("inf")
    # absolute bound ~ γ * sum |a_i w_i| over four products; use conservative sum of abs integrands
    abs_bound = gamma * (
        (a_n64.abs() @ w_n64.abs().T)
        + (a_n64.abs() @ d_w.abs().T)
        + (d_a.abs() @ w_n64.abs().T)
        + (d_a.abs() @ d_w.abs().T)
        + (a_h64.abs() @ w_h64.abs().T)
    )
    ok = bool(torch.all(resid <= abs_bound + 1e-15).item())
    return {
        "ok": ok,
        "max_abs_residual": float(resid.max().item()),
        "max_abs_bound": float(abs_bound.max().item()),
        "gamma_k": float(gamma),
        "num_checked": int(resid.numel()),
    }


def decompose_linear_error(
    a_n: torch.Tensor,
    a_h: torch.Tensor,
    w_n: torch.Tensor,
    w_h: torch.Tensor,
    *,
    path_id: str,
    layer_idx: int,
    module_name: str,
    projection: str,
    phase: str,
    sample_id: str,
    with_shapley: bool = True,
    with_fp64_audit: bool = False,
) -> LinearDecompositionRecord:
    """Y_H - Y_N = ΔW A_N + W_N ΔA + ΔW ΔA  (via F.linear / X @ W.T)."""
    a_n = a_n.float()
    a_h = a_h.float()
    w_n = w_n.float()
    w_h = w_h.float()
    d_a = a_h - a_n
    d_w = w_h - w_n

    y_n = F.linear(a_n, w_n)
    y_h = F.linear(a_h, w_h)
    term_dw_an = F.linear(a_n, d_w)
    term_wn_da = F.linear(d_a, w_n)
    term_dw_da = F.linear(d_a, d_w)
    recon = y_n + term_dw_an + term_wn_da + term_dw_da
    resid = y_h - recon
    y_h_e = float((y_h * y_h).sum().item())
    resid_e = float((resid * resid).sum().item())
    residual_rel = resid_e / y_h_e if y_h_e > 0 else resid_e

    def _e(t: torch.Tensor) -> float:
        return float((t * t).sum().item())

    def _cross(u: torch.Tensor, v: torch.Tensor) -> float:
        return float((u * v).sum().item())

    extras: dict[str, Any] = {
        "output_nmse": compute_pair_metrics(y_n, y_h)["nmse"],
        "y_h_energy": y_h_e,
    }
    if with_shapley:
        extras.update(shapley_wa(a_n, a_h, w_n, w_h))
    if with_fp64_audit:
        extras["fp64_audit"] = fp64_decomposition_audit(a_n, a_h, w_n, w_h)

    return LinearDecompositionRecord(
        path_id=path_id,
        layer_idx=layer_idx,
        module_name=module_name,
        projection=projection,
        phase=phase,
        sample_id=sample_id,
        energy_wn_an=_e(y_n),
        energy_delta_w_an=_e(term_dw_an),
        energy_wn_delta_a=_e(term_wn_da),
        energy_delta_w_delta_a=_e(term_dw_da),
        cross_dw_an_wn_da=_cross(term_dw_an, term_wn_da),
        cross_dw_an_dw_da=_cross(term_dw_an, term_dw_da),
        cross_wn_da_dw_da=_cross(term_wn_da, term_dw_da),
        residual_rel=residual_rel,
        extras=extras,
    )


def verify_decomposition_identity(
    a_n: torch.Tensor,
    a_h: torch.Tensor,
    w_n: torch.Tensor,
    w_h: torch.Tensor,
    atol: float = 1e-5,
) -> dict[str, Any]:
    rec = decompose_linear_error(
        a_n,
        a_h,
        w_n,
        w_h,
        path_id="P2_matched_semantic",
        layer_idx=0,
        module_name="test",
        projection="down_proj",
        phase="prefill",
        sample_id="t0",
        with_fp64_audit=True,
    )
    ok = rec.residual_rel <= atol and rec.extras.get("fp64_audit", {}).get("ok", False)
    shapley_ok = rec.extras.get("shapley_residual_rel", 1.0) <= atol
    return {
        "ok": ok and shapley_ok,
        "residual_rel": rec.residual_rel,
        "shapley_residual_rel": rec.extras.get("shapley_residual_rel"),
        "fp64_audit": rec.extras.get("fp64_audit"),
        "record": rec.to_dict(),
    }
