"""M1–M4: MLP nonlinear error propagation with product decomposition."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics


def local_gain(delta_in: torch.Tensor, delta_out: torch.Tensor) -> dict[str, Any]:
    nin = float(torch.linalg.vector_norm(delta_in.float()).item())
    nout = float(torch.linalg.vector_norm(delta_out.float()).item())
    if nin == 0.0:
        return {"gain": 0.0, "gain_status": "zero_input_error", "norm_in": nin, "norm_out": nout}
    return {"gain": nout / nin, "gain_status": "ok", "norm_in": nin, "norm_out": nout}


def product_exact_decomposition(
    g_n: torch.Tensor,
    g_h: torch.Tensor,
    u_n: torch.Tensor,
    u_h: torch.Tensor,
) -> dict[str, float]:
    """g_H u_H - g_N u_N = δg u_N + g_N δu + δg δu."""
    dg = g_h.float() - g_n.float()
    du = u_h.float() - u_n.float()
    gn = g_n.float()
    un = u_n.float()
    term_dg_un = dg * un
    term_gn_du = gn * du
    term_dg_du = dg * du
    total = g_h.float() * u_h.float() - gn * un
    recon = term_dg_un + term_gn_du + term_dg_du
    resid = total - recon

    def _e(t: torch.Tensor) -> float:
        return float((t * t).sum().item())

    te = _e(total)
    return {
        "energy_total": te,
        "energy_dg_un": _e(term_dg_un),
        "energy_gn_du": _e(term_gn_du),
        "energy_dg_du": _e(term_dg_du),
        "residual_rel": _e(resid) / te if te > 0 else _e(resid),
        "cross_share": _e(term_dg_du) / te if te > 0 else 0.0,
    }


@torch.no_grad()
def mlp_stage_metrics(
    mlp_input: torch.Tensor,
    w_gate_n: torch.Tensor,
    w_gate_h: torch.Tensor,
    w_up_n: torch.Tensor,
    w_up_h: torch.Tensor,
    w_down_n: torch.Tensor,
    w_down_h: torch.Tensor,
) -> dict[str, Any]:
    """Frozen-input MLP path for source (N) vs target (H) weights."""
    x = mlp_input.float()
    gate_n = F.linear(x, w_gate_n.float())
    gate_h = F.linear(x, w_gate_h.float())
    up_n = F.linear(x, w_up_n.float())
    up_h = F.linear(x, w_up_h.float())
    silu_n = F.silu(gate_n)
    silu_h = F.silu(gate_h)
    prod_n = silu_n * up_n
    prod_h = silu_h * up_h
    down_n = F.linear(prod_n, w_down_n.float())
    down_h = F.linear(prod_h, w_down_h.float())

    stages = {
        "gate_proj_out": (gate_n, gate_h),
        "silu_gate": (silu_n, silu_h),
        "up_proj_out": (up_n, up_h),
        "product": (prod_n, prod_h),
        "down_proj_out": (down_n, down_h),
    }
    metrics = {name: compute_pair_metrics(a, b) for name, (a, b) in stages.items()}
    # Gains along chain
    order = ["gate_proj_out", "silu_gate", "product", "down_proj_out"]
    gains = {}
    for i in range(len(order) - 1):
        a0, b0 = stages[order[i]]
        a1, b1 = stages[order[i + 1]]
        gains[f"{order[i]}->{order[i+1]}"] = local_gain(b0 - a0, b1 - a1)

    prod_decomp = product_exact_decomposition(silu_n, silu_h, up_n, up_h)
    return {
        "stage_metrics": metrics,
        "gains": gains,
        "product_decomposition": prod_decomp,
        "hypothesis_id": "H5-MLP",
        "evidence_class": "controlled_causal_evidence",
    }
