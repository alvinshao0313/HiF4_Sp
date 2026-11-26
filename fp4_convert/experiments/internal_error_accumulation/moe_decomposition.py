"""MoE q / p_expert / p_router decomposition with energy closure."""
from __future__ import annotations

import torch

from .math_utils import as_f64, cosine, l2, squared_l2


def moe_three_way_decompose(
    *,
    y0: torch.Tensor,
    y00: torch.Tensor,
    y10: torch.Tensor,
    y11: torch.Tensor,
    delta_R_A: torch.Tensor,
) -> dict:
    """Formal definitions:
    q = y00 - y0
    p_expert = y10 - y00
    p_router = y11 - y10
    delta_M = y11 - y0 = q + p_expert + p_router
    p = y11 - y00 = p_expert + p_router
    """
    y0v, y00v, y10v, y11v = as_f64(y0), as_f64(y00), as_f64(y10), as_f64(y11)
    q = y00v - y0v
    p_expert = y10v - y00v
    p_router = y11v - y10v
    p = y11v - y00v
    delta_M = y11v - y0v
    rem = delta_M - (q + p_expert + p_router)
    rem_p = p - (p_expert + p_router)
    scale = max(float(delta_M.abs().max()), 1e-300)
    tol = 8 * torch.finfo(torch.float64).eps * scale
    if float(rem.abs().max()) > tol or float(rem_p.abs().max()) > tol:
        raise RuntimeError(
            f"MoE q/p_expert/p_router closure failed: rem={float(rem.abs().max())} "
            f"rem_p={float(rem_p.abs().max())} tol={tol}"
        )
    dRA = as_f64(delta_R_A)
    g_terms = {
        "q_sq": squared_l2(q),
        "p_expert_sq": squared_l2(p_expert),
        "p_router_sq": squared_l2(p_router),
        "2_q_p_expert": 2.0 * float((q * p_expert).sum()),
        "2_q_p_router": 2.0 * float((q * p_router).sum()),
        "2_p_expert_p_router": 2.0 * float((p_expert * p_router).sum()),
        "2_dRA_q": 2.0 * float((dRA * q).sum()),
        "2_dRA_p_expert": 2.0 * float((dRA * p_expert).sum()),
        "2_dRA_p_router": 2.0 * float((dRA * p_router).sum()),
    }
    g_m = squared_l2(dRA + delta_M) - squared_l2(dRA)
    g_sum = sum(g_terms.values())
    # Algebraic identity in FP64: G_M = ||dRA+dM||^2 - ||dRA||^2 expands to the terms above.
    if abs(g_sum - g_m) > 8 * torch.finfo(torch.float64).eps * max(abs(g_m), 1.0):
        raise RuntimeError(f"MoE energy expansion failed: g_sum={g_sum} G_M={g_m}")
    return {
        "q_l2": l2(q),
        "p_l2": l2(p),
        "p_expert_l2": l2(p_expert),
        "p_router_l2": l2(p_router),
        "delta_M_l2": l2(delta_M),
        "cos_q_p": cosine(q, p),
        "cos_q_p_expert": cosine(q, p_expert),
        "cos_q_p_router": cosine(q, p_router),
        "closure_max_abs": float(rem.abs().max()),
        "G_M": g_m,
        "energy_terms": g_terms,
        "pass": True,
    }


def router_id_weight_split(
    *,
    y_r0: torch.Tensor,
    y_r1_on_s0: torch.Tensor,
    y_r1: torch.Tensor,
) -> dict:
    """p_weight = M1(x1;r1_on_S0)-M1(x1;r0); p_id = M1(x1;r1)-M1(x1;r1_on_S0)."""
    a, b, c = as_f64(y_r0), as_f64(y_r1_on_s0), as_f64(y_r1)
    p_weight = b - a
    p_id = c - b
    p_router = c - a
    rem = p_router - (p_weight + p_id)
    tol = 8 * torch.finfo(torch.float64).eps * max(float(p_router.abs().max()), 1e-300)
    if float(rem.abs().max()) > tol:
        raise RuntimeError(f"p_router=p_weight+p_id closure failed: {float(rem.abs().max())}")
    return {
        "p_weight_l2": l2(p_weight),
        "p_id_l2": l2(p_id),
        "p_router_l2": l2(p_router),
        "closure_max_abs": float(rem.abs().max()),
        "pass": True,
    }
