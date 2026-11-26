"""Strict residual error ledger over real vLLM captures."""
from __future__ import annotations

from typing import Any

import torch

from .bf16_residual import assert_runtime_add_identity, bf16_fused_residual_add, rounding_residue
from .config import NUM_LAYERS
from .math_utils import as_f64, cosine, l2, rel_l2, squared_l2


def _req(index: dict, key: tuple) -> torch.Tensor:
    if key not in index:
        raise KeyError(f"missing capture key {key}")
    return index[key]


def index_capture_records(records: list[dict]) -> dict[tuple, torch.Tensor]:
    out: dict[tuple, torch.Tensor] = {}
    for row in records:
        key = (
            str(row["sample_key"]),
            int(row["decode_index"]),
            row["layer"] if row["layer"] is None else int(row["layer"]),
            str(row["boundary"]),
            str(row["role"]),
            int(row["tp_rank"]),
        )
        if key in out:
            raise RuntimeError(f"duplicate capture key: {key}")
        out[key] = row["tensor"]
    return out


def extract_layer_tensors(
    index: dict[tuple, torch.Tensor],
    *,
    sample_key: str,
    decode_index: int,
    rank: int = 0,
) -> dict[str, Any]:
    """Map formal residual ledger symbols from captured boundaries."""
    sk, di, r = str(sample_key), int(decode_index), int(rank)
    R: list[torch.Tensor] = []
    R_A: list[torch.Tensor] = []
    A: list[torch.Tensor] = []
    M: list[torch.Tensor] = []
    # R_0 = layer0 layer_in.branch
    R0 = _req(index, (sk, di, 0, "layer_in", "branch", r))
    R.append(R0)
    for layer in range(NUM_LAYERS):
        A_l = _req(index, (sk, di, layer, "o_proj", "tp_reduced", r))
        M_l = _req(index, (sk, di, layer, "moe_out", "tp_reduced", r))
        R_A_l = _req(index, (sk, di, layer, "post_attn_norm", "updated_residual", r))
        A.append(A_l)
        M.append(M_l)
        R_A.append(R_A_l)
        if layer + 1 < NUM_LAYERS:
            R_next = _req(index, (sk, di, layer + 1, "input_norm", "updated_residual", r))
        else:
            R_next = _req(index, (sk, di, None, "final_norm", "updated_residual", r))
        R.append(R_next)
    final_hidden = _req(index, (sk, di, None, "final_norm", "normalized", r))
    return {
        "R": R,  # len 49: R_0 .. R_48
        "R_A": R_A,  # len 48
        "A": A,
        "M": M,
        "final_hidden": final_hidden,
    }


def check_tp_replica(
    index: dict[tuple, torch.Tensor],
    *,
    sample_key: str,
    decode_index: int,
    boundaries: list[tuple[str, str]],
    layers: list[int | None],
) -> dict:
    failures = []
    checked = 0
    for layer in layers:
        for boundary, role in boundaries:
            if boundary == "input_norm" and role == "updated_residual" and layer == 0:
                continue
            if boundary in {"final_norm"} and layer is not None:
                continue
            if boundary != "final_norm" and layer is None:
                continue
            k0 = (sample_key, decode_index, layer, boundary, role, 0)
            k1 = (sample_key, decode_index, layer, boundary, role, 1)
            if k0 not in index or k1 not in index:
                failures.append({"key": [sample_key, decode_index, layer, boundary, role], "reason": "missing"})
                continue
            a, b = as_f64(index[k0]), as_f64(index[k1])
            checked += 1
            if not torch.equal(a, b):
                failures.append(
                    {
                        "key": [sample_key, decode_index, layer, boundary, role],
                        "max_abs": float((a - b).abs().max()),
                        "l2": float(torch.linalg.vector_norm(a - b)),
                    }
                )
    return {"checked": checked, "failures": failures, "pass": not failures}


def runtime_residual_closure_for_variant(
    tensors: dict[str, Any],
    *,
    repeatability_max_abs: float,
    repeatability_l2: float,
) -> dict:
    rows = []
    for layer in range(NUM_LAYERS):
        R_l = tensors["R"][layer]
        A_l = tensors["A"][layer]
        R_A = tensors["R_A"][layer]
        M_l = tensors["M"][layer]
        R_next = tensors["R"][layer + 1]
        attn_gate = assert_runtime_add_identity(
            R_A, R_l, A_l,
            repeatability_max_abs=repeatability_max_abs,
            repeatability_l2=repeatability_l2,
        )
        moe_gate = assert_runtime_add_identity(
            R_next, R_A, M_l,
            repeatability_max_abs=repeatability_max_abs,
            repeatability_l2=repeatability_l2,
        )
        eps_A = rounding_residue(R_A, R_l, A_l)
        eps_M = rounding_residue(R_next, R_A, M_l)
        rows.append(
            {
                "layer": layer,
                "attn_identity": attn_gate,
                "moe_identity": moe_gate,
                "eps_A_l2": l2(eps_A),
                "eps_M_l2": l2(eps_M),
            }
        )
    return {"pass": True, "layers": rows}


def cross_variant_ledger(
    e0: dict[str, Any],
    e1: dict[str, Any],
) -> dict:
    """Build delta R / A / M ledger with BF16 rounding residues."""
    rows = []
    delta_R = []
    delta_A = []
    delta_M = []
    delta_eps_A = []
    delta_eps_M = []
    for layer in range(NUM_LAYERS):
        dR = as_f64(e1["R"][layer]) - as_f64(e0["R"][layer])
        dA = as_f64(e1["A"][layer]) - as_f64(e0["A"][layer])
        dRA = as_f64(e1["R_A"][layer]) - as_f64(e0["R_A"][layer])
        dM = as_f64(e1["M"][layer]) - as_f64(e0["M"][layer])
        dR_next = as_f64(e1["R"][layer + 1]) - as_f64(e0["R"][layer + 1])
        eps_A0 = rounding_residue(e0["R_A"][layer], e0["R"][layer], e0["A"][layer])
        eps_A1 = rounding_residue(e1["R_A"][layer], e1["R"][layer], e1["A"][layer])
        eps_M0 = rounding_residue(e0["R"][layer + 1], e0["R_A"][layer], e0["M"][layer])
        eps_M1 = rounding_residue(e1["R"][layer + 1], e1["R_A"][layer], e1["M"][layer])
        d_eps_A = eps_A1 - eps_A0
        d_eps_M = eps_M1 - eps_M0
        rem_A = dRA - (dR + dA + d_eps_A)
        rem_M = dR_next - (dRA + dM + d_eps_M)
        # FP64 arithmetic roundoff only
        scale_A = max(float(dRA.abs().max()), float(dR.abs().max()), float(dA.abs().max()), float(d_eps_A.abs().max()), 1e-300)
        scale_M = max(float(dR_next.abs().max()), float(dRA.abs().max()), float(dM.abs().max()), float(d_eps_M.abs().max()), 1e-300)
        tol_A = 8 * torch.finfo(torch.float64).eps * scale_A
        tol_M = 8 * torch.finfo(torch.float64).eps * scale_M
        if float(rem_A.abs().max()) > tol_A or float(rem_M.abs().max()) > tol_M:
            raise RuntimeError(
                f"cross-variant residual closure failed at layer {layer}: "
                f"rem_A={float(rem_A.abs().max())} tol_A={tol_A} rem_M={float(rem_M.abs().max())} tol_M={tol_M}"
            )
        g_A = squared_l2(dRA) - squared_l2(dR)
        g_M = squared_l2(dR_next) - squared_l2(dRA)
        rows.append(
            {
                "layer": layer,
                "delta_R_l2": float(torch.linalg.vector_norm(dR)),
                "delta_A_l2": float(torch.linalg.vector_norm(dA)),
                "delta_R_A_l2": float(torch.linalg.vector_norm(dRA)),
                "delta_M_l2": float(torch.linalg.vector_norm(dM)),
                "delta_R_next_l2": float(torch.linalg.vector_norm(dR_next)),
                "G_A": g_A,
                "G_M": g_M,
                "cos_dR_dA": cosine(dR, dA),
                "cos_dRA_dM": cosine(dRA, dM),
                "delta_eps_A_l2": float(torch.linalg.vector_norm(d_eps_A)),
                "delta_eps_M_l2": float(torch.linalg.vector_norm(d_eps_M)),
            }
        )
        delta_R.append(dR)
        delta_A.append(dA)
        delta_M.append(dM)
        delta_eps_A.append(d_eps_A)
        delta_eps_M.append(d_eps_M)
    dR0 = as_f64(e1["R"][0]) - as_f64(e0["R"][0])
    dR48 = as_f64(e1["R"][NUM_LAYERS]) - as_f64(e0["R"][NUM_LAYERS])
    reconstructed = dR0.clone()
    for layer in range(NUM_LAYERS):
        reconstructed = reconstructed + delta_A[layer] + delta_M[layer] + delta_eps_A[layer] + delta_eps_M[layer]
    rem_final = dR48 - reconstructed
    scale = max(float(dR48.abs().max()), 1e-300)
    tol = 32 * torch.finfo(torch.float64).eps * scale * NUM_LAYERS
    if float(rem_final.abs().max()) > tol:
        raise RuntimeError(
            f"48-layer final residual closure failed: max_abs={float(rem_final.abs().max())} tol={tol}"
        )
    u = dR48
    u_sq = float((u * u).sum())
    attributions = []
    for layer in range(NUM_LAYERS):
        s_A = float((delta_A[layer] * u).sum()) / u_sq if u_sq else None
        s_M = float((delta_M[layer] * u).sum()) / u_sq if u_sq else None
        attributions.append({"layer": layer, "s_A": s_A, "s_M": s_M})
    return {
        "pass": True,
        "delta_R0_l2": float(torch.linalg.vector_norm(dR0)),
        "delta_R48_l2": float(torch.linalg.vector_norm(dR48)),
        "final_closure_max_abs": float(rem_final.abs().max()),
        "final_closure_tol": tol,
        "rows": rows,
        "attributions": attributions,
        "delta_A": delta_A,
        "delta_M": delta_M,
        "delta_R": delta_R + [dR48],
        "R48_0_sq": squared_l2(e0["R"][NUM_LAYERS]),
        "delta_R48": dR48,
        "final_hidden_rel_l2": rel_l2(
            as_f64(e1["final_hidden"]) - as_f64(e0["final_hidden"]),
            e0["final_hidden"],
        ),
    }


def coherence_ratio(vectors: list[torch.Tensor]) -> float | None:
    if not vectors:
        return None
    total = torch.zeros_like(as_f64(vectors[0]))
    energy = 0.0
    for v in vectors:
        vv = as_f64(v)
        total = total + vv
        energy += float((vv * vv).sum())
    if energy == 0.0:
        return None
    return float((total * total).sum()) / energy


def reject_layer0_state_reset(layer: int) -> None:
    if int(layer) == 0:
        raise ValueError(
            "layer0 state-reset is forbidden: input_norm does not return "
            "(normalized, updated_residual); layer0 is only a residual-ledger anchor"
        )
