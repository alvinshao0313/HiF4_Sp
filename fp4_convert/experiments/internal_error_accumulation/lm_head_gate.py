"""LM Head / raw-logit reconstruction identity gate."""
from __future__ import annotations

import torch

from .math_utils import as_f64, l2


def reconstruct_logits_from_hidden(lm_head_weight: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """logits = hidden @ W^T for captured final_norm.normalized."""
    w = lm_head_weight.detach().to(device="cpu", dtype=torch.float32)
    h = hidden.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if h.numel() != w.shape[1]:
        raise RuntimeError(f"hidden dim {h.numel()} != lm_head in_features {w.shape[1]}")
    return (h @ w.T).to(dtype=torch.float32)


def lm_head_raw_logit_identity(
    *,
    captured_logits: torch.Tensor,
    reconstructed_logits: torch.Tensor,
    repeatability_max_abs: float,
    repeatability_l2: float,
) -> dict:
    a = as_f64(captured_logits)
    b = as_f64(reconstructed_logits)
    if a.shape != b.shape:
        raise RuntimeError(f"logit shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    delta = a - b
    max_abs = float(delta.abs().max())
    l2v = float(torch.linalg.vector_norm(delta))
    passed = max_abs <= float(repeatability_max_abs) and l2v <= float(repeatability_l2)
    result = {
        "identity_pass": passed,
        "max_abs": max_abs,
        "l2": l2v,
        "repeatability_max_abs": float(repeatability_max_abs),
        "repeatability_l2": float(repeatability_l2),
        "cosine": float((a * b).sum()) / (l2(a) * l2(b)) if l2(a) and l2(b) else None,
    }
    if not passed:
        raise RuntimeError(f"LM Head raw-logit reconstruction identity failed: {result}")
    return result


def assert_lm_head_params_match(w0: torch.Tensor, w1: torch.Tensor) -> dict:
    a, b = as_f64(w0), as_f64(w1)
    if a.shape != b.shape:
        raise RuntimeError("E0/E1 LM Head shape mismatch")
    if not torch.equal(a, b):
        raise RuntimeError(
            f"E0/E1 LM Head parameters differ: max_abs={float((a-b).abs().max())}"
        )
    return {"pass": True, "numel": int(a.numel()), "shape": list(w0.shape)}
