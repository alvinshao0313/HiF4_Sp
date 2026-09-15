"""Shared tensor / KL / Fisher utilities for mechanism analysis."""
from __future__ import annotations

import math

import torch


def as_f64(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(type(x))
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("tensor must be finite floating point")
    return x.detach().to(device="cpu", dtype=torch.float64).reshape(-1)


def l2(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(as_f64(x)))


def squared_l2(x: torch.Tensor) -> float:
    v = as_f64(x)
    return float((v * v).sum())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    aa, bb = as_f64(a), as_f64(b)
    na, nb = float(torch.linalg.vector_norm(aa)), float(torch.linalg.vector_norm(bb))
    if na == 0.0 or nb == 0.0:
        return None
    return float((aa * bb).sum()) / (na * nb)


def rel_l2(err: torch.Tensor, ref: torch.Tensor) -> float | None:
    den = l2(ref)
    if den == 0.0:
        return None
    return l2(err) / den


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    x = as_f64(logits)
    x = x - x.max()
    ex = torch.exp(x)
    return ex / ex.sum()


def exact_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p||q) with p=softmax(p_logits), q=softmax(q_logits)."""
    p = softmax_probs(p_logits)
    q = softmax_probs(q_logits)
    return float((p * (torch.log(p.clamp_min(1e-300)) - torch.log(q.clamp_min(1e-300)))).sum())


def fisher_quadratic_kl(p_logits: torch.Tensor, delta_logits: torch.Tensor) -> float:
    """0.5 * delta^T F_0 delta = 0.5 * Var_{j~p0}[delta_j]."""
    p = softmax_probs(p_logits)
    d = as_f64(delta_logits)
    mean = float((p * d).sum())
    var = float((p * (d - mean) ** 2).sum())
    return 0.5 * var


def fisher_inner(p_logits: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> float:
    p = softmax_probs(p_logits)
    vv, ww = as_f64(v), as_f64(w)
    # v^T F w = sum_i p_i v_i w_i - (sum p_i v_i)(sum p_i w_i)
    return float((p * vv * ww).sum() - (p * vv).sum() * (p * ww).sum())


def fisher_cosine(p_logits: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> float | None:
    num = fisher_inner(p_logits, v, w)
    den_v = fisher_inner(p_logits, v, v)
    den_w = fisher_inner(p_logits, w, w)
    if den_v <= 0.0 or den_w <= 0.0:
        return None
    return num / math.sqrt(den_v * den_w)


def sample_bootstrap_ci(
    values: list[float],
    *,
    seed: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> dict:
    if not values:
        raise ValueError("empty bootstrap values")
    g = torch.Generator()
    g.manual_seed(int(seed))
    x = torch.tensor(values, dtype=torch.float64)
    n = x.numel()
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    means = x[idx].mean(dim=1).sort().values
    lo = float(means[int(math.floor((alpha / 2) * n_boot))])
    hi = float(means[min(n_boot - 1, int(math.floor((1 - alpha / 2) * n_boot)))])
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "n": n,
        "n_boot": n_boot,
    }
