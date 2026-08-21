"""Stratified bootstrap correlation helpers (no scipy dependency)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Stable average ranks for Spearman (1-based ranks, ties averaged)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_x = x[order]
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        # average rank of positions i..j (1-based)
        avg = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if den == 0.0:
        return 0.0
    return float(np.dot(x, y) / den)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _as_1d(a: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(a), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("expected 1-D series")
    return arr


def _cluster_bootstrap_indices(
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample clusters (primary unit), keep all members of selected clusters."""
    unique = np.unique(cluster_ids)
    if unique.size == 0:
        return np.zeros(0, dtype=np.int64)
    chosen = rng.choice(unique, size=unique.size, replace=True)
    parts = [np.flatnonzero(cluster_ids == c) for c in chosen]
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(parts)


def pearson_with_bootstrap(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    seed: int,
    repeats: int = 1000,
    cluster_ids: Sequence[int] | np.ndarray | None = None,
    confidence: float = 0.95,
) -> dict[str, float]:
    xa = _as_1d(x)
    ya = _as_1d(y)
    if xa.shape != ya.shape:
        raise ValueError("x/y length mismatch")
    if xa.size < 2:
        return {
            "estimate": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "repeats": float(repeats),
            "n": float(xa.size),
        }
    est = _pearson(xa, ya)
    rng = np.random.default_rng(seed)
    if cluster_ids is None:
        # Forbidden for group-level primary analysis; allowed only when caller
        # already aggregated to sample/tensor clusters.
        clusters = np.arange(xa.size)
    else:
        clusters = np.asarray(list(cluster_ids))
        if clusters.shape != xa.shape:
            raise ValueError("cluster_ids length mismatch")
    boots = []
    for _ in range(repeats):
        idx = _cluster_bootstrap_indices(clusters, rng)
        if idx.size < 2:
            boots.append(0.0)
            continue
        boots.append(_pearson(xa[idx], ya[idx]))
    boots_arr = np.sort(np.asarray(boots, dtype=np.float64))
    alpha = 1.0 - confidence
    lo = float(np.quantile(boots_arr, alpha / 2))
    hi = float(np.quantile(boots_arr, 1.0 - alpha / 2))
    return {
        "estimate": float(est),
        "ci_low": lo,
        "ci_high": hi,
        "repeats": float(repeats),
        "n": float(xa.size),
        "n_clusters": float(np.unique(clusters).size),
    }


def mean_ci(
    values: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Sample mean with normal-approx CI; n<2 → degenerate CI at mean."""
    xa = _as_1d(values)
    if xa.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0.0}
    m = float(xa.mean())
    if xa.size < 2:
        return {"mean": m, "ci_low": m, "ci_high": m, "n": float(xa.size)}
    se = float(xa.std(ddof=1) / math.sqrt(xa.size))
    # z for 95% ≈ 1.96; keep simple (no t-table dependency)
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else 1.959963984540054
    return {
        "mean": m,
        "ci_low": m - z * se,
        "ci_high": m + z * se,
        "n": float(xa.size),
    }


def spearman_with_bootstrap(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    seed: int,
    repeats: int = 1000,
    cluster_ids: Sequence[int] | np.ndarray | None = None,
    confidence: float = 0.95,
) -> dict[str, float]:
    xa = _as_1d(x)
    ya = _as_1d(y)
    if xa.shape != ya.shape:
        raise ValueError("x/y length mismatch")
    if xa.size < 2:
        return {
            "estimate": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "repeats": float(repeats),
            "n": float(xa.size),
        }
    est = _spearman(xa, ya)
    rng = np.random.default_rng(seed)
    if cluster_ids is None:
        clusters = np.arange(xa.size)
    else:
        clusters = np.asarray(list(cluster_ids))
        if clusters.shape != xa.shape:
            raise ValueError("cluster_ids length mismatch")
    boots = []
    for _ in range(repeats):
        idx = _cluster_bootstrap_indices(clusters, rng)
        if idx.size < 2:
            boots.append(0.0)
            continue
        boots.append(_spearman(xa[idx], ya[idx]))
    boots_arr = np.sort(np.asarray(boots, dtype=np.float64))
    alpha = 1.0 - confidence
    lo = float(np.quantile(boots_arr, alpha / 2))
    hi = float(np.quantile(boots_arr, 1.0 - alpha / 2))
    return {
        "estimate": float(est),
        "ci_low": lo,
        "ci_high": hi,
        "repeats": float(repeats),
        "n": float(xa.size),
        "n_clusters": float(np.unique(clusters).size),
    }
