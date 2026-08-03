"""Audit whether the G4-level proxy cost can rank real G64 outcomes.

The hierarchical search uses ``c4_cost`` (S1P2 oracle over 4 channels) to pick
G4 tuples, but the deployed cost is ``c64_cost`` (real HiF4 over the enclosing
64-channel block, whose shared scales couple all 64 channels). This module
quantifies the rank correlation between the two on deterministic candidate
sets, without any SciPy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .config import SearchConfig
from .objective import ChannelStatistics, c4_cost, c64_cost


@dataclass(frozen=True)
class ProxyAuditResult:
    spearman: float
    pearson: float
    top1_match: bool
    top5_overlap: float
    n_candidates: int


def _average_ranks(values: Sequence[float]) -> torch.Tensor:
    """1-based average ranks (ties share the mean rank)."""
    v = torch.as_tensor(list(values), dtype=torch.float64)
    order = torch.argsort(v, stable=True)
    sorted_v = v[order]
    ranks = torch.empty(v.numel(), dtype=torch.float64)
    i = 0
    n = v.numel()
    while i < n:
        j = i
        while j + 1 < n and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float64) - a.to(torch.float64).mean()
    b = b.to(torch.float64) - b.to(torch.float64).mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom.item()) == 0.0:
        return 0.0
    return float(((a * b).sum() / denom).item())


def _rank_correlations(
    proxy_costs: Sequence[float], real_costs: Sequence[float]
) -> ProxyAuditResult:
    if len(proxy_costs) != len(real_costs):
        raise ValueError("proxy/real cost lists must have equal length")
    n = len(proxy_costs)
    if n < 2:
        raise ValueError(f"need >= 2 candidates, got {n}")
    proxy_r = _average_ranks(proxy_costs)
    real_r = _average_ranks(real_costs)
    spearman = _pearson(proxy_r, real_r)
    pearson = _pearson(
        torch.as_tensor(list(proxy_costs), dtype=torch.float64),
        torch.as_tensor(list(real_costs), dtype=torch.float64),
    )
    proxy_order = torch.argsort(
        torch.as_tensor(list(proxy_costs), dtype=torch.float64), stable=True
    )
    real_order = torch.argsort(
        torch.as_tensor(list(real_costs), dtype=torch.float64), stable=True
    )
    k = min(5, n)
    top5_overlap = len(set(proxy_order[:k].tolist()) & set(real_order[:k].tolist())) / k
    return ProxyAuditResult(
        spearman=spearman,
        pearson=pearson,
        top1_match=bool(proxy_order[0].item() == real_order[0].item()),
        top5_overlap=top5_overlap,
        n_candidates=n,
    )


def audit_g4_proxy_ranking(
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    base_g64_channels: Sequence[int],
    candidate_g4_groups: Sequence[Sequence[int]],
    stats: ChannelStatistics,
    config: SearchConfig,
) -> ProxyAuditResult:
    """Rank candidates by ``c4_cost`` vs real ``c64_cost`` inside one block.

    ``base_g64_channels`` is the fixed real 64-channel block. The variable pool
    is the union of all candidate channels; the remaining channels of the pool
    (those not in a given candidate) fill the block deterministically so the
    real cost always sees exactly 64 channels.
    """
    base = [int(c) for c in base_g64_channels]
    if len(base) != 64:
        raise ValueError(f"base_g64_channels must have 64 channels, got {len(base)}")
    if len(set(base)) != 64:
        raise ValueError("base_g64_channels must be unique")
    candidates = [tuple(int(c) for c in g) for g in candidate_g4_groups]
    if len(candidates) < 2:
        raise ValueError(f"need >= 2 candidates, got {len(candidates)}")
    base_set = set(base)
    for g in candidates:
        if not (1 <= len(g) <= 4):
            raise ValueError(f"candidate must have 1..4 channels, got {len(g)}")
        if not set(g) <= base_set:
            raise ValueError(f"candidate {g} has channels outside base block")
    pool = sorted(set().union(*(set(g) for g in candidates)))
    fixed = [c for c in base if c not in pool]
    if len(fixed) + len(pool) != 64:
        raise ValueError("pool/fixed decomposition inconsistent")

    proxy_costs = [
        c4_cost(list(g), activation, weight_rows, stats, config) for g in candidates
    ]
    real_costs: list[float] = []
    for g in candidates:
        rest = [c for c in pool if c not in set(g)]
        full64 = fixed + list(g) + rest
        if len(full64) != 64:
            raise ValueError(f"assembled block has {len(full64)} channels, expected 64")
        real_costs.append(c64_cost(full64, activation, weight_rows, stats, config))
    return _rank_correlations(proxy_costs, real_costs)
