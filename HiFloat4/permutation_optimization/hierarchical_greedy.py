"""Hierarchical greedy construction of G4 / G8 / G64 and local refinement."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .candidate_selection import (
    CandidateDecision,
    CandidateMetrics,
    select_candidate,
)
from .config import LayerSearchResult, SearchConfig
from .objective import (
    ChannelStatistics,
    DeploymentDownContext,
    DeploymentMLPContext,
    build_channel_statistics,
    build_quantized_swiglu_activation,
    c4_cost,
    c64_cost,
    full_layout_hif4_loss,
    pair_compatibility_cost,
    precompute_neighbor_pair_costs,
    sample_weight_rows,
)
from .split_utils import RowSplit, apply_row_split, make_row_split


def _precompute_pair_costs(
    stats: ChannelStatistics,
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    eps: float,
) -> dict[tuple[int, int], float]:
    return precompute_neighbor_pair_costs(activation, weight_rows, stats, eps)


def _pair_cost(
    cache: dict[tuple[int, int], float],
    i: int,
    j: int,
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    eps: float,
) -> float:
    if i == j:
        return 0.0
    a, b = (i, j) if i < j else (j, i)
    key = (a, b)
    if key not in cache:
        cache[key] = pair_compatibility_cost(i, j, activation, weight_rows, stats, eps)
    return cache[key]


def _group_proxy_cost(
    members: Sequence[int],
    cache: dict[tuple[int, int], float],
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    eps: float,
) -> float:
    """Full pairwise mean proxy (reference; used by tests)."""
    if len(members) <= 1:
        return 0.0
    total = 0.0
    n = 0
    for a_i in range(len(members)):
        for b_i in range(a_i + 1, len(members)):
            total += _pair_cost(
                cache, members[a_i], members[b_i], activation, weight_rows, stats, eps
            )
            n += 1
    return total / max(n, 1)


def _pair_cost_adj(
    adj: list[dict[int, float]],
    cache: dict[tuple[int, int], float],
    i: int,
    j: int,
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    eps: float,
) -> float:
    if i == j:
        return 0.0
    d = adj[i]
    if j in d:
        return d[j]
    # Lazy fill (rare: non-neighbor edges during window fallback).
    val = _pair_cost(cache, i, j, activation, weight_rows, stats, eps)
    adj[i][j] = val
    adj[j][i] = val
    return val


def _expand_proxy_adj(
    state: Sequence[int],
    state_proxy: float,
    nxt: int,
    adj: list[dict[int, float]],
    cache: dict[tuple[int, int], float],
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    eps: float,
) -> float:
    k = len(state)
    if k == 0:
        return 0.0
    old_edges = k * (k - 1) // 2
    old_sum = state_proxy * old_edges if old_edges > 0 else 0.0
    add = 0.0
    for m in state:
        add += _pair_cost_adj(adj, cache, m, nxt, activation, weight_rows, stats, eps)
    return (old_sum + add) / (old_edges + k)


def _sorted_insert(state: tuple[int, ...], nxt: int) -> tuple[int, ...]:
    """Insert nxt into an already-sorted tuple (same as tuple(sorted(state+(nxt,))))."""
    out: list[int] = []
    inserted = False
    for x in state:
        if not inserted and nxt < x:
            out.append(nxt)
            inserted = True
        out.append(x)
    if not inserted:
        out.append(nxt)
    return tuple(out)


def build_g4_groups(
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    config: SearchConfig,
) -> list[tuple[int, int, int, int]]:
    """Local-neighbor beam search + exact C4 rerank → disjoint 4-tuples."""
    d_ff = activation.shape[1]
    if d_ff % 64 != 0:
        raise ValueError(f"d_ff must be divisible by 64, got {d_ff}")
    if d_ff % 4 != 0:
        raise ValueError(f"d_ff must be divisible by 4, got {d_ff}")

    eps = config.eps
    pair_cache = _precompute_pair_costs(stats, activation, weight_rows, eps)
    neighbors: list[list[int]] = [row.tolist() for row in stats.neighbors]
    # Symmetric adjacency for O(1) pair lookup in the hot beam loop.
    adj: list[dict[int, float]] = [{} for _ in range(d_ff)]
    for (a, b), c in pair_cache.items():
        adj[a][b] = c
        adj[b][a] = c
    unused_flags = bytearray(b"\x01") * d_ff
    unused_count = d_ff

    order = torch.argsort(stats.primary_scale, stable=True).tolist()
    pos = {ch: i for i, ch in enumerate(order)}
    beam_w = config.beam_width_g4
    cand_win = config.candidate_window
    rerank_k = config.exact_rerank_g4

    def isolation(ch: int) -> float:
        costs: list[float] = []
        for j in neighbors[ch]:
            if unused_flags[j] and j != ch:
                costs.append(_pair_cost_adj(adj, pair_cache, ch, j, activation, weight_rows, stats, eps))
            if len(costs) >= 3:
                break
        if not costs:
            for j in order:
                if unused_flags[j] and j != ch:
                    costs.append(
                        _pair_cost_adj(adj, pair_cache, ch, j, activation, weight_rows, stats, eps)
                    )
                if len(costs) >= 3:
                    break
        if not costs:
            return 0.0
        return sum(costs) / len(costs)

    iso_scores = [isolation(ch) for ch in range(d_ff)]
    seed_order = sorted(range(d_ff), key=lambda c: (iso_scores[c], -c), reverse=True)
    seed_cursor = 0

    groups: list[tuple[int, int, int, int]] = []
    while unused_count > 0:
        while seed_cursor < len(seed_order) and not unused_flags[seed_order[seed_cursor]]:
            seed_cursor += 1
        if seed_cursor >= len(seed_order):
            seed = max(
                (c for c in range(d_ff) if unused_flags[c]),
                key=lambda c: (iso_scores[c], -c),
            )
        else:
            seed = seed_order[seed_cursor]
            seed_cursor += 1

        beam: list[tuple[tuple[int, ...], float]] = [((seed,), 0.0)]
        for _step in range(3):
            candidates: dict[tuple[int, ...], float] = {}
            for state, state_proxy in beam:
                neigh: list[int] = []
                seen: set[int] = set()
                for m in state:
                    for j in neighbors[m]:
                        if unused_flags[j] and j not in state and j not in seen:
                            seen.add(j)
                            neigh.append(j)
                if not neigh:
                    seed_pos = pos[state[0]]
                    radius = cand_win
                    while not neigh:
                        lo = max(0, seed_pos - radius)
                        hi = min(d_ff, seed_pos + radius + 1)
                        for p in range(lo, hi):
                            ch = order[p]
                            if unused_flags[ch] and ch not in state and ch not in seen:
                                seen.add(ch)
                                neigh.append(ch)
                        if lo == 0 and hi == d_ff:
                            break
                        radius = min(d_ff, radius * 2 if radius > 0 else 1)
                    if not neigh:
                        for ch in range(d_ff):
                            if unused_flags[ch] and ch not in state:
                                neigh.append(ch)
                neigh.sort()
                for nxt in neigh:
                    new_members = _sorted_insert(state, nxt)
                    if new_members in candidates:
                        continue
                    candidates[new_members] = _expand_proxy_adj(
                        state,
                        state_proxy,
                        nxt,
                        adj,
                        pair_cache,
                        activation,
                        weight_rows,
                        stats,
                        eps,
                    )
            if not candidates:
                raise RuntimeError(
                    f"Could not expand G4 beam from seed {seed}; unused={unused_count}"
                )
            ranked = sorted(candidates.items(), key=lambda kv: (kv[1], kv[0]))
            beam = ranked[:beam_w]

        complete = [(t, c) for t, c in beam if len(t) == 4]
        if len(complete) < 1:
            raise RuntimeError("G4 beam produced no complete 4-tuples")
        complete.sort(key=lambda kv: (kv[1], kv[0]))
        rerank = complete[:rerank_k]
        best_tuple = None
        best_c4 = math.inf
        for t, _ in rerank:
            cost = c4_cost(t, activation, weight_rows, stats, config)
            if cost < best_c4 or (cost == best_c4 and (best_tuple is None or t < best_tuple)):
                best_c4 = cost
                best_tuple = t
        assert best_tuple is not None and len(best_tuple) == 4
        groups.append(best_tuple)  # type: ignore[arg-type]
        for ch in best_tuple:
            unused_flags[ch] = 0
            unused_count -= 1

    assert len(groups) == d_ff // 4
    return groups


def _g4_peak_trajectories(
    g4_groups: Sequence[Sequence[int]],
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """peak_A: [T, n_g4], peak_W: [R, n_g4]."""
    act = activation.detach().to(device="cpu", dtype=torch.float32)
    w = weight_rows.detach().to(device="cpu", dtype=torch.float32)
    idx = torch.tensor([list(g) for g in g4_groups], dtype=torch.long)  # [n, 4]
    peak_a = act[:, idx].abs().amax(dim=2)
    peak_w = w[:, idx].abs().amax(dim=2)
    return peak_a, peak_w


def _g4_pair_penalty_matrix(
    peak_a: torch.Tensor,
    peak_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Dense [n,n] G4 pair penalties from peak trajectories (symmetric, diag=inf).

    Uses CUDA when available (same float64 formula; finite entries match CPU to ~1e-14).
    Multi-process searches share one GPU via a file lock so workers do not OOM.
    """

    def _one(peak: torch.Tensor, block: int) -> torch.Tensor:
        n = peak.shape[1]
        device = peak.device
        out = torch.zeros(n, n, dtype=torch.float64, device=device)
        p = peak.to(dtype=torch.float64)
        for i0 in range(0, n, block):
            i1 = min(n, i0 + block)
            pi = p[:, i0:i1].unsqueeze(2)  # [T, bi, 1]
            pj = p.unsqueeze(1)  # [T, 1, n]
            log_gap = (torch.log2((pi + eps) / (pj + eps))).abs()
            penalty = torch.relu(log_gap - 1.0) ** 2
            row_w = pi * pi + pj * pj
            num = (penalty * row_w).sum(dim=0)  # [bi, n]
            den = row_w.sum(dim=0).clamp_min(eps)
            out[i0:i1] = num / den
            del log_gap, penalty, row_w, num, den, pi
        out.fill_diagonal_(float("inf"))
        return out

    def _compute_cpu() -> torch.Tensor:
        return 0.5 * _one(peak_a.cpu(), block=128) + 0.5 * _one(peak_w.cpu(), block=128)

    if not torch.cuda.is_available():
        return _compute_cpu()

    # Serialize GPU matrix builds across ProcessPool workers on a shared device.
    import fcntl
    from pathlib import Path

    lock_path = Path("/tmp/hif4_perm_g4_penalty_cuda.lock")
    lock_path.touch(exist_ok=True)
    with lock_path.open("a") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            device = torch.device("cuda")
            # Smaller blocks: peak alloc ~ T*block*n*8 bytes (avoid multi-GB spikes).
            block = 64
            a = peak_a.to(device=device, dtype=torch.float32)
            w = peak_w.to(device=device, dtype=torch.float32)
            out = 0.5 * _one(a, block=block) + 0.5 * _one(w, block=block)
            out_cpu = out.detach().cpu()
            del out, a, w
            torch.cuda.empty_cache()
            return out_cpu
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return _compute_cpu()


def pair_g4_into_g8(
    g4_groups: Sequence[Sequence[int]],
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    config: SearchConfig,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Greedy pair G4s by peak-trajectory 1-bit-tolerant penalty."""
    n = len(g4_groups)
    if n % 2 != 0:
        raise ValueError(f"Number of G4 groups must be even, got {n}")
    peak_a, peak_w = _g4_peak_trajectories(g4_groups, activation, weight_rows)
    cost = _g4_pair_penalty_matrix(peak_a, peak_w, config.eps)  # [n,n]
    unused_mask = torch.ones(n, dtype=torch.bool)
    g8: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    # Initial best-partner cost for hardest-first ordering.
    best_c = cost.min(dim=1).values  # [n]
    seed_order = torch.argsort(best_c, descending=True).tolist()

    for seed in seed_order:
        if not bool(unused_mask[seed].item()):
            continue
        row = cost[seed].clone()
        row[~unused_mask] = float("inf")
        partner = int(torch.argmin(row).item())
        if (not bool(unused_mask[partner].item())) or not math.isfinite(
            float(row[partner].item())
        ):
            raise RuntimeError("Failed to find G4 partner")
        a, b = (seed, partner) if seed < partner else (partner, seed)
        g8.append((tuple(g4_groups[a]), tuple(g4_groups[b])))
        unused_mask[seed] = False
        unused_mask[partner] = False
    return g8


def _g8_flat(g8: Sequence[Sequence[Sequence[int]]]) -> list[int]:
    out: list[int] = []
    for g4a, g4b in g8:  # type: ignore[misc]
        out.extend(list(g4a))
        out.extend(list(g4b))
    return out


def _g8_peak_and_features(
    g8_groups: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return peak_A [T,n], peak_W [R,n], primary_scale [n], features [n,12]."""
    # Proxies stay on CPU; CUDA quantile on wide tensors is far slower here.
    act = activation.detach().to(device="cpu", dtype=torch.float32)
    w = weight_rows.detach().to(device="cpu", dtype=torch.float32)
    idx = torch.tensor(
        [list(g4a) + list(g4b) for g4a, g4b in g8_groups], dtype=torch.long
    )  # [n, 8]
    peak_a = act[:, idx].abs().amax(dim=2)  # [T, n]
    peak_w = w[:, idx].abs().amax(dim=2)  # [R, n]

    def _feat6(peak: torch.Tensor) -> torch.Tensor:
        # peak: [rows, n]
        rms = torch.sqrt((peak * peak).mean(dim=0).clamp_min(eps))
        q50 = torch.quantile(peak, 0.50, dim=0)
        q90 = torch.quantile(peak, 0.90, dim=0)
        q99 = torch.quantile(peak, 0.99, dim=0)
        log_rms = torch.log2(rms.clamp_min(eps))
        log_q50 = torch.log2(q50.clamp_min(eps))
        log_q90 = torch.log2(q90.clamp_min(eps))
        log_q99 = torch.log2(q99.clamp_min(eps))
        log_spread = torch.log2((q99 / q50.clamp_min(eps)).clamp_min(eps))
        near_zero = (peak < 0.01 * q99.unsqueeze(0)).float().mean(dim=0)
        near_zero = torch.where(q99 <= eps, torch.ones_like(near_zero), near_zero)
        return torch.stack([log_rms, log_q50, log_q90, log_q99, log_spread, near_zero], dim=1)

    fa = _feat6(peak_a)
    fw = _feat6(peak_w)
    feats = torch.cat([fa, fw], dim=1)
    mean = feats.mean(dim=0, keepdim=True)
    std = feats.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    feats = (feats - mean) / std
    primary = 0.5 * (fa[:, 3] + fw[:, 3])
    return peak_a, peak_w, primary, feats


def _g8_conflict(
    i: int,
    j: int,
    peak_a: torch.Tensor,
    peak_w: torch.Tensor,
    eps: float,
    cache: dict[tuple[int, int], float],
) -> float:
    """Scalar G8 conflict (kept for tests / fallback). Prefer matrix lookup in pack."""
    if i == j:
        return 0.0
    key = (i, j) if i < j else (j, i)
    if key in cache:
        return cache[key]

    def _one(peak: torch.Tensor) -> float:
        # Match `_g4_pair_penalty_matrix` (float64) for identical values.
        p = peak.to(torch.float64)
        p1 = p[:, i]
        p2 = p[:, j]
        log_gap = (torch.log2((p1 + eps) / (p2 + eps))).abs()
        penalty = torch.relu(log_gap - 1.0) ** 2
        row_w = p1 * p1 + p2 * p2
        den = row_w.sum()
        if float(den.item()) <= eps:
            return 0.0
        return float((penalty * row_w).sum().item() / den.item())

    val = 0.5 * _one(peak_a) + 0.5 * _one(peak_w)
    cache[key] = val
    return val


def _expand_g8_conflict_proxy(
    state: Sequence[int],
    state_proxy: float,
    nxt: int,
    conflict_rows: list[list[float]],
) -> float:
    """Incremental mean G8-conflict after appending ``nxt``."""
    k = len(state)
    if k == 0:
        return 0.0
    old_edges = k * (k - 1) // 2
    old_sum = state_proxy * old_edges if old_edges > 0 else 0.0
    add = 0.0
    row = conflict_rows[nxt]
    for m in state:
        add += row[m]
    return (old_sum + add) / (old_edges + k)


def pack_g8_into_g64(
    g8_groups: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    config: SearchConfig,
) -> list[list[list[list[int]]]]:
    """Pack eight G8s into each G64 via local beam + real C64."""
    n = len(g8_groups)
    if n % 8 != 0:
        raise ValueError(f"Number of G8 groups must be divisible by 8, got {n}")
    eps = config.eps
    peak_a, peak_w, primary, _feats = _g8_peak_and_features(
        g8_groups, activation, weight_rows, eps
    )
    # Same formula as repeated _g8_conflict; one dense matrix replaces .item() storms.
    conflict = _g4_pair_penalty_matrix(peak_a, peak_w, eps)
    conflict_rows = conflict.tolist()
    order = torch.argsort(primary, stable=True).tolist()
    pos_of = {g8_i: p for p, g8_i in enumerate(order)}
    unused_flags = bytearray(b"\x01") * n
    unused_count = n

    def isolation(i: int) -> float:
        p = pos_of[i]
        costs: list[float] = []
        half = config.candidate_window
        while len(costs) < 3:
            lo = max(0, p - half)
            hi = min(n, p + half + 1)
            for q in range(lo, hi):
                j = order[q]
                if j != i:
                    costs.append(conflict_rows[i][j])
                if len(costs) >= 3:
                    break
            if lo == 0 and hi == n:
                break
            half = min(n, half * 2 if half > 0 else 1)
        if not costs:
            return 0.0
        return sum(costs[:3]) / len(costs[:3])

    iso_scores = {i: isolation(i) for i in range(n)}
    seed_order = sorted(range(n), key=lambda i: (iso_scores[i], -i), reverse=True)
    seed_cursor = 0

    def local_candidates(members: Sequence[int]) -> list[int]:
        member_set = set(members)
        cand: set[int] = set()
        half = config.candidate_window
        for m in members:
            p = pos_of[m]
            lo = max(0, p - half)
            hi = min(n, p + half + 1)
            for q in range(lo, hi):
                j = order[q]
                if unused_flags[j] and j not in member_set:
                    cand.add(j)
        if not cand:
            for j in sorted(i for i in range(n) if unused_flags[i]):
                if j not in member_set:
                    cand.add(j)
                if len(cand) >= config.neighbor_k:
                    break
        return sorted(cand)

    g64_out: list[list[list[list[int]]]] = []
    while unused_count > 0:
        while seed_cursor < len(seed_order) and not unused_flags[seed_order[seed_cursor]]:
            seed_cursor += 1
        if seed_cursor >= len(seed_order):
            seed = next(i for i in range(n) if unused_flags[i])
        else:
            seed = seed_order[seed_cursor]
            seed_cursor += 1

        beam: list[tuple[tuple[int, ...], float]] = [((seed,), 0.0)]
        for _ in range(7):
            cand_map: dict[tuple[int, ...], float] = {}
            for state, state_proxy in beam:
                for nxt in local_candidates(state):
                    new_state = tuple(sorted(state + (nxt,)))
                    if new_state in cand_map:
                        continue
                    cand_map[new_state] = _expand_g8_conflict_proxy(
                        state, state_proxy, nxt, conflict_rows
                    )
            if not cand_map:
                raise RuntimeError("G64 beam expansion failed")
            ranked = sorted(cand_map.items(), key=lambda kv: (kv[1], kv[0]))
            beam = ranked[: config.beam_width_g64]

        complete = [t for t, _ in beam if len(t) == 8]
        if not complete:
            raise RuntimeError("No complete G64 candidates")
        best_set = None
        best_cost = math.inf
        for g8_idx_tuple in complete:
            channels: list[int] = []
            for gi in g8_idx_tuple:
                g4a, g4b = g8_groups[gi]
                channels.extend(list(g4a))
                channels.extend(list(g4b))
            assert len(channels) == 64
            cost = c64_cost(channels, activation, weight_rows, stats, config)
            if cost < best_cost or (
                cost == best_cost and (best_set is None or g8_idx_tuple < best_set)
            ):
                best_cost = cost
                best_set = g8_idx_tuple
        assert best_set is not None
        block: list[list[list[int]]] = []
        for gi in best_set:
            g4a, g4b = g8_groups[gi]
            block.append([list(g4a), list(g4b)])
            unused_flags[gi] = 0
            unused_count -= 1
        g64_out.append(block)
    return g64_out


def flatten_hierarchy(g64_groups: Sequence) -> torch.Tensor:
    """Flatten G64→G8→G4→channel into perm[new]=old."""
    channels: list[int] = []
    for block in g64_groups:
        for g8 in block:
            for g4 in g8:
                channels.extend(list(g4))
    perm = torch.tensor(channels, dtype=torch.long)
    d = perm.numel()
    if d % 64 != 0:
        raise ValueError(f"flatten_hierarchy length {d} not divisible by 64")
    uniq = torch.unique(perm)
    if uniq.numel() != d:
        raise ValueError("flatten_hierarchy produced duplicate or missing channels")
    return perm


def _channels_of_g64(block) -> list[int]:
    out: list[int] = []
    for g8 in block:
        for g4 in g8:
            out.extend(list(g4))
    return out


def _validate_hierarchy(g64_groups: list, d_ff: int) -> None:
    flat = flatten_hierarchy(g64_groups)
    if flat.numel() != d_ff:
        raise RuntimeError(f"hierarchy size {flat.numel()} != d_ff {d_ff}")
    if torch.unique(flat).numel() != d_ff:
        raise RuntimeError("hierarchy has duplicates or missing channels")


def refine_hierarchy(
    g64_groups: list,
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    config: SearchConfig,
) -> tuple[list, list[float]]:
    """Best-improvement G8/G4/channel swaps on the worst blocks."""
    d_ff = activation.shape[1]
    _validate_hierarchy(g64_groups, d_ff)
    # Deep copy mutable structure
    import copy

    groups = copy.deepcopy(g64_groups)
    n_blocks = len(groups)
    # One batched HiF4 pass instead of n_blocks individual c64 calls.
    init_perm = flatten_hierarchy(groups)
    mean0, block_losses = full_layout_hif4_loss(
        init_perm, activation, weight_rows, stats, config
    )
    history: list[float] = [float(mean0)]

    def total_loss() -> float:
        return float(sum(block_losses) / n_blocks)

    # Neighbor maps for G8/G4 based on features of their channels' primary scale.
    # Build G8 identity list across blocks: (block_idx, g8_idx)
    def g8_primary(block_i: int, g8_i: int) -> float:
        ch = []
        for g4 in groups[block_i][g8_i]:
            ch.extend(g4)
        return float(stats.primary_scale[ch].mean().item())

    def g4_primary(block_i: int, g8_i: int, g4_i: int) -> float:
        ch = groups[block_i][g8_i][g4_i]
        return float(stats.primary_scale[ch].mean().item())

    max_swaps = config.refine_max_swaps_per_stage
    for _pass in range(config.refine_passes):
        # ---- G8 swaps ----
        improved = True
        swaps_done = 0
        while improved and swaps_done < max_swaps:
            improved = False
            worst = sorted(range(n_blocks), key=lambda b: (-block_losses[b], b))[
                : config.refine_bad_blocks
            ]
            best_delta = 0.0
            best_move = None  # (b1, g8_1, b2, g8_2)
            # Candidate G8s: near in primary scale
            all_g8 = [(b, g) for b in range(n_blocks) for g in range(8)]
            prim = {(b, g): g8_primary(b, g) for b, g in all_g8}
            ordered = sorted(all_g8, key=lambda bg: prim[bg])
            pos = {bg: i for i, bg in enumerate(ordered)}
            for b1 in worst:
                for g1 in range(8):
                    p = pos[(b1, g1)]
                    lo = max(0, p - config.candidate_window)
                    hi = min(len(ordered), p + config.candidate_window + 1)
                    for q in range(lo, hi):
                        b2, g2 = ordered[q]
                        if (b2, g2) == (b1, g1):
                            continue
                        if b1 == b2:
                            continue  # swap within same block is noop for C64 of that block only; skip
                        # Try swap
                        g8_a = groups[b1][g1]
                        g8_b = groups[b2][g2]
                        groups[b1][g1] = g8_b
                        groups[b2][g2] = g8_a
                        new_l1 = c64_cost(
                            _channels_of_g64(groups[b1]), activation, weight_rows, stats, config
                        )
                        new_l2 = c64_cost(
                            _channels_of_g64(groups[b2]), activation, weight_rows, stats, config
                        )
                        delta = (block_losses[b1] + block_losses[b2]) - (new_l1 + new_l2)
                        # revert
                        groups[b1][g1] = g8_a
                        groups[b2][g2] = g8_b
                        if delta > best_delta + config.improvement_tol:
                            best_delta = delta
                            best_move = (b1, g1, b2, g2, new_l1, new_l2)
            if best_move is not None and best_delta > config.improvement_tol:
                b1, g1, b2, g2, new_l1, new_l2 = best_move
                groups[b1][g1], groups[b2][g2] = groups[b2][g2], groups[b1][g1]
                block_losses[b1] = new_l1
                block_losses[b2] = new_l2
                history.append(total_loss())
                improved = True
                swaps_done += 1
                _validate_hierarchy(groups, d_ff)
            else:
                break

        # ---- G4 swaps ----
        improved = True
        swaps_done = 0
        while improved and swaps_done < max_swaps:
            improved = False
            worst = sorted(range(n_blocks), key=lambda b: (-block_losses[b], b))[
                : config.refine_bad_blocks
            ]
            best_delta = 0.0
            best_move = None
            all_g4 = [
                (b, g8i, g4i)
                for b in range(n_blocks)
                for g8i in range(8)
                for g4i in range(2)
            ]
            prim4 = {k: g4_primary(*k) for k in all_g4}
            ordered4 = sorted(all_g4, key=lambda k: prim4[k])
            pos4 = {k: i for i, k in enumerate(ordered4)}
            for b1 in worst:
                for g8_1 in range(8):
                    for g4_1 in range(2):
                        p = pos4[(b1, g8_1, g4_1)]
                        lo = max(0, p - config.candidate_window)
                        hi = min(len(ordered4), p + config.candidate_window + 1)
                        for q in range(lo, hi):
                            b2, g8_2, g4_2 = ordered4[q]
                            if (b2, g8_2, g4_2) == (b1, g8_1, g4_1):
                                continue
                            if b1 == b2:
                                continue
                            g4_a = groups[b1][g8_1][g4_1]
                            g4_b = groups[b2][g8_2][g4_2]
                            groups[b1][g8_1][g4_1] = g4_b
                            groups[b2][g8_2][g4_2] = g4_a
                            new_l1 = c64_cost(
                                _channels_of_g64(groups[b1]),
                                activation,
                                weight_rows,
                                stats,
                                config,
                            )
                            new_l2 = c64_cost(
                                _channels_of_g64(groups[b2]),
                                activation,
                                weight_rows,
                                stats,
                                config,
                            )
                            delta = (block_losses[b1] + block_losses[b2]) - (new_l1 + new_l2)
                            groups[b1][g8_1][g4_1] = g4_a
                            groups[b2][g8_2][g4_2] = g4_b
                            if delta > best_delta + config.improvement_tol:
                                best_delta = delta
                                best_move = (b1, g8_1, g4_1, b2, g8_2, g4_2, new_l1, new_l2)
            if best_move is not None and best_delta > config.improvement_tol:
                b1, g8_1, g4_1, b2, g8_2, g4_2, new_l1, new_l2 = best_move
                groups[b1][g8_1][g4_1], groups[b2][g8_2][g4_2] = (
                    groups[b2][g8_2][g4_2],
                    groups[b1][g8_1][g4_1],
                )
                block_losses[b1] = new_l1
                block_losses[b2] = new_l2
                history.append(total_loss())
                improved = True
                swaps_done += 1
                _validate_hierarchy(groups, d_ff)
            else:
                break

        # ---- Channel swaps ----
        improved = True
        swaps_done = 0
        while improved and swaps_done < max_swaps:
            improved = False
            worst = sorted(range(n_blocks), key=lambda b: (-block_losses[b], b))[
                : config.refine_bad_blocks
            ]
            best_delta = 0.0
            best_move = None
            # Map channel -> (block, g8, g4, pos_in_g4)
            loc: dict[int, tuple[int, int, int, int]] = {}
            for b in range(n_blocks):
                for g8i in range(8):
                    for g4i in range(2):
                        for pi, ch in enumerate(groups[b][g8i][g4i]):
                            loc[ch] = (b, g8i, g4i, pi)
            for b1 in worst:
                chs1 = _channels_of_g64(groups[b1])
                for ch1 in chs1:
                    # Limit to a prefix of the neighbor list for tractable best-improvement.
                    for ch2 in stats.neighbors[ch1].tolist()[: min(8, config.neighbor_k)]:
                        if ch2 not in loc:
                            continue
                        b2, g8_2, g4_2, p2 = loc[ch2]
                        if b1 == b2:
                            continue
                        b1_, g8_1, g4_1, p1 = loc[ch1]
                        # Swap
                        groups[b1_][g8_1][g4_1][p1], groups[b2][g8_2][g4_2][p2] = (
                            groups[b2][g8_2][g4_2][p2],
                            groups[b1_][g8_1][g4_1][p1],
                        )
                        new_l1 = c64_cost(
                            _channels_of_g64(groups[b1_]),
                            activation,
                            weight_rows,
                            stats,
                            config,
                        )
                        new_l2 = c64_cost(
                            _channels_of_g64(groups[b2]),
                            activation,
                            weight_rows,
                            stats,
                            config,
                        )
                        delta = (block_losses[b1_] + block_losses[b2]) - (new_l1 + new_l2)
                        # revert
                        groups[b1_][g8_1][g4_1][p1], groups[b2][g8_2][g4_2][p2] = (
                            groups[b2][g8_2][g4_2][p2],
                            groups[b1_][g8_1][g4_1][p1],
                        )
                        if delta > best_delta + config.improvement_tol:
                            best_delta = delta
                            best_move = (
                                b1_,
                                g8_1,
                                g4_1,
                                p1,
                                b2,
                                g8_2,
                                g4_2,
                                p2,
                                new_l1,
                                new_l2,
                            )
            if best_move is not None and best_delta > config.improvement_tol:
                b1_, g8_1, g4_1, p1, b2, g8_2, g4_2, p2, new_l1, new_l2 = best_move
                groups[b1_][g8_1][g4_1][p1], groups[b2][g8_2][g4_2][p2] = (
                    groups[b2][g8_2][g4_2][p2],
                    groups[b1_][g8_1][g4_1][p1],
                )
                block_losses[b1_] = new_l1
                block_losses[b2] = new_l2
                history.append(total_loss())
                improved = True
                swaps_done += 1
                _validate_hierarchy(groups, d_ff)
            else:
                break

    # Monotone check
    for i in range(1, len(history)):
        if history[i] > history[i - 1] + 1e-12:
            raise RuntimeError(f"loss history not monotone: {history}")
    return groups, history


def _random_perm(d_ff: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randperm(d_ff, generator=gen)


def _swap_ranges(perm: torch.Tensor, start1: int, start2: int, width: int) -> torch.Tensor:
    new = perm.clone()
    r1 = slice(start1, start1 + width)
    r2 = slice(start2, start2 + width)
    new[r1] = perm[r2]
    new[r2] = perm[r1]
    return new


def _enumerate_local_moves(
    perm: torch.Tensor,
    block_losses: list[float],
    stats: ChannelStatistics,
) -> list[tuple[float, str, torch.Tensor]]:
    """Deterministic bounded move set: (proxy priority, tie-break key, perm).

    A. swap two G4 slots within a high-loss G64 block;
    B. swap two G8 slots between adjacent high-loss blocks;
    C. single-channel swap between a high-loss block and its nearest-scale block.
    Priority is the (summed) current real block loss of the affected blocks;
    only the top ``refine_candidates_per_round`` get real-loss evaluation.
    """
    d_ff = int(perm.numel())
    n_blocks = d_ff // 64
    ps = stats.primary_scale
    moves: list[tuple[float, str, torch.Tensor]] = []
    worst_blocks = sorted(range(n_blocks), key=lambda b: (-block_losses[b], b))
    block_ps = [float(ps[perm[64 * b : 64 * b + 64]].mean().item()) for b in range(n_blocks)]

    for b in worst_blocks[:8]:
        pairs_a = [(i, j) for i in range(16) for j in range(i + 1, 16)][:8]
        for i, j in pairs_a:
            moves.append(
                (
                    block_losses[b],
                    f"A_{b}_{i}_{j}",
                    _swap_ranges(perm, 64 * b + 4 * i, 64 * b + 4 * j, 4),
                )
            )
        for b2 in (b - 1, b + 1):
            if 0 <= b2 < n_blocks:
                pairs_b = [(i, j) for i in range(8) for j in range(8)][:4]
                for i, j in pairs_b:
                    moves.append(
                        (
                            block_losses[b] + block_losses[b2],
                            f"B_{b}_{b2}_{i}_{j}",
                            _swap_ranges(perm, 64 * b + 8 * i, 64 * b2 + 8 * j, 8),
                        )
                    )
        if n_blocks < 2:
            continue
        others = sorted(
            (abs(block_ps[b2] - block_ps[b]), b2)
            for b2 in range(n_blocks)
            if b2 != b
        )
        b2 = others[0][1]
        block_ch = perm[64 * b2 : 64 * b2 + 64]
        for pos_in_block in (0, 16, 32, 48):
            p = 64 * b + pos_in_block
            ch = perm[p]
            q = 64 * b2 + int(
                torch.argmin((ps[block_ch] - ps[ch]).abs()).item()
            )
            new = perm.clone()
            new[p] = perm[q]
            new[q] = perm[p]
            moves.append(
                (
                    block_losses[b] + block_losses[b2],
                    f"C_{b}_{b2}_{pos_in_block}",
                    new,
                )
            )
    return moves


def seeded_local_refine(
    seed_perms: dict[str, torch.Tensor],
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    config: SearchConfig,
) -> tuple[torch.Tensor, str, dict]:
    """Budgeted best-improvement local search from each seed permutation.

    Only accepts strict real ``full_layout_hif4_loss`` improvements, so the
    returned permutation is never worse than its source seed under the real
    proxy. Returns (best_perm, source_seed_name, info).
    """
    from .objective import batched_full_layout_hif4_loss

    results: dict[str, tuple[torch.Tensor, float, dict]] = {}
    for seed_name, seed_perm in seed_perms.items():
        perm = seed_perm.detach().to(dtype=torch.long).clone()
        loss, block_losses = full_layout_hif4_loss(
            perm, activation, weight_rows, stats, config
        )
        evaluated = 0
        rounds_done = 0
        history = [float(loss)]
        for _ in range(config.refine_max_rounds):
            moves = _enumerate_local_moves(perm, block_losses, stats)
            if not moves:
                break
            moves.sort(key=lambda m: (-m[0], m[1]))
            cand = [m[2] for m in moves[: config.refine_candidates_per_round]]
            losses = batched_full_layout_hif4_loss(
                torch.stack(cand), activation, weight_rows, stats, config
            )
            evaluated += int(losses.numel())
            best_i = int(torch.argmin(losses).item())
            best_loss = float(losses[best_i].item())
            if best_loss < loss - config.refine_min_proxy_gain:
                perm = cand[best_i]
                loss = best_loss
                _, block_losses = full_layout_hif4_loss(
                    perm, activation, weight_rows, stats, config
                )
                history.append(float(loss))
                rounds_done += 1
            else:
                break
        results[seed_name] = (
            perm,
            float(loss),
            {
                "evaluated_candidates": evaluated,
                "rounds": rounds_done,
                "history": history,
            },
        )
    best_seed = min(results, key=lambda s: (results[s][1], s))
    best_perm, best_loss, info = results[best_seed]
    out = dict(info)
    out["source_seed"] = best_seed
    out["final_proxy_loss"] = best_loss
    out["by_seed"] = {
        name: {
            "evaluated_candidates": r[2]["evaluated_candidates"],
            "rounds": r[2]["rounds"],
            "final_proxy_loss": r[1],
        }
        for name, r in results.items()
    }
    return best_perm, best_seed, out


def _build_candidate_pool(
    d_ff: int, stats: ChannelStatistics, hier_perm: torch.Tensor
) -> list[tuple[str, torch.Tensor, bool]]:
    """(name, permutation, eligible_for_deployment) for every candidate.

    Random permutations are diagnostic-only negative controls.
    """
    identity = torch.arange(d_ff, dtype=torch.long)
    q99_desc = torch.argsort(stats.primary_scale, descending=True, stable=True).to(
        dtype=torch.long
    )
    q99_asc = torch.argsort(stats.primary_scale, descending=False, stable=True).to(
        dtype=torch.long
    )
    return [
        ("identity", identity, True),
        ("q99_sort_desc", q99_desc, True),
        ("q99_sort_asc", q99_asc, True),
        ("hierarchical", hier_perm, True),
        ("random_seed_43", _random_perm(d_ff, 43), False),
        ("random_seed_44", _random_perm(d_ff, 44), False),
        ("random_seed_45", _random_perm(d_ff, 45), False),
    ]


def _evaluate_pool_on_contexts(
    pool: list[tuple[str, torch.Tensor, bool]],
    contexts: list,
) -> list[CandidateMetrics]:
    out: list[CandidateMetrics] = []
    for name, perm, eligible in pool:
        split_metrics = tuple(ctx.evaluate(perm) for ctx in contexts)
        out.append(
            CandidateMetrics(
                name=name,
                permutation=perm,
                split_metrics=split_metrics,
                eligible_for_deployment=eligible,
            )
        )
    return out


def _validation_context_rows(
    n_rows: int, config: SearchConfig
) -> list[tuple[int, torch.Tensor]]:
    """Validation row indices for each validation seed (evaluation only).

    Search is restricted to the fixed ``config.seed`` search rows; these seeds
    only generate evaluation rows and are recorded for audit.
    """
    out: list[tuple[int, torch.Tensor]] = []
    for vseed in config.validation_seeds:
        vsplit = make_row_split(n_rows, config.validation_fraction, vseed)
        out.append((vseed, vsplit.validation_idx))
    return out


def _assemble_result(
    layer_name: str,
    decision: CandidateDecision,
    candidates: list[CandidateMetrics],
    proxy_losses: dict[str, float],
    g4: list,
    g8: list,
    g64: list,
    loss_hist: list[float],
    extra: dict,
) -> LayerSearchResult:
    agg = decision.aggregate_metrics
    structured = [
        c for c in candidates if c.eligible_for_deployment and c.name != "identity"
    ]
    best_structured = min(
        structured, key=lambda c: (agg[c.name]["mean_total_nrmse"], c.name)
    )
    baseline_metrics: dict[str, dict[str, float]] = {}
    for c in candidates:
        entry = {
            "hif4_loss": float(proxy_losses[c.name]),
            "output_nrmse": float(agg[c.name]["mean_total_nrmse"]),
        }
        entry.update({k: float(v) for k, v in agg[c.name].items()})
        baseline_metrics[c.name] = entry
    g4_lists = [list(t) for t in g4]
    g8_lists = [list(a) + list(b) for a, b in g8]
    g64_lists = [_channels_of_g64(b) for b in g64]
    return LayerSearchResult(
        layer_name=layer_name,
        permutation=decision.selected_permutation,
        candidate_permutation=best_structured.permutation,
        baseline_metrics=baseline_metrics,
        identity_hif4_loss=float(proxy_losses["identity"]),
        optimized_hif4_loss=float(proxy_losses[best_structured.name]),
        identity_output_nrmse=float(agg["identity"]["mean_total_nrmse"]),
        optimized_output_nrmse=float(agg[best_structured.name]["mean_total_nrmse"]),
        accepted=decision.accepted,
        g4_groups=g4_lists,
        g8_groups=g8_lists,
        g64_groups=g64_lists,
        extra=extra,
    )


def optimize_layer_permutation(
    layer_name: str,
    mlp_input: torch.Tensor,
    up_weight: torch.Tensor | None = None,
    gate_weight: torch.Tensor | None = None,
    down_weight: torch.Tensor | None = None,
    config: SearchConfig | None = None,
) -> LayerSearchResult:
    """Full single-layer search over a unified candidate pool.

    Full path: pass ``mlp_input, up_weight, gate_weight, down_weight``; search
    uses the real W4A4 SwiGLU activation on the fixed seed-42 search rows;
    acceptance is decided by deployment-consistent ``total_nrmse`` on
    independent validation splits (``config.validation_seeds``).

    Down-only path (tests / no up-gate): ``optimize_layer_permutation(name, act, down_weight=wd, config=cfg)``
    where ``mlp_input`` is the down input activation and ``down_weight`` is the
    down projection weight [d_model, d_ff]; same pool and acceptance rule.
    """
    if config is None:
        raise ValueError("config must be provided")
    full_path = up_weight is not None and gate_weight is not None and down_weight is not None
    eval_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not full_path:
        # Down-only path: mlp_input is down activation; down_weight is [d_model, d_ff].
        if down_weight is None or up_weight is not None or gate_weight is not None:
            raise ValueError(
                "Down-only path expects optimize_layer_permutation(name, act, down_weight=wd, config=cfg)"
            )
        act = mlp_input.detach().to(device="cpu", dtype=torch.float32).contiguous()
        wd = down_weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
        d_ff = act.shape[1]
        if d_ff % 64 != 0:
            raise ValueError(f"d_ff must be divisible by 64, got {d_ff}")
        if wd.shape[1] != d_ff:
            raise ValueError(f"down_weight shape {wd.shape} mismatches act d_ff={d_ff}")
        split = make_row_split(act.shape[0], config.validation_fraction, config.seed)
        search_act, val_act = apply_row_split(act, split)
        weight_rows = sample_weight_rows(wd, config.weight_rows, config.seed)
        stats = build_channel_statistics(search_act, weight_rows, config)
        g4 = build_g4_groups(search_act, weight_rows, stats, config)
        g8 = pair_g4_into_g8(g4, search_act, weight_rows, config)
        g64 = pack_g8_into_g64(g8, search_act, weight_rows, stats, config)
        g64, loss_hist = refine_hierarchy(g64, search_act, weight_rows, stats, config)
        hier_perm = flatten_hierarchy(g64)

        pool = _build_candidate_pool(d_ff, stats, hier_perm)
        refinement_info: dict = {"enabled": False}
        if config.refine_enabled:
            seed_perms = {
                name: perm for name, perm, _e in pool if name in config.refine_seed_candidates
            }
            refined_perm, _source, refinement_info = seeded_local_refine(
                seed_perms, search_act, weight_rows, stats, config
            )
            refinement_info["enabled"] = True
            pool = pool + [("hierarchical_refined", refined_perm, True)]
        proxy_losses = {
            name: float(full_layout_hif4_loss(perm, search_act, weight_rows, stats, config)[0])
            for name, perm, _ in pool
        }
        val_rows = _validation_context_rows(act.shape[0], config)
        contexts = [
            DeploymentDownContext(
                act.index_select(0, rows.to(device=act.device)), wd, eval_device
            )
            for _seed, rows in val_rows
        ]
        candidates = _evaluate_pool_on_contexts(pool, contexts)
        decision = select_candidate(candidates, config)
        extra = {
            "loss_history": loss_hist,
            "split": _split_metadata(split, config.seed),
            "validation_seeds": [int(s) for s in config.validation_seeds],
            "validation_split_indices": {
                str(int(s)): rows.tolist() for s, rows in val_rows
            },
            "selected_candidate": decision.selected_name,
            "rejection_reason": decision.rejection_reason,
            "candidate_metrics": {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in decision.aggregate_metrics.items()
            },
            "refinement": refinement_info,
            "proxy_audit": (
                _run_proxy_audit(g64, search_act, weight_rows, stats, config)
                if config.proxy_audit_enabled
                else {"enabled": False}
            ),
            "candidate_permutations": {
                name: perm.detach().to(device="cpu", dtype=torch.long).contiguous()
                for name, perm, _e in pool
            },
            "path": "down_only",
        }
        return _assemble_result(
            layer_name, decision, candidates, proxy_losses, g4, g8, g64, loss_hist, extra
        )

    x = mlp_input.detach().to(device="cpu", dtype=torch.float32).contiguous()
    wu = up_weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    wg = gate_weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    wd = down_weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if x.ndim != 2 or wu.ndim != 2 or wg.ndim != 2 or wd.ndim != 2:
        raise ValueError("X/Wu/Wg/Wd must be 2-D")
    if wu.shape != wg.shape:
        raise ValueError(f"up/gate weight shape mismatch: {wu.shape} vs {wg.shape}")
    d_ff, d_model = wu.shape
    if x.shape[1] != d_model or wd.shape != (d_model, d_ff):
        raise ValueError(
            f"shape mismatch: X {x.shape}, Wu {wu.shape}, Wd {wd.shape}"
        )
    if d_ff % 64 != 0:
        raise ValueError(f"d_ff must be divisible by 64, got {d_ff}")
    if d_model % 64 != 0:
        raise ValueError(f"d_model must be divisible by 64 for X quant, got {d_model}")

    act = build_quantized_swiglu_activation(x, wu, wg).to(device="cpu").contiguous()
    split = make_row_split(x.shape[0], config.validation_fraction, config.seed)
    search_x, val_x = apply_row_split(x, split)
    search_act, val_act = apply_row_split(act, split)
    weight_rows = sample_weight_rows(wd, config.weight_rows, config.seed)
    stats = build_channel_statistics(search_act, weight_rows, config)

    g4 = build_g4_groups(search_act, weight_rows, stats, config)
    g8 = pair_g4_into_g8(g4, search_act, weight_rows, config)
    g64 = pack_g8_into_g64(g8, search_act, weight_rows, stats, config)
    g64, loss_hist = refine_hierarchy(g64, search_act, weight_rows, stats, config)

    hier_perm = flatten_hierarchy(g64)
    pool = _build_candidate_pool(d_ff, stats, hier_perm)
    refinement_info = {"enabled": False}
    if config.refine_enabled:
        seed_perms = {
            name: perm for name, perm, _e in pool if name in config.refine_seed_candidates
        }
        refined_perm, _source, refinement_info = seeded_local_refine(
            seed_perms, search_act, weight_rows, stats, config
        )
        refinement_info["enabled"] = True
        pool = pool + [("hierarchical_refined", refined_perm, True)]
    proxy_losses = {
        name: float(full_layout_hif4_loss(perm, search_act, weight_rows, stats, config)[0])
        for name, perm, _ in pool
    }
    val_rows = _validation_context_rows(x.shape[0], config)
    contexts = [
        DeploymentMLPContext(
            x.index_select(0, rows.to(device=x.device)), wu, wg, wd, eval_device
        )
        for _seed, rows in val_rows
    ]
    candidates = _evaluate_pool_on_contexts(pool, contexts)
    decision = select_candidate(candidates, config)
    extra = {
        "loss_history": loss_hist,
        "split": _split_metadata(split, config.seed),
        "validation_seeds": [int(s) for s in config.validation_seeds],
        "validation_split_indices": {
            str(int(s)): rows.tolist() for s, rows in val_rows
        },
        "selected_candidate": decision.selected_name,
        "rejection_reason": decision.rejection_reason,
        "candidate_metrics": {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in decision.aggregate_metrics.items()
        },
        "search_x_rows": int(search_x.shape[0]),
        "validation_x_rows": int(val_x.shape[0]),
        "validation_act_rows": int(val_act.shape[0]),
        "refinement": refinement_info,
        "proxy_audit": (
            _run_proxy_audit(g64, search_act, weight_rows, stats, config)
            if config.proxy_audit_enabled
            else {"enabled": False}
        ),
        "candidate_permutations": {
            name: perm.detach().to(device="cpu", dtype=torch.long).contiguous()
            for name, perm, _e in pool
        },
        "path": "full_mlp",
    }
    return _assemble_result(
        layer_name, decision, candidates, proxy_losses, g4, g8, g64, loss_hist, extra
    )


def _run_proxy_audit(
    g64: list,
    search_act: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    config: SearchConfig,
) -> dict:
    """C4-proxy vs real-G64 ranking audit on the first constructed block."""
    import itertools
    from dataclasses import asdict

    from .proxy_audit import audit_g4_proxy_ranking

    base = _channels_of_g64(g64[0])
    pool16 = base[:16]
    combos = list(itertools.combinations(pool16, 4))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(config.seed)
    sel = torch.randperm(len(combos), generator=gen)[: config.proxy_audit_candidates].tolist()
    candidates = [combos[i] for i in sel]
    result = audit_g4_proxy_ranking(search_act, weight_rows, base, candidates, stats, config)
    return asdict(result)


def _split_metadata(split: RowSplit, seed: int) -> dict:
    search_set = set(split.search_idx.tolist())
    val_set = set(split.validation_idx.tolist())
    return {
        "seed": seed,
        "search_seed": seed,
        "search_rows": int(split.search_idx.numel()),
        "validation_rows": int(split.validation_idx.numel()),
        "overlap_rows": len(search_set & val_set),
        "search_indices": split.search_idx.tolist(),
        "validation_indices": split.validation_idx.tolist(),
    }


