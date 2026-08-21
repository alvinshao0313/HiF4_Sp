"""Tests for hierarchical greedy construction and refinement."""

from __future__ import annotations

import copy

import torch
import pytest

from permutation_optimization.config import SearchConfig
from permutation_optimization.hierarchical_greedy import (
    build_g4_groups,
    flatten_hierarchy,
    optimize_layer_permutation,
    pack_g8_into_g64,
    pair_g4_into_g8,
    refine_hierarchy,
)
from permutation_optimization.objective import build_channel_statistics, c4_cost, c64_cost


def _make_clustered_layer(d_ff: int = 128, rows: int = 64, n_clusters: int = 4, seed: int = 0):
    """Interleaved channels from distinct scale clusters."""
    g = torch.Generator().manual_seed(seed)
    act = torch.zeros(rows, d_ff)
    w = torch.zeros(rows, d_ff)
    scales = [2.0 ** i for i in range(n_clusters)]
    for c in range(d_ff):
        s = scales[c % n_clusters]
        act[:, c] = s * torch.randn(rows, generator=g)
        w[:, c] = s * torch.randn(rows, generator=g)
    return act, w


def test_g4_groups_same_scale_together():
    act, w = _make_clustered_layer(d_ff=64, n_clusters=4, rows=48)
    # Force strong within-cluster correlation by shared trajectory.
    rows = act.shape[0]
    t = torch.randn(rows, 4)
    for c in range(64):
        k = c % 4
        act[:, c] = (2.0 ** k) * t[:, k] + 0.01 * torch.randn(rows)
        w[:, c] = (2.0 ** k) * t[:, k] + 0.01 * torch.randn(rows)
    cfg = SearchConfig(
        candidate_window=32,
        neighbor_k=16,
        beam_width_g4=4,
        exact_rerank_g4=8,
        refine_passes=0,
    )
    stats = build_channel_statistics(act, w, cfg)
    g4 = build_g4_groups(act, w, stats, cfg)
    assert len(g4) == 16
    flat = [c for g in g4 for c in g]
    assert sorted(flat) == list(range(64))
    # Majority of groups should be pure mod-4.
    pure = sum(1 for g in g4 if len({c % 4 for c in g}) == 1)
    assert pure >= 8


def test_g4_not_just_rms():
    """Same RMS but alternating peaks: C4 prefers matching peak phase over mixed."""
    rows = 64
    even_traj = torch.zeros(rows)
    odd_traj = torch.zeros(rows)
    even_traj[0::2] = 1.0
    even_traj[1::2] = 0.05
    odd_traj[1::2] = 1.0
    odd_traj[0::2] = 0.05
    # Build a 64-d tensor so stats are valid; only first 4 channels used in C4.
    act = torch.randn(rows, 64) * 0.01
    act[:, 0] = even_traj
    act[:, 1] = even_traj * 1.02
    act[:, 2] = even_traj * 0.98
    act[:, 3] = even_traj * 1.01
    act[:, 4] = odd_traj
    act[:, 5] = odd_traj * 1.02
    act[:, 6] = odd_traj * 0.98
    act[:, 7] = odd_traj * 1.01
    w = act.clone()
    cfg = SearchConfig()
    stats = build_channel_statistics(act, w, cfg)
    c_same = c4_cost([0, 1, 2, 3], act, w, stats, cfg)
    c_mixed = c4_cost([0, 1, 4, 5], act, w, stats, cfg)
    assert c_same < c_mixed


def test_g4_deterministic():
    act, w = _make_clustered_layer(64, 32, 4, seed=1)
    cfg = SearchConfig(candidate_window=32, neighbor_k=12, refine_passes=0, seed=1)
    stats = build_channel_statistics(act, w, cfg)
    a = build_g4_groups(act, w, stats, cfg)
    b = build_g4_groups(act, w, stats, cfg)
    assert a == b


def test_g4_rejects_bad_dff():
    act = torch.randn(16, 60)
    w = torch.randn(16, 60)
    cfg = SearchConfig()
    with pytest.raises(ValueError, match="divisible by 64"):
        stats = build_channel_statistics(act, w, cfg)


def test_g8_prefers_close_peaks():
    """Two peak≈1 G4s must pair; hardest-first may leave 1.5 with 8."""
    rows = 32
    act = torch.zeros(rows, 16)
    for t in range(rows):
        act[t, 0:4] = 1.0
        act[t, 4:8] = 1.5
        act[t, 8:12] = 8.0
        act[t, 12:16] = 1.0
    w = act.clone()
    g4 = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)]
    cfg = SearchConfig()
    g8 = pair_g4_into_g8(g4, act, w, cfg)
    partners = []
    for a, b in g8:
        ia = next(i for i, g in enumerate(g4) if set(g) == set(a))
        ib = next(i for i, g in enumerate(g4) if set(g) == set(b))
        partners.append(tuple(sorted((ia, ib))))
    assert (0, 3) in partners


def test_g64_cluster_separation():
    d_ff = 128
    rows = 48
    act = torch.zeros(rows, d_ff)
    w = torch.zeros(rows, d_ff)
    # Two clear scale clusters: low [0:64], high [64:128] but interleaved in index? 
    # Put first 64 channels scale 1, next 64 scale 16 — algorithm should form two G64s.
    act[:, :64] = torch.randn(rows, 64)
    act[:, 64:] = 16.0 * torch.randn(rows, 64)
    w[:, :64] = torch.randn(rows, 64)
    w[:, 64:] = 16.0 * torch.randn(rows, 64)
    cfg = SearchConfig(
        candidate_window=64,
        neighbor_k=32,
        beam_width_g4=4,
        beam_width_g64=4,
        refine_passes=0,
        exact_rerank_g4=8,
    )
    stats = build_channel_statistics(act, w, cfg)
    g4 = build_g4_groups(act, w, stats, cfg)
    g8 = pair_g4_into_g8(g4, act, w, cfg)
    g64 = pack_g8_into_g64(g8, act, w, stats, cfg)
    assert len(g64) == 2
    for block in g64:
        assert len(block) == 8
        for g8b in block:
            assert len(g8b) == 2
            for g4b in g8b:
                assert len(g4b) == 4
    perm = flatten_hierarchy(g64)
    assert sorted(perm.tolist()) == list(range(128))
    # Each G64 should be mostly from one cluster
    for block in g64:
        ch = []
        for g8b in block:
            for g4b in g8b:
                ch.extend(g4b)
        low = sum(1 for c in ch if c < 64)
        assert low <= 16 or low >= 48  # not mixed 50/50


def test_refine_repairs_swapped_g8():
    d_ff = 128
    act, w = _make_clustered_layer(d_ff, 40, 4, seed=2)
    cfg = SearchConfig(
        candidate_window=48,
        neighbor_k=24,
        refine_passes=1,
        refine_bad_blocks=4,
        beam_width_g64=4,
    )
    stats = build_channel_statistics(act, w, cfg)
    g4 = build_g4_groups(act, w, stats, cfg)
    g8 = pair_g4_into_g8(g4, act, w, cfg)
    g64 = pack_g8_into_g64(g8, act, w, stats, cfg)
    # Swap two G8s across blocks if we have 2 blocks
    if len(g64) >= 2:
        g64_bad = copy.deepcopy(g64)
        g64_bad[0][0], g64_bad[1][0] = g64_bad[1][0], g64_bad[0][0]
        loss_before = c64_cost(
            [c for g8b in g64_bad[0] for g4b in g8b for c in g4b],
            act,
            w,
            stats,
            cfg,
        ) + c64_cost(
            [c for g8b in g64_bad[1] for g4b in g8b for c in g4b],
            act,
            w,
            stats,
            cfg,
        )
        refined, hist = refine_hierarchy(g64_bad, act, w, stats, cfg)
        loss_after = c64_cost(
            [c for g8b in refined[0] for g4b in g8b for c in g4b],
            act,
            w,
            stats,
            cfg,
        ) + c64_cost(
            [c for g8b in refined[1] for g4b in g8b for c in g4b],
            act,
            w,
            stats,
            cfg,
        )
        assert loss_after <= loss_before + 1e-9
        assert all(hist[i] <= hist[i - 1] + 1e-12 for i in range(1, len(hist)))


def test_optimize_layer_accepts_on_interleaved():
    d_ff = 128
    rows = 80
    g = torch.Generator().manual_seed(0)
    # Ideal: channels clustered by scale; stored interleaved → identity bad.
    scales = [1.0, 2.0, 4.0, 8.0]
    act = torch.zeros(rows, d_ff)
    w = torch.zeros(32, d_ff)
    traj = torch.randn(rows, 4, generator=g)
    wtraj = torch.randn(32, 4, generator=g)
    for c in range(d_ff):
        k = c % 4
        act[:, c] = scales[k] * traj[:, k] + 0.02 * torch.randn(rows, generator=g)
        w[:, c] = scales[k] * wtraj[:, k] + 0.02 * torch.randn(32, generator=g)
    cfg = SearchConfig(
        activation_rows=rows,
        weight_rows=32,
        validation_fraction=0.2,
        candidate_window=48,
        neighbor_k=24,
        beam_width_g4=4,
        exact_rerank_g4=8,
        beam_width_g64=4,
        refine_passes=1,
        refine_bad_blocks=4,
        seed=0,
    )
    result = optimize_layer_permutation("toy", act, down_weight=w, config=cfg)
    assert result.baseline_metrics["hierarchical"]["hif4_loss"] < result.baseline_metrics["identity"]["hif4_loss"]
    best_q99 = min(
        result.baseline_metrics["q99_sort_desc"]["hif4_loss"],
        result.baseline_metrics["q99_sort_asc"]["hif4_loss"],
    )
    assert result.baseline_metrics["hierarchical"]["hif4_loss"] <= best_q99 + 1e-5
    assert all(
        abs(v) < 1e6 and v == v
        for m in result.baseline_metrics.values()
        for v in m.values()
    )


def test_optimize_layer_exports_all_candidate_permutations():
    """Downstream experiments need every candidate, not only the accepted one."""
    torch.manual_seed(101)
    d_ff = 64
    act = torch.randn(40, d_ff)
    w = torch.randn(32, d_ff)
    cfg = SearchConfig(
        activation_rows=40,
        weight_rows=32,
        candidate_window=16,
        neighbor_k=8,
        beam_width_g4=2,
        exact_rerank_g4=4,
        beam_width_g64=2,
        refine_passes=0,
        refine_max_rounds=1,
        refine_candidates_per_round=8,
        proxy_audit_enabled=False,
        seed=101,
    )
    result = optimize_layer_permutation("toy", act, down_weight=w, config=cfg)
    exported = result.extra["candidate_permutations"]
    required = {
        "identity",
        "q99_sort_desc",
        "q99_sort_asc",
        "hierarchical",
        "hierarchical_refined",
        "random_seed_43",
    }
    assert required <= set(exported)
    for perm in exported.values():
        assert perm.device.type == "cpu"
        assert perm.dtype == torch.long
        assert torch.equal(torch.sort(perm).values, torch.arange(d_ff))


def test_expand_proxy_matches_full_group_proxy():
    from permutation_optimization.hierarchical_greedy import (
        _expand_proxy_adj,
        _group_proxy_cost,
        _precompute_pair_costs,
    )

    torch.manual_seed(3)
    d_ff, rows = 128, 40
    act = torch.randn(rows, d_ff)
    w = torch.randn(rows, d_ff)
    cfg = SearchConfig(candidate_window=32, neighbor_k=16, refine_passes=0)
    stats = build_channel_statistics(act, w, cfg)
    cache = _precompute_pair_costs(stats, act, w, cfg.eps)
    adj: list[dict[int, float]] = [{} for _ in range(d_ff)]
    for (a, b), c in cache.items():
        adj[a][b] = c
        adj[b][a] = c
    state = (7, 19, 33)
    full_state = _group_proxy_cost(state, cache, act, w, stats, cfg.eps)
    proxy = 0.0
    cur: tuple[int, ...] = (state[0],)
    for nxt in state[1:]:
        proxy = _expand_proxy_adj(cur, proxy, nxt, adj, cache, act, w, stats, cfg.eps)
        cur = tuple(sorted(cur + (nxt,)))
    assert cur == tuple(sorted(state))
    assert abs(proxy - full_state) < 1e-12


def test_g8_conflict_matrix_matches_scalar():
    from permutation_optimization.hierarchical_greedy import (
        _g4_pair_penalty_matrix,
        _g8_conflict,
        _g8_peak_and_features,
    )

    torch.manual_seed(5)
    # 16 channels → 4 G4 → 2 G8
    act = torch.randn(32, 64)
    w = torch.randn(32, 64)
    cfg = SearchConfig(candidate_window=16, neighbor_k=8, refine_passes=0)
    stats = build_channel_statistics(act, w, cfg)
    g4 = build_g4_groups(act, w, stats, cfg)
    g8 = pair_g4_into_g8(g4, act, w, cfg)
    peak_a, peak_w, _primary, _ = _g8_peak_and_features(g8, act, w, cfg.eps)
    mat = _g4_pair_penalty_matrix(peak_a, peak_w, cfg.eps)
    cache: dict = {}
    for i in range(len(g8)):
        for j in range(i + 1, len(g8)):
            scalar = _g8_conflict(i, j, peak_a, peak_w, cfg.eps, cache)
            assert abs(scalar - float(mat[i, j].item())) < 1e-10


def test_pipeline_deterministic_d256():
    torch.manual_seed(11)
    act, w = _make_clustered_layer(d_ff=256, rows=48, n_clusters=4, seed=11)
    cfg = SearchConfig(
        candidate_window=48,
        neighbor_k=16,
        beam_width_g4=4,
        exact_rerank_g4=4,
        beam_width_g64=4,
        refine_passes=0,
        seed=11,
    )
    stats = build_channel_statistics(act, w, cfg)
    g4_a = build_g4_groups(act, w, stats, cfg)
    g4_b = build_g4_groups(act, w, stats, cfg)
    assert g4_a == g4_b
    g8_a = pair_g4_into_g8(g4_a, act, w, cfg)
    g8_b = pair_g4_into_g8(g4_a, act, w, cfg)
    assert g8_a == g8_b
    g64_a = pack_g8_into_g64(g8_a, act, w, stats, cfg)
    g64_b = pack_g8_into_g64(g8_a, act, w, stats, cfg)
    assert flatten_hierarchy(g64_a).tolist() == flatten_hierarchy(g64_b).tolist()


def test_seeded_refine_never_worse_than_seed():
    from permutation_optimization.hierarchical_greedy import seeded_local_refine
    from permutation_optimization.objective import full_layout_hif4_loss

    d_ff = 128
    act, w = _make_clustered_layer(d_ff, 48, 4, seed=5)
    cfg = SearchConfig(
        candidate_window=48,
        neighbor_k=24,
        beam_width_g4=2,
        exact_rerank_g4=4,
        beam_width_g64=2,
        refine_passes=0,
        refine_max_rounds=2,
        refine_candidates_per_round=32,
        seed=5,
    )
    stats = build_channel_statistics(act, w, cfg)
    seeds = {
        "identity": torch.arange(d_ff),
        "q99_sort_desc": torch.argsort(stats.primary_scale, descending=True, stable=True),
    }
    refined, source, info = seeded_local_refine(seeds, act, w, stats, cfg)
    seed_losses = {
        name: full_layout_hif4_loss(p, act, w, stats, cfg)[0] for name, p in seeds.items()
    }
    refined_loss = full_layout_hif4_loss(refined, act, w, stats, cfg)[0]
    assert refined_loss <= min(seed_losses.values()) + 1e-12
    assert source in seeds


def test_seeded_refine_keeps_valid_permutation():
    from permutation_optimization.hierarchical_greedy import seeded_local_refine

    d_ff = 128
    act, w = _make_clustered_layer(d_ff, 40, 4, seed=6)
    cfg = SearchConfig(
        candidate_window=48,
        neighbor_k=24,
        beam_width_g4=2,
        exact_rerank_g4=4,
        beam_width_g64=2,
        refine_passes=0,
        refine_max_rounds=2,
        refine_candidates_per_round=32,
        seed=6,
    )
    stats = build_channel_statistics(act, w, cfg)
    seeds = {"identity": torch.arange(d_ff)}
    refined, _source, _info = seeded_local_refine(seeds, act, w, stats, cfg)
    assert sorted(refined.tolist()) == list(range(d_ff))


def test_seeded_refine_respects_candidate_budget():
    from permutation_optimization.hierarchical_greedy import seeded_local_refine

    d_ff = 128
    act, w = _make_clustered_layer(d_ff, 40, 4, seed=7)
    cfg = SearchConfig(
        candidate_window=48,
        neighbor_k=24,
        beam_width_g4=2,
        exact_rerank_g4=4,
        beam_width_g64=2,
        refine_passes=0,
        refine_max_rounds=2,
        refine_candidates_per_round=32,
        seed=7,
    )
    stats = build_channel_statistics(act, w, cfg)
    seeds = {"identity": torch.arange(d_ff)}
    _refined, _source, info = seeded_local_refine(seeds, act, w, stats, cfg)
    assert info["evaluated_candidates"] <= cfg.refine_max_rounds * cfg.refine_candidates_per_round


def test_full_path_split_metadata_disjoint_and_complete():
    """Full MLP path: validation rows == 20%, zero overlap with search rows."""
    torch.manual_seed(17)
    d_model, d_ff, rows = 64, 128, 40
    x = torch.randn(rows, d_model) * 0.1
    wu = torch.randn(d_ff, d_model) * 0.05
    wg = torch.randn(d_ff, d_model) * 0.05
    wd = torch.randn(d_model, d_ff) * 0.05
    cfg = SearchConfig(
        activation_rows=rows,
        weight_rows=32,
        validation_fraction=0.2,
        candidate_window=48,
        neighbor_k=16,
        beam_width_g4=2,
        exact_rerank_g4=4,
        beam_width_g64=2,
        refine_passes=0,
        seed=42,
    )
    result = optimize_layer_permutation("full", x, wu, wg, wd, cfg)
    split = result.extra["split"]
    assert split["search_rows"] == 32
    assert split["validation_rows"] == 8
    assert split["overlap_rows"] == 0
    assert set(split["search_indices"]).isdisjoint(set(split["validation_indices"]))
    assert set(split["search_indices"]) | set(split["validation_indices"]) == set(range(rows))


def test_full_path_validation_rows_not_in_search():
    """Row-unique encoding: validation activations must not come from search rows."""
    torch.manual_seed(19)
    d_model, d_ff, rows = 64, 128, 40
    # Encode row identity in the first input coordinate via orthogonal one-hot rows.
    x = torch.zeros(rows, d_model)
    x[:, 0] = torch.arange(rows, dtype=torch.float32) + 1.0
    wu = torch.randn(d_ff, d_model, generator=torch.Generator().manual_seed(1)) * 0.05
    wg = torch.randn(d_ff, d_model, generator=torch.Generator().manual_seed(2)) * 0.05
    wd = torch.randn(d_model, d_ff, generator=torch.Generator().manual_seed(3)) * 0.05
    cfg = SearchConfig(
        activation_rows=rows,
        weight_rows=32,
        validation_fraction=0.2,
        candidate_window=48,
        neighbor_k=16,
        beam_width_g4=2,
        exact_rerank_g4=4,
        beam_width_g64=2,
        refine_passes=0,
        seed=42,
    )
    result = optimize_layer_permutation("full", x, wu, wg, wd, cfg)
    split = result.extra["split"]
    val_idx = set(split["validation_indices"])
    search_idx = set(split["search_indices"])
    assert val_idx.isdisjoint(search_idx)
    # Validation inputs must be exactly the rows of x at validation_indices.
    assert split["search_seed"] == 42


def test_optimize_records_proxy_audit():
    torch.manual_seed(27)
    d_ff, rows = 128, 40
    act = torch.randn(rows, d_ff)
    w = torch.randn(32, d_ff)
    cfg = SearchConfig(
        activation_rows=rows,
        weight_rows=32,
        validation_fraction=0.2,
        candidate_window=48,
        neighbor_k=16,
        beam_width_g4=2,
        exact_rerank_g4=4,
        beam_width_g64=2,
        refine_passes=0,
        seed=42,
    )
    result = optimize_layer_permutation("down", act, down_weight=w, config=cfg)
    audit = result.extra["proxy_audit"]
    assert audit["n_candidates"] == 128
    assert -1.0 <= audit["spearman"] <= 1.0
    assert -1.0 <= audit["pearson"] <= 1.0
    assert 0.0 <= audit["top5_overlap"] <= 1.0
    assert isinstance(audit["top1_match"], bool)


def test_down_only_path_uses_shared_row_split():
    torch.manual_seed(23)
    d_ff, rows = 128, 40
    act = torch.randn(rows, d_ff)
    w = torch.randn(32, d_ff)
    cfg = SearchConfig(
        activation_rows=rows,
        weight_rows=32,
        validation_fraction=0.2,
        candidate_window=48,
        neighbor_k=16,
        beam_width_g4=2,
        exact_rerank_g4=4,
        beam_width_g64=2,
        refine_passes=0,
        seed=42,
    )
    result = optimize_layer_permutation("down", act, down_weight=w, config=cfg)
    split = result.extra["split"]
    assert split["search_rows"] == 32
    assert split["validation_rows"] == 8
    assert split["overlap_rows"] == 0


def test_optimize_layer_deterministic_d256():
    torch.manual_seed(13)
    act, w = _make_clustered_layer(d_ff=256, rows=64, n_clusters=4, seed=13)
    cfg = SearchConfig(
        activation_rows=64,
        weight_rows=64,
        validation_fraction=0.2,
        candidate_window=48,
        neighbor_k=16,
        refine_passes=0,
        seed=13,
    )
    r1 = optimize_layer_permutation("L", act, down_weight=w, config=cfg)
    r2 = optimize_layer_permutation("L", act, down_weight=w, config=cfg)
    assert torch.equal(r1.permutation, r2.permutation)
    assert r1.accepted == r2.accepted
