"""GPU-vectorized per-group S0 + e8/e4 search for static weights."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from .formats import e6m2_neighbor_values, quantize_s1p2_magnitude, round_bfloat16
from .quantizer import HiF4QuantConfig, compute_reciprocal_s0, compute_s0, quantize_hif4

SearchBudget = Literal["fast", "full"]

FAST_OFFSETS = (-1, 0, 1)
FULL_OFFSETS = (-2, -1, 0, 1, 2)

# (e8, e4_left, e4_right) -> per-element total exponent over 8 elements
# elements: [0,1,2,3 | 4,5,6,7] -> left nibble e4, right nibble e4
_EXP_PATTERN_TABLE: list[list[int]] = []
for _e8 in (0, 1):
    for _e4l in (0, 1):
        for _e4r in (0, 1):
            _EXP_PATTERN_TABLE.append(
                [_e8 + _e4l] * 4 + [_e8 + _e4r] * 4
            )


def build_exp_patterns(device: torch.device | str) -> torch.Tensor:
    """Constant table [8_combos, 8_elements] of total exponents."""
    return torch.tensor(_EXP_PATTERN_TABLE, device=device, dtype=torch.float32)


def estimate_group_chunk_size(
    device: torch.device,
    *,
    k_s0: int,
    memory_budget_fraction: float = 0.25,
    bytes_per_elem_peak: float = 4.0 * 6,
) -> int:
    """Conservative G_chunk from free memory.

    Peak roughly: groups[G,8,8] + candidates broadcast intermediates.
    We budget for a logical [G, K, 8, 8, 8] FP32 error contribution plus a few
    working buffers (~6 FP32 tensors of that size as upper bound).
    """
    if device.type != "cuda":
        return 4096
    free_b, _total_b = torch.cuda.mem_get_info(device)
    budget = free_b * memory_budget_fraction
    # Per group peak ~ K * 8 * 8 * 8 * bytes_per_elem_peak
    per_group = k_s0 * 8 * 8 * 8 * bytes_per_elem_peak
    chunk = int(budget // max(per_group, 1.0))
    chunk = max(1, min(chunk, 65536))
    # Prefer multiples of 64 for stable shapes.
    return max(64, (chunk // 64) * 64)


@dataclass
class WeightSearchResult:
    reconstruction: torch.Tensor
    s0: torch.Tensor
    e8: torch.Tensor
    e4: torch.Tensor
    s0_index: torch.Tensor
    mse: float
    nmse: float
    groups_per_second: float
    elapsed_s: float
    peak_memory_bytes: int
    budget: str
    group_chunk_size: int


def _prepare_groups(weight: torch.Tensor, group_dim: int = -1) -> tuple[torch.Tensor, tuple[int, ...], int]:
    """Return groups [G, 8, 8], original moved shape, and normalized dim."""
    dim = group_dim % weight.ndim
    moved = weight.movedim(dim, -1).contiguous().to(torch.float32)
    if moved.shape[-1] % 64 != 0:
        raise ValueError(f"last dim must be divisible by 64, got {moved.shape[-1]}")
    shape = tuple(moved.shape)
    groups = moved.reshape(-1, 8, 8)
    return groups, shape, dim


def search_weight_groups(
    weight: torch.Tensor,
    *,
    budget: SearchBudget = "fast",
    group_chunk_size: int | None = None,
    memory_budget_fraction: float = 0.25,
    s0_mode: str = "hardware",
    s0_divisor: float = 7.0,
    enumerate_e8_e4: bool = True,
    e8_threshold: float = 4.0,
    e4_threshold: float = 2.0,
    device: str | torch.device | None = None,
) -> WeightSearchResult:
    """Vectorized per-64-group search. No Python loops over groups/combos."""
    if device is None:
        device = weight.device if weight.is_cuda else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    device = torch.device(device)
    offsets = FAST_OFFSETS if budget == "fast" else FULL_OFFSETS
    k_s0 = len(offsets)

    if group_chunk_size is None:
        group_chunk_size = estimate_group_chunk_size(
            device, k_s0=k_s0, memory_budget_fraction=memory_budget_fraction
        )
    if group_chunk_size < 1:
        raise ValueError("group_chunk_size must be >= 1")

    orig_dtype = weight.dtype
    groups_cpu_shape_dim = _prepare_groups(weight.detach())
    groups_all, moved_shape, norm_dim = groups_cpu_shape_dim
    g_total = groups_all.shape[0]
    groups_all = groups_all.to(device)

    exp_patterns = build_exp_patterns(device)  # [8, 8]
    # Bit patterns for e8 / e4 packing from combo index 0..7
    combo_ids = torch.arange(8, device=device, dtype=torch.int64)
    combo_e8 = ((combo_ids >> 2) & 1).to(torch.float32)  # [8]
    combo_e4l = ((combo_ids >> 1) & 1).to(torch.float32)
    combo_e4r = (combo_ids & 1).to(torch.float32)

    best_s0 = torch.empty(g_total, device=device, dtype=torch.float32)
    best_s0_idx = torch.empty(g_total, device=device, dtype=torch.int64)
    best_e8 = torch.empty(g_total, 8, device=device, dtype=torch.float32)
    best_e4 = torch.empty(g_total, 16, device=device, dtype=torch.float32)
    best_recon = torch.empty(g_total, 8, 8, device=device, dtype=torch.float32)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    cfg_base = HiF4QuantConfig(
        s0_divisor=s0_divisor,
        e8_threshold=e8_threshold,
        e4_threshold=e4_threshold,
        s0_mode=s0_mode,  # type: ignore[arg-type]
    )

    for start in range(0, g_total, group_chunk_size):
        end = min(start + group_chunk_size, g_total)
        g = groups_all[start:end]  # [Gc, 8, 8]
        gc = g.shape[0]
        abs_g = g.abs()
        amax64 = abs_g.reshape(gc, -1).amax(dim=-1)
        nonzero = amax64 > 0
        s0_base = compute_s0(amax64, s0_divisor, cfg_base.s0_mode)
        # Neighbor candidates [Gc, K]
        s0_cands = e6m2_neighbor_values(s0_base, offsets)
        # Zero groups: keep candidate 0 as 1.0 for numerical safety; recon forced later.
        s0_cands = torch.where(
            nonzero.unsqueeze(-1),
            s0_cands,
            torch.ones_like(s0_cands),
        )
        recip = compute_reciprocal_s0(s0_cands, cfg_base.s0_mode)  # [Gc, K]

        # abs layout [Gc, 8_blocks, 8_elems]
        abs_blocks = abs_g  # already [Gc, 8, 8]
        signs = g.sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)

        if enumerate_e8_e4:
            # Broadcast: [Gc, K, 8, 8_combos, 8]
            # local_scale = s0 * 2^exp
            exp = exp_patterns.view(1, 1, 1, 8, 8)  # combos x elems
            s0_b = s0_cands.view(gc, k_s0, 1, 1, 1)
            recip_b = recip.view(gc, k_s0, 1, 1, 1)
            local = s0_b * torch.exp2(exp)
            # normalized magnitude
            norm = abs_blocks.view(gc, 1, 8, 1, 8) * (recip_b / torch.exp2(exp))
            payload = quantize_s1p2_magnitude(norm)
            recon = signs.view(gc, 1, 8, 1, 8) * local * payload
            # error vs original
            err = g.view(gc, 1, 8, 1, 8) - recon
            block_err = (err * err).sum(dim=-1)  # [Gc, K, 8, 8_combos]
            best_combo_err, best_combo = block_err.min(dim=-1)  # [Gc, K, 8]
            group_err = best_combo_err.sum(dim=-1)  # [Gc, K]
            best_k_err, best_k = group_err.min(dim=-1)  # [Gc]

            # Gather S0
            s0_sel = s0_cands.gather(1, best_k.unsqueeze(1)).squeeze(1)
            # Gather combo per block: best_combo[gc, k, 8] -> index by best_k
            combo_sel = best_combo.gather(
                1, best_k.view(gc, 1, 1).expand(gc, 1, 8)
            ).squeeze(1)  # [Gc, 8]

            e8_sel = combo_e8[combo_sel]  # [Gc, 8]
            e4l = combo_e4l[combo_sel]
            e4r = combo_e4r[combo_sel]
            e4_sel = torch.stack([e4l, e4r], dim=-1).reshape(gc, 16)

            # Reconstruct with selected patterns only (compact).
            exp_sel = exp_patterns[combo_sel]  # [Gc, 8, 8]
            s0_b2 = s0_sel.view(gc, 1, 1)
            recip_sel = compute_reciprocal_s0(s0_sel, cfg_base.s0_mode).view(gc, 1, 1)
            local2 = s0_b2 * torch.exp2(exp_sel)
            norm2 = abs_blocks * (recip_sel / torch.exp2(exp_sel))
            payload2 = quantize_s1p2_magnitude(norm2)
            recon2 = signs * local2 * payload2
        else:
            # S0 search only; e8/e4 from fixed thresholds per candidate S0.
            amax8 = abs_blocks.amax(dim=-1)  # [Gc, 8]
            abs_4 = abs_blocks.reshape(gc, 8, 2, 4)
            amax4 = abs_4.amax(dim=-1)  # [Gc, 8, 2]

            recip_b = recip.view(gc, k_s0, 1)
            e8 = (amax8.unsqueeze(1) * recip_b >= e8_threshold).to(torch.float32)  # [Gc,K,8]
            e8_per4 = e8.unsqueeze(-1).expand(gc, k_s0, 8, 2)
            e4 = (
                amax4.unsqueeze(1) * recip_b.unsqueeze(-1) / torch.exp2(e8_per4)
                >= e4_threshold
            ).to(torch.float32)  # [Gc,K,8,2]

            exp_elem = e8.unsqueeze(-1).expand(gc, k_s0, 8, 8).clone()
            # left 4 / right 4
            exp_elem[:, :, :, 0:4] = exp_elem[:, :, :, 0:4] + e4[:, :, :, 0:1]
            exp_elem[:, :, :, 4:8] = exp_elem[:, :, :, 4:8] + e4[:, :, :, 1:2]

            s0_b = s0_cands.view(gc, k_s0, 1, 1)
            recip4 = recip.view(gc, k_s0, 1, 1)
            local = s0_b * torch.exp2(exp_elem)
            norm = abs_blocks.unsqueeze(1) * (recip4 / torch.exp2(exp_elem))
            payload = quantize_s1p2_magnitude(norm)
            recon = signs.unsqueeze(1) * local * payload
            err = g.unsqueeze(1) - recon
            group_err = (err * err).sum(dim=(-1, -2))  # [Gc, K]
            best_k_err, best_k = group_err.min(dim=-1)

            s0_sel = s0_cands.gather(1, best_k.unsqueeze(1)).squeeze(1)
            gather_idx = best_k.view(gc, 1, 1).expand(gc, 1, 8)
            e8_sel = e8.gather(1, gather_idx).squeeze(1)
            e4_g = e4.gather(1, best_k.view(gc, 1, 1, 1).expand(gc, 1, 8, 2)).squeeze(1)
            e4_sel = e4_g.reshape(gc, 16)
            recon2 = recon.gather(
                1, best_k.view(gc, 1, 1, 1).expand(gc, 1, 8, 8)
            ).squeeze(1)

        recon2 = torch.where(nonzero.view(gc, 1, 1), recon2, torch.zeros_like(recon2))
        s0_sel = torch.where(nonzero, s0_sel, torch.ones_like(s0_sel))

        best_s0[start:end] = s0_sel
        best_s0_idx[start:end] = best_k
        best_e8[start:end] = e8_sel
        best_e4[start:end] = e4_sel
        best_recon[start:end] = recon2

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    peak_mem = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    gps = g_total / elapsed if elapsed > 0 else float("inf")

    recon_moved = best_recon.reshape(moved_shape)
    reconstruction = recon_moved.movedim(-1, norm_dim).to(dtype=orig_dtype)

    # Metrics on device then scalar
    w32 = weight.detach().to(device=device, dtype=torch.float32)
    r32 = reconstruction.to(device=device, dtype=torch.float32)
    err_e = ((w32 - r32) ** 2).sum()
    ref_e = (w32 * w32).sum()
    mse = float(err_e.item() / max(w32.numel(), 1))
    nmse = float((err_e / ref_e).item()) if float(ref_e.item()) > 0 else 0.0

    groups_per_row = moved_shape[-1] // 64
    leading = moved_shape[:-1]
    s0_view = best_s0.reshape(leading + (groups_per_row,))
    e8_view = best_e8.reshape(leading + (groups_per_row, 8))
    e4_view = best_e4.reshape(leading + (groups_per_row, 16))

    return WeightSearchResult(
        reconstruction=reconstruction.cpu() if not weight.is_cuda else reconstruction,
        s0=s0_view,
        e8=e8_view,
        e4=e4_view,
        s0_index=best_s0_idx.reshape(leading + (groups_per_row,)),
        mse=mse,
        nmse=nmse,
        groups_per_second=gps,
        elapsed_s=elapsed,
        peak_memory_bytes=peak_mem,
        budget=budget,
        group_chunk_size=group_chunk_size,
    )


def brute_force_group_search_reference(
    group64: torch.Tensor,
    *,
    offsets: Sequence[int] = FAST_OFFSETS,
    s0_divisor: float = 7.0,
    s0_mode: str = "hardware",
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slow per-group reference for correctness tests. Input shape [64]."""
    assert group64.numel() == 64
    g = group64.reshape(8, 8).to(torch.float32)
    abs_g = g.abs()
    amax64 = abs_g.amax()
    cfg = HiF4QuantConfig(s0_divisor=s0_divisor, s0_mode=s0_mode)  # type: ignore[arg-type]
    if float(amax64) == 0.0:
        z = torch.zeros_like(g)
        return z.reshape(64), 0.0, torch.tensor(1.0), torch.zeros(8), torch.zeros(16)

    s0_base = compute_s0(amax64.view(1), s0_divisor, cfg.s0_mode)[0]
    cands = e6m2_neighbor_values(s0_base.view(1), list(offsets))[0]
    exp_patterns = build_exp_patterns(g.device)
    best_err = float("inf")
    best = None
    for s0 in cands:
        recip = compute_reciprocal_s0(s0.view(1), cfg.s0_mode)[0]
        for ci in range(8):
            # Need independent combo per block — true reference enumerates 8^8 which is huge.
            # Instead: for each block independently choose best combo (same as vectorized).
            pass
    # Block-independent exact enumeration (matches production search objective).
    best_s0 = cands[0]
    best_recon = torch.zeros_like(g)
    best_e8 = torch.zeros(8)
    best_e4 = torch.zeros(16)
    best_total = float("inf")
    for s0 in cands:
        recip = compute_reciprocal_s0(s0.view(1), cfg.s0_mode)[0]
        recon_blocks = []
        e8_list = []
        e4_list = []
        total = 0.0
        for b in range(8):
            block = g[b]
            abs_b = abs_g[b]
            best_b_err = float("inf")
            best_b = None
            for ci in range(8):
                exp = exp_patterns[ci]
                local = s0 * torch.exp2(exp)
                norm = abs_b * (recip / torch.exp2(exp))
                payload = quantize_s1p2_magnitude(norm)
                signs = block.sign()
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                recon_b = signs * local * payload
                err = float(((block - recon_b) ** 2).sum().item())
                if err < best_b_err:
                    best_b_err = err
                    e8 = float((ci >> 2) & 1)
                    e4l = float((ci >> 1) & 1)
                    e4r = float(ci & 1)
                    best_b = (recon_b, e8, e4l, e4r)
            assert best_b is not None
            recon_blocks.append(best_b[0])
            e8_list.append(best_b[1])
            e4_list.extend([best_b[2], best_b[3]])
            total += best_b_err
        if total < best_total:
            best_total = total
            best_s0 = s0
            best_recon = torch.stack(recon_blocks, dim=0)
            best_e8 = torch.tensor(e8_list, dtype=torch.float32)
            best_e4 = torch.tensor(e4_list, dtype=torch.float32)
    return best_recon.reshape(64), best_total, best_s0, best_e8, best_e4


def standard_rtn_quantize(weight: torch.Tensor) -> torch.Tensor:
    return quantize_hif4(weight, config=HiF4QuantConfig()).reconstruction
