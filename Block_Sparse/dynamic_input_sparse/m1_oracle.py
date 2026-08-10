from __future__ import annotations

from dataclasses import dataclass

import torch

from Block_Sparse.dynamic_input_sparse.common import (
    all_ones_mask,
    check_divisible,
    flatten_tokens,
    ratio_to_keep_count,
)

# Production chunk size is frozen at 8. Tests may pass other values for invariance.
_DEFAULT_CHUNK = 8


@dataclass(frozen=True)
class M1OracleInternalResult:
    final_mask: torch.Tensor
    greedy_mask: torch.Tensor
    removal_order: torch.Tensor
    swap_count: torch.Tensor
    sse: torch.Tensor


def _weight_by_k(weight: torch.Tensor, kb: int, k_block: int) -> torch.Tensor:
    """Reshape W [Dout,Din] -> [Kb, Dout, 64] for batched partial products."""
    d_out, d_in = int(weight.shape[0]), int(weight.shape[1])
    if d_in != kb * k_block:
        raise ValueError(f"D_in={d_in} != Kb*K={kb * k_block}")
    # [Dout, Kb, 64] -> [Kb, Dout, 64]
    return (
        weight.reshape(d_out, kb, k_block)
        .permute(1, 0, 2)
        .contiguous()
    )


def _build_gram_multiweight(
    x_blocks: torch.Tensor,
    weights: list[torch.Tensor],
    k_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (G [Tc,Kb,Kb] FP64, sse_diag helper unused).

    G_joint = sum_m P_m @ P_m.T without persisting concatenated P.
    Partial products computed in FP32 then cast to FP64 before Gram multiply.
    """
    tc, kb, bk = (int(x_blocks.shape[0]), int(x_blocks.shape[1]), int(x_blocks.shape[2]))
    if bk != k_block:
        raise ValueError(f"x block width {bk} != {k_block}")
    device = x_blocks.device
    g = torch.zeros(tc, kb, kb, dtype=torch.float64, device=device)
    x32 = x_blocks.to(torch.float32)
    for w in weights:
        w_by_k = _weight_by_k(w.detach().to(torch.float32), kb, k_block)
        # P[t,k,o] = sum_d X[t,k,d] * W_by_k[k,o,d]
        # einsum: [Tc,Kb,64] x [Kb,Dout,64] -> [Tc,Kb,Dout]
        p32 = torch.einsum("tkd,kod->tko", x32, w_by_k)
        p64 = p32.to(torch.float64)
        g = g + torch.matmul(p64, p64.transpose(-1, -2))
        del p32, p64, w_by_k
    return g, x32


def _vectorized_backward_greedy(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return removal_order [Tc,Kb] and final r_dot_p after full removal (unused)."""
    tc, kb, _ = g.shape
    device = g.device
    active = torch.ones(tc, kb, dtype=torch.bool, device=device)
    diag = torch.diagonal(g, dim1=-2, dim2=-1).contiguous()
    r_dot_p = torch.zeros(tc, kb, dtype=torch.float64, device=device)
    base_error = torch.zeros(tc, dtype=torch.float64, device=device)
    removal = torch.empty(tc, kb, dtype=torch.int64, device=device)

    for step in range(kb):
        cand = base_error.unsqueeze(-1) + 2.0 * r_dot_p + diag
        cand = torch.where(active, cand, torch.full_like(cand, float("inf")))
        k_star = torch.argmin(cand, dim=-1)  # lower-index tie break
        removal[:, step] = k_star
        gather_idx = k_star.unsqueeze(-1)
        chosen = cand.gather(1, gather_idx).squeeze(-1)
        base_error = chosen
        # r_dot_p[t,:] += G[t,:,k_star]
        g_col = g.gather(2, k_star.view(tc, 1, 1).expand(tc, kb, 1)).squeeze(-1)
        r_dot_p = r_dot_p + g_col
        active.scatter_(1, gather_idx, False)

    return removal, r_dot_p


def _greedy_masks_from_order(
    removal_order: torch.Tensor, keep_count: int
) -> torch.Tensor:
    tc, kb = int(removal_order.shape[0]), int(removal_order.shape[1])
    n_remove = kb - keep_count
    mask = torch.ones(tc, kb, dtype=torch.bool, device=removal_order.device)
    if n_remove <= 0:
        return mask
    removed = removal_order[:, :n_remove]
    mask.scatter_(1, removed, False)
    return mask


def _vectorized_one_swap_refine(
    g: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1-swap refinement matching scalar M1 semantics.

    Returns (refined_mask, swap_count, sse).
    """
    tc, kb, _ = g.shape
    device = g.device
    s = mask.clone()
    r = ~s
    # Initialize r_dot_p = G @ 1_R  (residual = sum of removed block contributions)
    # For residual r = sum_{k in R} P_k, r_dot_p[b] = <r, P_b> = sum_{a in R} G[b,a]
    ones_r = r.to(dtype=g.dtype)
    r_dot_p = torch.matmul(g, ones_r.unsqueeze(-1)).squeeze(-1)
    # base_error = ||r||^2 = 1_R^T G 1_R
    base_error = torch.sum(ones_r * r_dot_p, dim=-1)

    swap_count = torch.zeros(tc, dtype=torch.int64, device=device)
    max_swaps = kb * kb
    # pair_index = a * Kb + b; eligible when a removed and b kept
    a_idx = torch.arange(kb, device=device).view(1, kb, 1).expand(tc, kb, kb)
    b_idx = torch.arange(kb, device=device).view(1, 1, kb).expand(tc, kb, kb)
    pair_index = a_idx * kb + b_idx

    for _ in range(max_swaps + 1):
        e = base_error
        thresh = -1e-7 * torch.clamp(e, min=1.0)
        # delta(a,b) for all pairs
        # -2*r_dot_p[a] + 2*r_dot_p[b] + G[a,a] + G[b,b] - 2*G[a,b]
        diag = torch.diagonal(g, dim1=-2, dim2=-1)
        delta = (
            -2.0 * r_dot_p.unsqueeze(-1)
            + 2.0 * r_dot_p.unsqueeze(-2)
            + diag.unsqueeze(-1)
            + diag.unsqueeze(-2)
            - 2.0 * g
        )
        # Only pairs with a in R and b in S
        eligible = r.unsqueeze(-1) & s.unsqueeze(-2)
        # delta shape is [Tc, a, b] with a along dim1, b along dim2 —
        # but we built: -2*r_dot_p[a] uses unsqueeze(-1) -> [Tc,Kb,1] for a
        # +2*r_dot_p[b] uses unsqueeze(-2) -> [Tc,1,Kb] for b
        # So delta[t,a,b] is correct.
        improved = eligible & (delta < thresh.view(tc, 1, 1))
        if not bool(improved.any().item()):
            break

        # Among improved pairs, pick minimal delta; ties -> minimal pair_index
        big = torch.full_like(delta, float("inf"))
        cand_delta = torch.where(improved, delta, big)
        flat_delta = cand_delta.reshape(tc, -1)
        flat_pair = pair_index.reshape(tc, -1)
        # Sort by (delta, pair_index): negate pair for secondary via lex on stacked
        # Use: primary = delta; for equal delta, smaller pair_index wins.
        # Encode: sort_key = delta + epsilon * pair_index / (Kb^2)
        # Safer: argsort delta, then among min-delta choose min pair.
        min_delta = flat_delta.min(dim=-1).values
        is_min = flat_delta <= (min_delta.unsqueeze(-1) + 1e-15)
        # Mask non-min to inf pair
        pair_for_tie = torch.where(
            is_min,
            flat_pair.to(torch.float64),
            torch.full_like(flat_pair, fill_value=kb * kb + 1, dtype=torch.float64),
        )
        best_flat = torch.argmin(pair_for_tie, dim=-1)
        # Tokens with no improvement: min_delta is +inf
        has_improve = torch.isfinite(min_delta)
        if not bool(has_improve.any().item()):
            break

        best_a = best_flat // kb
        best_b = best_flat % kb
        best_delta = flat_delta.gather(1, best_flat.unsqueeze(-1)).squeeze(-1)

        # Apply swaps only where has_improve
        if bool((swap_count[has_improve] >= max_swaps).any().item()):
            raise RuntimeError(
                f"1-swap refinement hit max_swaps={max_swaps} with remaining improvement"
            )

        # Update r_dot_p = r_dot_p - G[:,a] + G[:,b]
        g_a = g.gather(2, best_a.view(tc, 1, 1).expand(tc, kb, 1)).squeeze(-1)
        g_b = g.gather(2, best_b.view(tc, 1, 1).expand(tc, kb, 1)).squeeze(-1)
        r_dot_p = torch.where(
            has_improve.unsqueeze(-1),
            r_dot_p - g_a + g_b,
            r_dot_p,
        )
        base_error = torch.where(has_improve, base_error + best_delta, base_error)

        # Update masks: S[a]=True, S[b]=False, R flips
        # scatter needs care for tokens without improve — leave unchanged
        new_s = s.clone()
        new_r = r.clone()
        # For improved tokens set a kept, b removed
        rows = torch.arange(tc, device=device)
        improve_rows = rows[has_improve]
        if improve_rows.numel() > 0:
            new_s[improve_rows, best_a[has_improve]] = True
            new_s[improve_rows, best_b[has_improve]] = False
            new_r[improve_rows, best_a[has_improve]] = False
            new_r[improve_rows, best_b[has_improve]] = True
            swap_count[improve_rows] += 1
        s = new_s
        r = new_r
    else:
        raise RuntimeError(
            f"1-swap refinement hit max_swaps={max_swaps} with remaining improvement"
        )

    sse = base_error.clamp(min=0.0)
    return s, swap_count, sse


def _predict_chunk(
    x_flat_chunk: torch.Tensor,
    weights: list[torch.Tensor],
    keep_ratio: float,
    k_block: int,
    return_internal: bool,
) -> torch.Tensor | M1OracleInternalResult:
    tc, d_in = int(x_flat_chunk.shape[0]), int(x_flat_chunk.shape[1])
    kb = check_divisible(d_in, k_block, "D_in")
    if float(keep_ratio) == 1.0:
        mask = all_ones_mask(tc, kb, x_flat_chunk.device)
        if not return_internal:
            return mask
        return M1OracleInternalResult(
            final_mask=mask,
            greedy_mask=mask,
            removal_order=torch.arange(kb, device=x_flat_chunk.device)
            .view(1, kb)
            .expand(tc, kb)
            .contiguous(),
            swap_count=torch.zeros(tc, dtype=torch.int64, device=x_flat_chunk.device),
            sse=torch.zeros(tc, dtype=torch.float64, device=x_flat_chunk.device),
        )

    keep_count = ratio_to_keep_count(keep_ratio, kb)
    x_blocks = x_flat_chunk.reshape(tc, kb, k_block)
    g, _ = _build_gram_multiweight(x_blocks, weights, k_block)
    removal_order, _ = _vectorized_backward_greedy(g)
    greedy = _greedy_masks_from_order(removal_order, keep_count)
    final, swaps, sse = _vectorized_one_swap_refine(g, greedy)
    del g
    if not return_internal:
        return final
    return M1OracleInternalResult(
        final_mask=final,
        greedy_mask=greedy,
        removal_order=removal_order,
        swap_count=swaps,
        sse=sse,
    )


def predict_m1_full_output_mask_multiweight(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    keep_ratio: float,
    token_chunk_size: int = _DEFAULT_CHUNK,
    k_block_size: int = 64,
    return_internal: bool = False,
) -> torch.Tensor | M1OracleInternalResult:
    """Full-output exact input recovery for one or more weights sharing one MX."""
    if not weights:
        raise ValueError("weights must be non-empty")
    if int(k_block_size) != 64:
        raise ValueError(f"k_block_size must be 64, got {k_block_size}")
    if int(token_chunk_size) < 1:
        raise ValueError(f"token_chunk_size must be >= 1, got {token_chunk_size}")
    x_flat, _ = flatten_tokens(x)
    if not bool(torch.isfinite(x_flat).all().item()):
        raise ValueError("activation contains NaN/Inf")
    t = int(x_flat.shape[0])
    d_in = int(x_flat.shape[1])
    for i, w in enumerate(weights):
        if w.ndim != 2:
            raise ValueError(f"weights[{i}] must be 2D, got {tuple(w.shape)}")
        if int(w.shape[1]) != d_in:
            raise ValueError(
                f"weights[{i}] D_in={int(w.shape[1])} != x D_in={d_in}"
            )
        if not bool(torch.isfinite(w.detach()).all().item()):
            raise ValueError(f"weights[{i}] contains NaN/Inf")

    kb = check_divisible(d_in, k_block_size, "D_in")
    if float(keep_ratio) == 1.0:
        mask = all_ones_mask(t, kb, x_flat.device)
        if not return_internal:
            return mask
        return M1OracleInternalResult(
            final_mask=mask,
            greedy_mask=mask,
            removal_order=torch.arange(kb, device=x_flat.device)
            .view(1, kb)
            .expand(t, kb)
            .contiguous(),
            swap_count=torch.zeros(t, dtype=torch.int64, device=x_flat.device),
            sse=torch.zeros(t, dtype=torch.float64, device=x_flat.device),
        )

    finals: list[torch.Tensor] = []
    greeds: list[torch.Tensor] = []
    orders: list[torch.Tensor] = []
    swaps: list[torch.Tensor] = []
    sses: list[torch.Tensor] = []
    for start in range(0, t, int(token_chunk_size)):
        chunk = x_flat[start : start + int(token_chunk_size)]
        out = _predict_chunk(
            chunk, weights, keep_ratio, k_block_size, return_internal=True
        )
        assert isinstance(out, M1OracleInternalResult)
        finals.append(out.final_mask)
        greeds.append(out.greedy_mask)
        orders.append(out.removal_order)
        swaps.append(out.swap_count)
        sses.append(out.sse)

    result = M1OracleInternalResult(
        final_mask=torch.cat(finals, dim=0),
        greedy_mask=torch.cat(greeds, dim=0),
        removal_order=torch.cat(orders, dim=0),
        swap_count=torch.cat(swaps, dim=0),
        sse=torch.cat(sses, dim=0),
    )
    if return_internal:
        return result
    return result.final_mask


def predict_m1_full_output_mask(
    x: torch.Tensor,
    weight: torch.Tensor,
    keep_ratio: float,
    token_chunk_size: int = _DEFAULT_CHUNK,
    k_block_size: int = 64,
    return_internal: bool = False,
) -> torch.Tensor | M1OracleInternalResult:
    """Single-weight entrypoint; reuses the multiweight recovery path."""
    return predict_m1_full_output_mask_multiweight(
        x,
        [weight],
        keep_ratio,
        token_chunk_size=token_chunk_size,
        k_block_size=k_block_size,
        return_internal=return_internal,
    )
