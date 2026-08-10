from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExactRecoveryResult:
    masks_by_keep: dict[int, torch.Tensor]
    greedy_masks_by_keep: dict[int, torch.Tensor]
    removal_order: torch.Tensor
    swap_count_by_keep: dict[int, torch.Tensor]
    mse_by_keep: dict[int, torch.Tensor]


def _build_partial_matrix(
    x_block: torch.Tensor,
    w_blocks: torch.Tensor,
    output_mask_row: torch.Tensor,
) -> torch.Tensor:
    """Build P[Kb, D] with selected output blocks in ascending j order."""
    selected = torch.where(output_mask_row)[0]
    selected, _ = torch.sort(selected)
    kb = int(x_block.shape[0])
    if selected.numel() == 0:
        return torch.zeros(kb, 0, dtype=torch.float64, device=x_block.device)

    # FP32 batched matmul, identical layout to nested j-ascending loops:
    # for k: concat_j (X[k] @ W[j,k].T).reshape(-1)
    x32 = x_block.to(torch.float32)  # [Kb,32,64]
    w_sel = w_blocks[selected].to(torch.float32)  # [Jsel,Kb,32,64]
    w_k = w_sel.permute(1, 0, 2, 3).contiguous()  # [Kb,Jsel,32,64]
    # einsum: [Kb,32,64] x [Kb,Jsel,32,64] -> [Kb,Jsel,32,32]
    parts = torch.einsum("kmd,kjnd->kjmn", x32, w_k)
    return parts.reshape(kb, -1).to(torch.float64)


def _backward_greedy(
    P: torch.Tensor,
    min_keep: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    """Full removal-order permutation via residual-aware backward greedy."""
    kb, d = int(P.shape[0]), int(P.shape[1])
    device = P.device
    P = P.to(torch.float64)
    G = (P @ P.T).contiguous()
    active = torch.ones(kb, dtype=torch.bool, device=device)
    diag = torch.diagonal(G).contiguous()
    max_diag = float(diag.abs().max().item()) if kb > 0 else 1.0
    removal: list[int] = []

    r_vec = torch.zeros(d, dtype=torch.float64, device=device)
    r_dot_p = torch.zeros(kb, dtype=torch.float64, device=device)
    base_error = torch.zeros((), dtype=torch.float64, device=device)
    while int(active.sum().item()) > 0:
        cand_sse = base_error + 2.0 * r_dot_p + diag
        cand_sse = torch.where(active, cand_sse, torch.full_like(cand_sse, float("inf")))
        k_star = int(torch.argmin(cand_sse).item())
        if not bool(active[k_star].item()):
            break
        chosen = float(cand_sse[k_star].item())
        if chosen < -1e-5 * max(1.0, max_diag):
            raise RuntimeError(
                f"candidate SSE {chosen} is too negative (max_diag={max_diag})"
            )
        direct = float(torch.sum((r_vec + P[k_star]) ** 2).item())
        if abs(direct - chosen) >= 1e-5:
            raise RuntimeError(
                f"Gram candidate SSE {chosen} != direct residual {direct}"
            )
        base_error = cand_sse[k_star]
        # Tiny negative clamp for reporting only; sorting used pre-clamp value via chosen.
        r_dot_p = r_dot_p + G[:, k_star]
        r_vec = r_vec + P[k_star]
        active[k_star] = False
        removal.append(k_star)

    removal_order = torch.tensor(removal, dtype=torch.int64, device=device)
    if removal_order.numel() != kb or sorted(removal) != list(range(kb)):
        raise RuntimeError("removal_order must be a permutation of all K blocks")
    return removal_order, G, {}


def _greedy_mask_from_order(removal_order: torch.Tensor, keep: int, kb: int) -> torch.Tensor:
    removed = set(int(x) for x in removal_order[: kb - keep].tolist())
    mask = torch.ones(kb, dtype=torch.bool, device=removal_order.device)
    for idx in removed:
        mask[idx] = False
    return mask


def _swap_delta(
    r_dot_p: torch.Tensor,
    G: torch.Tensor,
    a: int,
    b: int,
) -> float:
    return (
        -2.0 * float(r_dot_p[a].item())
        + 2.0 * float(r_dot_p[b].item())
        + float(G[a, a].item())
        + float(G[b, b].item())
        - 2.0 * float(G[a, b].item())
    )


def _one_swap_refine(
    P: torch.Tensor,
    G: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, int, float]:
    kb, d = int(P.shape[0]), int(P.shape[1])
    device = P.device
    P = P.to(torch.float64)
    G = G.to(torch.float64)
    S = mask.clone()
    R = ~S
    if bool(R.any()):
        r = P[R].sum(dim=0)
    else:
        r = torch.zeros(d, dtype=torch.float64, device=device)
    base_error = torch.dot(r, r)
    r_dot_p = P @ r
    swaps = 0
    max_swaps = kb * kb

    while True:
        E = float(base_error.item())
        thresh = -1e-7 * max(1.0, E)
        best_delta = float("inf")
        best_pair: tuple[int, int] | None = None
        for a in torch.where(R)[0].tolist():
            for b in torch.where(S)[0].tolist():
                delta = _swap_delta(r_dot_p, G, a, b)
                if delta < thresh and (
                    best_pair is None
                    or delta < best_delta - 1e-15
                    or (abs(delta - best_delta) <= 1e-15 and (a, b) < best_pair)
                ):
                    best_delta = delta
                    best_pair = (a, b)
        if best_pair is None:
            break
        if swaps >= max_swaps:
            raise RuntimeError(
                f"1-swap refinement hit max_swaps={max_swaps} with remaining improvement"
            )
        a, b = best_pair
        r_before = P[R].sum(dim=0) if bool(R.any()) else torch.zeros(d, dtype=torch.float64, device=device)
        sse_before = float(torch.dot(r_before, r_before).item())
        r_dot_p = r_dot_p - G[:, a] + G[:, b]
        base_error = base_error + best_delta
        S[a] = True
        S[b] = False
        R[a] = False
        R[b] = True
        r_after = P[R].sum(dim=0) if bool(R.any()) else torch.zeros(d, dtype=torch.float64, device=device)
        sse_after = float(torch.dot(r_after, r_after).item())
        if not (sse_after < sse_before - 1e-12):
            raise RuntimeError(
                f"accepted swap did not strictly decrease SSE: {sse_before} -> {sse_after}"
            )
        swaps += 1

    sse = float(base_error.item())
    max_diag = float(torch.diagonal(G).abs().max().item()) if kb > 0 else 1.0
    if sse < 0:
        if sse < -1e-5 * max(1.0, max_diag):
            raise RuntimeError(f"SSE too negative: {sse}")
        sse = 0.0
    mse = sse / max(d, 1)
    return S, swaps, mse


def recover_input_masks_exact(
    x_blocks: torch.Tensor,
    w_blocks: torch.Tensor,
    output_mask: torch.Tensor,
    keep_counts: tuple[int, ...],
) -> ExactRecoveryResult:
    if x_blocks.ndim != 4 or w_blocks.ndim != 4:
        raise ValueError("x_blocks/w_blocks must be 4D")
    if output_mask.ndim != 2:
        raise ValueError("output_mask must be bool[A,Jb]")
    a, kb, br, bk = x_blocks.shape
    jb, kb_w, bo, bk_w = w_blocks.shape
    if kb != kb_w or bk != bk_w:
        raise ValueError("x/w K-block mismatch")
    if output_mask.shape != (a, jb):
        raise ValueError(
            f"output_mask shape {tuple(output_mask.shape)} != {(a, jb)}"
        )
    if not keep_counts:
        raise ValueError("keep_counts must be non-empty")
    keep_counts = tuple(sorted(set(int(k) for k in keep_counts)))
    for kc in keep_counts:
        if kc < 1 or kc > kb:
            raise ValueError(f"keep count {kc} out of range for Kb={kb}")

    device = x_blocks.device
    min_keep = min(keep_counts)
    removal_order = torch.empty(a, kb, dtype=torch.int64, device=device)
    greedy_masks: dict[int, torch.Tensor] = {
        kc: torch.empty(a, kb, dtype=torch.bool, device=device) for kc in keep_counts
    }
    final_masks: dict[int, torch.Tensor] = {
        kc: torch.empty(a, kb, dtype=torch.bool, device=device) for kc in keep_counts
    }
    swap_counts: dict[int, torch.Tensor] = {
        kc: torch.empty(a, dtype=torch.int64, device=device) for kc in keep_counts
    }
    mse_by_keep: dict[int, torch.Tensor] = {
        kc: torch.empty(a, dtype=torch.float32, device=device) for kc in keep_counts
    }

    x_blocks = x_blocks.to(torch.float32)
    w_blocks = w_blocks.to(torch.float32)

    for i in range(a):
        P = _build_partial_matrix(x_blocks[i], w_blocks, output_mask[i])
        order, G, _ = _backward_greedy(P, min_keep)
        removal_order[i] = order
        for kc in keep_counts:
            gmask = _greedy_mask_from_order(order, kc, kb)
            greedy_masks[kc][i] = gmask
            refined, n_swap, mse = _one_swap_refine(P, G, gmask)
            final_masks[kc][i] = refined
            swap_counts[kc][i] = n_swap
            mse_by_keep[kc][i] = mse

            # Verify stored mse vs direct recompute
            y = P.sum(dim=0)
            yhat = P[refined].sum(dim=0) if bool(refined.any()) else torch.zeros_like(y)
            direct = float(torch.mean((y - yhat) ** 2).item()) if P.shape[1] > 0 else 0.0
            if abs(direct - mse) >= 1e-5:
                raise RuntimeError(
                    f"stored MSE {mse} != direct {direct} (abs err >= 1e-5)"
                )

    return ExactRecoveryResult(
        masks_by_keep=final_masks,
        greedy_masks_by_keep=greedy_masks,
        removal_order=removal_order,
        swap_count_by_keep=swap_counts,
        mse_by_keep=mse_by_keep,
    )
