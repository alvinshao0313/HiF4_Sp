from __future__ import annotations

import math

import torch


def mask_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    if candidate.shape != reference.shape:
        raise ValueError(
            f"mask shape mismatch: {tuple(candidate.shape)} vs {tuple(reference.shape)}"
        )
    c = candidate.to(torch.bool)
    r = reference.to(torch.bool)
    inter = (c & r).sum().item()
    union = (c | r).sum().item()
    keep = int(r.sum().item())
    total = c.numel()
    overlap = float(inter / keep) if keep > 0 else 0.0
    iou = float(inter / union) if union > 0 else 0.0
    hamming = float((c == r).sum().item() / total) if total > 0 else 0.0
    if c.ndim >= 2:
        exact_row = float((c == r).all(dim=-1).float().mean().item())
    else:
        exact_row = float(torch.equal(c, r))
    return {
        "intersection": float(inter),
        "union": float(union),
        "iou": iou,
        "overlap": overlap,
        "hamming_agreement": hamming,
        "exact_row_match": exact_row,
    }


def reconstruct_real_output(
    x_blocks: torch.Tensor,
    w_blocks: torch.Tensor,
    input_mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct Y from real X/W using only selected input K blocks."""
    a, kb, br, bk = x_blocks.shape
    jb, kb_w, bo, bk_w = w_blocks.shape
    if kb != kb_w or input_mask.shape != (a, kb):
        raise ValueError("shape mismatch in reconstruct_real_output")
    x_blocks = x_blocks.to(torch.float32)
    w_blocks = w_blocks.to(torch.float32)
    # y_blocks[i,j] = sum_k M_X[i,k] * (X[i,k] @ W[j,k].T)
    # einsum over kept k via masking x
    x_masked = x_blocks * input_mask.to(x_blocks.dtype).view(a, kb, 1, 1)
    # [A,Kb,32,64] x [Jb,Kb,32,64] -> [A,Jb,32,32]
    y_blocks = torch.einsum("akmd,jknd->ajmn", x_masked, w_blocks)
    y = (
        y_blocks.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(a * br, jb * bo)
    )
    return y


def reconstruct_joint_sparse_output(
    x_blocks: torch.Tensor,
    w_blocks: torch.Tensor,
    output_mask: torch.Tensor,
    input_mask: torch.Tensor,
) -> torch.Tensor:
    a, kb, br, bk = x_blocks.shape
    jb = w_blocks.shape[0]
    bo = w_blocks.shape[2]
    y = reconstruct_real_output(x_blocks, w_blocks, input_mask)
    # Zero output blocks outside method output mask.
    y_blocks = y.reshape(a, br, jb, bo).permute(0, 2, 1, 3).contiguous()
    y_blocks = y_blocks * output_mask.to(y_blocks.dtype).view(a, jb, 1, 1)
    return y_blocks.permute(0, 2, 1, 3).contiguous().reshape(a * br, jb * bo)


def nrmse(y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    y_hat = y_hat.to(torch.float32)
    y = y.to(torch.float32)
    if mask is not None:
        # mask is bool[A,Jb] over 32x32 blocks
        a, jb = mask.shape
        br = y.shape[0] // a
        bo = y.shape[1] // jb
        m = torch.zeros_like(y, dtype=torch.bool)
        for i in range(a):
            for j in range(jb):
                if bool(mask[i, j]):
                    m[i * br : (i + 1) * br, j * bo : (j + 1) * bo] = True
        diff = (y - y_hat)[m]
        ref = y[m]
    else:
        diff = y - y_hat
        ref = y
    num = torch.linalg.vector_norm(diff).item()
    den = torch.linalg.vector_norm(ref).item() + 1e-12
    return float(num / den)


def mse(y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    y_hat = y_hat.to(torch.float32)
    y = y.to(torch.float32)
    if mask is not None:
        a, jb = mask.shape
        br = y.shape[0] // a
        bo = y.shape[1] // jb
        m = torch.zeros_like(y, dtype=torch.bool)
        for i in range(a):
            for j in range(jb):
                if bool(mask[i, j]):
                    m[i * br : (i + 1) * br, j * bo : (j + 1) * bo] = True
        return float(torch.mean((y[m] - y_hat[m]) ** 2).item())
    return float(torch.mean((y - y_hat) ** 2).item())


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    """Average ranks for ties, 1-based."""
    x = x.to(torch.float64)
    n = x.numel()
    order = torch.argsort(x, stable=True)
    ranks = torch.empty(n, dtype=torch.float64, device=x.device)
    i = 0
    vals = x[order]
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.reshape(-1).to(torch.float64)
    y = y.reshape(-1).to(torch.float64)
    if x.numel() != y.numel():
        raise ValueError("pearson_corr length mismatch")
    if x.numel() < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if float(denom.item()) == 0.0:
        return 0.0
    return float((x * y).sum().item() / denom.item())


def spearman_rank(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.reshape(-1).to(torch.float64)
    y = y.reshape(-1).to(torch.float64)
    if x.numel() != y.numel():
        raise ValueError("spearman_rank length mismatch")
    if x.numel() < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.sqrt((rx * rx).sum() * (ry * ry).sum())
    if float(denom.item()) == 0.0:
        return 0.0
    return float((rx * ry).sum().item() / denom.item())


def kendall_tau_b(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.reshape(-1).tolist()
    y = y.reshape(-1).tolist()
    n = len(x)
    if n < 2:
        return 0.0
    conc = disc = tie_x = tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                conc += 1
            else:
                disc += 1
    num = conc - disc
    den = math.sqrt((conc + disc + tie_x) * (conc + disc + tie_y))
    if den == 0.0:
        return 0.0
    return float(num / den)
