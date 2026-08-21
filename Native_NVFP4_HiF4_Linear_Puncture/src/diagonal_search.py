"""Channel-wise diagonal D search on HiF4 K=64 groups (calibration only)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from Native_NVFP4_HiF4_Linear_Puncture.src.config import DiagonalSearchConfig
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct

# Re-export for callers/tests that import config from this module.
__all__ = [
    "DiagonalSearchConfig",
    "DiagonalSearchResult",
    "group_output_error_via_gram",
    "search_channelwise_diagonal",
    "diagonal_result_to_row",
    "config_dict",
]


@dataclass
class DiagonalSearchResult:
    d: torch.Tensor
    log2_d: torch.Tensor
    group_cal_error_identity: torch.Tensor
    group_cal_error_searched: torch.Tensor
    group_kept_mask: torch.Tensor
    elapsed_s: float


def group_output_error_via_gram(
    a_target: torch.Tensor,
    w_target: torch.Tensor,
    a_ref: torch.Tensor,
    w_ref: torch.Tensor,
) -> torch.Tensor:
    """Exact ||A W_t^T - C W_r^T||_F^2 via 64×64 Grams (FP64)."""
    a_t = a_target.to(torch.float64)
    w_t = w_target.to(torch.float64)
    a_r = a_ref.to(torch.float64)
    w_r = w_ref.to(torch.float64)
    ata = a_t.T @ a_t
    wtw = w_t.T @ w_t
    ara = a_r.T @ a_r
    wrw = w_r.T @ w_r
    atc = a_t.T @ a_r
    wtd = w_t.T @ w_r
    # <X,Y>_F = sum_ij X_ij Y_ij  (not X_ij Y_ji)
    term1 = (ata * wtw).sum()
    term2 = (ara * wrw).sum()
    term3 = (atc * wtd).sum()
    return term1 + term2 - 2.0 * term3


def _linspace_indices(n: int, k: int) -> torch.Tensor:
    if n <= 0:
        return torch.zeros(0, dtype=torch.long)
    kk = min(k, n)
    if kk == n:
        return torch.arange(n, dtype=torch.long)
    return torch.linspace(0, n - 1, kk).round().long().unique_consecutive()


def _clip_log2(z: torch.Tensor, zmin: float, zmax: float) -> torch.Tensor:
    return torch.clamp(z, min=zmin, max=zmax)


def _eval_group_errors_batch(
    xg: torch.Tensor,
    ang: torch.Tensor,
    wg: torch.Tensor,
    d_batch: torch.Tensor,
) -> torch.Tensor:
    """Evaluate E_g for stacked diagonal candidates.

    xg: [N,64], ang: [N,64], wg: [O,64], d_batch: [C,64] → errors [C]
    """
    c = d_batch.shape[0]
    # [C,N,64], [C,O,64]
    xd = xg.unsqueeze(0) / d_batch.unsqueeze(1)
    wd = wg.unsqueeze(0) * d_batch.unsqueeze(1)
    # HiF4 over last dim (=64) for each candidate independently.
    ah = qdq_hif4_direct(xd.reshape(-1, 64)).to(torch.float32).reshape(c, -1, 64)
    wh = qdq_hif4_direct(wd.reshape(-1, 64)).to(torch.float32).reshape(
        c, wg.shape[0], 64
    )
    errors = []
    for i in range(c):
        errors.append(
            group_output_error_via_gram(ah[i], wh[i], ang, wg)
        )
    return torch.stack(errors)


def _search_one_group(
    xg: torch.Tensor,
    ang: torch.Tensor,
    wg: torch.Tensor,
    config: DiagonalSearchConfig,
) -> torch.Tensor:
    """Return log2_d for one 64-group (length 64)."""
    z = torch.zeros(64, dtype=torch.float32, device=xg.device)
    zmin = config.log2_scale_min
    zmax = config.log2_scale_max

    def sweep(offsets: tuple[float, ...], num_sweeps: int) -> None:
        off = torch.tensor(offsets, dtype=torch.float32, device=xg.device)
        for _ in range(num_sweeps):
            for j in range(64):
                base = z.clone()
                d_batch = []
                z_cands = []
                for o in off.tolist():
                    zj = _clip_log2(
                        torch.tensor(float(base[j]) + o), zmin, zmax
                    ).item()
                    z_cands.append(zj)
                    zz = base.clone()
                    zz[j] = zj
                    d_batch.append(torch.pow(torch.tensor(2.0, device=xg.device), zz))
                d_b = torch.stack(d_batch, dim=0)
                errs = _eval_group_errors_batch(xg, ang, wg, d_b)
                # Map errors back relative to offsets used (after clip may collide).
                # Re-evaluate choice among unique z candidates with tie-break on z.
                best_z = z_cands[0]
                best_e = float(errs[0].item())
                for zi, ei in zip(z_cands, errs.tolist()):
                    e = float(ei)
                    if e < best_e - 0.0:
                        best_e = e
                        best_z = zi
                    elif abs(e - best_e) <= 0.0:
                        if abs(zi) < abs(best_z) or (
                            abs(zi) == abs(best_z) and zi < best_z
                        ):
                            best_z = zi
                z[j] = best_z

    sweep(config.coarse_log2_offsets, config.num_coarse_sweeps)
    sweep(config.refine_log2_offsets, config.num_refine_sweeps)
    return z


def search_channelwise_diagonal(
    x_rot_cal: torch.Tensor,
    a_n_cal: torch.Tensor,
    w_n: torch.Tensor,
    config: DiagonalSearchConfig,
) -> DiagonalSearchResult:
    """Search per-channel D on calibration only. Does not accept validation tensors."""
    t0 = time.perf_counter()
    if x_rot_cal.ndim != 2 or a_n_cal.ndim != 2 or w_n.ndim != 2:
        raise ValueError("expected 2D tensors X[N,K], A[N,K], W[O,K]")
    if x_rot_cal.shape != a_n_cal.shape:
        raise ValueError("x_rot_cal and a_n_cal shape mismatch")
    n, k = x_rot_cal.shape
    o = w_n.shape[0]
    if w_n.shape[1] != k:
        raise ValueError("W_N K dim mismatch")
    gs = config.group_size
    if k % gs != 0:
        raise ValueError(f"K={k} not divisible by group_size={gs}")

    device = x_rot_cal.device
    x = x_rot_cal.to(device=device, dtype=torch.float32)
    a = a_n_cal.to(device=device, dtype=torch.float32)
    w = w_n.to(device=device, dtype=torch.float32)

    row_idx = _linspace_indices(n, config.search_token_rows_per_module).to(device)
    col_idx = _linspace_indices(o, config.search_output_channels_per_module).to(device)
    x_s = x.index_select(0, row_idx)
    a_s = a.index_select(0, row_idx)
    w_s = w.index_select(0, col_idx)

    num_groups = k // gs
    log2_d = torch.zeros(k, dtype=torch.float32)
    group_identity = torch.zeros(num_groups, dtype=torch.float64)
    group_searched = torch.zeros(num_groups, dtype=torch.float64)
    kept = torch.zeros(num_groups, dtype=torch.bool)

    ones = torch.ones(gs, dtype=torch.float32, device=device)

    for g in range(num_groups):
        k0 = g * gs
        k1 = k0 + gs
        xg = x_s[:, k0:k1]
        ang = a_s[:, k0:k1]
        wg = w_s[:, k0:k1]
        z_g = _search_one_group(xg, ang, wg, config)
        d_g = torch.pow(torch.tensor(2.0, device=device), z_g.to(device))

        # Full-calibration rollback gate.
        xg_full = x[:, k0:k1]
        ang_full = a[:, k0:k1]
        wg_full = w[:, k0:k1]
        e_id = float(
            group_output_error_via_gram(
                qdq_hif4_direct(xg_full / ones).to(torch.float32),
                qdq_hif4_direct(wg_full * ones).to(torch.float32),
                ang_full,
                wg_full,
            ).item()
        )
        e_se = float(
            group_output_error_via_gram(
                qdq_hif4_direct(xg_full / d_g).to(torch.float32),
                qdq_hif4_direct(wg_full * d_g).to(torch.float32),
                ang_full,
                wg_full,
            ).item()
        )
        group_identity[g] = e_id
        # Keep only strictly improved groups; equal error rolls back to identity.
        if e_se >= e_id:
            d_g = ones.clone()
            z_g = torch.zeros(gs, dtype=torch.float32)
            e_se = e_id
            kept[g] = False
        else:
            kept[g] = True
        group_searched[g] = e_se
        log2_d[k0:k1] = z_g.detach().cpu()

    d = torch.pow(torch.tensor(2.0), log2_d)
    # Equivalence gate on first min(32,N) cal rows (unquantized).
    n_eq = min(32, n)
    d_dev = d.to(device=device)
    y0 = F.linear(x[:n_eq], w)
    y1 = F.linear(x[:n_eq] / d_dev, w * d_dev)
    rel = torch.norm((y1 - y0).to(torch.float64)) / torch.clamp(
        torch.norm(y0.to(torch.float64)), min=config.eps
    )
    cos = torch.nn.functional.cosine_similarity(
        y0.reshape(1, -1).to(torch.float64),
        y1.reshape(1, -1).to(torch.float64),
    ).item()
    if float(rel.item()) > 1e-6 or cos < 0.9999999:
        raise RuntimeError(
            f"diagonal equivalence gate failed: relative_l2={float(rel):.3e}, cosine={cos}"
        )

    elapsed = time.perf_counter() - t0
    return DiagonalSearchResult(
        d=d.cpu(),
        log2_d=log2_d.cpu(),
        group_cal_error_identity=group_identity,
        group_cal_error_searched=group_searched,
        group_kept_mask=kept,
        elapsed_s=elapsed,
    )


def diagonal_result_to_row(module_name: str, result: DiagonalSearchResult, k_dim: int) -> dict[str, Any]:
    d = result.d
    log2 = result.log2_d
    id_e = float(result.group_cal_error_identity.sum().item())
    se_e = float(result.group_cal_error_searched.sum().item())
    recovery = float("nan") if id_e == 0 else (id_e - se_e) / id_e
    return {
        "module_name": module_name,
        "k_dim": k_dim,
        "num_k_groups": int(result.group_kept_mask.numel()),
        "num_groups_kept": int(result.group_kept_mask.sum().item()),
        "num_groups_rolled_back": int((~result.group_kept_mask).sum().item()),
        "cal_identity_error_energy": id_e,
        "cal_searched_error_energy": se_e,
        "cal_recovery": recovery,
        "diag_scale_min": float(d.min()),
        "diag_scale_p10": float(torch.quantile(d, 0.10)),
        "diag_scale_median": float(torch.quantile(d, 0.50)),
        "diag_scale_p90": float(torch.quantile(d, 0.90)),
        "diag_scale_max": float(d.max()),
        "fraction_d_eq_1": float((d == 1).to(torch.float64).mean()),
        "fraction_d_at_lower_bound": float((log2 <= -4.0 + 1e-12).to(torch.float64).mean()),
        "fraction_d_at_upper_bound": float((log2 >= 4.0 - 1e-12).to(torch.float64).mean()),
        "elapsed_s": result.elapsed_s,
    }


def config_dict(config: DiagonalSearchConfig) -> dict[str, Any]:
    return asdict(config)
