"""AX1: HiF4 S0 divisor oracle search."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_with_divisor
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics


def candidate_alphas(
    divisor_min: float = 4.0,
    divisor_max: float = 10.0,
    coarse_step: float = 0.125,
) -> torch.Tensor:
    n = int(round((divisor_max - divisor_min) / coarse_step)) + 1
    return torch.linspace(divisor_min, divisor_max, n, dtype=torch.float64)


def _fine_alphas(center: float, *, half_width: float = 0.125, fine_points: int = 33) -> torch.Tensor:
    lo = center - half_width
    hi = center + half_width
    return torch.linspace(lo, hi, fine_points, dtype=torch.float64)


def _energy_vs_ref(recons: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """recons [A, ...], ref [...] -> [A] error energies."""
    diff = recons - ref.unsqueeze(0)
    return (diff.double().reshape(diff.shape[0], -1) ** 2).sum(dim=1)


def _batch_hif4_alphas(
    x_bf16: torch.Tensor,
    alphas: torch.Tensor,
    *,
    alpha_chunk: int = 8,
    group_size: int = 64,
) -> torch.Tensor:
    """Quantize x with each alpha; return [A, *x.shape] float32."""
    orig_shape = tuple(x_bf16.shape)
    flat = x_bf16.reshape(-1, orig_shape[-1])
    outs: list[torch.Tensor] = []
    for i in range(0, alphas.numel(), alpha_chunk):
        chunk = alphas[i : i + alpha_chunk]
        chunk_out: list[torch.Tensor] = []
        for alpha in chunk:
            view = quantize_hif4_with_divisor(
                flat,
                divisor=float(alpha.item()),
                group_size=group_size,
                group_dim=-1,
                output_dtype=torch.float32,
            )
            chunk_out.append(view.metadata["values_fp32"].reshape(flat.shape))
        outs.append(torch.stack(chunk_out, dim=0))
    stacked = torch.cat(outs, dim=0)
    return stacked.reshape(stacked.shape[0], *orig_shape)


def _sub16_stats(groups64: torch.Tensor) -> dict[str, float]:
    """groups64 [G, 64] -> sub16 dispersion stats."""
    sub = groups64.reshape(groups64.shape[0], 4, 16)
    amax = sub.abs().amax(dim=-1)
    rms = torch.sqrt((sub * sub).mean(dim=-1).clamp_min(1e-12))
    amax_pos = amax[amax > 0]
    log2_range = 0.0
    if amax_pos.numel() > 1:
        log2_range = float((amax_pos.max().log2() - amax_pos.min().log2()).item())
    amax_max = amax.max(dim=-1).values
    amax_mean = amax.mean(dim=-1)
    ratio = float((amax_max / amax_mean.clamp_min(1e-12)).mean().item())
    energy = (sub * sub).sum(dim=-1)
    share_max = float((energy.max(dim=-1).values / energy.sum(dim=-1).clamp_min(1e-12)).mean().item())
    return {
        "sub16_amax_ratio": ratio,
        "sub16_rms_ratio": float((rms.max(dim=-1).values / rms.mean(dim=-1).clamp_min(1e-12)).mean().item()),
        "sub16_energy_share_max": share_max,
        "sub16_log2_amax_range": log2_range,
    }


def _distribution_stats(x: torch.Tensor) -> dict[str, float]:
    xf = x.float().reshape(-1)
    amax = float(xf.abs().max().item())
    rms = float(torch.sqrt((xf * xf).mean()).item())
    max_over_rms = amax / (rms + 1e-12)
    mean = float(xf.mean().item())
    var = float(((xf - mean) ** 2).mean().item())
    kurt = float(((xf - mean) ** 4).mean().item() / (var * var + 1e-12))
    return {
        "amax": amax,
        "rms": rms,
        "max_over_rms": max_over_rms,
        "kurtosis": kurt,
    }


@torch.no_grad()
def search_s0_divisor_oracle(
    x_bf16: torch.Tensor,
    a_n: torch.Tensor,
    w_n: torch.Tensor,
    *,
    alphas: torch.Tensor | None = None,
    alpha_chunk: int = 8,
    fine_half_width: float = 0.125,
    fine_points: int = 33,
) -> dict[str, Any]:
    """Coarse+fine alpha search; vectorize alphas as batch dim."""
    x = x_bf16.float()
    a_n_f = a_n.float()
    a_h = quantize_hif4_with_divisor(x_bf16, divisor=7.0, output_dtype=torch.float32)
    a_h_f = a_h.metadata["values_fp32"].float()
    w = w_n.float()

    e_full_a = float(((a_h_f - a_n_f) ** 2).sum().item())
    delta_full = a_h_f - a_n_f
    e_full_y = float((F.linear(delta_full, w) ** 2).sum().item())

    coarse = alphas if alphas is not None else candidate_alphas()
    coarse = coarse.to(dtype=torch.float64)
    recons_coarse = _batch_hif4_alphas(x_bf16, coarse, alpha_chunk=alpha_chunk)
    err_nv_coarse = _energy_vs_ref(recons_coarse, a_n_f)
    err_x_coarse = _energy_vs_ref(recons_coarse, x)
    best_nv_idx = int(err_nv_coarse.argmin().item())
    best_x_idx = int(err_x_coarse.argmin().item())
    alpha_nv_c = float(coarse[best_nv_idx].item())
    alpha_x_c = float(coarse[best_x_idx].item())

    fine_nv = _fine_alphas(alpha_nv_c, half_width=fine_half_width, fine_points=fine_points)
    fine_x = _fine_alphas(alpha_x_c, half_width=fine_half_width, fine_points=fine_points)
    fine_all = torch.unique(torch.cat([fine_nv, fine_x]))
    recons_fine = _batch_hif4_alphas(x_bf16, fine_all, alpha_chunk=alpha_chunk)
    err_nv_fine = _energy_vs_ref(recons_fine, a_n_f)
    err_x_fine = _energy_vs_ref(recons_fine, x)
    alpha_oracle_nvfp4 = float(fine_all[int(err_nv_fine.argmin().item())].item())
    alpha_oracle_bf16 = float(fine_all[int(err_x_fine.argmin().item())].item())

    a_oracle = quantize_hif4_with_divisor(
        x_bf16, divisor=alpha_oracle_nvfp4, output_dtype=torch.float32
    ).metadata["values_fp32"].float()
    a_current = a_h_f

    def _recovery(a_v: torch.Tensor) -> tuple[float, float]:
        e_a = float(((a_v - a_n_f) ** 2).sum().item())
        e_y = float((F.linear(a_v - a_n_f, w) ** 2).sum().item())
        r_a = 1.0 - e_a / e_full_a if e_full_a > 0 else 0.0
        r_y = 1.0 - e_y / e_full_y if e_full_y > 0 else 0.0
        return r_a, r_y

    r_a_oracle, r_y_oracle = _recovery(a_oracle)
    r_a_current, r_y_current = _recovery(a_current)

    groups = x.reshape(-1, 64)
    gstats = _sub16_stats(groups)
    dstats = _distribution_stats(x)

    # per-group stats at alpha=7
    g64 = groups
    amax64 = g64.abs().amax(dim=-1)
    per_group = {
        "alpha_current": 7.0,
        "group_amax_mean": float(amax64.mean().item()),
        "group_amax_max": float(amax64.max().item()),
        "num_groups": int(g64.shape[0]),
    }

    return {
        "alpha_current": 7.0,
        "alpha_oracle_nvfp4": alpha_oracle_nvfp4,
        "alpha_oracle_bf16": alpha_oracle_bf16,
        "activation_recovery": r_a_oracle,
        "output_recovery": r_y_oracle,
        "activation_recovery_current": r_a_current,
        "output_recovery_current": r_y_current,
        "e_full_activation": e_full_a,
        "e_full_output": e_full_y,
        "per_group_alpha7": per_group,
        **gstats,
        **dstats,
    }


def _select_groups(
    x_bf16: torch.Tensor,
    a_n: torch.Tensor,
    a_h: torch.Tensor,
    *,
    top_k: int,
    random_k: int,
    energy_k: int,
    seed: int,
) -> torch.Tensor:
    """Return selected K-dim group indices [S] (group_size=64)."""
    hidden = x_bf16.shape[-1]
    n = hidden // 64
    if n == 0:
        return torch.tensor([], dtype=torch.long)
    flat_x = x_bf16.reshape(-1, hidden).float()
    flat_n = a_n.reshape(-1, hidden).float()
    flat_h = a_h.reshape(-1, hidden).float()
    err = torch.zeros(n, dtype=torch.float64)
    ref_energy = torch.zeros(n, dtype=torch.float64)
    for gi in range(n):
        sl = slice(gi * 64, (gi + 1) * 64)
        err[gi] = ((flat_h[:, sl] - flat_n[:, sl]) ** 2).sum()
        ref_energy[gi] = (flat_n[:, sl] ** 2).sum()
    top_k = min(top_k, n)
    random_k = min(random_k, n)
    energy_k = min(energy_k, n)
    top_idx = torch.topk(err, k=top_k, largest=True).indices
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rand_idx = torch.randperm(n, generator=gen)[:random_k]
    target = ref_energy.median()
    energy_idx = torch.topk((ref_energy - target).abs(), k=energy_k, largest=False).indices
    return torch.unique(torch.cat([top_idx.cpu(), rand_idx, energy_idx.cpu()]))


@torch.no_grad()
def search_output_aware_group_alphas(
    x: torch.Tensor,
    a_n: torch.Tensor,
    w_n: torch.Tensor,
    alphas: torch.Tensor,
    *,
    top_k: int = 256,
    random_k: int = 256,
    energy_k: int = 256,
    seed: int = 20260810,
    alpha_chunk: int = 8,
) -> list[dict[str, Any]]:
    """Output-aware oracle on selected groups; alphas batched per group batch."""
    x_f = x.float()
    a_n_f = a_n.float()
    a_h_f = quantize_hif4_with_divisor(x, divisor=7.0, output_dtype=torch.float32)
    a_h_f = a_h_f.metadata["values_fp32"].float()
    w = w_n.float()

    group_idx = _select_groups(
        x, a_n, a_h_f, top_k=top_k, random_k=random_k, energy_k=energy_k, seed=seed
    )
    rows: list[dict[str, Any]] = []
    hidden = x.shape[-1]

    for gi in group_idx.tolist():
        sl_start = gi * 64
        sl_end = sl_start + 64
        x_g = x_f.reshape(-1, hidden)[:, sl_start:sl_end]
        a_n_g = a_n_f.reshape(-1, hidden)[:, sl_start:sl_end]
        w_g = w[:, sl_start:sl_end]

        recons = _batch_hif4_alphas(x_g, alphas, alpha_chunk=alpha_chunk)
        delta = recons - a_n_g.unsqueeze(0)
        out_err = (torch.einsum("agk,ok->ago", delta, w_g) ** 2).sum(dim=(1, 2))
        best = int(out_err.argmin().item())
        rows.append(
            {
                "group_index": int(gi),
                "alpha_oracle_output": float(alphas[best].item()),
                "output_error_energy": float(out_err[best].item()),
                "selection": "mixed",
            }
        )
    return rows
