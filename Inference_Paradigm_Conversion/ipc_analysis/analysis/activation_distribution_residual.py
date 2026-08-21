"""Pure stats for NVFP4 W4A4 activation distribution / NVFP4→HiF4 residual viz."""

from __future__ import annotations

import math
from typing import Any

import torch

_EPS = 1e-12

_ENERGY_CURVE_FRAC = (
    0.0,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    0.7,
    1.0,
)


def _as_flat_f32(a: torch.Tensor) -> torch.Tensor:
    return a.detach().to(torch.float32).reshape(-1)


def residual_element_stats(a_nvfp4: torch.Tensor, a_hif4: torch.Tensor) -> dict[str, float]:
    if a_nvfp4.shape != a_hif4.shape:
        raise ValueError(f"shape mismatch {tuple(a_nvfp4.shape)} vs {tuple(a_hif4.shape)}")
    an = _as_flat_f32(a_nvfp4)
    ah = _as_flat_f32(a_hif4)
    delta = ah - an
    abs_d = delta.abs()
    n = int(an.numel())
    if n == 0:
        raise ValueError("empty tensors")
    mean_d = float(delta.mean().item())
    rms = float(torch.sqrt((delta * delta).mean()).item())
    an_energy = float((an * an).sum().item())
    both_nz = (an != 0) & (ah != 0)
    if int(both_nz.sum().item()) == 0:
        sign_flip = 0.0
    else:
        sign_flip = float(((an[both_nz] * ah[both_nz]) < 0).float().mean().item())
    # Pearson on |A_N| vs |ΔA|
    x = an.abs()
    y = abs_d
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float(torch.sqrt((x_c * x_c).sum() * (y_c * y_c).sum()).item())
    pearson = float((x_c * y_c).sum().item() / denom) if denom > 0 else 0.0
    qs = torch.quantile(abs_d, torch.tensor([0.5, 0.9, 0.95, 0.99, 0.999], device=abs_d.device))
    return {
        "numel": float(n),
        "mean_delta": mean_d,
        "median_delta": float(delta.median().item()),
        "std_delta": float(delta.std(unbiased=False).item()),
        "mae": float(abs_d.mean().item()),
        "rms": rms,
        "max_abs": float(abs_d.max().item()),
        "q50_abs": float(qs[0].item()),
        "q90_abs": float(qs[1].item()),
        "q95_abs": float(qs[2].item()),
        "q99_abs": float(qs[3].item()),
        "q999_abs": float(qs[4].item()),
        "bias_over_rms": mean_d / (rms + _EPS),
        "nmse_hif4_vs_nvfp4": float((delta * delta).sum().item() / (an_energy + _EPS)),
        "pearson_abs_an_abs_delta": pearson,
        "sign_flip_rate": sign_flip,
    }


def zero_transition_stats(a_nvfp4: torch.Tensor, a_hif4: torch.Tensor) -> dict[str, float]:
    if a_nvfp4.shape != a_hif4.shape:
        raise ValueError(f"shape mismatch {tuple(a_nvfp4.shape)} vs {tuple(a_hif4.shape)}")
    an = _as_flat_f32(a_nvfp4)
    ah = _as_flat_f32(a_hif4)
    n = float(an.numel())
    if n == 0:
        raise ValueError("empty tensors")
    nz_an = an == 0
    nz_ah = ah == 0
    both_zero = float((nz_an & nz_ah).sum().item()) / n
    nv_zero_hf_nonzero = float((nz_an & ~nz_ah).sum().item()) / n
    nv_nonzero_hf_zero = float((~nz_an & nz_ah).sum().item()) / n
    both_nonzero = float((~nz_an & ~nz_ah).sum().item()) / n
    s = both_zero + nv_zero_hf_nonzero + nv_nonzero_hf_zero + both_nonzero
    if abs(s - 1.0) > 1e-6:
        raise RuntimeError(f"zero transition probs sum to {s}, not 1")
    nv_zero_rate = float(nz_an.float().mean().item())
    hf_zero_rate = float(nz_ah.float().mean().item())
    return {
        "both_zero": both_zero,
        "nv_zero_hf_nonzero": nv_zero_hf_nonzero,
        "nv_nonzero_hf_zero": nv_nonzero_hf_zero,
        "both_nonzero": both_nonzero,
        "nv_zero_rate": nv_zero_rate,
        "hf_zero_rate": hf_zero_rate,
        "hf_minus_nv_zero_rate": hf_zero_rate - nv_zero_rate,
    }


def residual_energy_concentration(delta: torch.Tensor) -> dict[str, float | list[float]]:
    d = _as_flat_f32(delta)
    n = int(d.numel())
    if n == 0:
        raise ValueError("empty delta")
    energy = d * d
    total = float(energy.sum().item())
    if total <= 0:
        zeros = [0.0] * len(_ENERGY_CURVE_FRAC)
        return {
            "top_0p1pct_energy_share": 0.0,
            "top_1pct_energy_share": 0.0,
            "top_5pct_energy_share": 0.0,
            "top_10pct_energy_share": 0.0,
            "curve_fraction_elements": list(_ENERGY_CURVE_FRAC),
            "curve_fraction_energy": zeros,
        }
    sorted_e, _ = torch.sort(energy, descending=True)
    csum = torch.cumsum(sorted_e, dim=0)
    curve_e: list[float] = []
    for frac in _ENERGY_CURVE_FRAC:
        if frac <= 0:
            curve_e.append(0.0)
            continue
        k = max(1, int(math.ceil(frac * n)))
        k = min(k, n)
        curve_e.append(float(csum[k - 1].item() / total))

    def share(pct: float) -> float:
        k = max(1, int(math.ceil(pct * n)))
        k = min(k, n)
        return float(csum[k - 1].item() / total)

    return {
        "top_0p1pct_energy_share": share(0.001),
        "top_1pct_energy_share": share(0.01),
        "top_5pct_energy_share": share(0.05),
        "top_10pct_energy_share": share(0.10),
        "curve_fraction_elements": list(_ENERGY_CURVE_FRAC),
        "curve_fraction_energy": curve_e,
    }


def activation_quantile_residual_curve(
    a_nvfp4: torch.Tensor,
    delta: torch.Tensor,
    num_bins: int = 32,
) -> dict[str, list[float]]:
    if a_nvfp4.shape != delta.shape:
        raise ValueError("shape mismatch")
    an = _as_flat_f32(a_nvfp4)
    d = _as_flat_f32(delta)
    n = int(an.numel())
    if n == 0:
        raise ValueError("empty")
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    abs_an = an.abs()
    # Equal-frequency bins via rank quantiles of |A_N|.
    order = torch.argsort(abs_an)
    abs_sorted = abs_an[order]
    d_sorted = d[order]
    an_sorted = an[order]
    edges = [0]
    for b in range(1, num_bins):
        edges.append(int(round(b * n / num_bins)))
    edges.append(n)
    # Ensure nondecreasing unique-ish edges
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]
    abs_an_lo: list[float] = []
    abs_an_hi: list[float] = []
    count: list[float] = []
    mean_abs_an: list[float] = []
    mean_abs_delta: list[float] = []
    rms_delta: list[float] = []
    mean_delta: list[float] = []
    zero_transition_rate: list[float] = []
    for b in range(num_bins):
        lo_i, hi_i = edges[b], edges[b + 1]
        if hi_i <= lo_i:
            # empty bin (possible with tiny n): still emit a row
            abs_an_lo.append(float("nan"))
            abs_an_hi.append(float("nan"))
            count.append(0.0)
            mean_abs_an.append(float("nan"))
            mean_abs_delta.append(float("nan"))
            rms_delta.append(float("nan"))
            mean_delta.append(float("nan"))
            zero_transition_rate.append(float("nan"))
            continue
        aa = abs_sorted[lo_i:hi_i]
        dd = d_sorted[lo_i:hi_i]
        aa_signed = an_sorted[lo_i:hi_i]
        # zero transition relative to A_N / A_H = A_N+delta
        ah = aa_signed + dd
        zt = ((aa_signed == 0) != (ah == 0)).float().mean()
        abs_an_lo.append(float(aa.min().item()))
        abs_an_hi.append(float(aa.max().item()))
        count.append(float(aa.numel()))
        mean_abs_an.append(float(aa.mean().item()))
        mean_abs_delta.append(float(dd.abs().mean().item()))
        rms_delta.append(float(torch.sqrt((dd * dd).mean()).item()))
        mean_delta.append(float(dd.mean().item()))
        zero_transition_rate.append(float(zt.item()))
    if int(sum(count)) != n:
        raise RuntimeError(f"quantile bin counts {sum(count)} != numel {n}")
    return {
        "abs_an_lo": abs_an_lo,
        "abs_an_hi": abs_an_hi,
        "count": count,
        "mean_abs_an": mean_abs_an,
        "mean_abs_delta": mean_abs_delta,
        "rms_delta": rms_delta,
        "mean_delta": mean_delta,
        "zero_transition_rate": zero_transition_rate,
    }


def _sub16_dispersion_per_token(x_tg: torch.Tensor) -> torch.Tensor:
    """x_tg: [T, 64] -> d_t [T]."""
    t = x_tg.shape[0]
    blocks = x_tg.reshape(t, 4, 16).abs().amax(dim=-1)  # [T,4]
    pos_max = blocks.amax(dim=-1)
    pos_min = torch.where(blocks > 0, blocks, torch.full_like(blocks, float("inf"))).amin(dim=-1)
    return torch.where(
        (pos_max > 0) & torch.isfinite(pos_min) & (pos_min > 0),
        torch.log2(pos_max) - torch.log2(pos_min),
        torch.zeros_like(pos_max),
    )


def group64_residual_stats(
    x_in: torch.Tensor,
    a_nvfp4: torch.Tensor,
    a_hif4: torch.Tensor,
) -> list[dict[str, float | int]]:
    if x_in.shape != a_nvfp4.shape or x_in.shape != a_hif4.shape:
        raise ValueError("shape mismatch among x_in / a_nvfp4 / a_hif4")
    x = x_in.detach().to(torch.float32)
    an = a_nvfp4.detach().to(torch.float32)
    ah = a_hif4.detach().to(torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
        an = an.unsqueeze(0)
        ah = ah.unsqueeze(0)
    elif x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
        an = an.reshape(-1, an.shape[-1])
        ah = ah.reshape(-1, ah.shape[-1])
    t, k = x.shape
    if k % 64 != 0:
        raise ValueError(f"K={k} must be divisible by 64")
    g = k // 64
    xg = x.reshape(t, g, 64)
    ang = an.reshape(t, g, 64)
    ahg = ah.reshape(t, g, 64)
    dg = ahg - ang

    amax64 = xg.abs().amax(dim=-1)  # [T,G]
    blocks = xg.reshape(t, g, 4, 16).abs().amax(dim=-1)  # [T,G,4]
    pos_max = blocks.amax(dim=-1)
    pos_min = torch.where(blocks > 0, blocks, torch.full_like(blocks, float("inf"))).amin(dim=-1)
    disp = torch.where(
        (pos_max > 0) & torch.isfinite(pos_min) & (pos_min > 0),
        torch.log2(pos_max) - torch.log2(pos_min),
        torch.zeros_like(pos_max),
    )  # [T,G]

    n_el = float(t * 64)
    sum_x2 = (xg * xg).sum(dim=(0, 2))
    sum_an2 = (ang * ang).sum(dim=(0, 2))
    sum_ah2 = (ahg * ahg).sum(dim=(0, 2))
    sum_d2 = (dg * dg).sum(dim=(0, 2))
    mae = dg.abs().mean(dim=(0, 2))
    max_abs = dg.abs().amax(dim=(0, 2))
    nz_hf0 = ((ang != 0) & (ahg == 0)).float().mean(dim=(0, 2))
    both_nz = (ang != 0) & (ahg != 0)
    flips = both_nz & ((ang * ahg) < 0)
    both_n = both_nz.sum(dim=(0, 2)).to(torch.float32).clamp_min(1.0)
    sflip = flips.sum(dim=(0, 2)).to(torch.float32) / both_n

    rows: list[dict[str, float | int]] = []
    amax_max = amax64.amax(dim=0).tolist()
    amax_mean = amax64.mean(dim=0).tolist()
    mean_disp = disp.mean(dim=0).tolist()
    q90_disp = torch.quantile(disp, 0.9, dim=0).tolist()
    rms_x = torch.sqrt(sum_x2 / n_el).tolist()
    rms_an = torch.sqrt(sum_an2 / n_el).tolist()
    rms_ah = torch.sqrt(sum_ah2 / n_el).tolist()
    rms_d = torch.sqrt(sum_d2 / n_el).tolist()
    mae_l = mae.tolist()
    max_l = max_abs.tolist()
    de_l = sum_d2.tolist()
    an_e_l = sum_an2.tolist()
    nz_l = nz_hf0.tolist()
    sf_l = sflip.tolist()
    for gi in range(g):
        an_e = float(an_e_l[gi])
        d_e = float(de_l[gi])
        rows.append(
            {
                "group_idx": int(gi),
                "num_tokens": int(t),
                "amax64_x_max": float(amax_max[gi]),
                "amax64_x_mean": float(amax_mean[gi]),
                "rms_x": float(rms_x[gi]),
                "rms_an": float(rms_an[gi]),
                "rms_ah": float(rms_ah[gi]),
                "rms_delta": float(rms_d[gi]),
                "mae_delta": float(mae_l[gi]),
                "max_abs_delta": float(max_l[gi]),
                "residual_energy": d_e,
                "delta_energy_over_an_energy": d_e / (an_e + _EPS),
                "mean_sub16_dispersion": float(mean_disp[gi]),
                "q90_sub16_dispersion": float(q90_disp[gi]),
                "nv_nonzero_hf_zero_rate": float(nz_l[gi]),
                "sign_flip_rate": float(sf_l[gi]),
            }
        )
    return rows


def build_token_group_residual_map(delta: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    d = delta.detach().to(torch.float32)
    if d.ndim == 1:
        d = d.unsqueeze(0)
    elif d.ndim > 2:
        d = d.reshape(-1, d.shape[-1])
    t, k = d.shape
    if k % group_size != 0:
        raise ValueError(f"K={k} must be divisible by {group_size}")
    g = k // group_size
    blocks = d.reshape(t, g, group_size)
    return torch.sqrt((blocks * blocks).mean(dim=-1))


def basic_tensor_moments(x: torch.Tensor) -> dict[str, float]:
    v = _as_flat_f32(x)
    if v.numel() == 0:
        raise ValueError("empty")
    return {
        "mean": float(v.mean().item()),
        "std": float(v.std(unbiased=False).item()),
        "rms": float(torch.sqrt((v * v).mean()).item()),
        "zero_rate": float((v == 0).float().mean().item()),
    }


def flatten_stats_for_csv(
    residual: dict[str, float],
    zero_tr: dict[str, float],
    energy: dict[str, float | list[float]],
) -> dict[str, Any]:
    """Prefix residual_/zero_transition_/energy_ keys for capture summary rows."""
    out: dict[str, Any] = {}
    for k, v in residual.items():
        out[f"residual_{k}"] = v
    for k, v in zero_tr.items():
        out[f"zero_transition_{k}"] = v
    for k, v in energy.items():
        if isinstance(v, list):
            continue
        out[f"energy_{k}"] = v
    return out
