"""AX3: NVFP4 vs HiF4 theoretical grids and real occupancy."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from ChuanCi.nvfp4_hif4_torch import E2M1_VALUES, E4M3FN_VALUES, E6M2_VALUES
from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_with_divisor
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation
from NVFP4.torch_fake import FP4_E2M1_MAX, cast_to_fp4_e2m1

# Formal S1P2 magnitudes used by HiF4 hardware path.
_S1P2_MAGNITUDES = torch.tensor(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=torch.float32
)


def enumerate_nvfp4_payload_grid() -> torch.Tensor:
    """Enumerate E2M1 payload levels via cast_to_fp4_e2m1 on fine linspace."""
    xs = torch.linspace(-FP4_E2M1_MAX, FP4_E2M1_MAX, 8192, dtype=torch.float32)
    cast = cast_to_fp4_e2m1(xs)
    pos = cast[cast >= 0].unique().sort().values
    return pos


def enumerate_hif4_payload_grid() -> torch.Tensor:
    """Enumerate S1P2 positive payload levels."""
    return _S1P2_MAGNITUDES.clone().sort().values


def enumerate_nvfp4_e4m3_scale_values() -> torch.Tensor:
    """Legal NVFP4 block-scale values = non-negative E4M3FN codebook (ChuanCi).

    Matches `ipc_analysis/formats/nvfp4.py` / `cast_to_fp8_e4m3fn` local scale.
    Outer FP32 per-tensor `input_global_scale` is excluded from the internal grid.
    """
    out = E4M3FN_VALUES.to(torch.float32).unique().sort().values
    if out.numel() != E4M3FN_VALUES.numel():
        raise RuntimeError("E4M3FN codebook contains unexpected duplicates")
    if float(out.min()) < 0:
        raise RuntimeError("E4M3FN scale codebook must be non-negative")
    return out


def enumerate_hif4_s0_values() -> torch.Tensor:
    """Legal HiF4 hardware S0 values = formal E6M2 codebook in ChuanCi."""
    out = E6M2_VALUES.to(torch.float32).unique().sort().values
    if out.numel() != E6M2_VALUES.numel():
        raise RuntimeError("E6M2 codebook contains unexpected duplicates")
    if float(out.min()) <= 0:
        raise RuntimeError("HiF4 S0 / E6M2 values must be strictly positive")
    return out


def _signed_from_magnitudes(mags: torch.Tensor) -> torch.Tensor:
    """Build unique signed set including a single 0 from positive magnitudes."""
    m = mags.to(torch.float32).reshape(-1)
    m = m[m >= 0].unique()
    signed = torch.cat([-m.flip(0), m])
    # Collapse +0/-0.
    return torch.unique(signed).sort().values


def _unique_finite_sorted(values: torch.Tensor) -> torch.Tensor:
    v = values.reshape(-1).to(torch.float32)
    v = v[torch.isfinite(v)]
    return torch.unique(v).sort().values


def enumerate_nvfp4_full_internal_grid() -> tuple[torch.Tensor, dict[str, Any]]:
    """Full internal set after removing FP32 per-tensor scale: unique(E4M3FN × signed E2M1).

    Dequant without g: A ≈ E2M1_payload × E4M3FN_block_scale.
    """
    scales = enumerate_nvfp4_e4m3_scale_values()
    payloads = _signed_from_magnitudes(E2M1_VALUES)
    raw = (scales.unsqueeze(1) * payloads.unsqueeze(0)).reshape(-1)
    num_raw = int(raw.numel())
    grid = _unique_finite_sorted(raw)
    meta = {
        "scale_format": {
            "name": "E4M3FN (ChuanCi.nvfp4_hif4_torch.E4M3FN_VALUES / cast_to_fp8_e4m3fn)",
            "num_scale_values": int(scales.numel()),
            "min_scale": float(scales.min().item()),
            "max_scale": float(scales.max().item()),
            "codes": "0..126",
        },
        "payload_format": "signed_E2M1",
        "num_raw_combinations": num_raw,
        "num_unique_values": int(grid.numel()),
        "duplicate_ratio": 1.0 - (float(grid.numel()) / float(num_raw)),
    }
    return grid, meta


def enumerate_hif4_full_internal_grid() -> tuple[torch.Tensor, dict[str, Any]]:
    """Full internal set: unique(± S0 × 2^(e8+e4) × S1P2)."""
    s0 = enumerate_hif4_s0_values()
    payloads = _signed_from_magnitudes(_S1P2_MAGNITUDES)
    e_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    chunks: list[torch.Tensor] = []
    for e8, e4 in e_pairs:
        factor = float(2 ** (e8 + e4))
        chunks.append((s0.unsqueeze(1) * (payloads.unsqueeze(0) * factor)).reshape(-1))
    raw = torch.cat(chunks, dim=0)
    num_raw = int(raw.numel())
    grid = _unique_finite_sorted(raw)
    meta = {
        "s0_format": {
            "name": "E6M2 (ChuanCi.nvfp4_hif4_torch.E6M2_VALUES / round_e6m2)",
            "num_scale_values": int(s0.numel()),
            "min_scale": float(s0.min().item()),
            "max_scale": float(s0.max().item()),
            "codes": "0..254",
        },
        "hierarchy": {"e8": [0, 1], "e4": [0, 1]},
        "payload_format": "signed_S1P2",
        "num_raw_combinations": num_raw,
        "num_unique_values": int(grid.numel()),
        "duplicate_ratio": 1.0 - (float(grid.numel()) / float(num_raw)),
    }
    return grid, meta


def full_grid_stats(grid: torch.Tensor, *, num_raw_combinations: int) -> dict[str, float]:
    g = grid.to(torch.float32).reshape(-1)
    if int(torch.unique(g).numel()) != int(g.numel()):
        raise ValueError("full grid must already be unique")
    pos = g[g > 0]
    neg = g[g < 0]
    abs_nz = g[g != 0].abs()
    return {
        "num_raw_combinations": float(num_raw_combinations),
        "num_unique_values": float(g.numel()),
        "num_positive_values": float(pos.numel()),
        "num_negative_values": float(neg.numel()),
        "num_zeros": float((g == 0).sum().item()),
        "min_nonzero_abs": float(abs_nz.min().item()) if abs_nz.numel() else 0.0,
        "max_abs": float(g.abs().max().item()) if g.numel() else 0.0,
        "duplicate_ratio": 1.0 - (float(g.numel()) / float(num_raw_combinations)),
    }


def theoretical_grid_stats(grid: torch.Tensor) -> dict[str, float]:
    """Legacy payload-only stats (normalized |q|/max for positive grid)."""
    g = grid[grid > 0].float()
    if g.numel() == 0:
        return {
            "num_positive_levels": 0.0,
            "min_norm": 0.0,
            "median_spacing": 0.0,
            "max_spacing": 0.0,
            "spacing_cv": 0.0,
        }
    norm = g / g.max()
    spacings = norm[1:] - norm[:-1] if norm.numel() > 1 else torch.tensor([0.0])
    med = float(spacings.median().item()) if spacings.numel() else 0.0
    mx = float(spacings.max().item()) if spacings.numel() else 0.0
    cv = float(spacings.std().item() / (spacings.mean().item() + 1e-12)) if spacings.numel() > 1 else 0.0
    return {
        "num_positive_levels": float(g.numel()),
        "min_norm": float(norm.min().item()),
        "median_spacing": med,
        "max_spacing": mx,
        "spacing_cv": cv,
    }


def _payload_occurrence(payload_abs: torch.Tensor, grid: torch.Tensor, *, eps: float = 1e-6) -> dict[str, float]:
    """Occupancy stats for payload magnitudes (existing AX3 occupancy path)."""
    p = payload_abs.reshape(-1).float()
    if p.numel() == 0:
        return {
            "zero_rate": 0.0,
            "boundary_rate": 0.0,
            "mid_rate": 0.0,
            "entropy": 0.0,
            "norm_entropy": 0.0,
            "effective_codes": 0.0,
            "max_code_prob": 0.0,
        }
    g = grid.to(device=p.device, dtype=torch.float32).reshape(-1)
    gmax = float(g.abs().max().item())
    boundary = gmax
    dist = (p.unsqueeze(-1) - g.unsqueeze(0)).abs()
    idx = dist.argmin(dim=-1)
    counts = torch.bincount(idx, minlength=g.numel()).float()
    probs = counts / counts.sum()
    nonzero_probs = probs[probs > 0]
    entropy = float(-(nonzero_probs * nonzero_probs.log()).sum().item())
    k = float(g.numel())
    norm_entropy = entropy / math.log(k) if k > 1 else 0.0
    zero_rate = float((p <= eps).float().mean().item())
    boundary_rate = float((p >= boundary - eps).float().mean().item())
    mid_rate = float(1.0 - zero_rate - boundary_rate)
    return {
        "zero_rate": zero_rate,
        "boundary_rate": boundary_rate,
        "mid_rate": mid_rate,
        "entropy": entropy,
        "norm_entropy": norm_entropy,
        "effective_codes": float((probs > 0).sum().item()),
        "max_code_prob": float(probs.max().item()),
    }


def compare_local_scales(nv_meta: dict, hf_meta: dict) -> dict[str, float]:
    """Compare NVFP4 vs HiF4 effective local scale distributions."""
    nv_e4 = nv_meta.get("e4m3_local_scale")
    nv_raw = nv_meta.get("raw_local_scale")
    hf_local = hf_meta.get("local_scale")
    out: dict[str, float] = {}
    if nv_e4 is not None and torch.is_tensor(nv_e4):
        t = nv_e4.float().reshape(-1)
        out["nv_e4m3_local_mean"] = float(t.mean().item())
        out["nv_e4m3_local_std"] = float(t.std().item())
    if nv_raw is not None and torch.is_tensor(nv_raw):
        t = nv_raw.float().reshape(-1)
        out["nv_raw_local_mean"] = float(t.mean().item())
        out["nv_raw_local_std"] = float(t.std().item())
    if hf_local is not None and torch.is_tensor(hf_local):
        t = hf_local.float().reshape(-1)
        out["hf_local_mean"] = float(t.mean().item())
        out["hf_local_std"] = float(t.std().item())
    g = nv_meta.get("input_global_scale")
    if g is not None:
        out["nv_global_scale"] = float(g) if not torch.is_tensor(g) else float(g.reshape(()).item())
    top = hf_meta.get("top_scale")
    if top is not None and torch.is_tensor(top):
        t = top.float().reshape(-1)
        out["hf_s0_mean"] = float(t.mean().item())
        out["hf_s0_std"] = float(t.std().item())
    return out


def grid_distance(payload_abs: torch.Tensor, grid: torch.Tensor, local_scale: torch.Tensor) -> dict[str, float]:
    """Distance to nearest normalized payload grid point (occupancy analysis)."""
    p = payload_abs.float().reshape(-1)
    ls = local_scale.float().reshape(-1).to(device=p.device)
    g = grid.to(device=p.device, dtype=torch.float32).reshape(-1)
    if p.numel() == 0 or ls.numel() == 0:
        return {"mean_grid_distance": 0.0, "mean_nearest_distance": 0.0}
    if ls.numel() == 1:
        ls = ls.expand(p.numel())
    elif ls.numel() != p.numel():
        if p.numel() % ls.numel() != 0:
            raise ValueError(
                f"payload len {p.numel()} not divisible by local_scale len {ls.numel()}"
            )
        ls = ls.repeat_interleave(p.numel() // ls.numel())
    ls = ls.clamp_min(1e-12)
    norm = (p / ls).clamp_min(0)
    gnorm = g / g.max().clamp_min(1e-12)
    dist = (norm.unsqueeze(-1) - gnorm.unsqueeze(0)).abs()
    nearest = dist.min(dim=-1).values
    return {
        "mean_grid_distance": float(nearest.mean().item()),
        "mean_nearest_distance": float(nearest.mean().item()),
    }


@torch.no_grad()
def occupancy_before_after_oracle_s0(
    x_bf16: torch.Tensor,
    a_n: torch.Tensor,
    alpha_oracle: float,
    *,
    nv_meta: dict | None = None,
) -> dict[str, Any]:
    """Compare HiF4 occupancy at alpha=7 vs oracle alpha."""
    hf_grid = enumerate_hif4_payload_grid()
    a7 = quantize_hif4_with_divisor(x_bf16, divisor=7.0, output_dtype=torch.float32)
    a_or = quantize_hif4_with_divisor(x_bf16, divisor=alpha_oracle, output_dtype=torch.float32)
    p7 = a7.metadata["payload_magnitude"].abs()
    por = a_or.metadata["payload_magnitude"].abs()
    occ7 = _payload_occurrence(p7, hf_grid)
    occ_or = _payload_occurrence(por, hf_grid)
    return {
        "alpha_current": 7.0,
        "alpha_oracle": alpha_oracle,
        "occupancy_alpha7": occ7,
        "occupancy_oracle": occ_or,
        "zero_rate_delta": occ_or["zero_rate"] - occ7["zero_rate"],
        "boundary_rate_delta": occ_or["boundary_rate"] - occ7["boundary_rate"],
        "entropy_delta": occ_or["entropy"] - occ7["entropy"],
    }


@torch.no_grad()
def analyze_grid_occupancy_row(
    x_bf16: torch.Tensor,
    a_n: torch.Tensor,
    a_h: torch.Tensor,
    w_n: torch.Tensor,
    nv_meta: dict,
    hf_meta: dict,
    *,
    alpha_oracle: float | None = None,
) -> dict[str, Any]:
    """Single capture row for AX3 CSV (real occupancy; unchanged semantics)."""
    nv_grid = enumerate_nvfp4_payload_grid()
    hf_grid = enumerate_hif4_payload_grid()
    nv_payload = nv_meta.get("e2m1_payload")
    hf_payload = hf_meta.get("payload_magnitude")
    if nv_payload is None or hf_payload is None:
        raise KeyError("payload metadata missing")
    nv_occ = _payload_occurrence(nv_payload.abs(), nv_grid)
    hf_occ = _payload_occurrence(hf_payload.abs(), hf_grid)
    scale_cmp = compare_local_scales(nv_meta, hf_meta)
    hf_local = hf_meta["local_scale"]
    nv_effective = nv_meta["e4m3_local_scale"]
    nv_dist = grid_distance(nv_payload.abs(), nv_grid, nv_effective)
    hf_dist = grid_distance(hf_payload.abs(), hf_grid, hf_local)
    delta = a_h.float() - a_n.float()
    out_err = float((torch.nn.functional.linear(delta, w_n.float()) ** 2).sum().item())
    row: dict[str, Any] = {
        **{f"nv_occ_{k}": v for k, v in nv_occ.items()},
        **{f"hf_occ_{k}": v for k, v in hf_occ.items()},
        **scale_cmp,
        **{f"nv_dist_{k}": v for k, v in nv_dist.items()},
        **{f"hf_dist_{k}": v for k, v in hf_dist.items()},
        "conversion_output_error": out_err,
    }
    if alpha_oracle is not None:
        occ_cmp = occupancy_before_after_oracle_s0(x_bf16, a_n, alpha_oracle, nv_meta=nv_meta)
        row["oracle_s0_occupancy"] = occ_cmp
    return row


def _shared_hist_counts(a: torch.Tensor, b: torch.Tensor, *, bins: int) -> dict[str, Any]:
    lo = float(min(a.min().item(), b.min().item()))
    hi = float(max(a.max().item(), b.max().item()))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError(f"invalid shared hist range: [{lo}, {hi}]")
    edges = torch.linspace(lo, hi, bins + 1, dtype=torch.float64)
    ca = torch.histogram(a.to(torch.float64), bins=edges).hist
    cb = torch.histogram(b.to(torch.float64), bins=edges).hist
    return {
        "bin_edges": edges.to(torch.float32).tolist(),
        "nvfp4_counts": ca.to(torch.int64).tolist(),
        "hif4_counts": cb.to(torch.int64).tolist(),
        "x_min": lo,
        "x_max": hi,
        "bins": bins,
    }


def _log2_abs_hist(a: torch.Tensor, b: torch.Tensor, *, bins: int) -> dict[str, Any]:
    an = a[a != 0].abs().log2()
    bn = b[b != 0].abs().log2()
    return _shared_hist_counts(an, bn, bins=bins)


def density_comparison_summary(nv: torch.Tensor, hf: torch.Tensor, *, bins: int = 64) -> dict[str, Any]:
    """Compare unique-point density by log2(|x|) bins and near zero."""
    log_hist = _log2_abs_hist(nv, hf, bins=bins)
    edges = torch.tensor(log_hist["bin_edges"], dtype=torch.float32)
    c_nv = torch.tensor(log_hist["nvfp4_counts"], dtype=torch.float32)
    c_hf = torch.tensor(log_hist["hif4_counts"], dtype=torch.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nv_denser = (c_nv > c_hf).nonzero(as_tuple=False).reshape(-1)
    hf_denser = (c_hf > c_nv).nonzero(as_tuple=False).reshape(-1)

    def _ranges(idxs: torch.Tensor) -> list[dict[str, float]]:
        if idxs.numel() == 0:
            return []
        # merge contiguous denser bins
        ranges = []
        start = int(idxs[0].item())
        prev = start
        for i in idxs[1:].tolist():
            if i == prev + 1:
                prev = i
                continue
            ranges.append(
                {
                    "log2_abs_lo": float(edges[start].item()),
                    "log2_abs_hi": float(edges[prev + 1].item()),
                    "center_log2": float(centers[start : prev + 1].mean().item()),
                }
            )
            start = prev = i
        ranges.append(
            {
                "log2_abs_lo": float(edges[start].item()),
                "log2_abs_hi": float(edges[prev + 1].item()),
                "center_log2": float(centers[start : prev + 1].mean().item()),
            }
        )
        return ranges

    nv_abs = nv[nv != 0].abs()
    hf_abs = hf[hf != 0].abs()
    common_lo = float(max(nv_abs.min().item(), hf_abs.min().item()))
    common_hi = float(min(nv_abs.max().item(), hf_abs.max().item()))
    if not (common_hi > common_lo > 0):
        raise RuntimeError(
            f"no overlapping positive dynamic range for near-zero view: "
            f"[{common_lo}, {common_hi}]"
        )
    in_common = torch.cat(
        [
            nv_abs[(nv_abs >= common_lo) & (nv_abs <= common_hi)],
            hf_abs[(hf_abs >= common_lo) & (hf_abs <= common_hi)],
        ]
    )
    # 1% quantile inside the shared visible range → real-valued near-zero window.
    q = float(torch.quantile(in_common, 0.01).item())
    q = max(q, common_lo)
    near = (-q, q)
    nv_near = int(((nv >= near[0]) & (nv <= near[1])).sum().item())
    hf_near = int(((hf >= near[0]) & (hf <= near[1])).sum().item())
    return {
        "log2_hist": log_hist,
        "nvfp4_denser_log2_ranges": _ranges(nv_denser),
        "hif4_denser_log2_ranges": _ranges(hf_denser),
        "near_zero_window": {
            "x_min": near[0],
            "x_max": near[1],
            "quantile": 0.01,
            "common_abs_lo": common_lo,
            "common_abs_hi": common_hi,
        },
        "near_zero_unique_counts": {"nvfp4": nv_near, "hif4": hf_near},
    }


def build_theoretical_grid_json(out_dir: Path | str | None = None) -> dict[str, Any]:
    """Theory grids: keep payload fields; add full internal unique sets."""
    nv_payload = enumerate_nvfp4_payload_grid()
    hf_payload = enumerate_hif4_payload_grid()
    nv_full, nv_meta = enumerate_nvfp4_full_internal_grid()
    hf_full, hf_meta = enumerate_hif4_full_internal_grid()
    if nv_full.numel() <= nv_payload.numel() or hf_full.numel() <= hf_payload.numel():
        raise RuntimeError("full internal grids must be larger than payload codebooks")

    dens = density_comparison_summary(nv_full, hf_full, bins=64)
    linear_hist = _shared_hist_counts(nv_full, hf_full, bins=128)

    payload = {
        # legacy keys (payload codebook only)
        "nvfp4_grid": nv_payload.tolist(),
        "hif4_grid": hf_payload.tolist(),
        "nvfp4_stats": theoretical_grid_stats(nv_payload),
        "hif4_stats": theoretical_grid_stats(hf_payload),
        "nvfp4_max": float(nv_payload.max().item()),
        "hif4_max": float(hf_payload.max().item()),
        # explicit payload aliases
        "nvfp4_payload_grid": nv_payload.tolist(),
        "hif4_payload_grid": hf_payload.tolist(),
        # full internal unique sets (also stored as .pt when out_dir given)
        "nvfp4_full_internal_grid": nv_full.tolist(),
        "hif4_full_internal_grid": hf_full.tolist(),
        "nvfp4_full_stats": full_grid_stats(
            nv_full, num_raw_combinations=int(nv_meta["num_raw_combinations"])
        ),
        "hif4_full_stats": full_grid_stats(
            hf_full, num_raw_combinations=int(hf_meta["num_raw_combinations"])
        ),
        "nvfp4_scale_format": nv_meta["scale_format"],
        "hif4_s0_format": hf_meta["s0_format"],
        "nvfp4_full_meta": nv_meta,
        "hif4_full_meta": hf_meta,
        "full_grid_linear_histogram": linear_hist,
        "full_grid_density_comparison": dens,
        "note": (
            "full_internal_grid = unique representable values after removing outer FP32 "
            "per-tensor scale; NVFP4 uses E4M3FN×signed E2M1; "
            "HiF4 uses S0×2^(e8+e4)×signed S1P2."
        ),
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(nv_full, out / "ax3_nvfp4_full_internal_grid.pt")
        torch.save(hf_full, out / "ax3_hif4_full_internal_grid.pt")
        payload["nvfp4_full_internal_grid_path"] = str(out / "ax3_nvfp4_full_internal_grid.pt")
        payload["hif4_full_internal_grid_path"] = str(out / "ax3_hif4_full_internal_grid.pt")
    return payload
