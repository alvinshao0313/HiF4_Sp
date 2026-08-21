"""A5: H2 controlled interventions on real 64-group activations."""

from __future__ import annotations

from typing import Any

import torch

from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics

OUTLIER_SWEEP = (1.0, 2.0, 4.0, 8.0)
DISPERSION_D = (0.0, 0.5, 1.0, 1.5, 2.0)
SCALE_UTIL = (0.25, 0.5, 1.0, 2.0, 4.0)


def _as_groups64(x: torch.Tensor) -> torch.Tensor:
    """Flatten last dim into [N, 64] groups."""
    x = x.float()
    hidden = x.shape[-1]
    usable = hidden - (hidden % 64)
    if usable < 64:
        raise ValueError("need last dim >= 64 and divisible after trim")
    flat = x[..., :usable].reshape(-1, 64)
    return flat


def _restore_rms(g: torch.Tensor, rms0: torch.Tensor) -> torch.Tensor:
    rms1 = g.pow(2).mean(dim=-1).sqrt().clamp_min(1e-12)
    return g * (rms0 / rms1).unsqueeze(-1)


def intervene_single_outlier(group: torch.Tensor, factor: float) -> torch.Tensor:
    """group [64]; amplify abs-argmax element by factor, restore RMS."""
    g = group.float().clone()
    rms0 = g.pow(2).mean().sqrt().clamp_min(1e-12)
    idx = int(g.abs().argmax().item())
    g[idx] = g[idx] * factor
    return _restore_rms(g.unsqueeze(0), rms0.unsqueeze(0)).squeeze(0)


def intervene_double_outlier(group: torch.Tensor, factor: float) -> torch.Tensor:
    g = group.float().clone()
    rms0 = g.pow(2).mean().sqrt().clamp_min(1e-12)
    abs_g = g.abs()
    top2 = torch.topk(abs_g, k=min(2, g.numel())).indices
    g[top2] = g[top2] * factor
    return _restore_rms(g.unsqueeze(0), rms0.unsqueeze(0)).squeeze(0)


def intervene_max_over_rms(group: torch.Tensor, target_ratio: float) -> torch.Tensor:
    """Keep RMS, scale so max/RMS ≈ target_ratio."""
    g = group.float().clone()
    rms0 = g.pow(2).mean().sqrt().clamp_min(1e-12)
    idx = int(g.abs().argmax().item())
    # set peak to target_ratio * rms, keep sign
    sign = torch.sign(g[idx]).clamp_min(1.0) if g[idx] == 0 else torch.sign(g[idx])
    g[idx] = sign * target_ratio * rms0
    return _restore_rms(g.unsqueeze(0), rms0.unsqueeze(0)).squeeze(0)


def intervene_kurtosis(group: torch.Tensor, spike: float) -> torch.Tensor:
    """Raise kurtosis by concentrating energy on one element (RMS fixed)."""
    g = group.float().clone()
    rms0 = g.pow(2).mean().sqrt().clamp_min(1e-12)
    g2 = g * (1.0 / max(spike, 1e-6))
    idx = int(g.abs().argmax().item())
    energy_rest = g2.pow(2).sum() - g2[idx].pow(2)
    target_energy = 64.0 * rms0.pow(2)
    peak_sq = (target_energy - energy_rest).clamp_min(0)
    sign = torch.sign(g[idx])
    if float(sign.item()) == 0.0:
        sign = torch.tensor(1.0, device=g.device)
    g2[idx] = sign * peak_sq.sqrt()
    return _restore_rms(g2.unsqueeze(0), rms0.unsqueeze(0)).squeeze(0)


def intervene_equalize_4elem(group: torch.Tensor) -> torch.Tensor:
    """Equalize 16 chunks of 4 by RMS within 64-group; restore total RMS."""
    g = group.float().reshape(16, 4)
    rms0 = group.float().pow(2).mean().sqrt().clamp_min(1e-12)
    chunk_rms = g.pow(2).mean(dim=-1).sqrt().clamp_min(1e-12)
    gm = torch.exp(chunk_rms.log().mean())
    g2 = g * (gm / chunk_rms).unsqueeze(-1)
    return _restore_rms(g2.reshape(1, 64), rms0.unsqueeze(0)).squeeze(0)


def intervene_permute(group: torch.Tensor, seed: int) -> torch.Tensor:
    g = group.float().reshape(16, 4)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    perm = torch.randperm(16, generator=gen)
    return g[perm].reshape(64).to(group.device)


def format_errors(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, float]:
    """x: arbitrary shape ending with K%64==0."""
    xb = x.to(torch.bfloat16)
    a_n = quantize_nvfp4_activation(xb, scale).dequantized.float()
    a_h = quantize_hif4_tensor(x.float(), output_dtype=torch.float32).metadata["values_fp32"].float()
    xref = x.float()
    return {
        "nmse_nvfp4": compute_pair_metrics(xref, a_n)["nmse"],
        "nmse_hif4": compute_pair_metrics(xref, a_h)["nmse"],
        "nmse_h_vs_n": compute_pair_metrics(a_n, a_h)["nmse"],
        "error_energy_h_vs_n": float(((a_h - a_n) ** 2).sum().item()),
    }


def run_a5_on_groups(
    groups: torch.Tensor,
    scale: torch.Tensor,
    *,
    seed: int = 20260810,
    max_groups: int = 256,
) -> list[dict[str, Any]]:
    """groups: [N,64] real activation groups."""
    if groups.ndim != 2 or groups.shape[1] != 64:
        raise ValueError(f"expected [N,64], got {tuple(groups.shape)}")
    n = min(groups.shape[0], max_groups)
    if n < groups.shape[0]:
        idx = torch.linspace(0, groups.shape[0] - 1, n).round().long()
        groups = groups[idx]
    else:
        idx = torch.arange(n)

    rows: list[dict[str, Any]] = []
    scale = scale.reshape(()).to(dtype=torch.float32)

    for i in range(groups.shape[0]):
        g0 = groups[i]
        base = format_errors(g0.view(1, 64), scale)
        base_row = {
            "group_index": int(idx[i].item()) if torch.is_tensor(idx[i]) else int(idx[i]),
            "intervention": "baseline",
            "setting": "original",
            **base,
        }
        rows.append(base_row)

        for f in OUTLIER_SWEEP:
            g = intervene_single_outlier(g0, f)
            m = format_errors(g.view(1, 64), scale)
            rows.append(
                {
                    "group_index": base_row["group_index"],
                    "intervention": "single_outlier",
                    "setting": f"factor={f}",
                    **m,
                }
            )
            g2 = intervene_double_outlier(g0, f)
            m2 = format_errors(g2.view(1, 64), scale)
            rows.append(
                {
                    "group_index": base_row["group_index"],
                    "intervention": "double_outlier",
                    "setting": f"factor={f}",
                    **m2,
                }
            )

        for ratio in (2.0, 4.0, 8.0, 16.0):
            g = intervene_max_over_rms(g0, ratio)
            m = format_errors(g.view(1, 64), scale)
            rows.append(
                {
                    "group_index": base_row["group_index"],
                    "intervention": "max_over_rms",
                    "setting": f"ratio={ratio}",
                    **m,
                }
            )

        for spike in (2.0, 4.0, 8.0):
            g = intervene_kurtosis(g0, spike)
            m = format_errors(g.view(1, 64), scale)
            rows.append(
                {
                    "group_index": base_row["group_index"],
                    "intervention": "kurtosis",
                    "setting": f"spike={spike}",
                    **m,
                }
            )

        g = intervene_equalize_4elem(g0)
        m = format_errors(g.view(1, 64), scale)
        rows.append(
            {
                "group_index": base_row["group_index"],
                "intervention": "equalize_4elem",
                "setting": "equalized",
                **m,
            }
        )

        g = intervene_permute(g0, seed=seed + base_row["group_index"])
        m = format_errors(g.view(1, 64), scale)
        rows.append(
            {
                "group_index": base_row["group_index"],
                "intervention": "permute_4elem",
                "setting": "permuted",
                **m,
            }
        )

        # sidecar global-scale utilization: scale *= u, keep group shape
        for u in SCALE_UTIL:
            m = format_errors(g0.view(1, 64), scale * u)
            rows.append(
                {
                    "group_index": base_row["group_index"],
                    "intervention": "sidecar_scale_util",
                    "setting": f"u={u}",
                    **m,
                }
            )
    return rows
