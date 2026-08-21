"""W4 / S*: controlled interventions on 16→64 dispersion (mechanism probes)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics

DISPERSION_DOSES = (0.0, 0.5, 1.0, 1.5, 2.0)


def _group_rms(g: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((g * g).mean().clamp_min(0))


def equalize_sub16_rms(group64: torch.Tensor) -> torch.Tensor:
    """Intervention 2: equalize four 16-subblock RMS to geometric mean; keep total RMS.

    Preserves per-subblock unit shape/sign; mechanism probe only.
    group64: [4, 16]
    """
    if group64.shape != (4, 16):
        raise ValueError(f"expected (4,16), got {tuple(group64.shape)}")
    g = group64.to(torch.float32).clone()
    total_rms0 = _group_rms(g)
    sub_rms = torch.sqrt((g * g).mean(dim=-1).clamp_min(1e-12))  # [4]
    # geometric mean of positive RMS
    log_gm = sub_rms.log().mean()
    gm = torch.exp(log_gm)
    scales = gm / sub_rms
    g2 = g * scales.unsqueeze(-1)
    total_rms1 = _group_rms(g2)
    if total_rms1 > 0:
        g2 = g2 * (total_rms0 / total_rms1)
    return g2


def apply_dispersion_dose(group64: torch.Tensor, d: float) -> torch.Tensor:
    """Intervention 3: scale four subblocks by [2^{-d}, 2^{-d/3}, 2^{d/3}, 2^{d}].

    Keeps within-block relative shape and restores 64-group total RMS.
    """
    if group64.shape != (4, 16):
        raise ValueError(f"expected (4,16), got {tuple(group64.shape)}")
    g = group64.to(torch.float32).clone()
    total_rms0 = _group_rms(g)
    factors = torch.tensor(
        [2.0 ** (-d), 2.0 ** (-d / 3.0), 2.0 ** (d / 3.0), 2.0 ** d],
        dtype=torch.float32,
        device=g.device,
    )
    g2 = g * factors.unsqueeze(-1)
    total_rms1 = _group_rms(g2)
    if total_rms1 > 0:
        g2 = g2 * (total_rms0 / total_rms1)
    return g2


def permute_4elem_chunks(group64: torch.Tensor, seed: int) -> torch.Tensor:
    """Intervention 4: permute 4-element chunks within 64-group; keep multiset.

    Flattens to 16 chunks of 4, shuffles chunk order, reshapes back to [4,16].
    """
    flat = group64.to(torch.float32).reshape(-1)  # 64
    chunks = flat.reshape(16, 4)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    perm = torch.randperm(16, generator=g)
    return chunks[perm].reshape(4, 16).to(group64.device)


def hif4_group_error(
    group64: torch.Tensor,
    *,
    variant: str = "full",
    activation_row: torch.Tensor | None = None,
) -> dict[str, float]:
    """Quantize one flattened 64-group as [1,64] weight row slice."""
    w = group64.to(torch.float32).reshape(1, 64)
    view = quantize_hif4_tensor(w, variant=variant, output_dtype=torch.float32)
    w_h = view.metadata["values_fp32"].to(torch.float32)
    m = compute_pair_metrics(w, w_h)
    out: dict[str, float] = {
        "weight_nmse": m["nmse"],
        "weight_error_energy": m["error_energy"],
        "is_standard_hif4": 1.0 if variant in {"full", "group64_full_hierarchy"} else 0.0,
    }
    if activation_row is not None:
        a = activation_row.to(torch.float32).reshape(1, 64)
        y0 = F.linear(a, w)
        y1 = F.linear(a, w_h)
        mo = compute_pair_metrics(y0, y1)
        out["output_nmse"] = mo["nmse"]
        out["output_error_energy"] = mo["error_energy"]
    return out


def run_w4_interventions_on_groups(
    groups: torch.Tensor,
    *,
    max_groups: int = 2048,
    seed: int = 20260810,
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    """groups: [N, 4, 16] real weight groups. Returns per-group intervention records."""
    if groups.ndim != 3 or groups.shape[1:] != (4, 16):
        raise ValueError(f"expected [N,4,16], got {tuple(groups.shape)}")
    n = groups.shape[0]
    take = min(n, max_groups)
    # Deterministic subsample
    if take < n:
        idx = torch.linspace(0, n - 1, take).round().long()
        groups = groups[idx]
    else:
        idx = torch.arange(n)

    rows: list[dict[str, Any]] = []
    torch.manual_seed(seed)
    # Shared synthetic activation for output-aware ranking
    a = torch.randn(1, 64, device=device, dtype=torch.float32)

    for i in range(groups.shape[0]):
        g0 = groups[i].to(device=device, dtype=torch.float32)
        amax = g0.abs().amax(dim=-1)
        amax_pos = amax[amax > 0]
        if amax_pos.numel() == 0:
            log2_range = 0.0
        else:
            log2_range = float((amax_pos.max().log2() - amax_pos.min().log2()).item())

        base: dict[str, Any] = {
            "group_index": int(idx[i].item()) if torch.is_tensor(idx[i]) else int(idx[i]),
            "sub16_log2_amax_range": log2_range,
        }

        # Intervention 1: group size counterfactuals
        for variant, tag in [
            ("group16_full_hierarchy", "group16"),
            ("group32_full_hierarchy", "group32"),
            ("full", "group64"),
        ]:
            m = hif4_group_error(g0, variant=variant, activation_row=a)
            rows.append(
                {
                    **base,
                    "intervention": "group_size",
                    "setting": tag,
                    "is_standard_hif4": tag == "group64",
                    "output_error_energy": m.get("output_error_energy", 0.0),
                    "weight_error_energy": m["weight_error_energy"],
                }
            )

        # Intervention 2: equalization
        g_eq = equalize_sub16_rms(g0)
        m0 = hif4_group_error(g0, variant="full", activation_row=a)
        m1 = hif4_group_error(g_eq, variant="full", activation_row=a)
        rows.append(
            {
                **base,
                "intervention": "equalize_sub16_rms",
                "setting": "original",
                "is_standard_hif4": True,
                "output_error_energy": m0.get("output_error_energy", 0.0),
                "weight_error_energy": m0["weight_error_energy"],
            }
        )
        rows.append(
            {
                **base,
                "intervention": "equalize_sub16_rms",
                "setting": "equalized",
                "is_standard_hif4": True,
                "output_error_energy": m1.get("output_error_energy", 0.0),
                "weight_error_energy": m1["weight_error_energy"],
                "recoverable_vs_original": (
                    1.0
                    - m1.get("output_error_energy", 0.0)
                    / m0.get("output_error_energy", 1.0)
                    if m0.get("output_error_energy", 0.0) > 0
                    else 0.0
                ),
            }
        )

        # Intervention 3: dose sweep
        for d in DISPERSION_DOSES:
            gd = apply_dispersion_dose(g0, d)
            md = hif4_group_error(gd, variant="full", activation_row=a)
            rows.append(
                {
                    **base,
                    "intervention": "dispersion_dose",
                    "setting": f"d={d}",
                    "dose": d,
                    "is_standard_hif4": True,
                    "output_error_energy": md.get("output_error_energy", 0.0),
                    "weight_error_energy": md["weight_error_energy"],
                }
            )

        # Intervention 4: hierarchy layout permutation
        g_perm = permute_4elem_chunks(g0, seed=seed + int(base["group_index"]))
        mp = hif4_group_error(g_perm, variant="full", activation_row=a)
        rows.append(
            {
                **base,
                "intervention": "hierarchy_permute",
                "setting": "permuted_4elem_chunks",
                "is_standard_hif4": True,
                "output_error_energy": mp.get("output_error_energy", 0.0),
                "weight_error_energy": mp["weight_error_energy"],
                "original_output_error_energy": m0.get("output_error_energy", 0.0),
            }
        )
    return rows
