"""HiF4 weight direct RTN and hierarchical scale-search variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from HiFloat4.hif4_scale_threshold_optimization.src.quantizer import quantize_hif4
from HiFloat4.hif4_scale_threshold_optimization.src.weight_search import (
    search_weight_groups,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import STANDARD_HIF4_CONFIG
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import ensure_dir
from Native_NVFP4_HiF4_Linear_Puncture.src.metrics import error_energy, nmse, recovery_ratio


@dataclass
class WeightVariant:
    variant_id: str
    reconstruction: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


def build_hif4_direct_weight(w_n: torch.Tensor) -> WeightVariant:
    result = quantize_hif4(w_n.to(torch.float32), config=STANDARD_HIF4_CONFIG)
    return WeightVariant(
        variant_id="hif4_rtn",
        reconstruction=result.reconstruction.to(torch.float32),
        metadata={
            "group_size": STANDARD_HIF4_CONFIG.group_size,
            "s0_mode": STANDARD_HIF4_CONFIG.s0_mode,
            "s0_divisor": STANDARD_HIF4_CONFIG.s0_divisor,
            "e8_threshold": STANDARD_HIF4_CONFIG.e8_threshold,
            "e4_threshold": STANDARD_HIF4_CONFIG.e4_threshold,
        },
    )


def build_hif4_greedy_weight(
    w_n: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    memory_budget_fraction: float = 0.25,
) -> WeightVariant:
    result = search_weight_groups(
        w_n,
        budget="full",
        enumerate_e8_e4=True,
        s0_mode="hardware",
        s0_divisor=7.0,
        e8_threshold=4.0,
        e4_threshold=2.0,
        memory_budget_fraction=memory_budget_fraction,
        device=device,
    )
    recon = result.reconstruction.detach().to(torch.float32).cpu()
    meta: dict[str, Any] = {}
    if hasattr(result, "s0_index"):
        s0_index = result.s0_index.detach().reshape(-1).to(torch.int64).cpu()
        fractions = [
            float((s0_index == i).to(torch.float64).mean().item()) if s0_index.numel() else float("nan")
            for i in range(5)
        ]
        meta["fraction_s0_index"] = fractions
    for key in (
        "mse",
        "nmse",
        "elapsed_s",
        "groups_per_second",
        "group_chunk_size",
        "budget",
        "peak_memory_bytes",
    ):
        if hasattr(result, key):
            meta[key] = getattr(result, key)
    return WeightVariant(variant_id="hif4_greedy", reconstruction=recon, metadata=meta)


def weight_variant_row(
    module_name: str,
    w_n: torch.Tensor,
    direct: WeightVariant,
    greedy: WeightVariant,
) -> dict[str, Any]:
    w_ref = w_n.detach().to(dtype=torch.float32, device="cpu")
    w_rtn = direct.reconstruction.detach().to(dtype=torch.float32, device="cpu")
    w_greedy = greedy.reconstruction.detach().to(dtype=torch.float32, device="cpu")
    rtn_err = error_energy(w_rtn, w_ref)
    greedy_err = error_energy(w_greedy, w_ref)
    fractions = greedy.metadata.get("fraction_s0_index", [float("nan")] * 5)
    row = {
        "module_name": module_name,
        "numel": int(w_ref.numel()),
        "rtn_weight_nmse": nmse(w_rtn, w_ref),
        "greedy_weight_nmse": nmse(w_greedy, w_ref),
        "weight_nmse_recovery": recovery_ratio(rtn_err, greedy_err),
        "greedy_elapsed_s": greedy.metadata.get("elapsed_s"),
        "groups_per_second": greedy.metadata.get("groups_per_second"),
        "group_chunk_size": greedy.metadata.get("group_chunk_size"),
    }
    for i in range(5):
        row[f"fraction_s0_index_{i}"] = fractions[i] if i < len(fractions) else float("nan")
    return row


def write_weight_variants_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)
