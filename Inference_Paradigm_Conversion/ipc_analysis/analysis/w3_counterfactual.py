"""W3: HiF4 counterfactual variants — recoverable error under idealization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.analysis.weight_conversion import (
    iter_linear_weight_names,
)
from Inference_Paradigm_Conversion.ipc_analysis.config import resolve_representative_layers
from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import (
    load_nvfp4_qat_dequant_weight,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import (
    VARIANT_CONFIGS,
    quantize_hif4_tensor,
)
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (
    atomic_write_json,
    ensure_dir,
    write_csv,
)
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics

# Variants that are legal HiF4 or explicit probes.
W3_VARIANTS = [
    "full",
    "continuous_s0",
    "bf16_s0_no_e6m2",
    "continuous_payload_clipped",
    "rounded_payload_no_clip_probe",
    "no_exp8",
    "no_exp4",
    "no_exp8_exp4",
    "group16_full_hierarchy",
    "group32_full_hierarchy",
    "group64_full_hierarchy",
]


def recoverable_fraction(e_full: float, e_cf: float) -> float:
    """R_cf = 1 - E_cf / E_full ; clamp for numerical noise."""
    if e_full <= 0:
        return 0.0 if e_cf <= 0 else float("-inf")
    return 1.0 - (e_cf / e_full)


@torch.no_grad()
def evaluate_variants_on_weight(
    w_n: torch.Tensor,
    *,
    activation: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    variants: list[str] | None = None,
) -> dict[str, Any]:
    """Compare counterfactuals in weight space and optional output space."""
    variants = variants or W3_VARIANTS
    w_n = w_n.to(device=device, dtype=torch.float32)
    full_view = quantize_hif4_tensor(w_n, variant="full", output_dtype=torch.float32)
    w_full = full_view.metadata["values_fp32"].to(torch.float32)
    e_full_w = float(((w_full - w_n) ** 2).sum().item())
    y_n = None
    e_full_y = None
    if activation is not None:
        a = activation.to(device=device, dtype=torch.float32)
        y_n = F.linear(a, w_n)
        y_full = F.linear(a, w_full)
        e_full_y = float(((y_full - y_n) ** 2).sum().item())

    rows = []
    for variant in variants:
        view = quantize_hif4_tensor(w_n, variant=variant, output_dtype=torch.float32)
        w_cf = view.metadata["values_fp32"].to(torch.float32)
        m_w = compute_pair_metrics(w_n.cpu(), w_cf.cpu())
        e_w = float(m_w["error_energy"])
        row: dict[str, Any] = {
            "variant": variant,
            "weight_nmse": m_w["nmse"],
            "weight_error_energy": e_w,
            "R_cf_weight": recoverable_fraction(e_full_w, e_w),
            "scale_mode": view.metadata["scale_mode"],
            "payload_format": view.metadata["payload_format"],
            "group_size": view.metadata["group_size"],
            "legal_hif4": variant != "rounded_payload_no_clip_probe",
        }
        if activation is not None and y_n is not None and e_full_y is not None:
            y_cf = F.linear(a, w_cf)
            m_y = compute_pair_metrics(y_n.cpu(), y_cf.cpu())
            e_y = float(m_y["error_energy"])
            row["output_nmse"] = m_y["nmse"]
            row["output_error_energy"] = e_y
            row["R_cf_output"] = recoverable_fraction(e_full_y, e_y)
        rows.append(row)
    return {
        "e_full_weight": e_full_w,
        "e_full_output": e_full_y,
        "variants": rows,
    }


def run_w3_representative(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: str = "cuda:0",
    shard_id: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    import json

    out_dir = ensure_dir(out_dir)
    with (checkpoint / "config.json").open("r", encoding="utf-8") as f:
        num_layers = int(json.load(f)["num_hidden_layers"])
    layers = resolve_representative_layers(num_layers)
    names = list(iter_linear_weight_names(checkpoint, layer_indices=layers))
    names = [t for i, t in enumerate(names) if i % num_shards == shard_id]

    all_rows: list[dict[str, Any]] = []
    for tensor_name, layer_idx, projection in names:
        view = load_nvfp4_qat_dequant_weight(checkpoint, tensor_name, device=device)
        # Synthetic activation for output-space ranking (unit-ish); real A from AL run
        # is preferred when available — here use Gaussian with seed for ranking.
        torch.manual_seed(20260810 + layer_idx)
        a = torch.randn(64, view.dequantized.shape[1], device=device, dtype=torch.float32)
        result = evaluate_variants_on_weight(
            view.dequantized, activation=a, device=device
        )
        for row in result["variants"]:
            all_rows.append(
                {
                    "tensor_name": tensor_name,
                    "layer_idx": layer_idx,
                    "projection": projection,
                    **row,
                }
            )
        print(f"[W3] shard{shard_id} L{layer_idx}.{projection} done", flush=True)

    write_csv(out_dir / f"w3_variants_shard{shard_id}.csv", all_rows)
    # Rank by mean R_cf_output across tensors (higher = more recoverable by idealizing)
    from collections import defaultdict

    by_v: dict[str, list[float]] = defaultdict(list)
    for r in all_rows:
        if "R_cf_output" in r:
            by_v[r["variant"]].append(float(r["R_cf_output"]))
    ranking = sorted(
        (
            {
                "variant": v,
                "mean_R_cf_output": sum(xs) / len(xs),
                "n": len(xs),
            }
            for v, xs in by_v.items()
        ),
        key=lambda d: d["mean_R_cf_output"],
        reverse=True,
    )
    summary = {
        "shard_id": shard_id,
        "num_tensors": len(names),
        "ranking_by_mean_R_cf_output": ranking,
        "note": (
            "R_cf is recoverable error under idealization; "
            "do not interpret differences of two NMSE as independent shares"
        ),
        "evidence_class": "controlled_causal_evidence",
    }
    atomic_write_json(out_dir / f"w3_summary_shard{shard_id}.json", summary)
    return summary
