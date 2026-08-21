"""A1–A5: same-source activation three-format comparison and NVFP4→HiF4 delta."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.mxfp8 import quantize_mxfp8_activation
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (
    GzipJsonlWriter,
    atomic_write_json,
    ensure_dir,
    write_csv,
)
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics


def fair_triple_qdq(
    x_bf16: torch.Tensor,
    input_global_scale: torch.Tensor,
) -> dict[str, Any]:
    """A1: MXFP8 / NVFP4 / HiF4 QDQ on identical BF16 pre-quant activation."""
    if x_bf16.dtype != torch.bfloat16:
        raise TypeError("pre-quant activation must be bfloat16")
    a_m = quantize_mxfp8_activation(x_bf16, output_dtype=torch.bfloat16)
    a_n = quantize_nvfp4_activation(
        x_bf16, input_global_scale, output_dtype=torch.bfloat16
    )
    a_h = quantize_hif4_tensor(
        x_bf16.float(), group_dim=-1, output_dtype=torch.bfloat16
    )
    # Compare each format against the shared BF16 pre-quant reference X
    m_vs_x = compute_pair_metrics(x_bf16.float(), a_m.dequantized.float())
    n_vs_x = compute_pair_metrics(x_bf16.float(), a_n.dequantized.float())
    h_vs_x = compute_pair_metrics(x_bf16.float(), a_h.dequantized.float())
    h_vs_n = compute_pair_metrics(a_n.dequantized.float(), a_h.dequantized.float())
    return {
        "A_M": a_m,
        "A_N": a_n,
        "A_H": a_h,
        "metrics": {
            "MXFP8_vs_X": m_vs_x,
            "NVFP4_vs_X": n_vs_x,
            "HiF4_vs_X": h_vs_x,
            "HiF4_vs_NVFP4": h_vs_n,
        },
    }


def activation_output_delta_on_weight(
    x_bf16: torch.Tensor,
    w_n: torch.Tensor,
    input_global_scale: torch.Tensor,
) -> dict[str, Any]:
    """A3: e_A = (A_H - A_N) @ W_N.T  (same as F.linear direction)."""
    triple = fair_triple_qdq(x_bf16, input_global_scale)
    a_n = triple["A_N"].dequantized.float()
    a_h = triple["A_H"].dequantized.float()
    delta_a = a_h - a_n
    w = w_n.float()
    e_a = F.linear(delta_a, w)
    y_n = F.linear(a_n, w)
    y_h = F.linear(a_h, w)
    return {
        "delta_a_metrics": compute_pair_metrics(a_n, a_h),
        "output_delta_metrics": compute_pair_metrics(y_n, y_h),
        "e_a_energy": float((e_a * e_a).sum().item()),
        "y_n_energy": float((y_n * y_n).sum().item()),
    }


def mxfp8_modulation_of_weight_error(
    x_bf16: torch.Tensor,
    w_n: torch.Tensor,
    w_h: torch.Tensor,
    input_global_scale: torch.Tensor,
) -> dict[str, float]:
    """A4: same ΔW under A_M / A_N / X."""
    triple = fair_triple_qdq(x_bf16, input_global_scale)
    delta_w = w_h.float() - w_n.float()
    a_m = triple["A_M"].dequantized.float()
    a_n = triple["A_N"].dequantized.float()
    x = x_bf16.float()
    e_m = F.linear(a_m, delta_w)
    e_n = F.linear(a_n, delta_w)
    e_x = F.linear(x, delta_w)
    return {
        "energy_AM_dW": float((e_m * e_m).sum().item()),
        "energy_AN_dW": float((e_n * e_n).sum().item()),
        "energy_X_dW": float((e_x * e_x).sum().item()),
    }


def analyze_activation_batch(
    records: list[dict[str, Any]],
    out_dir: Path,
    *,
    shard_id: int = 0,
) -> dict[str, Any]:
    """records: dicts with x_bf16, scale, optional w_n/w_h, layer/module/phase meta."""
    out_dir = ensure_dir(out_dir)
    rows = []
    group_path = out_dir / f"activation_group_records_shard{shard_id}.jsonl.gz"
    with GzipJsonlWriter(group_path) as writer:
        for rec in records:
            x = rec["x_bf16"]
            scale = rec["input_global_scale"]
            triple = fair_triple_qdq(x, scale)
            metrics = triple["metrics"]
            row = {
                "sample_id": rec.get("sample_id", ""),
                "layer_idx": rec.get("layer_idx", -1),
                "module_name": rec.get("module_name", ""),
                "projection": rec.get("projection", ""),
                "phase": rec.get("phase", ""),
                "prompt_family": rec.get("prompt_family", ""),
                "nmse_mxfp8": metrics["MXFP8_vs_X"]["nmse"],
                "nmse_nvfp4": metrics["NVFP4_vs_X"]["nmse"],
                "nmse_hif4": metrics["HiF4_vs_X"]["nmse"],
                "nmse_hif4_vs_nvfp4": metrics["HiF4_vs_NVFP4"]["nmse"],
                "path_id": "P2_matched_semantic",
            }
            if "w_n" in rec:
                od = activation_output_delta_on_weight(x, rec["w_n"], scale)
                row["output_delta_nmse"] = od["output_delta_metrics"]["nmse"]
                if "w_h" in rec:
                    mod = mxfp8_modulation_of_weight_error(
                        x, rec["w_n"], rec["w_h"], scale
                    )
                    row.update(mod)
            rows.append(row)
            writer.write(row)
    write_csv(out_dir / f"activation_tensor_summary_shard{shard_id}.csv", rows)
    summary = {
        "shard_id": shard_id,
        "num_records": len(rows),
        "mean_nmse_hif4_vs_nvfp4": (
            sum(r["nmse_hif4_vs_nvfp4"] for r in rows) / len(rows) if rows else 0.0
        ),
        "evidence_class": "observational_correlation",
    }
    atomic_write_json(out_dir / f"activation_summary_shard{shard_id}.json", summary)
    return summary
