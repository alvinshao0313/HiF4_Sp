"""L3/L4: compare raw weight NMSE vs activation-weighted vs empirical output error."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.analysis.weight_conversion import (
    convert_weight_w0,
    iter_linear_weight_names,
)
from Inference_Paradigm_Conversion.ipc_analysis.config import resolve_representative_layers
from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import (
    load_nvfp4_qat_dequant_weight,
)
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (
    atomic_write_json,
    ensure_dir,
    write_csv,
)
from Inference_Paradigm_Conversion.ipc_analysis.metrics.statistics import (
    pearson_with_bootstrap,
    spearman_with_bootstrap,
)
from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics


def diagonal_activation_weighted_error(
    delta_w: torch.Tensor,
    a: torch.Tensor,
) -> float:
    """sum_j E[a_j^2] ||ΔW_{:,j}||^2  with E over tokens (dim 0)."""
    # a: [T, K], delta_w: [O, K]
    a = a.float()
    dw = delta_w.float()
    ea2 = (a * a).mean(dim=0)  # [K]
    col_sq = (dw * dw).sum(dim=0)  # [K]
    return float((ea2 * col_sq).sum().item())


def empirical_output_error(delta_w: torch.Tensor, a: torch.Tensor) -> float:
    y = F.linear(a.float(), delta_w.float())
    return float((y * y).sum().item())


def compute_three_predictors(
    w_n: torch.Tensor,
    w_h: torch.Tensor,
    a: torch.Tensor,
) -> dict[str, float]:
    dw = w_h.float() - w_n.float()
    raw = compute_pair_metrics(w_n.float(), w_h.float())
    return {
        "raw_weight_nmse": raw["nmse"],
        "raw_weight_error_energy": raw["error_energy"],
        "diag_act_weighted_error": diagonal_activation_weighted_error(dw, a),
        "empirical_output_error": empirical_output_error(dw, a),
    }


def run_l3_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: str = "cuda:0",
    shard_id: int = 0,
    num_shards: int = 1,
    tokens_per_tensor: int = 256,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    torch.set_num_threads(4)

    with (checkpoint / "config.json").open("r", encoding="utf-8") as f:
        num_layers = int(json.load(f)["num_hidden_layers"])
    layers = resolve_representative_layers(num_layers)
    names = list(iter_linear_weight_names(checkpoint, layer_indices=layers))
    names = [t for i, t in enumerate(names) if i % num_shards == shard_id]

    rows: list[dict[str, Any]] = []
    for tensor_name, layer_idx, projection in names:
        view = load_nvfp4_qat_dequant_weight(checkpoint, tensor_name, device=device_t)
        conv = convert_weight_w0(view.dequantized, device=device_t)
        w_n = conv["W_N"]
        w_h = conv["W_H_FP32"]
        assert isinstance(w_n, torch.Tensor) and isinstance(w_h, torch.Tensor)
        torch.manual_seed(20260810 + layer_idx * 17 + hash(projection) % 997)
        a = torch.randn(tokens_per_tensor, w_n.shape[1], device=device_t, dtype=torch.float32)
        pred = compute_three_predictors(w_n, w_h, a)
        rows.append(
            {
                "tensor_name": tensor_name,
                "layer_idx": layer_idx,
                "projection": projection,
                **pred,
                "hypothesis_id": "H4",
                "evidence_class": "observational_correlation",
            }
        )
        print(f"[L3] shard{shard_id} L{layer_idx}.{projection}", flush=True)

    write_csv(out_dir / f"l3_predictors_shard{shard_id}.csv", rows)

    # Within-shard correlations vs empirical_output_error (cluster by layer)
    y = [float(r["empirical_output_error"]) for r in rows]
    clusters = [int(r["layer_idx"]) for r in rows]
    summary = {"shard_id": shard_id, "num_tensors": len(rows), "correlations": {}}
    for name, key in [
        ("raw_weight_nmse", "raw_weight_nmse"),
        ("diag_act_weighted_error", "diag_act_weighted_error"),
        ("raw_weight_error_energy", "raw_weight_error_energy"),
    ]:
        x = [float(r[key]) for r in rows]
        summary["correlations"][name] = {
            "spearman": spearman_with_bootstrap(
                x, y, seed=20260810, repeats=500, cluster_ids=clusters
            ),
            "pearson": pearson_with_bootstrap(
                x, y, seed=20260810, repeats=500, cluster_ids=clusters
            ),
        }
    atomic_write_json(out_dir / f"l3_summary_shard{shard_id}.json", summary)
    return summary
