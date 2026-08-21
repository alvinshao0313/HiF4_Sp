"""W4 runner: 16→64 dispersion causal interventions on representative weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.synthetic_mechanisms import (
    run_w4_interventions_on_groups,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.weight_conversion import (
    iter_linear_weight_names,
    split_64_to_sub16_along_k,
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
    spearman_with_bootstrap,
)


def run_w4_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: str = "cuda:0",
    shard_id: int = 0,
    num_shards: int = 1,
    max_groups_per_tensor: int = 1024,
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

    all_rows: list[dict[str, Any]] = []
    for tensor_name, layer_idx, projection in names:
        view = load_nvfp4_qat_dequant_weight(checkpoint, tensor_name, device=device_t)
        w = view.dequantized.to(torch.float32)
        # [out, n64, 4, 16] -> [N, 4, 16]
        g = split_64_to_sub16_along_k(w.cpu()).reshape(-1, 4, 16)
        recs = run_w4_interventions_on_groups(
            g,
            max_groups=max_groups_per_tensor,
            seed=20260810 + layer_idx,
            device=device_t,
        )
        for r in recs:
            r.update(
                {
                    "tensor_name": tensor_name,
                    "layer_idx": layer_idx,
                    "projection": projection,
                    "evidence_class": "controlled_causal_evidence",
                    "hypothesis_id": "H1",
                }
            )
            all_rows.append(r)
        print(
            f"[W4] shard{shard_id} L{layer_idx}.{projection} groups={len(recs)}",
            flush=True,
        )

    write_csv(out_dir / f"w4_interventions_shard{shard_id}.csv", all_rows)

    # Summaries
    def _mean(rows: list[dict], key: str) -> float:
        xs = [float(r[key]) for r in rows if key in r and r[key] != ""]
        return sum(xs) / len(xs) if xs else 0.0

    group_size = [r for r in all_rows if r["intervention"] == "group_size"]
    by_setting: dict[str, list[float]] = {"group16": [], "group32": [], "group64": []}
    for r in group_size:
        by_setting.setdefault(r["setting"], []).append(float(r["output_error_energy"]))

    eq = [r for r in all_rows if r["intervention"] == "equalize_sub16_rms"]
    eq_rec = [float(r["recoverable_vs_original"]) for r in eq if "recoverable_vs_original" in r]

    dose = [r for r in all_rows if r["intervention"] == "dispersion_dose"]
    # Spearman dose vs output error (cluster by group_index within tensor)
    if dose:
        # Integer cluster ids: hash tensor+group into stable ints
        cluster_ids = [
            hash((r["tensor_name"], int(r["group_index"]))) % (10**9) for r in dose
        ]
        corr = spearman_with_bootstrap(
            [float(r["dose"]) for r in dose],
            [float(r["output_error_energy"]) for r in dose],
            seed=20260810,
            repeats=500,
            cluster_ids=cluster_ids,
        )
    else:
        corr = {"estimate": 0.0}

    summary = {
        "shard_id": shard_id,
        "num_rows": len(all_rows),
        "group_size_mean_output_error": {
            k: (sum(v) / len(v) if v else 0.0) for k, v in by_setting.items()
        },
        "equalize_mean_recoverable": sum(eq_rec) / len(eq_rec) if eq_rec else 0.0,
        "dispersion_dose_spearman_vs_output_error": corr,
        "h1_status": "partial_causal_evidence",
        "notes": (
            "H1 upgrade to causal_supported requires W2 observational + group16/32/64 "
            "+ equalization/dose + S1 synthetic together"
        ),
    }
    atomic_write_json(out_dir / f"w4_summary_shard{shard_id}.json", summary)
    return summary
