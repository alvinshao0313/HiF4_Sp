"""Representative-layer MLP frozen-input propagation (M1–M4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from Inference_Paradigm_Conversion.ipc_analysis.analysis.mlp_propagation import mlp_stage_metrics
from Inference_Paradigm_Conversion.ipc_analysis.analysis.weight_conversion import convert_weight_w0
from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
    load_source_model_for_capture,
)
from Inference_Paradigm_Conversion.ipc_analysis.capture.prompts import discovery_items
from Inference_Paradigm_Conversion.ipc_analysis.config import resolve_representative_layers
from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import (
    load_nvfp4_qat_dequant_weight,
)
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (
    atomic_write_json,
    ensure_dir,
    write_csv,
)


@torch.no_grad()
def run_mlp_repr_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: str = "cuda:0",
    shard_id: int = 0,
    num_shards: int = 1,
    samples_per_family: int = 8,
    max_seq_len: int = 128,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    torch.set_num_threads(4)

    with (checkpoint / "config.json").open("r", encoding="utf-8") as f:
        num_layers = int(json.load(f)["num_hidden_layers"])
    rep = resolve_representative_layers(num_layers)
    prompts = [p for i, p in enumerate(discovery_items(samples_per_family)) if i % num_shards == shard_id]

    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    weight_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def get_wh(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if name not in weight_cache:
            w = load_nvfp4_qat_dequant_weight(checkpoint, name, device=device_t).dequantized
            conv = convert_weight_w0(w, device=device_t)
            weight_cache[name] = (conv["W_N"].to(device_t), conv["W_H_FP32"].to(device_t))  # type: ignore
        return weight_cache[name]

    rows: list[dict[str, Any]] = []
    for item in prompts:
        enc = tok(item.text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        batch = {k: v.to(device_t) for k, v in enc.items()}
        # Capture mlp inputs on rep layers via hooks
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx in rep:
            mod = model.model.layers[layer_idx].mlp

            def make_hook(li=layer_idx):
                def hook(_m, inputs):
                    x = inputs[0].detach()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    # keep up to 64 tokens
                    captured[li] = x[:64].to(torch.bfloat16)

                return hook

            handles.append(mod.register_forward_pre_hook(make_hook()))
        try:
            model(**batch)
        finally:
            for h in handles:
                h.remove()

        for layer_idx, x in captured.items():
            prefix = f"model.layers.{layer_idx}.mlp"
            wg_n, wg_h = get_wh(f"{prefix}.gate_proj.weight")
            wu_n, wu_h = get_wh(f"{prefix}.up_proj.weight")
            wd_n, wd_h = get_wh(f"{prefix}.down_proj.weight")
            m = mlp_stage_metrics(x.float(), wg_n, wg_h, wu_n, wu_h, wd_n, wd_h)
            row = {
                "sample_id": item.sample_id,
                "prompt_family": item.family,
                "layer_idx": layer_idx,
                "phase": "prefill",
                "product_cross_share": m["product_decomposition"]["cross_share"],
                "product_residual_rel": m["product_decomposition"]["residual_rel"],
                "energy_dg_un": m["product_decomposition"]["energy_dg_un"],
                "energy_gn_du": m["product_decomposition"]["energy_gn_du"],
                "energy_dg_du": m["product_decomposition"]["energy_dg_du"],
                "nmse_product": m["stage_metrics"]["product"]["nmse"],
                "nmse_down": m["stage_metrics"]["down_proj_out"]["nmse"],
                "gain_product_to_down": m["gains"]["product->down_proj_out"]["gain"],
                "gain_status": m["gains"]["product->down_proj_out"]["gain_status"],
                "hypothesis_id": "H5-MLP",
                "evidence_class": "controlled_causal_evidence",
            }
            rows.append(row)
        print(f"[M] shard{shard_id} {item.sample_id} layers={list(captured)}", flush=True)

    write_csv(out_dir / f"mlp_propagation_shard{shard_id}.csv", rows)
    summary = {
        "shard_id": shard_id,
        "num_rows": len(rows),
        "mean_product_cross_share": (
            sum(r["product_cross_share"] for r in rows) / len(rows) if rows else 0.0
        ),
        "mean_nmse_product": (
            sum(r["nmse_product"] for r in rows) / len(rows) if rows else 0.0
        ),
    }
    atomic_write_json(out_dir / f"mlp_summary_shard{shard_id}.json", summary)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return summary
