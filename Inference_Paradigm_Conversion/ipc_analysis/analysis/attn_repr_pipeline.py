"""Representative-layer full-attention propagation runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.attention_propagation import (
    detect_linear_attn_modules,
    full_attention_propagation,
)
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
def run_attn_repr_shard(
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
        cfgj = json.load(f)
    num_layers = int(cfgj["num_hidden_layers"])
    num_heads = int(cfgj["num_attention_heads"])
    num_kv = int(cfgj["num_key_value_heads"])
    head_dim = int(cfgj.get("head_dim", cfgj["hidden_size"] // num_heads))
    rep = resolve_representative_layers(num_layers)
    prompts = [p for i, p in enumerate(discovery_items(samples_per_family)) if i % num_shards == shard_id]

    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    linear_attn = detect_linear_attn_modules(model)
    weight_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def get_wh(name: str):
        if name not in weight_cache:
            w = load_nvfp4_qat_dequant_weight(checkpoint, name, device=device_t).dequantized
            conv = convert_weight_w0(w, device=device_t)
            weight_cache[name] = (conv["W_N"].to(device_t), conv["W_H_FP32"].to(device_t))  # type: ignore
        return weight_cache[name]

    rows: list[dict[str, Any]] = []
    for item in prompts:
        enc = tok(item.text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        batch = {k: v.to(device_t) for k, v in enc.items()}
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx in rep:
            mod = model.model.layers[layer_idx].self_attn

            def make_hook(li=layer_idx):
                def hook(_m, args, kwargs):
                    # Qwen3Attention is called with hidden_states=... (kwargs only)
                    if args:
                        x = args[0]
                    else:
                        x = kwargs["hidden_states"]
                    x = x.detach()
                    captured[li] = x[:, :64].to(torch.bfloat16)  # cap tokens

                return hook

            handles.append(mod.register_forward_pre_hook(make_hook(), with_kwargs=True))
        try:
            model(**batch)
        finally:
            for h in handles:
                h.remove()

        for layer_idx, x in captured.items():
            pref = f"model.layers.{layer_idx}.self_attn"
            wqn, wqh = get_wh(f"{pref}.q_proj.weight")
            wkn, wkh = get_wh(f"{pref}.k_proj.weight")
            wvn, wvh = get_wh(f"{pref}.v_proj.weight")
            won, woh = get_wh(f"{pref}.o_proj.weight")
            m = full_attention_propagation(
                x.float(),
                wqn, wqh, wkn, wkh, wvn, wvh, won, woh,
                num_heads=num_heads,
                num_kv_heads=num_kv,
                head_dim=head_dim,
            )
            rows.append(
                {
                    "sample_id": item.sample_id,
                    "prompt_family": item.family,
                    "layer_idx": layer_idx,
                    "attention_kind": "full_attention",
                    "nmse_q": m["stage_metrics"]["q_proj"]["nmse"],
                    "nmse_k": m["stage_metrics"]["k_proj"]["nmse"],
                    "nmse_logits": m["stage_metrics"]["attn_logits"]["nmse"],
                    "nmse_av": m["stage_metrics"]["av_output"]["nmse"],
                    "nmse_o": m["stage_metrics"]["o_proj_output"]["nmse"],
                    "nmse_residual": m["stage_metrics"]["residual_output"]["nmse"],
                    "kl_st": m["kl_js"]["kl_st"],
                    "js": m["kl_js"]["js"],
                    "top_attended_flip_rate": m["top_attended_flip_rate"],
                    "entropy_change": m["entropy_change"],
                    "logits_gain": m["logits_gain_from_qk"]["gain"],
                    "logits_gain_status": m["logits_gain_from_qk"]["gain_status"],
                    "hypothesis_id": "H6-Attention",
                    "evidence_class": "controlled_causal_evidence",
                }
            )
        print(f"[T] shard{shard_id} {item.sample_id}", flush=True)

    write_csv(out_dir / f"attention_propagation_shard{shard_id}.csv", rows)
    summary = {
        "shard_id": shard_id,
        "num_rows": len(rows),
        "linear_attn_modules": linear_attn,
        "linear_attn_present": bool(linear_attn),
        "mean_kl_st": sum(r["kl_st"] for r in rows) / len(rows) if rows else 0.0,
        "mean_flip": sum(r["top_attended_flip_rate"] for r in rows) / len(rows) if rows else 0.0,
        "mean_logits_gain": sum(r["logits_gain"] for r in rows) / len(rows) if rows else 0.0,
        "note": "full_attention only; linear_attn schema not mixed",
    }
    atomic_write_json(out_dir / f"attention_summary_shard{shard_id}.json", summary)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return summary
