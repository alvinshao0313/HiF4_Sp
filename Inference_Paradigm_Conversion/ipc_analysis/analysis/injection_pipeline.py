"""Network injection experiment runner (N1/N2/N4/N5/N6 + N7 oracle repair)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.analysis.network_injection import (
    MaskSpec,
    logits_distance,
    oracle_repair_groups,
    prefix_suffix_boundaries,
    shapley_format_shift,
    sub16_dispersion_risk,
    with_conversion_mask,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.weight_conversion import convert_weight_w0
from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
    load_source_model_for_capture,
)
from Inference_Paradigm_Conversion.ipc_analysis.capture.prompts import (
    discovery_items,
    validation_items,
)
from Inference_Paradigm_Conversion.ipc_analysis.config import (
    LINEAR_PROJECTIONS,
    resolve_representative_layers,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import (
    load_nvfp4_qat_dequant_weight,
)
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (
    atomic_write_json,
    ensure_dir,
    write_csv,
)


@torch.no_grad()
def _forward_logits(model, batch) -> torch.Tensor:
    out = model(**batch)
    return out.logits


def run_injection_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: str = "cuda:0",
    shard_id: int = 0,
    num_shards: int = 1,
    max_seq_len: int = 128,
    mode: str = "n1_n2",  # n1_n2 | prefix_suffix | oracle
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    torch.set_num_threads(4)

    with (checkpoint / "config.json").open("r", encoding="utf-8") as f:
        num_layers = int(json.load(f)["num_hidden_layers"])
    rep = resolve_representative_layers(num_layers)
    prompts = discovery_items(8)  # small teacher-forced set
    # shard prompts
    prompts = [p for i, p in enumerate(prompts) if i % num_shards == shard_id]

    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    rows: list[dict[str, Any]] = []

    def encode(text: str):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        return {k: v.to(device_t) for k, v in enc.items()}

    if mode == "n1_n2":
        # N1: single linear on rep layers; N2: single layer
        jobs: list[MaskSpec] = []
        for li in rep:
            jobs.append(MaskSpec(kind="single_layer", layer_idx=li))
            for proj in LINEAR_PROJECTIONS:
                jobs.append(MaskSpec(kind="single_linear", layer_idx=li, projection=proj))
        jobs = [j for i, j in enumerate(jobs) if i % 1 == 0]  # all on this shard's prompts

        for item in prompts:
            batch = encode(item.text)
            base = _forward_logits(model, batch)
            for spec in jobs:
                with with_conversion_mask(model, spec, path_id="P1_semantic") as converted:
                    y = _forward_logits(model, batch)
                dist = logits_distance(base, y)
                rows.append(
                    {
                        "sample_id": item.sample_id,
                        "mask_kind": spec.kind,
                        "layer_idx": spec.layer_idx if spec.layer_idx is not None else -1,
                        "projection": spec.projection or "",
                        "prefix_k": -1,
                        "num_converted": len(converted),
                        "path_id": "P1_semantic",
                        **dist,
                    }
                )
            print(f"[N] shard{shard_id} {item.sample_id} n1n2 done", flush=True)

    elif mode == "prefix_suffix":
        bounds = prefix_suffix_boundaries(num_layers)
        # shard bounds
        bounds_p = [b for i, b in enumerate(bounds) if i % num_shards == shard_id]
        for item in prompts[:2]:  # keep cheap
            batch = encode(item.text)
            base = _forward_logits(model, batch)
            # N4 prefix
            for k in bounds_p:
                spec = MaskSpec(kind="prefix_layers", prefix_k=k)
                with with_conversion_mask(model, spec, path_id="P1_semantic") as converted:
                    y = _forward_logits(model, batch)
                dist = logits_distance(base, y)
                rows.append(
                    {
                        "sample_id": item.sample_id,
                        "mask_kind": "prefix_layers",
                        "layer_idx": -1,
                        "projection": "",
                        "prefix_k": k,
                        "num_converted": len(converted),
                        "path_id": "P1_semantic",
                        **dist,
                    }
                )
            # N5 suffix: convert layers >= L-k
            for k in bounds_p:
                start = num_layers - k
                spec = MaskSpec(kind="suffix_layers", suffix_k=k, layer_idx=start)
                with with_conversion_mask(model, spec, path_id="P1_semantic") as converted:
                    y = _forward_logits(model, batch)
                dist = logits_distance(base, y)
                rows.append(
                    {
                        "sample_id": item.sample_id,
                        "mask_kind": "suffix_layers",
                        "layer_idx": start,
                        "projection": "",
                        "prefix_k": k,  # reuse field as converted count from end
                        "num_converted": len(converted),
                        "path_id": "P1_semantic",
                        **dist,
                    }
                )
            print(f"[N] shard{shard_id} {item.sample_id} prefix/suffix done", flush=True)

    elif mode == "oracle":
        # N7: discovery scores → validation repair on one rep weight
        li = rep[1]  # middle
        proj = "down_proj"
        tname = f"model.layers.{li}.mlp.{proj}.weight"
        w_view = load_nvfp4_qat_dequant_weight(checkpoint, tname, device=device_t)
        conv = convert_weight_w0(w_view.dequantized, device=device_t)
        w_n = conv["W_N"].to(device_t)  # type: ignore
        w_h = conv["W_H_FP32"].to(device_t)  # type: ignore
        risk = sub16_dispersion_risk(w_n)
        # discovery: only to freeze ranking (risk already from weights, no data leak)
        val = [p for i, p in enumerate(validation_items(8)) if i % num_shards == shard_id]
        module_name = f"model.layers.{li}.mlp.{proj}"
        linear = dict(model.named_modules())[module_name]
        assert isinstance(linear, torch.nn.Linear)
        backup = linear.weight.detach().cpu().clone()

        for frac in (0.01, 0.05, 0.10):
            repaired = oracle_repair_groups(w_n, w_h, risk, top_frac=frac, mode="restore_source")
            random_r = oracle_repair_groups(w_n, w_h, risk, top_frac=frac, mode="random")
            for item in val:
                batch = encode(item.text)
                with torch.no_grad():
                    linear.weight.copy_(backup.to(device_t))
                    y_src = _forward_logits(model, batch)
                    linear.weight.copy_(w_h.to(dtype=linear.weight.dtype))
                    y_full = _forward_logits(model, batch)
                    linear.weight.copy_(repaired.to(dtype=linear.weight.dtype))
                    y_rep = _forward_logits(model, batch)
                    linear.weight.copy_(random_r.to(dtype=linear.weight.dtype))
                    y_rand = _forward_logits(model, batch)
                    linear.weight.copy_(backup.to(device_t))
                d_full = logits_distance(y_src, y_full)
                d_rep = logits_distance(y_src, y_rep)
                d_rand = logits_distance(y_src, y_rand)
                rows.append(
                    {
                        "sample_id": item.sample_id,
                        "mask_kind": "oracle_repair",
                        "layer_idx": li,
                        "projection": proj,
                        "top_frac": frac,
                        "path_id": "P1_semantic",
                        "kl_full": d_full["kl_last"],
                        "kl_repaired": d_rep["kl_last"],
                        "kl_random": d_rand["kl_last"],
                        "recoverable_kl": (
                            1.0 - d_rep["kl_last"] / d_full["kl_last"]
                            if d_full["kl_last"] > 0
                            else 0.0
                        ),
                        "random_recoverable_kl": (
                            1.0 - d_rand["kl_last"] / d_full["kl_last"]
                            if d_full["kl_last"] > 0
                            else 0.0
                        ),
                    }
                )
        print(f"[N7] shard{shard_id} oracle repair done", flush=True)
    else:
        raise ValueError(mode)

    write_csv(out_dir / f"injection_{mode}_shard{shard_id}.csv", rows)
    summary = {"shard_id": shard_id, "mode": mode, "num_rows": len(rows)}
    atomic_write_json(out_dir / f"injection_{mode}_summary_shard{shard_id}.json", summary)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def run_shapley_linear_demo(
    a_s: torch.Tensor,
    a_t: torch.Tensor,
    w_n: torch.Tensor,
    w_h: torch.Tensor,
) -> dict[str, Any]:
    """Unit-level N0 demo for Linear."""
    f_s = lambda x: F.linear(x, w_n)  # noqa: E731
    f_t = lambda x: F.linear(x, w_h)  # noqa: E731
    return shapley_format_shift(f_s, f_t, a_s, a_t)
