"""Representative-layer A2 activation counterfactual runner."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from Inference_Paradigm_Conversion.ipc_analysis.analysis.a2_counterfactual import (
    hif4_activation_counterfactuals,
    nvfp4_activation_counterfactuals,
)
from Inference_Paradigm_Conversion.ipc_analysis.capture.activation_capture import (
    capture_linear_inputs,
)
from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
    load_source_model_for_capture,
)
from Inference_Paradigm_Conversion.ipc_analysis.capture.prompts import discovery_items
from Inference_Paradigm_Conversion.ipc_analysis.config import (
    LINEAR_PROJECTIONS,
    resolve_representative_layers,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import (
    load_nvfp4_qat_dequant_weight,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import (
    load_nvfp4_activation_scales,
    resolve_activation_scale_path,
    resolve_nvfp4_scale_for_module,
)
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (
    GzipJsonlWriter,
    atomic_write_json,
    ensure_dir,
    write_csv,
)

_PROJ_RE = re.compile(
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


def _proj_of(name: str) -> str | None:
    m = _PROJ_RE.search(name)
    return m.group(1) if m else None


@torch.no_grad()
def run_a2_repr_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: str = "cuda:0",
    shard_id: int = 0,
    num_shards: int = 1,
    samples_per_family: int = 8,
    max_seq_len: int = 128,
    decode_steps: int = 4,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    torch.set_num_threads(4)
    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)

    with (checkpoint / "config.json").open("r", encoding="utf-8") as f:
        num_layers = int(json.load(f)["num_hidden_layers"])
    rep_layers = set(resolve_representative_layers(num_layers))
    prompts = [p for i, p in enumerate(discovery_items(samples_per_family)) if i % num_shards == shard_id]

    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))
    weight_cache: dict[str, torch.Tensor] = {}

    def module_filter(name: str, _mod: nn.Module) -> bool:
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_idx not in rep_layers:
            return False
        return _proj_of(name) in LINEAR_PROJECTIONS

    rows: list[dict[str, Any]] = []
    group_path = out_dir / f"activation_group_records_shard{shard_id}.jsonl.gz"

    with GzipJsonlWriter(group_path, flush_every=32) as group_writer:
        for item in prompts:
            enc = tok(item.text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            batch = {k: v.to(device_t) for k, v in enc.items()}
            if "attention_mask" not in batch:
                batch["attention_mask"] = torch.ones_like(batch["input_ids"])

            caps = capture_linear_inputs(
                model,
                batch,
                phase="prefill",
                module_filter=module_filter,
                sample_id=item.sample_id,
                max_tokens_per_module=2048,
                max_raw_tokens_per_module=64,
            )

            # short decode for phase split
            warm = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
            )
            past = warm.past_key_values
            next_id = warm.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([batch["input_ids"], next_id], dim=-1)
            attn = torch.cat(
                [
                    batch["attention_mask"],
                    torch.ones((1, 1), device=device_t, dtype=batch["attention_mask"].dtype),
                ],
                dim=-1,
            )
            caps_dec: list = []
            for step in range(decode_steps):
                step_batch = {
                    "input_ids": input_ids[:, -1:],
                    "attention_mask": attn,
                    "past_key_values": past,
                    "use_cache": True,
                }
                if step == decode_steps - 1:
                    step_caps, out = capture_linear_inputs(
                        model,
                        step_batch,
                        phase="decode",
                        module_filter=module_filter,
                        sample_id=f"{item.sample_id}_d{step}",
                        max_tokens_per_module=64,
                        max_raw_tokens_per_module=64,
                        return_outputs=True,
                    )
                    caps_dec.extend(step_caps)
                else:
                    out = model(**step_batch)
                past = out.past_key_values
                next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_id], dim=-1)
                attn = torch.cat(
                    [attn, torch.ones((1, 1), device=device_t, dtype=attn.dtype)],
                    dim=-1,
                )

            for cap in list(caps) + list(caps_dec):
                proj = _proj_of(cap.module_name)
                if proj is None:
                    continue
                x = cap.extras.get("stat_sample")
                if x is None or x.numel() == 0:
                    x = cap.tensor
                if x.numel() == 0:
                    continue
                x = x.to(device=device_t, dtype=torch.bfloat16)
                k = x.shape[-1]
                usable = k - (k % 64)
                if usable < 64:
                    continue
                # Cap tokens for A2 cost (oracle scale search is O(grid))
                if x.ndim == 2 and x.shape[0] > 32:
                    x = x[:32]
                elif x.ndim == 3 and x.shape[1] > 32:
                    x = x[:, :32]
                x = x[..., :usable]
                scale = resolve_nvfp4_scale_for_module(scales, cap.module_name).to(device_t)

                wname = cap.module_name + ".weight"
                if wname not in weight_cache:
                    w = load_nvfp4_qat_dequant_weight(checkpoint, wname, device=device_t).dequantized
                    if w.shape[1] != usable:
                        w = w[:, :usable]
                    weight_cache[wname] = w.detach()
                w_n = weight_cache[wname]

                nv = nvfp4_activation_counterfactuals(x, scale, w_n=w_n)
                hf = hif4_activation_counterfactuals(x, w_n=w_n)
                base = {
                    "sample_id": item.sample_id,
                    "prompt_family": item.family,
                    "layer_idx": cap.layer_idx,
                    "module_name": cap.module_name,
                    "projection": proj,
                    "phase": cap.phase,
                    "path_id": "P2_matched_semantic",
                    "hypothesis_id": "H2",
                }
                for block, fmt in ((nv, "nvfp4"), (hf, "hif4")):
                    for v in block["variants"]:
                        # exclude oracle boundary hits from main recoverable ranking later
                        row = {**base, "format": fmt, **v}
                        if (
                            fmt == "nvfp4"
                            and v["variant"] == "nv_oracle_global_scale"
                            and v.get("oracle_scale_search_boundary_hit")
                        ):
                            row["exclude_from_main_rcf"] = True
                        else:
                            row["exclude_from_main_rcf"] = False
                        rows.append(row)
                        group_writer.write(row)
            print(f"[A2] shard{shard_id} {item.sample_id}", flush=True)

    write_csv(out_dir / f"a2_variants_shard{shard_id}.csv", rows)
    summary = {
        "shard_id": shard_id,
        "num_rows": len(rows),
        "num_prompts": len(prompts),
        "representative_layers": sorted(rep_layers),
    }
    atomic_write_json(out_dir / f"a2_summary_shard{shard_id}.json", summary)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return summary
