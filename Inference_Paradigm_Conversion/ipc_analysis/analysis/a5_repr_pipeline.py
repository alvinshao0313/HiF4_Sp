"""Capture real activation groups on representative layers; run A5 interventions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from Inference_Paradigm_Conversion.ipc_analysis.analysis.a5_interventions import (
    _as_groups64,
    run_a5_on_groups,
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
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import (
    load_nvfp4_activation_scales,
    resolve_activation_scale_path,
    resolve_nvfp4_scale_for_module,
)
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (
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
def run_a5_repr_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: str = "cuda:0",
    shard_id: int = 0,
    num_shards: int = 1,
    samples_per_family: int = 4,
    max_seq_len: int = 128,
    max_groups_per_module: int = 64,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    torch.set_num_threads(4)
    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)

    with (checkpoint / "config.json").open("r", encoding="utf-8") as f:
        num_layers = int(json.load(f)["num_hidden_layers"])
    rep = set(resolve_representative_layers(num_layers))
    prompts = [p for i, p in enumerate(discovery_items(samples_per_family)) if i % num_shards == shard_id]

    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))

    def module_filter(name: str, _mod: nn.Module) -> bool:
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        return layer_idx in rep and _proj_of(name) in LINEAR_PROJECTIONS

    rows: list[dict[str, Any]] = []
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
            max_raw_tokens_per_module=128,
        )
        for cap in caps:
            proj = _proj_of(cap.module_name)
            if proj is None:
                continue
            x = cap.extras.get("stat_sample")
            if x is None or x.numel() == 0:
                x = cap.tensor
            if x.numel() == 0:
                continue
            x = x.to(device="cpu", dtype=torch.float32)
            try:
                groups = _as_groups64(x)
            except ValueError:
                continue
            scale = resolve_nvfp4_scale_for_module(scales, cap.module_name).cpu()
            inter = run_a5_on_groups(
                groups, scale, max_groups=max_groups_per_module, seed=20260810 + cap.layer_idx
            )
            for r in inter:
                rows.append(
                    {
                        "sample_id": item.sample_id,
                        "prompt_family": item.family,
                        "layer_idx": cap.layer_idx,
                        "module_name": cap.module_name,
                        "projection": proj,
                        "phase": cap.phase,
                        "path_id": "P2_matched_semantic",
                        "hypothesis_id": "H2",
                        **r,
                    }
                )
        print(f"[A5] shard{shard_id} {item.sample_id} rows_so_far={len(rows)}", flush=True)

    write_csv(out_dir / f"a5_interventions_shard{shard_id}.csv", rows)
    summary = {"shard_id": shard_id, "num_rows": len(rows), "num_prompts": len(prompts)}
    atomic_write_json(out_dir / f"a5_summary_shard{shard_id}.json", summary)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return summary
