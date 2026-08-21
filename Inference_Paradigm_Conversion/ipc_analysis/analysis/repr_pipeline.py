"""Representative-layer capture + activation + linear decomposition pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_conversion import (
    fair_triple_qdq,
    mxfp8_modulation_of_weight_error,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.linear_decomposition import (
    decompose_linear_error,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.weight_conversion import (
    convert_weight_w0,
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
from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
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
def run_repr_al_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: str,
    shard_id: int,
    num_shards: int,
    samples_per_family: int = 32,
    max_seq_len: int = 256,
    decode_steps: int = 8,
) -> dict[str, Any]:
    """Capture discovery prompts (sharded) on representative layers; run A/L."""
    out_dir = ensure_dir(out_dir)
    # Cap intra-op threads so multi-GPU shards do not oversubscribe the host.
    torch.set_num_threads(int(__import__("os").environ.get("TORCH_NUM_THREADS", "4")))
    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)

    with (checkpoint / "config.json").open("r", encoding="utf-8") as f:
        num_layers = int(json.load(f)["num_hidden_layers"])
    rep_layers = set(resolve_representative_layers(num_layers))

    prompts = discovery_items(samples_per_family)
    prompts = [p for i, p in enumerate(prompts) if i % num_shards == shard_id]

    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))

    # Cache HiF4 converted weights for rep layers only
    weight_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def module_filter(name: str, _mod: nn.Module) -> bool:
        # name like model.layers.4.mlp.down_proj
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_idx not in rep_layers:
            return False
        return _proj_of(name) in LINEAR_PROJECTIONS

    act_rows: list[dict[str, Any]] = []
    lin_rows: list[dict[str, Any]] = []
    raw_path = out_dir / f"repr_activation_raw_shard{shard_id}.jsonl.gz"

    with GzipJsonlWriter(raw_path, flush_every=32) as raw_writer:
        for item in prompts:
            enc = tok(
                item.text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            )
            batch = {k: v.to(device_t) for k, v in enc.items()}
            if "attention_mask" not in batch:
                batch["attention_mask"] = torch.ones_like(batch["input_ids"])

            # Prefill capture
            caps = capture_linear_inputs(
                model,
                batch,
                phase="prefill",
                module_filter=module_filter,
                sample_id=item.sample_id,
                max_tokens_per_module=4096,
                max_raw_tokens_per_module=256,
            )

            # Decode: generate with KV cache; capture on each new-token forward.
            input_ids = batch["input_ids"]
            attn = batch["attention_mask"]
            # Warm-start cache from the already-captured prefill sequence.
            warm = model(
                input_ids=input_ids,
                attention_mask=attn,
                use_cache=True,
            )
            past = warm.past_key_values
            next_id = warm.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=-1)
            attn = torch.cat(
                [attn, torch.ones((attn.shape[0], 1), device=device_t, dtype=attn.dtype)],
                dim=-1,
            )
            # Generate decode_steps tokens; only capture+analyze the final decode
            # step for A/L tables (phase label remains decode). Intermediate
            # steps still advance the KV cache so the final step is a true decode.
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
                        max_tokens_per_module=128,
                        max_raw_tokens_per_module=128,
                        return_outputs=True,
                    )
                    caps_dec.extend(step_caps)
                else:
                    out = model(**step_batch)
                past = out.past_key_values
                next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_id], dim=-1)
                attn = torch.cat(
                    [attn, torch.ones((attn.shape[0], 1), device=device_t, dtype=attn.dtype)],
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
                # Ensure last dim divisible by 64 for HiF4 and 32 for MXFP8
                k = x.shape[-1]
                usable = k - (k % 64)
                if usable < 64:
                    continue
                x = x[..., :usable]
                scale = resolve_nvfp4_scale_for_module(scales, cap.module_name).to(
                    device=device_t
                )

                weight_name = cap.module_name + ".weight"
                if weight_name not in weight_cache:
                    w_view = load_nvfp4_qat_dequant_weight(
                        checkpoint, weight_name, device=device_t
                    )
                    w_n = w_view.dequantized
                    # Match truncated K if needed (should not for formal shapes)
                    if w_n.shape[1] != x.shape[-1]:
                        w_n = w_n[:, : x.shape[-1]]
                    conv = convert_weight_w0(w_n, device=device_t)
                    w_h = conv["W_H_FP32"]
                    assert isinstance(w_h, torch.Tensor)
                    weight_cache[weight_name] = (
                        w_n.detach().to(device_t),
                        w_h.detach().to(device_t),
                    )
                w_n, w_h = weight_cache[weight_name]
                if w_n.shape[1] != x.shape[-1]:
                    w_n = w_n[:, : x.shape[-1]]
                    w_h = w_h[:, : x.shape[-1]]

                triple = fair_triple_qdq(x, scale)
                a_n = triple["A_N"].dequantized.float()
                a_h = triple["A_H"].dequantized.float()
                metrics = triple["metrics"]
                mod = mxfp8_modulation_of_weight_error(x, w_n, w_h, scale)
                lin = decompose_linear_error(
                    a_n,
                    a_h,
                    w_n,
                    w_h,
                    path_id="P2_matched_semantic",
                    layer_idx=cap.layer_idx,
                    module_name=cap.module_name,
                    projection=proj,
                    phase=cap.phase,
                    sample_id=item.sample_id,
                )
                # P1 path: same ΔW under MXFP8 activations
                a_m = triple["A_M"].dequantized.float()
                lin_p1 = decompose_linear_error(
                    a_m,
                    a_m,  # activation fixed MXFP8
                    w_n,
                    w_h,
                    path_id="P1_semantic",
                    layer_idx=cap.layer_idx,
                    module_name=cap.module_name,
                    projection=proj,
                    phase=cap.phase,
                    sample_id=item.sample_id,
                )

                act_row = {
                    "sample_id": item.sample_id,
                    "prompt_family": item.family,
                    "split": item.split,
                    "layer_idx": cap.layer_idx,
                    "module_name": cap.module_name,
                    "projection": proj,
                    "phase": cap.phase,
                    "nmse_mxfp8": metrics["MXFP8_vs_X"]["nmse"],
                    "nmse_nvfp4": metrics["NVFP4_vs_X"]["nmse"],
                    "nmse_hif4": metrics["HiF4_vs_X"]["nmse"],
                    "nmse_hif4_vs_nvfp4": metrics["HiF4_vs_NVFP4"]["nmse"],
                    **mod,
                    "path_id": "P2_matched_semantic",
                }
                act_rows.append(act_row)
                lin_rows.append(lin.to_dict())
                lin_rows.append(lin_p1.to_dict())
                raw_writer.write(
                    {
                        **act_row,
                        "linear_p2_residual_rel": lin.residual_rel,
                        "linear_p2_energy_delta_w_an": lin.energy_delta_w_an,
                        "linear_p2_energy_wn_delta_a": lin.energy_wn_delta_a,
                        "linear_p2_energy_delta_w_delta_a": lin.energy_delta_w_delta_a,
                        "linear_p1_energy_delta_w_an": lin_p1.energy_delta_w_an,
                    }
                )
            print(
                f"[AL] shard{shard_id} {item.sample_id} caps_prefill={len(caps)} caps_decode={len(caps_dec)}",
                flush=True,
            )

    write_csv(out_dir / f"activation_summary_shard{shard_id}.csv", act_rows)
    write_csv(out_dir / f"linear_decomp_shard{shard_id}.csv", lin_rows)
    summary = {
        "shard_id": shard_id,
        "num_shards": num_shards,
        "num_prompts": len(prompts),
        "num_activation_rows": len(act_rows),
        "num_linear_rows": len(lin_rows),
        "representative_layers": sorted(rep_layers),
        "device": str(device_t),
    }
    atomic_write_json(out_dir / f"repr_al_summary_shard{shard_id}.json", summary)
    # free model
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return summary
