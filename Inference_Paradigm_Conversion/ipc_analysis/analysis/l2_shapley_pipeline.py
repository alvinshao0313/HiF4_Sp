"""Re-run representative Linear decomp with Shapley + FP64 audit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_conversion import (
    fair_triple_qdq,
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
def run_l2_shapley_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: str = "cuda:0",
    shard_id: int = 0,
    num_shards: int = 1,
    samples_per_family: int = 4,
    max_seq_len: int = 128,
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
    weight_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def module_filter(name: str, _mod: nn.Module) -> bool:
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        return layer_idx in rep and _proj_of(name) in LINEAR_PROJECTIONS

    rows: list[dict[str, Any]] = []
    audit_fail = 0
    audit_ok = 0
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
            max_tokens_per_module=1024,
            max_raw_tokens_per_module=64,
        )
        for j, cap in enumerate(caps):
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
            if x.ndim >= 2 and x.reshape(-1, x.shape[-1]).shape[0] > 32:
                x = x.reshape(-1, x.shape[-1])[:32]
            x = x[..., :usable]
            scale = resolve_nvfp4_scale_for_module(scales, cap.module_name).to(device_t)
            wname = cap.module_name + ".weight"
            if wname not in weight_cache:
                w = load_nvfp4_qat_dequant_weight(checkpoint, wname, device=device_t).dequantized
                if w.shape[1] != usable:
                    w = w[:, :usable]
                conv = convert_weight_w0(w, device=device_t)
                weight_cache[wname] = (conv["W_N"].to(device_t), conv["W_H_FP32"].to(device_t))  # type: ignore
            w_n, w_h = weight_cache[wname]
            triple = fair_triple_qdq(x, scale)
            a_n = triple["A_N"].dequantized.float()
            a_h = triple["A_H"].dequantized.float()
            do_audit = j % 7 == 0
            rec = decompose_linear_error(
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
                with_shapley=True,
                with_fp64_audit=do_audit,
            )
            d = rec.to_dict()
            extras = d.pop("extras", {})
            row = {
                **{k: d[k] for k in d if k != "extras"},
                "energy_phi_w": extras.get("energy_phi_w"),
                "energy_phi_a": extras.get("energy_phi_a"),
                "cross_phi_w_phi_a": extras.get("cross_phi_w_phi_a"),
                "energy_delta_y": extras.get("energy_delta_y"),
                "shapley_residual_rel": extras.get("shapley_residual_rel"),
                "output_nmse": extras.get("output_nmse"),
            }
            if do_audit:
                aud = extras.get("fp64_audit") or {}
                row["fp64_audit_ok"] = aud.get("ok")
                row["fp64_max_abs_residual"] = aud.get("max_abs_residual")
                if aud.get("ok"):
                    audit_ok += 1
                else:
                    audit_fail += 1
            rows.append(row)
        print(f"[L2] shard{shard_id} {item.sample_id}", flush=True)

    write_csv(out_dir / f"l2_shapley_shard{shard_id}.csv", rows)
    ew = [float(r["energy_phi_w"]) for r in rows if r.get("energy_phi_w") not in (None, "")]
    ea = [float(r["energy_phi_a"]) for r in rows if r.get("energy_phi_a") not in (None, "")]
    summary = {
        "shard_id": shard_id,
        "num_rows": len(rows),
        "mean_energy_phi_w": sum(ew) / len(ew) if ew else 0.0,
        "mean_energy_phi_a": sum(ea) / len(ea) if ea else 0.0,
        "ratio_phi_a_over_phi_w": (sum(ea) / sum(ew)) if ew and sum(ew) > 0 else 0.0,
        "fp64_audit_ok": audit_ok,
        "fp64_audit_fail": audit_fail,
    }
    atomic_write_json(out_dir / f"l2_summary_shard{shard_id}.json", summary)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return summary
