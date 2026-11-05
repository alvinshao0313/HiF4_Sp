"""Identify the existing model without copying or rewriting its weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def model_identity(model_path, spec_path):
    model, sidecar = Path(model_path).resolve(), Path(spec_path).resolve()
    config = json.loads((model / "config.json").read_text())
    spec = torch.load(sidecar, map_location="cpu", weights_only=False)
    if config.get("model_type") != "qwen3_moe" or config.get("quantization_config"):
        raise ValueError("Ban requires the materialized Qwen3 MoE HiF4 export")
    if spec.get("variant") != "fusable_r64" or spec.get("runtime_abi_version") != 3:
        raise ValueError("Ban currently supports fusable_r64 runtime ABI 3")
    geometry = (config["num_hidden_layers"], config["num_experts"], config["num_experts_per_tok"])
    if geometry != (48, 128, 8) or geometry != tuple(spec[k] for k in ("num_layers", "num_experts", "top_k")):
        raise ValueError("Ban target must be the 48-layer/128-expert/top-8 matrix_norm model")
    index = model / "model.safetensors.index.json"
    files = []
    for name in sorted(set(json.loads(index.read_text())["weight_map"].values())):
        path = (model / name).resolve(strict=True)
        stat = path.stat()
        files.append([name, str(path), stat.st_size, stat.st_mtime_ns])
    # This describes file identity, rather than claiming a full weight-content SHA.
    return {"config_sha256": sha256(model / "config.json"), "sidecar_sha256": sha256(sidecar),
            "index_sha256": sha256(index), "export_sha256": sha256(model / "ablation_export.json"),
            "weight_files": files, "num_layers": geometry[0], "num_experts": geometry[1], "top_k": geometry[2]}


def load_artifact(path, identity):
    result = torch.load(path, map_location="cpu", weights_only=False)
    if result.get("format") != "hif4_ban_v1" or result.get("model_identity") != identity:
        raise ValueError("Ban calibration does not match the model/sidecar/weight files")
    sensitivity = result["layer_sensitivity"]
    if (sensitivity.shape != (identity["num_layers"],) or not torch.isfinite(sensitivity).all()
            or not ((sensitivity >= 0) & (sensitivity <= 1)).all()):
        raise ValueError("invalid Ban layer_sensitivity")
    if not 0 <= result["r_min"] < result["r_max"] <= 1 + 1e-6:
        raise ValueError("invalid Ban r_min/r_max")
    return result
