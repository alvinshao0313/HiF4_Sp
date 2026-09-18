"""Independent, atomic, layer-sharded artifacts and epoch-level resume state."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

KIND = "non_equivalent_reconstruction"
SCHEMA_VERSION = 1


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_initialization(path, model_path):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import select_layer_diag
    initial = load(path)
    for key, expected in (("model_type", "qwen3_moe"), ("diag_mode", "fusable"),
                          ("use_r64", True), ("num_layers", 48), ("rot_order", "diag_then_rot")):
        if initial.get(key) != expected:
            raise ValueError(f"initialization {key} must be {expected!r}")
    if initial["source_model"] != model_path:
        raise ValueError("initialization source_model differs from model_path")
    if set(initial["layers"]) != {str(i) for i in range(48)}:
        raise ValueError("initialization must contain all 48 layers")
    return {str(i): select_layer_diag(initial["layers"][str(i)], "adopted") for i in range(48)}


def load_manifest(run_dir, *, require_complete=False):
    path = Path(run_dir) / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("kind") != KIND or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("not a supported non-equivalent artifact")
    if require_complete and manifest["completed_layers"] != list(range(48)):
        raise ValueError("materialization requires all 48 completed layers")
    for index in manifest["completed_layers"]:
        if not (Path(run_dir) / f"layers/{index:03d}/selected.pt").is_file():
            raise FileNotFoundError(f"missing selected layer {index}")
    return manifest


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value
