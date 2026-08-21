"""Small IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_pt(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(obj, p)


def load_pt(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def module_capture_stem(module_name: str) -> str:
    """``model.layers.2.self_attn.q_proj`` -> ``layer02_q_proj``."""
    parts = module_name.split(".")
    layer_idx = None
    proj = parts[-1]
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            layer_idx = int(parts[i + 1])
            break
    if layer_idx is None:
        raise ValueError(f"cannot parse layer index from {module_name}")
    return f"layer{layer_idx:02d}_{proj}"
