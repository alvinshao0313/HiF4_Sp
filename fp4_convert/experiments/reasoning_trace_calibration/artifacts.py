"""Atomic artifacts and binding to this experiment's inputs and implementation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..non_equivalent_reconstruction.artifact import atomic_save, load, sha256, write_json
from .config import REPO, PROTOCOL


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def tensor_digest(tensor):
    t = tensor.detach().cpu().contiguous()
    return digest([str(t.dtype), list(t.shape), hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()])


def state_digest(state):
    return digest({key: tensor_digest(value) for key, value in sorted(state.items())})


def sources():
    package = Path(__file__).parent
    paths = list(package.rglob("*.py"))
    for directory in ("module_design_ablation", "non_equivalent_reconstruction",
                      "e2e_diag_reconstruction/core", "e2e_diag_reconstruction/data",
                      "e2e_diag_reconstruction/training", "e2e_diag_reconstruction/evaluation",
                      "downstream_accuracy_recovery"):
        paths.extend((package.parent / directory).rglob("*.py"))
    paths += [REPO / "main.py", REPO / "tasks/aime.py",
              REPO / "3rdparty/vllm/vllm/model_executor/models/qwen3_moe.py"]
    return {str(path.relative_to(REPO)): sha256(path) for path in sorted(set(paths))}


def initialize(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json({"kind": "reasoning_trace_calibration", "created_at": now(), "protocol": PROTOCOL,
                "protocol_sha256": digest(PROTOCOL), "status": "preparing"}, root / "manifest.json")
    return root


def validate_root(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("kind") != "reasoning_trace_calibration" or manifest.get("protocol_sha256") != digest(PROTOCOL):
        raise ValueError("run protocol differs from the confirmed experiment")
    return root


def bind_sources(root):
    root = validate_root(root)
    path = root / "source_hashes.json"
    current = sources()
    if path.exists():
        if read_json(path) != current:
            raise ValueError("run dependencies changed after qualification")
    else:
        write_json(current, path)
    return current


def verify_sources(root):
    expected = read_json(Path(root) / "source_hashes.json")
    current = sources()
    changed = [key for key in expected if current.get(key) != expected[key]]
    if changed or current.keys() != expected.keys():
        raise ValueError(f"qualified dependencies changed: {changed}")
