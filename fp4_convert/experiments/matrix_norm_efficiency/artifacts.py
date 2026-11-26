"""Atomic artifacts and explicit bindings for this independent run."""
from pathlib import Path

from ..reasoning_trace_calibration.artifacts import (
    atomic_save, digest, load, now, read_json, sha256, state_digest, tensor_digest, write_json,
)
from .config import PROTOCOL, REPO


def sources():
    package = Path(__file__).parent
    paths = list(package.glob("*.py"))
    for directory in ("module_design_ablation", "non_equivalent_reconstruction",
                      "e2e_diag_reconstruction/core", "e2e_diag_reconstruction/data",
                      "e2e_diag_reconstruction/training", "e2e_diag_reconstruction/evaluation",
                      "reasoning_trace_calibration", "downstream_accuracy_recovery"):
        paths.extend(p for p in (package.parent / directory).rglob("*.py") if "tests" not in p.parts)
    paths += [package.parent / "ban_trace_calibration/processes.py",
              REPO / "main.py", *sorted((REPO / "tasks").glob("*.py")),
              REPO / "3rdparty/vllm/vllm/model_executor/models/qwen3_moe.py"]
    return {str(path.relative_to(REPO)): sha256(path) for path in sorted(set(paths))}


def initialize(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json({"kind": "matrix_norm_efficiency", "created_at": now(), "protocol": PROTOCOL,
                "protocol_sha256": digest(PROTOCOL), "status": "preparing"}, root / "manifest.json")
    return root


def validate_root(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if (manifest.get("kind") != "matrix_norm_efficiency"
            or manifest.get("protocol_sha256") != digest(PROTOCOL)):
        raise ValueError("run protocol differs from this experiment")
    return root


def bind_sources(root):
    root = validate_root(root)
    path = root / "source_hashes.json"
    current = sources()
    if path.exists() and read_json(path) != current:
        raise ValueError("run dependencies changed after qualification")
    if not path.exists():
        write_json(current, path)
    return current


def verify_sources(root):
    expected = read_json(validate_root(root) / "source_hashes.json")
    current = sources()
    changed = [key for key in expected if current.get(key) != expected[key]]
    if changed or current.keys() != expected.keys():
        raise ValueError(f"qualified dependencies changed: {changed}")
