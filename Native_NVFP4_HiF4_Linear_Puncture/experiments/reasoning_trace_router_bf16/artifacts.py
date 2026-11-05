"""Bind an independent output directory to the immutable B experiment."""
from pathlib import Path

from ..reasoning_trace_calibration import artifacts as original
from ..reasoning_trace_calibration.artifacts import (
    atomic_save, digest, load, now, read_json, sha256, state_digest, write_json,
)
from ..reasoning_trace_calibration.prepare import require_isolation
from ..reasoning_trace_calibration.training import samples_for
from .config import CANDIDATE, NATIVE, ORIGIN, PROTOCOL, REPO


def sources():
    return {**original.sources(), **{
        str(p.relative_to(REPO)): sha256(p) for p in sorted(Path(__file__).parent.rglob("*.py"))
    }}


def initialize(root, source):
    root, source = Path(root).resolve(), original.validate_root(source)
    original.verify_sources(source)
    require_isolation(source)
    train, val, hashes = samples_for(source, ORIGIN)
    model = source / "models" / ORIGIN
    exported = read_json(model / "export.json")
    if not exported["complete"] or exported["qualification"]:
        raise ValueError("a complete formal B export is required")
    root.mkdir(parents=True, exist_ok=False)
    for name, expected in exported["shards_sha256"].items():
        if sha256(model / name) != expected:
            raise ValueError(f"B export shard changed: {name}")
    baseline = read_json(source / "comparison.json")["models"][ORIGIN]
    manifest = {
        "kind": "reasoning_trace_router_bf16", "created_at": now(), "status": "prepared",
        "protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL),
        "source_run": str(source), "initialization": str(model),
        "initialization_export_sha256": sha256(model / "export.json"),
        "calibration_sha256": hashes, "source_isolation_sha256": sha256(source / "isolation/manifest.json"),
        "teacher_index_sha256": sha256(NATIVE / "model.safetensors.index.json"),
        "teacher_config_sha256": sha256(NATIVE / "config.json"),
        "baseline_B": baseline,
        "samples": {split: [{"id": s.sample_id, "tokens": s.input_ids.numel()} for s in values]
                    for split, values in (("train", train), ("val", val))},
    }
    write_json(manifest, root / "manifest.json")
    write_json(sources(), root / "source_hashes.json")
    return root


def verify(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if (manifest["kind"] != "reasoning_trace_router_bf16"
            or manifest["protocol_sha256"] != digest(PROTOCOL)):
        raise ValueError("experiment protocol changed")
    if read_json(root / "source_hashes.json") != sources():
        raise ValueError("qualified source dependencies changed")
    original.verify_sources(Path(manifest["source_run"]))
    if sha256(Path(manifest["initialization"]) / "export.json") != manifest["initialization_export_sha256"]:
        raise ValueError("B initialization changed")
    if (sha256(NATIVE / "model.safetensors.index.json") != manifest["teacher_index_sha256"]
            or sha256(NATIVE / "config.json") != manifest["teacher_config_sha256"]):
        raise ValueError("native teacher version changed")
    return root, manifest
