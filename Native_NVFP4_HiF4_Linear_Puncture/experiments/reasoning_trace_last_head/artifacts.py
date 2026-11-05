"""Immutable bindings, including B's unfused last-layer parameters."""
from pathlib import Path

from ..reasoning_trace_calibration import artifacts as original
from ..reasoning_trace_calibration.artifacts import (
    atomic_save, digest, load, now, read_json, sha256, state_digest, write_json,
)
from ..reasoning_trace_calibration.prepare import require_isolation
from ..reasoning_trace_calibration.training import samples_for, training_dir
from .config import NATIVE, ORIGIN, PROTOCOL, REPO


def sources():
    paths = list(Path(__file__).parent.rglob("*.py"))
    paths += list((Path(__file__).parent.parent / "reasoning_trace_router_bf16").rglob("*.py"))
    return {**original.sources(), **{str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)}}


def initialize(root, source):
    root, source = Path(root).resolve(), original.validate_root(source)
    original.verify_sources(source)
    require_isolation(source)
    train, val, hashes = samples_for(source, ORIGIN)
    model = source / "models" / ORIGIN
    exported = read_json(model / "export.json")
    if not exported["complete"] or exported["qualification"]:
        raise ValueError("a complete formal B export is required")
    selected = training_dir(source, ORIGIN) / "layers/047/selected.pt"
    for name, expected in exported["shards_sha256"].items():
        if sha256(model / name) != expected:
            raise ValueError(f"B export shard changed: {name}")
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "reasoning_trace_last_head", "created_at": now(), "status": "prepared",
        "protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL),
        "source_run": str(source), "initialization": str(model),
        "initialization_export_sha256": sha256(model / "export.json"),
        "layer_parameters": str(selected), "layer_parameters_sha256": sha256(selected),
        "calibration_sha256": hashes, "source_isolation_sha256": sha256(source / "isolation/manifest.json"),
        "teacher_index_sha256": sha256(NATIVE / "model.safetensors.index.json"),
        "teacher_config_sha256": sha256(NATIVE / "config.json"),
        "baseline_B": read_json(source / "comparison.json")["models"][ORIGIN],
        "samples": {split: [{"id": s.sample_id, "tokens": s.input_ids.numel(),
                              "token_sha256": state_digest({"ids": s.input_ids, "mask": s.loss_mask})}
                             for s in values] for split, values in (("train", train), ("val", val))},
    }
    write_json(manifest, root / "manifest.json")
    bound = sources()
    write_json(bound, root / "source_hashes.json")
    for name in bound:
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
    plan = (Path(__file__).parent / "PLAN.md").read_text().replace("../../../docs/", "../../../../docs/")
    (root / "PLAN.md").write_text(plan, encoding="utf-8")
    return root


def verify(root):
    root = Path(root).resolve()
    m = read_json(root / "manifest.json")
    if m["kind"] != "reasoning_trace_last_head" or m["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("experiment protocol changed")
    if read_json(root / "source_hashes.json") != sources():
        raise ValueError("qualified source dependencies changed")
    original.verify_sources(Path(m["source_run"]))
    for path, expected in ((Path(m["initialization"]) / "export.json", m["initialization_export_sha256"]),
                           (Path(m["layer_parameters"]), m["layer_parameters_sha256"]),
                           (NATIVE / "config.json", m["teacher_config_sha256"]),
                           (NATIVE / "model.safetensors.index.json", m["teacher_index_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"bound initialization changed: {path}")
    return root, m


def selected_samples(manifest, qualification=False):
    train, val, hashes = samples_for(Path(manifest["source_run"]), ORIGIN)
    if hashes != manifest["calibration_sha256"]:
        raise ValueError("B's fixed calibration data changed")
    if qualification:
        train, val = [sorted(v, key=lambda s: s.input_ids.numel(), reverse=True)[:4] for v in (train, val)]
    return train, val
