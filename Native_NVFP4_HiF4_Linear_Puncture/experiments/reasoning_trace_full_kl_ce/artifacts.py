"""Bind original B, immutable old input caches, and this independent source tree."""
import os
from pathlib import Path

from ..reasoning_trace_last_head import artifacts as previous
from ..reasoning_trace_calibration.training import samples_for
from ..reasoning_trace_calibration.prepare import require_isolation
from ..reasoning_trace_last_head.artifacts import atomic_save, digest, load, now, read_json, sha256, state_digest, write_json
from .config import NATIVE, ORIGIN, PROTOCOL, REPO


def sources():
    return {**previous.sources(), **{str(p.relative_to(REPO)): sha256(p)
                                    for p in sorted(Path(__file__).parent.rglob("*.py"))}}


def initialize(root, source, cache_run, selection_policy):
    root, source, cache_run = Path(root).resolve(), Path(source).resolve(), Path(cache_run).resolve()
    _, old = previous.verify(cache_run)
    require_isolation(source)
    if Path(old["source_run"]) != source:
        raise ValueError("reused B cache belongs to a different source run")
    train, val, hashes = samples_for(source, ORIGIN)
    if hashes != old["calibration_sha256"]:
        raise ValueError("reused calibration hashes differ")
    for name, expected in read_json(Path(old["initialization"]) / "export.json")["shards_sha256"].items():
        if sha256(Path(old["initialization"]) / name) != expected:
            raise ValueError(f"original B shard changed: {name}")
    root.mkdir(parents=True, exist_ok=False)
    keys = ("initialization", "initialization_export_sha256", "layer_parameters", "layer_parameters_sha256",
            "calibration_sha256", "teacher_index_sha256", "teacher_config_sha256", "baseline_B", "samples",
            "source_isolation_sha256")
    manifest = {**{k: old[k] for k in keys}, "kind": "reasoning_trace_full_kl_ce", "created_at": now(),
                "source_run": str(source), "cache_run": str(cache_run), "status": "prepared",
                "protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL), "selection_policy": selection_policy,
                "selection_contract_sha256": digest({"protocol": PROTOCOL, "selection_policy": selection_policy}),
                "cache_source_manifest_sha256": sha256(cache_run / "cache/student/manifest.json"),
                "development_pools_sha256": sha256(source / "isolation/pools.json"),
                "B_development_sha256": sha256(source / "development" / ORIGIN / "metrics.json")}
    write_json(manifest, root / "manifest.json")
    bound = sources()
    write_json(bound, root / "source_hashes.json")
    for name in bound:
        path = root / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO / name).read_bytes())
    plan = (Path(__file__).parent / "PLAN.md").read_text().replace("../../../docs/", "../../../../docs/")
    (root / "PLAN.md").write_text(plan, encoding="utf-8")
    return root


def verify(root):
    root = Path(root).resolve()
    m = read_json(root / "manifest.json")
    if (m["kind"] != "reasoning_trace_full_kl_ce" or m["protocol_sha256"] != digest(PROTOCOL)
            or m["selection_contract_sha256"] != digest({"protocol": PROTOCOL, "selection_policy": m["selection_policy"]})
            or read_json(root / "source_hashes.json") != sources()):
        raise ValueError("fixed protocol or source dependencies changed")
    _, old = previous.verify(m["cache_run"])
    for key in ("initialization_export_sha256", "layer_parameters_sha256", "calibration_sha256",
                "teacher_index_sha256", "teacher_config_sha256"):
        if m[key] != old[key]:
            raise ValueError(f"reused B cache contract changed: {key}")
    for path, expected in ((Path(m["source_run"]) / "isolation/pools.json", m["development_pools_sha256"]),
                           (Path(m["source_run"]) / "development" / ORIGIN / "metrics.json", m["B_development_sha256"]),
                           (Path(m["cache_run"]) / "cache/student/manifest.json", m["cache_source_manifest_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"bound input changed: {path}")
    return root, m


def selected_samples(m, qualification=False):
    train, val = previous.selected_samples(m, False)
    if qualification:
        train, val = [sorted(values, key=lambda s: s.input_ids.numel(), reverse=True)[:4]
                      + sorted(values, key=lambda s: s.input_ids.numel())[:4] for values in (train, val)]
    return train, val


def reuse_student_cache(root):
    root, m = verify(root)
    old_dir = Path(m["cache_run"]) / "cache/student"
    prior = read_json(old_dir / "manifest.json")
    if (not prior["complete"] or prior["prefix_layers"] != 47 or not prior["independent_prefix"]
            or prior["initialization_export_sha256"] != m["initialization_export_sha256"]
            or prior["calibration_sha256"] != m["calibration_sha256"]):
        raise ValueError("B fixed-prefix cache does not match initialization")
    directory = root / "cache/student"
    directory.mkdir(parents=True, exist_ok=False)
    for record in prior["samples"]:
        source = old_dir / record["path"]
        if sha256(source) != record["sha256"]:
            raise ValueError("old student input payload changed")
        os.link(source, directory / record["path"])
    write_json({"complete": True, "role": "student", "prefix_layers": 47,
                "independent_prefix": True, "samples": prior["samples"],
                "calibration_sha256": m["calibration_sha256"],
                "initialization_export_sha256": m["initialization_export_sha256"],
                "reused_from": str(old_dir), "original_manifest_sha256": sha256(old_dir / "manifest.json"),
                "reason": "loss/checkpoint selection do not change frozen B layers 0..46"}, directory / "manifest.json")
