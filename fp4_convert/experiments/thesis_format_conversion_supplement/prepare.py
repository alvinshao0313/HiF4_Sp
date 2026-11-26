"""Bind existing inputs and models without modifying historical runs."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

from .artifacts import read_json, write_json, sha256, run_root, load
from .config import *


def prepare(root):
    from ..reasoning_trace_calibration.training import samples_for
    root = run_root(root)
    root.mkdir(parents=True, exist_ok=False)
    write_json(PROTOCOL, root / "protocol.json")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "plan": str(PLAN),
                "plan_sha256": sha256(PLAN), "allocated_gpus": [0, 1], "status": "preparing",
                "native_snapshot": str(NATIVE), "tasks": TASKS,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    write_json(manifest, root / "manifest.json")
    (root / "initial_worktree.diff").write_text(subprocess.check_output(["git", "diff"], text=True))
    calibration = {}
    for role, candidate in (("ours_a20", "matrix_norm_s1k"), ("b20", "matrix_norm_native_traces")):
        train, val, hashes = samples_for(SOURCE_RUN, candidate)
        calibration[role] = {"sha256": hashes, "splits": {k: {
            "ids": [s.sample_id for s in samples], "count": len(samples),
            "tokens": sum(s.input_ids.numel() for s in samples),
            "max_length": max(s.input_ids.numel() for s in samples)} for k, samples in (("train", train), ("val", val))}}
    write_json(calibration, root / "calibration.json")
    data = read_json(EFFICIENCY / "evaluation/data/manifest.json")
    data["tasks"] = {k: data["tasks"][k] for k in REASONING}
    for entry in data["tasks"].values():
        for item in entry["splits"].values():
            source, target = EFFICIENCY / item["path"], root / item["path"]
            if sha256(source) != item["sha256"]:
                raise ValueError(f"pinned data changed: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    write_json(data, root / "evaluation/data/manifest.json")
    # All existing models use identical tokenizer assets; freshly audit the three actual tasks.
    from .evaluation import audit_model
    write_json(audit_model(root, NATIVE), root / "evaluation/prompt_audit.json")
    inventory = {}
    for role, (source, candidate) in HISTORY.items():
        model = NATIVE if role == "native" else source / "models" / candidate
        entry = {"source_run": str(source), "candidate": candidate, "model_path": str(model),
                 "model_present": (model / "model.safetensors.index.json").exists()}
        if entry["model_present"]:
            index = read_json(model / "model.safetensors.index.json")
            shards = sorted(set(index["weight_map"].values()))
            entry["missing_shards"] = [s for s in shards if not (model / s).is_file()]
            if entry["missing_shards"]:
                raise FileNotFoundError(entry["missing_shards"])
            marker = model / "export.json"
            recorded = read_json(marker).get("shards_sha256", {}) if marker.exists() else {}
            observed = {}
            for shard in shards:
                observed[shard] = sha256(model / shard)
                if shard in recorded and observed[shard] != recorded[shard]:
                    raise ValueError(f"export shard changed: {model / shard}")
            entry["shards_sha256"] = observed
            entry["assets_sha256"] = {p.name: sha256(p) for p in model.iterdir()
                if p.is_file() and p.suffix in (".json", ".pt", ".jinja") and p.name != "tokenizer.json"}
            entry["weight_bytes"] = sum((model / s).stat().st_size for s in shards)
        training = source / "training" / candidate / "manifest.json"
        if training.exists():
            saved = read_json(training)
            entry["training"] = {k: saved.get(k) for k in ("complete", "seconds", "completed_layers", "calibration_sha256")}
            entry["training_manifest"] = str(training)
            entry["training_manifest_sha256"] = sha256(training)
        inventory[role] = entry
        write_json(inventory, root / "inventory.json")
        print(f"bound {role}: model_present={entry['model_present']}", flush=True)
    from .report import collect
    collect(root)
    manifest["status"] = "prepared"
    write_json(manifest, root / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    prepare(parser.parse_args().run)
