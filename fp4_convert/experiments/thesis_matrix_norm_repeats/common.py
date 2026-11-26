"""Bind the two existing matrix_norm models and their reusable first rounds."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

from ..thesis_format_conversion_supplement.artifacts import (
    freeze_sources, read_json, run_root, sha256, verify_sources, write_json)
from ..thesis_format_conversion_supplement.config import REPO, PROTOCOL, TASKS
from ..matrix_norm_efficiency.pinned import manifest as data_manifest

MODULE = "fp4_convert.experiments.thesis_matrix_norm_repeats"
ROLES = ("ours_a20", "b20")


def sources():
    paths = list(Path(__file__).parent.glob("*.py"))
    for package in ("downstream_accuracy_recovery", "matrix_norm_efficiency"):
        paths += list((Path(__file__).parent.parent / package).glob("*.py"))
    paths += [Path(__file__).parent.parent / "ban_trace_calibration/processes.py"]
    return {str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)}


def verify(root):
    verify_sources(root)
    if sources() != read_json(root / "repeat_source_hashes.json"):
        raise ValueError("repeat evaluation sources changed")


def prepare(root, source):
    root, source = run_root(root), run_root(source)
    verify_sources(source)
    root.mkdir(parents=True, exist_ok=False)
    for name in ("inventory.json", "summary.json", "reuse.json", "additional_native_reuse.json"):
        if (source / name).is_file():
            shutil.copyfile(source / name, root / name)
    shutil.copytree(source / "evaluation/data", root / "evaluation/data")
    shutil.copyfile(source / "evaluation/prompt_audit.json", root / "evaluation/prompt_audit.json")
    data_manifest(root)
    write_json(PROTOCOL, root / "protocol.json")
    inventory, reuse = read_json(root / "inventory.json"), read_json(root / "reuse.json")
    binding = {}
    for role in ROLES:
        entry = inventory[role]
        model = Path(entry["model_path"])
        export = read_json(model / "export.json")
        if (not export["complete"] or export["qualification"] or export["candidate"] != entry["candidate"]
                or sha256(entry["training_manifest"]) != entry["training_manifest_sha256"]):
            raise ValueError(f"wrong or changed training/export identity: {role}")
        for name, expected in {**entry["shards_sha256"], **entry["assets_sha256"]}.items():
            if sha256(model / name) != expected:
                raise ValueError(f"model changed: {model / name}")
        for task, evidence in reuse[role].items():
            if evidence.get("status") == "not_reused":
                raise ValueError(f"first-round evidence rejected: {role}/{task}")
            for path_key, hash_key in (("metrics", "metrics_sha256"), ("details", "details_sha256")):
                if path_key in evidence:
                    expected = evidence.get(hash_key, evidence.get("sha256"))
                    if sha256(evidence[path_key]) != expected:
                        raise ValueError(f"historical evidence changed: {role}/{task}/{path_key}")
            for token in evidence.get("tokens", []):
                if sha256(token.get("path", token.get("record"))) != token["sha256"]:
                    raise ValueError(f"historical raw-token evidence changed: {role}/{task}")
        rounds = read_json(root / "summary.json")["rows"][role]["round_scores"]
        if any(set(rounds[t]) != {"1"} for t in TASKS):
            raise ValueError(f"expected exactly one audited historical round for {role}")
        binding[role] = {"model_path": str(model), "candidate": entry["candidate"],
                         "training_calibration_sha256": entry["training"]["calibration_sha256"],
                         "weight_files_verified": len(entry["shards_sha256"]),
                         "weight_bytes": entry["weight_bytes"], "reused_rounds": rounds,
                         "evidence": reuse[role]}
        print(f"Verified {role}: complete original model and all five first-round results", flush=True)
    write_json(binding, root / "reuse_binding.json")
    write_json({"created_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared",
                "allocated_gpu_pair": "4,5", "roles": ROLES, "tasks": TASKS,
                "repetitions": 5, "new_rounds": [2, 3, 4, 5], "source_run": str(source),
                "authorization": "2026-11-18 user allocates GPU4,5 and requests both S1K and native-trace matrix_norm",
                "comparison_boundary": "A20 matches GPTQ S1K; B20 uses different self-generated calibration",
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()},
               root / "manifest.json")
    freeze_sources(root)
    write_json(sources(), root / "repeat_source_hashes.json")
    verify(root)
