"""Reuse verified calibration, rematch a new expert budget, and regenerate all evaluations."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from ..ban.artifact import load_artifact, model_identity
from ..ban.evaluate import REPO, file_sha256
from .config import budget_config, now, read, save, source_identity


def prepare(source, output, target_experts):
    source, root = Path(source).resolve(strict=True), Path(output).resolve()
    budget = budget_config(target_experts)
    if root.exists():
        raise FileExistsError(f"new budget requires a new output directory: {root}")
    original = read(source / "manifest.json")
    state = read(source / "status.json")
    # Only orchestration/budget/report changes are allowed when reusing the
    # expensive calibration. Algorithm, runtime, sampling and task sources stay pinned.
    prefix = "Native_NVFP4_HiF4_Linear_Puncture/experiments/ban_trace_calibration/"
    editable = {prefix + name for name in ("config.py", "stages.py", "report.py", "prepare.py", "run.py", "tests.py")}
    changes = {}
    for name, digest in original["code"]["source_sha256"].items():
        current = file_sha256(REPO / name)
        if current != digest:
            if name not in editable:
                raise ValueError(f"calibration/task dependency changed: {name}")
            changes[name] = {"source_sha256": digest, "current_sha256": current}
    for name, digest in original["input_sha256"].items():
        if file_sha256(name) != digest:
            raise ValueError(f"source input changed: {name}")
    if model_identity(original["model"], original["spec"]) != original["model_identity"]:
        raise ValueError("model/sidecar identity changed")
    inputs = dict(original["input_sha256"])
    for label in ("diagnostic", "trace/calibrate"):
        phase = state["phases"].get(label, {})
        if phase.get("status") != "complete" or phase.get("exit_code") != 0:
            raise ValueError(f"source calibration phase incomplete: {label}")
    for path in (source / "calibration/formal", Path(original["c4_artifact"]).parent):
        record = read(path / "manifest.json")
        artifact = path / "c4_ban.pt"
        if record["status"] != "complete" or file_sha256(artifact) != record["artifact_sha256"]:
            raise ValueError(f"calibration completion/hash mismatch: {path}")
        parsed = load_artifact(artifact, original["model_identity"])
        if (parsed["phase"], parsed["num_samples"], parsed["k_pruned"], parsed["topn"], parsed["kl_positions"]) != (
                "formal", 512, 3, 1000, "last"):
            raise ValueError("calibration protocol differs")
        # Calibration records every sibling file, including downstream reporters
        # it never imports. Compare the actual calibration/forward dependencies.
        dependencies = ("algorithm.py", "artifact.py", "calibrate.py", "data.py", "diagnose.py",
                        "engine.py", "worker.py", "moe.py", "runtime.py")
        for name in dependencies:
            if file_sha256(Path(__file__).parent.parent / "ban" / name) != record["code_sha256"][name]:
                raise ValueError(f"calibration implementation changed: {name}")
        for item in (artifact, path / "manifest.json", path / "input_ids.pt"):
            inputs[str(item)] = file_sha256(item)
    if not torch.equal(torch.load(source / "calibration/formal/input_ids.pt", weights_only=True),
                       torch.load(source / "calibration_inputs.pt", weights_only=True)):
        raise ValueError("saved calibration windows differ from the prepared raw tokens")
    proof = read(source / "preflight.json")
    if not proof["passed"]:
        raise ValueError("source diagnostic preflight failed")
    for name, digest in proof["artifact_sha256"].items():
        if file_sha256(name) != digest:
            raise ValueError(f"source diagnostic evidence changed: {name}")
        inputs[name] = digest
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    for name in ("calibration_inputs.pt", "diagnostic_inputs.pt", "windows.json", "budget_inputs.json",
                 "datasets.json", "lcb_input_audit.json", "mmlu_pro_input_audit.json", "diagnostic_verified.json"):
        (root / name).symlink_to(source / name)
        inputs[str(source / name)] = file_sha256(source / name)
    for name in ("calibration/formal", "calibration/diagnostic"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(source / name, target_is_directory=True)
    manifest = copy.deepcopy(original)
    manifest.update(created_at=now(), budget=budget, code=source_identity(), input_sha256=inputs,
                    rebudget_source=str(source), reused_phases={"diagnostic": str(source), "trace/calibrate": str(source)})
    save(root / "manifest.json", manifest)
    save(root / "reuse.json", {"source": str(source), "created_at": now(),
         "source_manifest_sha256": file_sha256(source / "manifest.json"), "budget": budget,
         "source_changes": changes, "reused": ["calibration", "diagnostic", "pinned inputs"],
         "fresh": ["TP2 preflight", "both budget searches", "all qualifications", "all formal predictions"]})
    phases = {label: {**copy.deepcopy(state["phases"][label]), "reused_from": str(source)}
              for label in manifest["reused_phases"]}
    save(root / "status.json", {"status": "prepared", "created_at": now(), "phases": phases, "errors": []})
    print(f"BAN_REBUDGET_PREPARED {root} target={target_experts}", flush=True)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target_experts", type=float, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output, args.target_experts)
