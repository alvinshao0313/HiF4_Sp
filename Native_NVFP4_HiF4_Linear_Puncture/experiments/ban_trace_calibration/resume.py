"""Prepare this failed, pre-evaluation experiment in a new output directory."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import psutil
import torch

from ..ban.artifact import load_artifact, model_identity
from ..ban.evaluate import REPO, file_sha256
from .config import TASKS, now, read, save, source_identity
from .report import summarize


def prepare(source, output, controller_snapshot):
    source, root = Path(source).resolve(strict=True), Path(output).resolve()
    snapshot = Path(controller_snapshot).resolve(strict=True)
    if root.exists():
        raise FileExistsError(f"continuation output must be new: {root}")
    manifest, state = read(source / "manifest.json"), read(source / "status.json")
    if state["status"] != "failed":
        raise ValueError("this continuation requires the terminated failed source run")
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid == os.getpid():
            continue
        if "python" not in (process.info["name"] or "").lower():
            continue
        if any(str(source) == arg for arg in process.info["cmdline"] or []):
            raise ValueError(f"source run still has an active process: {process.pid}")
    controller = "Native_NVFP4_HiF4_Linear_Puncture/experiments/ban_trace_calibration/run.py"
    changes = {}
    for name, digest in manifest["code"]["source_sha256"].items():
        current = file_sha256(REPO / name)
        if current != digest:
            if name != controller or file_sha256(snapshot) != digest:
                raise ValueError(f"scientific source changed; cannot reuse outputs: {name}")
            changes[name] = {"before": digest, "after": current, "original_snapshot": str(snapshot)}
    for name, digest in manifest["input_sha256"].items():
        if file_sha256(name) != digest:
            raise ValueError(f"original input changed: {name}")
    if model_identity(manifest["model"], manifest["spec"]) != manifest["model_identity"]:
        raise ValueError("model/sidecar changed")
    reused = ("diagnostic", "preflight_budget", "trace/calibrate", "trace/budget",
              "qualification/trace/lcb", "qualification/trace/mmlu_pro")
    for label in reused:
        phase = state["phases"].get(label, {})
        if phase.get("status") != "complete" or phase.get("exit_code") != 0:
            raise ValueError(f"required source phase is incomplete: {label}")
    for arm in ("top8", "c4", "trace"):
        for task, definition in TASKS.items():
            for index in range(1, definition["rounds"] + 1):
                target = source / arm / task / f"round_{index:02d}"
                if list(target.rglob("results_*.json")):
                    raise ValueError("this source already contains formal outputs; reassess continuation scope")
    proof = read(source / "preflight.json")
    if not proof["passed"]:
        raise ValueError("original preflight did not pass")
    inputs = dict(manifest["input_sha256"])
    for name, digest in proof["artifact_sha256"].items():
        if file_sha256(name) != digest:
            raise ValueError("original preflight artifact changed")
        inputs[name] = digest
    artifact_path = source / "calibration/formal/c4_ban.pt"
    calibrated = read(source / "calibration/formal/manifest.json")
    if calibrated["status"] != "complete" or file_sha256(artifact_path) != calibrated["artifact_sha256"]:
        raise ValueError("formal calibration completion/hash mismatch")
    artifact = load_artifact(artifact_path, manifest["model_identity"])
    if (artifact["phase"], artifact["num_samples"], artifact["k_pruned"], artifact["topn"], artifact["kl_positions"]) != (
            "formal", 512, 3, 1000, "last"):
        raise ValueError("formal calibration protocol differs")
    if not torch.equal(torch.load(source / "calibration/formal/input_ids.pt", weights_only=True),
                       torch.load(source / "calibration_inputs.pt", weights_only=True)):
        raise ValueError("formal calibration input tokens differ from prepared windows")
    budget = read(source / "budget/trace/budget.json")
    best = min(budget["trials"], key=lambda t: abs(t["average_experts"] - 6.07))
    if (budget["status"] != "complete" or not budget["budget_matched"] or budget["diagnostic"]
            or budget["chosen_lambda"] != best["lambda"]
            or budget["chosen_average_experts"] != best["average_experts"]
            or abs(best["average_experts"] - 6.07) > 0.05 or Path(budget["artifact"]) != artifact_path):
        raise ValueError("original budget choice is invalid")
    expected = {r["id"]: r["input_tokens_sha256"] for r in read(source / "budget_inputs.json")}
    for trial in budget["trials"]:
        if ({r["id"]: r["input_tokens_sha256"] for r in trial["requests"]} != expected
                or trial["stats"]["selected"] != trial["stats"]["dispatched"]
                or trial["stats"]["tp_ranks"] != 2):
            raise ValueError("original budget input/routing coverage differs")
    for name in TASKS:
        target = source / "qualification/trace" / name
        if summarize(target, name, "trace", read(source / f"{name}_input_audit.json"), short=True) != read(target / "report.json"):
            raise ValueError("original short-test raw outputs no longer match")
        for path in target.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".parquet"):
                inputs[str(path)] = file_sha256(path)
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    files = ("calibration_inputs.pt", "diagnostic_inputs.pt", "windows.json", "budget_inputs.json",
             "datasets.json", "lcb_input_audit.json", "mmlu_pro_input_audit.json", "preflight.json", "diagnostic_verified.json")
    for name in files:
        (root / name).symlink_to(source / name)
        inputs[str(source / name)] = file_sha256(source / name)
    for name in ("calibration/formal", "calibration/diagnostic", "budget/trace",
                 "qualification/trace/lcb", "qualification/trace/mmlu_pro"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(source / name, target_is_directory=True)
    for path in (source / "manifest.json", source / "status.json", source / "calibration/formal/manifest.json",
                 source / "calibration/formal/input_ids.pt", artifact_path, source / "budget/trace/budget.json", snapshot):
        inputs[str(path)] = file_sha256(path)
    new_manifest = copy.deepcopy(manifest)
    new_manifest.update(created_at=now(), input_sha256=inputs, code=source_identity(),
                        continuation_source=str(source), reused_phases={label: str(source) for label in reused},
                        controller_changes=changes,
                        reuse_basis="Only controller process cleanup changed. Frozen model/data/algorithm/report sources, full calibration, budget inputs/routing and raw short outputs revalidated. No formal predictions reused.")
    save(root / "manifest.json", new_manifest)
    phases = {label: {**state["phases"][label], "reused_from": str(source)} for label in reused}
    save(root / "status.json", {"status": "preflight_complete", "created_at": now(), "phases": phases,
                               "errors": [], "continuation_source": str(source)})
    save(root / "continuation.json", {"source": str(source), "created_at": now(), "reused_phases": list(reused),
         "trace_lambda": budget["chosen_lambda"], "trace_development_average_experts": budget["chosen_average_experts"],
         "formal_predictions_reused": 0, "source_run_unchanged": True, "controller_changes": changes})
    print(f"BAN_TRACE_CONTINUATION_PREPARED {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--controller_snapshot", required=True)
    args = parser.parse_args()
    prepare(args.source, args.output, args.controller_snapshot)
