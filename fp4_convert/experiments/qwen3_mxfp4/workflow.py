"""Finite, fail-fast continuation: native -> direct -> R64; never retries."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import torch

from ..nvfp4_gpqa.run import run as run_evaluation
from .artifacts import PACKAGE, REPO, check_hashes, code_hashes, file_hash, save
from .evidence import comparison, summarize_phase, verify_runtime_audit


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(pid):
    process = psutil.Process(pid)
    return {"pid": pid, "created": process.create_time(), "command": process.cmdline()}


def alive(record):
    try:
        process = psutil.Process(record["pid"])
        return process.create_time() == record["created"] and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def prepare(root, baseline):
    root, baseline = Path(root).absolute(), Path(baseline).absolute()
    if (root / "workflow_manifest.json").exists():
        raise FileExistsError("Prepared workflow already exists")
    native_manifest = json.loads((baseline / "manifest.json").read_text())
    native_status = json.loads((baseline / "formal/status.json").read_text())
    check_hashes(native_manifest["source_sha256"])
    if native_status["status"] not in ("running", "completed"):
        raise ValueError("Native baseline has failed or is not running")
    if native_manifest["environment"]["CUDA_VISIBLE_DEVICES"] != "4,5,6,7":
        raise ValueError("Expected the authorized GPU4-7 baseline")
    compilation = json.loads((root / "offline_compile.json").read_text())
    if compilation["status"] != "passed" or len(compilation["kernels"]) != 10:
        raise ValueError("Actual kernel specializations have not compiled")
    if not (root / "cpu_tests.log").read_text().rstrip().endswith("OK"):
        raise ValueError("CPU format tests have not passed")
    if not (root / "cpu_integration_tests.log").read_text().rstrip().endswith("OK"):
        raise ValueError("CPU config/spawn integration tests have not passed")
    owned = []
    if native_status["status"] == "running":
        controller = psutil.Process(native_status["controller_pid"])
        if str(baseline) not in controller.cmdline():
            raise ValueError("Native controller identity does not match run directory")
        owned = [identity(process.pid) for process in [controller] + controller.children(recursive=True)]
    manifest = {"created_at": now(), "baseline": str(baseline), "source_sha256": code_hashes(),
                "baseline_source_sha256": native_manifest["source_sha256"],
                "baseline_processes": owned, "protocol": native_manifest["protocol"],
                "dataset": native_manifest["dataset"], "environment": native_manifest["environment"],
                "protocol_inputs_sha256": file_hash(baseline / "protocol_inputs.json"),
                "cpu_tests_sha256": file_hash(root / "cpu_tests.log"),
                "cpu_integration_tests_sha256": file_hash(root / "cpu_integration_tests.log"),
                "offline_compile_sha256": file_hash(root / "offline_compile.json"),
                "order": ["native", "direct_gpu_validation", "direct_probe", "direct_formal",
                          "r64_gpu_validation", "r64_probe", "r64_formal", "comparison"],
                "gpu_policy": "Wait for native completion and owned process release; only CUDA_VISIBLE_DEVICES=4,5,6,7; no process termination",
                "failure_policy": "Exit on first failure; no automatic retry, protocol modification, or scoring repair"}
    for variant in ("direct", "r64"):
        destination = root / variant
        destination.mkdir()
        candidate = copy.deepcopy(native_manifest)
        candidate["source_sha256"].update(manifest["source_sha256"])
        candidate.update(created_at=now(), status="prepared", source_snapshot=str(root / "checkpoints" / variant),
                         source_native_snapshot=native_manifest["source_snapshot"], variant=variant,
                         baseline=str(baseline), validation_phase="probe")
        candidate["checks"] = {"cpu_format_tests": str(root / "cpu_tests.log"),
                               "offline_sm80_compile": str(root / "offline_compile.json"), "gpu_inference": "not_run"}
        candidate["protocol"]["quantization"] = f"routed experts {variant} MXFP4 E2M1/E8M0 group32 floor W4A4; non-experts unchanged"
        for phase in ("probe", "formal"):
            directory = destination / phase
            directory.mkdir()
            command = candidate["commands"][phase]
            command[2] = str(Path(__file__).with_name("entry.py"))
            for flag, path in (("--model_path", directory / "model"), ("--output_dir", directory / "evaluation")):
                command[command.index(flag) + 1] = str(path)
        # Remove stale source-only filenames and historical validation claims.
        for key in ("model_files", "model_small_file_sha256", "parent_run", "initialization_fix", "resolved_blocker"):
            candidate.pop(key, None)
        save(destination / "manifest.json", candidate)
    save(root / "workflow_manifest.json", manifest)
    return manifest


def wait_baseline(root, manifest, status):
    baseline = Path(manifest["baseline"])
    status.update(status="waiting_baseline", phase="native", started_at=now())
    save(root / "workflow_status.json", status)
    print("Waiting for the original baseline to complete and release its own workers", flush=True)
    while True:
        current = json.loads((baseline / "formal/status.json").read_text())
        if current["status"] == "completed":
            break
        if current["status"] != "running" or not any(alive(p) for p in manifest["baseline_processes"]):
            raise RuntimeError("Native baseline failed/exited without complete verified results")
        time.sleep(30)
    deadline = time.monotonic() + 120
    while any(alive(record) for record in manifest["baseline_processes"]):
        if time.monotonic() >= deadline:
            raise RuntimeError("Native status completed but owned worker processes did not release")
        time.sleep(5)
    status.update(status="running", phase="verify_native")
    save(root / "workflow_status.json", status)
    summarize_phase(baseline, "formal", baseline / "protocol_inputs.json", root / "native_summary.json")


def run_command(root, status, name, command, env):
    stage = {"name": name, "status": "running", "started_at": now(), "command": command,
             "log": str(root / f"{name}.log")}
    status["stages"].append(stage)
    status.update(status="running", phase=name)
    with Path(stage["log"]).open("x") as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        stage["pid"] = process.pid
        save(root / "workflow_status.json", status)
        stage["returncode"] = process.wait()
    stage["finished_at"] = now()
    stage["status"] = "completed" if stage["returncode"] == 0 else "failed"
    save(root / "workflow_status.json", status)
    if stage["returncode"]:
        raise RuntimeError(f"{name} failed; see {stage['log']}")


def make_model_views(root, variant, phases=("probe", "formal")):
    checkpoint = root / "checkpoints" / variant
    conversion = json.loads((checkpoint / "conversion.json").read_text())
    if conversion["status"] != "completed" or not (checkpoint / "model.safetensors.index.json").exists():
        raise ValueError(f"Unverified checkpoint: {variant}")
    conversion_sources = {name: digest for name, digest in conversion["source_code_sha256"].items()
                          if Path(name).name in ("convert.py", "format.py", "fht32.py", "artifacts.py", "r64_transform.py")}
    check_hashes(conversion_sources)
    for phase in phases:
        model = root / variant / phase / "model"
        model.mkdir()
        for path in checkpoint.iterdir():
            if path.is_file() and path.name != "conversion.json":
                (model / path.name).symlink_to(path)


def verify_worker_release(directory):
    workers = {int(path.stem.split("-")[1]) for path in Path(directory).glob("worker-*.jsonl")}
    deadline = time.monotonic() + 120
    while True:
        query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                               check=True, capture_output=True, text=True)
        gpu_pids = {int(line.strip()) for line in query.stdout.splitlines() if line.strip().isdigit()}
        if not workers & gpu_pids:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Completed phase still owns GPU processes: {sorted(workers & gpu_pids)}")
        time.sleep(5)


def run(root):
    root = Path(root).absolute()
    if (root / "workflow_status.json").exists():
        raise FileExistsError("One-shot workflow already started; inspect existing evidence")
    manifest = json.loads((root / "workflow_manifest.json").read_text())
    status = {"status": "starting", "pid": os.getpid(), "stages": [], "started_at": now()}
    torch.set_num_threads(8)
    try:
        check_hashes(manifest["source_sha256"])
        check_hashes(manifest["baseline_source_sha256"])
        os.environ.update(manifest["environment"])
        wait_baseline(root, manifest, status)
        while True:
            weights = json.loads((root / "weights_status.json").read_text())
            if weights["status"] == "completed":
                break
            if weights["status"] != "running" or not psutil.pid_exists(weights["pid"]):
                raise RuntimeError("Weight conversion failed/exited")
            status.update(status="waiting_weights", phase="cpu_conversion")
            save(root / "workflow_status.json", status)
            time.sleep(30)
        baseline = Path(manifest["baseline"])
        if file_hash(baseline / "protocol_inputs.json") != manifest["protocol_inputs_sha256"]:
            raise ValueError("Original protocol inputs changed")
        for variant in ("direct", "r64"):
            check_hashes(manifest["source_sha256"])
            check_hashes(manifest["baseline_source_sha256"])
            make_model_views(root, variant)
            name = f"{variant}_gpu_validation"
            command = [sys.executable, "-u", "-m", PACKAGE + ".validate_gpu", "--model",
                       str(root / "checkpoints" / variant), "--output", str(root / f"{name}.json")]
            run_command(root, status, name, command, dict(os.environ))
            if json.loads((root / f"{name}.json").read_text())["status"] != "passed":
                raise ValueError(f"Missing successful GPU evidence for {variant}")
            for phase in ("probe", "formal"):
                phase_name = f"{variant}_{phase}"
                status.update(status="running", phase=phase_name)
                stage = {"name": phase_name, "status": "running", "started_at": now(),
                         "run_root": str(root / variant)}
                status["stages"].append(stage)
                save(root / "workflow_status.json", status)
                os.environ["QWEN3_MXFP4_AUDIT_DIR"] = str(root / variant / phase / "audit")
                print(f"Starting {phase_name}", flush=True)
                run_evaluation(root / variant, phase)
                summarize_phase(root / variant, phase, baseline / "protocol_inputs.json",
                                root / (f"{variant}_summary.json" if phase == "formal" else f"{variant}_probe_summary.json"))
                verify_runtime_audit(root / "checkpoints" / variant, root / variant / phase / "audit",
                                     root / f"{variant}_{phase}_runtime_audit.json")
                verify_worker_release(root / variant / phase / "audit")
                stage.update(status="completed", finished_at=now())
                save(root / "workflow_status.json", status)
        comparison(root)
        status.update(status="completed", phase="comparison", finished_at=now())
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=now())
        if status["stages"] and status["stages"][-1]["status"] == "running":
            status["stages"][-1]["status"] = "failed"
        raise
    finally:
        save(root / "workflow_status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.baseline is None:
            parser.error("prepare requires --baseline")
        prepare(args.root, args.baseline)
    else:
        run(args.root)
