"""User-authorized GPU split: direct on 0-3 alongside native on 4-7."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

import psutil
import torch

from ..nvfp4_gpqa.run import run as run_evaluation
from .artifacts import PACKAGE, check_hashes, code_hashes, file_hash, save
from .evidence import comparison, summarize_phase, verify_runtime_audit
from .workflow import make_model_views, now, run_command, verify_worker_release, wait_baseline


def prepare(root, previous):
    root, previous = Path(root).absolute(), Path(previous).absolute()
    if (root / "workflow_manifest.json").exists():
        raise FileExistsError(root / "workflow_manifest.json")
    old = json.loads((previous / "workflow_manifest.json").read_text())
    check_hashes(old["source_sha256"])
    check_hashes(old["baseline_source_sha256"])
    old_status = json.loads((previous / "workflow_status.json").read_text())
    if old_status["status"] != "waiting_baseline" or old_status["stages"]:
        raise ValueError("Only the unused waiting continuation may be replaced")
    controller = psutil.Process(old_status["pid"])
    if controller.children(recursive=True) or str(previous) not in controller.cmdline():
        raise ValueError("Old controller identity/child state changed")
    manifest = copy.deepcopy(old)
    manifest.update(created_at=now(), previous_run=str(previous), source_sha256=code_hashes(),
                    previous_controller={"pid": controller.pid, "created": controller.create_time(), "command": controller.cmdline()},
                    order=["direct_gpu_validation", "direct_probe", "direct_formal", "wait_native",
                           "r64_gpu_validation", "r64_probe", "r64_formal", "comparison"],
                    gpu_policy="Direct: GPU0-3. Native remains GPU4-7. R64 uses GPU4-7 after native and direct complete.",
                    placement_authorization="User: 用0-3先测试直转的吧")
    (root / "checkpoints").symlink_to(previous / "checkpoints", target_is_directory=True)
    for variant, devices in (("direct", "0,1,2,3"), ("r64", "4,5,6,7")):
        destination = root / variant
        destination.mkdir()
        candidate = json.loads((previous / variant / "manifest.json").read_text())
        candidate["source_sha256"].update(manifest["source_sha256"])
        candidate.update(created_at=now(), status="prepared", previous_run=str(previous))
        candidate["environment"]["CUDA_VISIBLE_DEVICES"] = devices
        candidate["protocol"]["cuda_visible_devices"] = devices
        for phase in ("probe", "formal"):
            directory = destination / phase
            directory.mkdir()
            command = candidate["commands"][phase]
            for flag, path in (("--model_path", directory / "model"), ("--output_dir", directory / "evaluation")):
                command[command.index(flag)+1] = str(path)
        save(destination / "manifest.json", candidate)
    save(root / "workflow_manifest.json", manifest)


def candidate(root, manifest, status, variant):
    check_hashes(manifest["source_sha256"])
    check_hashes(manifest["baseline_source_sha256"])
    reused = manifest.get("validated_direct_probe") if variant == "direct" else None
    phases = ("formal",) if reused else ("probe", "formal")
    make_model_views(root, variant, phases)
    environment = json.loads((root / variant / "manifest.json").read_text())["environment"]
    os.environ.update(environment)
    if variant == "r64":
        command = [sys.executable, "-u", "-m", PACKAGE + ".validate_gpu", "--model",
                   str(root / "checkpoints" / variant), "--output", str(root / "r64_gpu_validation.json")]
        run_command(root, status, "r64_gpu_validation", command, dict(os.environ))
    numerical_path = Path(manifest.get("gpu_validation_reports", {}).get(variant, root / f"{variant}_gpu_validation.json"))
    numerical = json.loads(numerical_path.read_text())
    if numerical["status"] != "passed" or numerical["variant"] != variant:
        raise ValueError(f"Missing successful GPU validation for {variant}")
    # Reporting/controller fixes do not change previously validated operators.
    # Every inference and numerical-check dependency must retain its exact hash.
    reporting = {"evidence.py", "workflow.py", "parallel_workflow.py"}
    check_hashes({name: digest for name, digest in numerical["source_sha256"].items()
                  if Path(name).name not in reporting})
    baseline = Path(manifest["baseline"])
    if file_hash(baseline / "protocol_inputs.json") != manifest["protocol_inputs_sha256"]:
        raise ValueError("Original protocol inputs changed")
    if reused:
        check_hashes(reused["inference_source_sha256"])
        for name in ("summary", "runtime_audit"):
            if file_hash(reused[name]) != reused[name + "_sha256"]:
                raise ValueError(f"Verified direct probe evidence changed: {name}")
        summary = json.loads(Path(reused["summary"]).read_text())
        audit = json.loads(Path(reused["runtime_audit"]).read_text())
        if summary["status"] != "verified" or summary["generations"] != 260 or audit["status"] != "passed":
            raise ValueError("Direct probe reuse requires complete verified execution and TP4 loading")
        status["stages"].append({"name": "direct_probe", "status": "reused", "run_root": reused["run_root"],
                                 "reason": "Only post-run reporting/controller changed; inference hashes unchanged"})
    for phase in phases:
        stage = {"name": f"{variant}_{phase}", "status": "running", "started_at": now(),
                 "run_root": str(root / variant), "devices": environment["CUDA_VISIBLE_DEVICES"]}
        status["stages"].append(stage)
        status.update(status="running", phase=stage["name"])
        save(root / "workflow_status.json", status)
        os.environ["QWEN3_MXFP4_AUDIT_DIR"] = str(root / variant / phase / "audit")
        print(f"Starting {stage['name']} on GPU {stage['devices']}", flush=True)
        run_evaluation(root / variant, phase)
        summarize_phase(root / variant, phase, baseline / "protocol_inputs.json",
                        root / (f"{variant}_summary.json" if phase == "formal" else f"{variant}_probe_summary.json"))
        verify_runtime_audit(root / "checkpoints" / variant, root / variant / phase / "audit",
                             root / f"{variant}_{phase}_runtime_audit.json")
        verify_worker_release(root / variant / phase / "audit")
        stage.update(status="completed", finished_at=now())
        save(root / "workflow_status.json", status)


def run(root):
    root = Path(root).absolute()
    if (root / "workflow_status.json").exists():
        raise FileExistsError("One-shot continuation already started")
    manifest = json.loads((root / "workflow_manifest.json").read_text())
    old = manifest["previous_controller"]
    if psutil.pid_exists(old["pid"]):
        process = psutil.Process(old["pid"])
        if process.create_time() == old["created"] and process.status() != psutil.STATUS_ZOMBIE:
            raise RuntimeError("Previous controller must be explicitly retired before takeover")
    status = {"status": "starting", "pid": os.getpid(), "started_at": now(), "stages": []}
    torch.set_num_threads(8)
    try:
        candidate(root, manifest, status, "direct")
        wait_baseline(root, manifest, status)
        candidate(root, manifest, status, "r64")
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.previous is None:
            parser.error("prepare requires --previous")
        prepare(args.root, args.previous)
    else:
        run(args.root)
