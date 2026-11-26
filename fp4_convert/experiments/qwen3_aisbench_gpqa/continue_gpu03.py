"""Finish the existing R16 run, then run G512 once on GPUs 0-3."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import psutil

from .common import PACKAGE, REPO, check_hashes, now, read, save


def wait_existing(identity):
    """A non-child exit code may be unknown; completion also needs artifacts."""
    try:
        process = psutil.Process(identity["pid"])
    except psutil.NoSuchProcess:
        return None
    if process.create_time() != identity["create_time"]:
        raise RuntimeError("The saved R16 PID was reused")
    if process.status() != psutil.STATUS_ZOMBIE:
        if [part for part in process.cmdline() if part] != identity["cmd"]:
            raise RuntimeError("The saved R16 process identity changed")
    return process.wait()


def completed(root, variant):
    phase = root / variant / "formal"
    status = read(phase / "status.json")
    summary = read(phase / "summary.json")
    runtime = read(phase / "runtime_validation.json")
    final = read(phase / "final_submission_summary.json")
    if (status["status"] != "completed" or status["returncode"] != 0
            or summary["status"] != "verified" or summary["generations"] != 1980
            or runtime["status"] != "passed" or final["generations"] != 1980):
        raise RuntimeError(f"{variant} is missing successful completion evidence")
    if not all(Path(path).is_file() for path in summary["raw_caches"]):
        raise FileNotFoundError(f"{variant} raw generation cache is missing")
    return final


def run(root, control):
    from ..qwen3_mxfp4.workflow import verify_worker_release
    from ..qwen3_mxfp4_rotation_eval.workflow import available

    root, control = Path(root).resolve(), Path(control).resolve()
    if control.parent != root:
        raise ValueError("The takeover record must belong to this run")
    target = control / "continuation_status.json"
    if target.exists():
        raise FileExistsError("This continuation has already started; no automatic retry")
    authorization = read(control / "authorization.json")
    if (authorization["allowed_devices"] != "0,1,2,3"
            or authorization["denied_devices"] != "4,5,6,7"
            or authorization["cancelled"] != ["dualscale_g256"]):
        raise ValueError("Unexpected resource or cancellation authorization")
    stop = read(control / "stop_record.json")
    if stop["status"] != "stopped":
        raise RuntimeError("The old controller has not been stopped")
    old = stop["controller"]
    if psutil.pid_exists(old["pid"]):
        process = psutil.Process(old["pid"])
        if process.create_time() == old["create_time"] and process.status() != psutil.STATUS_ZOMBIE:
            raise RuntimeError("The old controller still exists")
    check_hashes(read(root / "workflow_manifest.json")["source_sha256"])
    manifest_path = root / "dualscale_g512/manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != read(control / "placement_change.json")["after_sha256"]:
        raise ValueError("G512 placement manifest changed after validation")
    manifest = read(manifest_path)
    if (manifest["environment"]["CUDA_VISIBLE_DEVICES"] != "0,1,2,3"
            or manifest["protocol"]["cuda_visible_devices"] != "0,1,2,3"):
        raise ValueError("G512 must use only GPUs 0-3")
    state_path = root / "formal_workflow_status.json"
    state = read(state_path)
    state.update(status="running", pid=os.getpid(), continuation=str(control),
                 active_sequence=["r16x2", "dualscale_g512"], disabled_devices="4,5,6,7")
    local = {"status": "waiting_for_r16", "pid": os.getpid(), "started_at": now(),
             "devices": "0,1,2,3", "next_variant": "dualscale_g512"}
    save(state_path, state)
    save(target, local)
    active = "r16x2"
    try:
        identity = next(item for item in stop["protected_processes"]
                        if item["pid"] == state["stages"]["r16x2"]["pid"])
        print(f"Waiting for existing R16 PID {identity['pid']}; then G512 on GPU0-3", flush=True)
        observed_exit = wait_existing(identity)
        if observed_exit not in (None, 0):
            raise RuntimeError(f"R16 exited {observed_exit}")
        completed(root, "r16x2")
        verify_worker_release(root / "r16x2/formal/audit")
        state["stages"]["r16x2"].update(status="completed", exit_code=observed_exit,
                                               completion_verified_at=now(), completion_evidence="saved stage artifacts")
        save(state_path, state)
        active = "dualscale_g512"
        available("0,1,2,3")
        check_hashes(read(root / "workflow_manifest.json")["source_sha256"])
        command = [sys.executable, "-u", "-m", PACKAGE + ".workflow", "stage", "--root", str(root),
                   "--variant", active, "--phase", "formal"]
        log = root / active / "formal/controller.log"
        if (root / active / "formal/status.json").exists():
            raise FileExistsError("G512 has already started")
        record = {"status": "starting", "devices": "0,1,2,3", "started_at": now(),
                  "command": command, "log": str(log)}
        state["stages"][active] = record
        save(state_path, state)
        with log.open("x") as stream:
            proc = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"))
            record.update(status="running", pid=proc.pid)
            local.update(status="running_g512", stage_pid=proc.pid, command=command, log=str(log))
            save(state_path, state)
            save(target, local)
            code = proc.wait()
        record.update(exit_code=code, finished_at=now())
        if code:
            raise RuntimeError(f"G512 exited {code}; see {log}")
        completed(root, active)
        record["status"] = "completed"
        variants = ["native", "direct", "r64", "fht32", "r16x2", "dualscale_g512"]
        results = {variant: completed(root, variant) for variant in variants}
        save(root / "comparison.json", {
            "status": "awaiting_blind_review" if any(v["pending_review_count"] for v in results.values())
                      else "completed_with_cancellations",
            "results": results, "cancelled": ["dualscale_g256"], "stopped": ["direct_uos_c7p25"],
            "protocol": "AISBench prompt; original TP4 and BF16 KV protocol; remaining G512 on GPU0-3",
        })
        state["status"] = "completed_with_cancellations"
        local["status"] = "completed"
        print("R16 and G512 completed; no additional tasks scheduled", flush=True)
    except Exception as error:
        state.update(status="failed", error=str(error))
        state["stages"][active].update(status="failed", error=str(error), finished_at=now())
        local.update(status="failed", error=str(error))
        raise
    finally:
        local["finished_at"] = now()
        state["updated_at"] = now()
        save(target, local)
        save(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--control", required=True)
    args = parser.parse_args()
    run(args.root, args.control)
