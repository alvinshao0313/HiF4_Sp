"""Stop only the authorized previous G512 tree after measured KV benefit."""
import argparse
import os
from pathlib import Path
import signal
import subprocess

import psutil

from ..qwen3_aisbench_gpqa.common import now, read, save


def stop_previous(root):
    root = Path(root).resolve()
    target = root / "stop_record.json"
    if target.exists():
        raise FileExistsError(target)
    decision = read(root / "budget_decision.json")
    if not decision["increase"]:
        raise ValueError("No measured KV benefit; preserve the existing G512")
    auth = read(root / "authorization.json")
    old_root = Path(auth["old_g512_root"])
    old_state = read(old_root / "workflow_status.json")
    if old_state["status"] != "formal":
        raise ValueError("Previous G512 is not in its expected formal stage")
    controller = psutil.Process(old_state["pid"])
    if (controller.uids().real != os.getuid() or str(old_root) not in controller.cmdline()
            or "fp4_convert.experiments.qwen3_mxfp4_dualscale_tp2.workflow" not in controller.cmdline()):
        raise ValueError("Controller identity mismatch")
    processes = [controller] + controller.children(recursive=True)
    identities = [{"pid": p.pid, "created": p.create_time(), "command": p.cmdline(), "uid": p.uids().real}
                  for p in processes]
    if any(item["uid"] != os.getuid() for item in identities):
        raise ValueError("Process tree contains a different owner")
    gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    allowed = {line.split(",")[1].strip() for line in gpus.splitlines() if line.split(",")[0].strip() in ("0", "1")}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    affected = {int(line.split(",")[1]) for line in apps.splitlines() if line.split(",")[0].strip() in allowed}
    if not affected or not affected.issubset({p.pid for p in processes}):
        raise ValueError("GPU0-1 ownership differs from the authorized G512 tree")
    record = {"status": "stopping", "at": now(), "authorization": auth, "processes": identities,
              "old_workflow_status": old_state, "old_formal_status": read(old_root / "dualscale_g512/formal/status.json")}
    save(target, record)
    for p, identity in zip(processes, identities):
        try:
            if p.create_time() != identity["created"]:
                raise ValueError("PID was reused")
            p.send_signal(signal.SIGTERM)
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(processes, timeout=20)
    killed = []
    for p in alive:
        if p.status() != psutil.STATUS_ZOMBIE:
            p.kill()
            killed.append(p.pid)
    _, alive = psutil.wait_procs(alive, timeout=10)
    remaining = [p.pid for p in alive if p.status() != psutil.STATUS_ZOMBIE]
    record.update(status="stopped" if not remaining else "failed", finished_at=now(), killed=killed,
                  remaining=remaining, raw_outputs_preserved=True)
    save(target, record)
    if remaining:
        raise RuntimeError(f"Authorized processes still alive: {remaining}")
    old_state.update(status="superseded_by_authorized_kv_budget", replacement=str(root), stop_record=str(target))
    save(old_root / "workflow_status.json", old_state)
    formal = record["old_formal_status"]
    formal.update(status="stopped_by_user_for_kv_budget", stopped_at=now(), replacement=str(root / "dualscale_g512"))
    save(old_root / "dualscale_g512/formal/status.json", formal)
    print(f"Stopped only previous G512 tree: {[p.pid for p in processes]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    stop_previous(parser.parse_args().root)
