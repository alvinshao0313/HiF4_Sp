"""Join this stage's vLLM children before handing its GPUs to the next stage."""
from __future__ import annotations

import json
import os
import subprocess
import time

import psutil


def gpu_snapshot(pair):
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                                     "--format=csv,noheader,nounits"], text=True).splitlines()
    devices = {}
    for line in lines:
        index, uuid, memory = [part.strip() for part in line.split(",")]
        if index in pair.split(","):
            devices[index] = {"uuid": uuid, "memory_mib": int(memory), "pids": []}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader,nounits"], text=True)
    for line in apps.splitlines():
        if not line.strip():
            continue
        uuid, pid = [part.strip() for part in line.split(",")]
        for device in devices.values():
            if device["uuid"] == uuid:
                device["pids"].append(int(pid))
    return devices


def require_free(pair):
    state = gpu_snapshot(pair)
    for index in pair.split(","):
        device = state.get(index)
        if device is None or device["pids"] or device["memory_mib"] > 1024:
            raise RuntimeError(f"GPU {index} is unavailable; no process will be stopped or shared; "
                               + json.dumps(state, sort_keys=True))


def join_session(session_id, timeout=60):
    """Wait only for this Popen(start_new_session=True) session; never signal it."""
    started, owned = time.monotonic(), {session_id}
    while True:
        members = []
        for process in psutil.process_iter(["pid", "status"]):
            try:
                if os.getsid(process.pid) == session_id:
                    owned.add(process.pid)
                    if process.info["status"] != psutil.STATUS_ZOMBIE:
                        members.append(process)
            except (OSError, psutil.Error):
                continue
        if not members:
            return {"session_id": session_id, "owned_pids": sorted(owned),
                    "child_wait_seconds": time.monotonic() - started}
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError(f"stage children did not exit within {timeout}s: {[p.pid for p in members]}")
        # psutil is supported by this Python build; os.pidfd_open is unavailable.
        psutil.wait_procs(members, timeout=min(0.2, remaining))


def finish_stage(session_id, pair, timeout=30):
    result = join_session(session_id)
    started = time.monotonic()
    while True:
        state = gpu_snapshot(pair)
        foreign = [pid for device in state.values() for pid in device["pids"]
                   if pid not in result["owned_pids"] and psutil.pid_exists(pid)]
        if foreign:
            # A newly occupied GPU is not ours to wait for or release. The
            # completed calculation remains valid; the next stage fails safely.
            return {**result, "gpu_state": state, "gpu_available": False,
                    "foreign_pids": foreign, "release_wait_seconds": time.monotonic() - started}
        if len(state) == 2 and all(not d["pids"] and d["memory_mib"] <= 1024 for d in state.values()):
            return {**result, "gpu_state": state, "gpu_available": True,
                    "release_wait_seconds": time.monotonic() - started}
        if time.monotonic() - started >= timeout:
            raise RuntimeError(f"stage GPU contexts did not clear within {timeout}s: {state}")
        time.sleep(0.2)
