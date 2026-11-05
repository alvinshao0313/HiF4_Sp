"""Sequential, detached-session-friendly execution; no resource waiting or retries."""
from __future__ import annotations

import argparse
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from .artifacts import now, read_json, sha256, verify_inputs, verify_sources, write_json
from .config import MODULE, REPO, TASKS
from .prepare import prepare


class GPUsUnavailable(RuntimeError):
    pass


def parse_gpus(value):
    try:
        gpus = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--gpus requires comma-separated GPU indices") from error
    if len(gpus) != 8 or set(gpus) != set(range(8)):
        raise ValueError("the approved TP8 allocation is GPU 0-7, each exactly once")
    return gpus


def check_gpus(gpus):
    def query(argument):
        result = subprocess.run(["nvidia-smi", argument, "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    devices = {int(index): (uuid.strip(), int(memory)) for index, uuid, memory in query("--query-gpu=index,uuid,memory.used")}
    processes = query("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory")
    busy = []
    for index in gpus:
        if index not in devices:
            raise GPUsUnavailable(f"GPU {index} is absent")
        uuid, memory = devices[index]
        jobs = [row for row in processes if row[0].strip() == uuid]
        if jobs or memory > 256:
            busy.append({"gpu": index, "used_mib": memory, "processes": jobs})
    if busy:
        raise GPUsUnavailable(f"required GPUs are occupied: {busy}")


def stages(root, gpus):
    def evaluation(candidate, smoke):
        for task in TASKS:
            name = f"{candidate}_{task}_" + ("qualification" if smoke else "formal")
            command = [sys.executable, "-u", "-m", MODULE + ".evaluate", "--run", str(root),
                       "--candidate", candidate, "--task", task]
            if smoke:
                command.append("--smoke")
            folder = "qualification/evaluation" if smoke else "evaluation"
            artifacts = [f"{folder}/{candidate}/{task}/{filename}" for filename in
                         ("metrics.json", "verified_scores.json", "raw_cache_validation.json")]
            yield name, command, gpus, artifacts, task == "aime25"

    yield "generation_qualification", [sys.executable, "-u", "-m", MODULE + ".generate", "--run", str(root), "--smoke"], gpus, ["qualification/generation/manifest.json"], False
    yield from evaluation("native_nvfp4", True)
    yield "generation", [sys.executable, "-u", "-m", MODULE + ".generate", "--run", str(root)], gpus, ["generation/manifest.json"], False
    yield from evaluation("native_nvfp4", False)
    for qualification in (True, False):
        name = "training_qualification" if qualification else "training"
        command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                   "--module", MODULE + ".training", "--run", str(root)]
        if qualification:
            command.append("--qualification")
        artifact = "qualification/training/manifest.json" if qualification else "training/manifest.json"
        yield name, command, gpus[:4], [artifact], False
    yield "export", [sys.executable, "-u", "-m", MODULE + ".export", "--run", str(root)], gpus[:1], ["models/matrix_norm/export.json"], False
    yield from evaluation("matrix_norm", True)
    yield from evaluation("matrix_norm", False)


def run(root, gpus, *, stage="all"):
    root = prepare(root)
    verify_sources(root)
    if stage == "prepare":
        return
    if stage == "report":
        from .report import report
        with (root / "runner.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = report(root)
            state_path = root / "status.json"
            state = read_json(state_path) if state_path.is_file() else {"stages": {}}
            state.update(status="complete" if result["complete"] else "awaiting_aime_review",
                         updated_at=now(), report_finished_at=now())
            state.pop("error", None)
            write_json(state, state_path)
        return
    with (root / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "status.json"
        state = read_json(state_path) if state_path.is_file() else {"stages": {}, "created_at": now()}
        state.update(status="running", pid=os.getpid(), started_at=now(), gpus=list(gpus))
        state.pop("error", None)
        write_json(state, state_path)
        try:
            for name, command, devices, artifacts, review_allowed in stages(root, gpus):
                previous = state["stages"].get(name)
                if previous and previous["status"] in {"complete", "awaiting_aime_review"}:
                    for filename, expected in previous["artifacts_sha256"].items():
                        if sha256(root / filename) != expected:
                            raise ValueError(f"completed stage artifact changed: {filename}")
                    continue
                verify_sources(root)
                verify_inputs(root)
                check_gpus(devices)
                logfile = root / "logs" / f"{name}.log"
                logfile.parent.mkdir(exist_ok=True)
                # Failed or partial stages are retained, never silently retried.
                if logfile.exists():
                    raise FileExistsError(f"stage already has a log; inspect it before an explicit new RUN: {logfile}")
                row = {"status": "running", "command": command, "gpus": list(devices), "log": str(logfile), "started_at": now()}
                state["stages"][name] = row
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices)),
                       "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "8",
                       "HF_ENDPOINT": "https://huggingface.co", "PYTHONUNBUFFERED": "1"}
                with logfile.open("w", encoding="utf-8") as log:
                    child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    row["pid"] = child.pid
                    write_json(state, state_path)
                    code = child.wait()
                row.update(exit_code=code, finished_at=now())
                if code != 0 and not (code == 21 and review_allowed):
                    row["status"] = "failed"
                    raise RuntimeError(f"{name} failed with exit {code}: {logfile}")
                row.update(status="awaiting_aime_review" if code == 21 else "complete",
                           artifacts_sha256={p: sha256(root / p) for p in artifacts})
                write_json(state, state_path)
                print({"event": "stage_complete", "stage": name, "exit_code": code}, flush=True)
            from .report import report
            result = report(root)
            state["status"] = "complete" if result["complete"] else "awaiting_aime_review"
        except GPUsUnavailable as error:
            state.update(status="blocked_resources", error=str(error))
            raise
        except Exception as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            state["updated_at"] = now()
            write_json(state, state_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", type=parse_gpus)
    parser.add_argument("--stage", choices=("prepare", "all", "report"), default="all")
    args = parser.parse_args()
    run(args.run, args.gpus, stage=args.stage)
