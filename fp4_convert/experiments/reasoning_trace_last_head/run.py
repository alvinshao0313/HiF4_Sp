"""One finite experiment pipeline; no retries, reservation jobs, or monitoring daemon."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .artifacts import initialize, now, read_json, verify, write_json
from .config import CANDIDATE, MODULE, REPO


def start(root, module, *arguments, gpus):
    argv = [sys.executable, "-u", "-m", MODULE + "." + module, "--run", str(root), *arguments]
    ledger = root / "commands.jsonl"
    number = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    logfile = root / "logs" / f"{number:03d}_{module}_{gpus.replace(',', '_')}.log"
    logfile.parent.mkdir(exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"}
    with logfile.open("x") as log:
        child = subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
    record = {"time": now(), "command": argv, "pid": child.pid, "gpus": gpus, "log": str(logfile), "status": "started"}
    with ledger.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    return child, record


def finish(root, job):
    child, record = job
    code = child.wait()
    with (root / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps({**record, "time": now(), "status": "exited", "exit_code": code}) + "\n")
    if code and not ("aime25" in record["command"] and code == 21):
        raise RuntimeError(f"child exited {code}: {record['log']}")
    return code


def command(root, module, *arguments, gpus):
    return finish(root, start(root, module, *arguments, gpus=gpus))


def resources(root, *, inference=False):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"], text=True)
    observed = {}
    for line in raw.splitlines():
        i, total, free = [int(x.strip()) for x in line.split(",")]
        if i in (6, 7):
            observed[i] = {"total_MiB": total, "free_MiB": free}
    write_json({"time": now(), "gpus": observed}, root / "resources_latest.json")
    for i, row in observed.items():
        # vLLM reserves 90% of each GPU; training's measured peak is recorded after qualification.
        q = root / "qualification/training/manifest.json"
        needed = (row["total_MiB"] * .9 + 1024 if inference else
                  read_json(q)["peak_reserved"] / 2**20 + 2048 if q.exists() and i == 6 else 24 * 1024)
        if row["free_MiB"] < needed:
            raise RuntimeError(f"GPU {i} needs {needed:.0f} MiB free; has {row['free_MiB']}; existing jobs are untouched")


def qualify(root):
    complete = root / "qualification/complete.json"
    if complete.exists():
        if read_json(complete)["source_hashes"] != read_json(root / "source_hashes.json"):
            raise ValueError("qualification sources changed")
        return
    command(root, "validation", gpus="")
    command(root, "export", "--qualification", "--zero-step", gpus="6")
    resources(root, inference=True)
    command(root, "qualify", "--qualification", "--zero-step", gpus="6,7")
    resources(root)
    pending = []
    for role, gpu in (("teacher", "6"), ("student", "7")):
        path = root / "cache" / role / "manifest.json"
        if not path.exists():
            pending.append(start(root, "cache", "--role", role, gpus=gpu))
    errors = []
    for job in pending:
        try:
            finish(root, job)
        except Exception as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError("; ".join(errors))
    command(root, "training", "--qualification", gpus="6")
    command(root, "export", "--qualification", gpus="6")
    resources(root, inference=True)
    command(root, "qualify", "--qualification", gpus="6,7")
    write_json({"passed": True, "time": now(), "source_hashes": read_json(root / "source_hashes.json")}, complete)


def execute(root, stage):
    root, manifest = verify(root)
    manifest.update(status="running", stage=stage, pid=os.getpid(), train_gpu="6", inference_gpus="6,7")
    write_json(manifest, root / "manifest.json")
    try:
        if stage in ("qualify", "experiment"):
            qualify(root)
        if stage in ("train", "experiment"):
            q = read_json(root / "qualification/complete.json")
            if not q["passed"] or q["source_hashes"] != read_json(root / "source_hashes.json"):
                raise ValueError("applicable completed qualification is required")
            resources(root)
            command(root, "training", gpus="6")
        if stage in ("export", "experiment"):
            command(root, "export", gpus="6")
            resources(root, inference=True)
            command(root, "qualify", gpus="6,7")
        if stage in ("evaluate", "experiment"):
            pending = []
            for task in ("arc", "mmlu_pro", "lcb", "gsm8k", "aime25"):
                resources(root, inference=True)
                if command(root, "evaluation", "--task", task, gpus="6,7") == 21:
                    pending.append(str(root / "evaluation" / CANDIDATE / task / "corrected_responses.json"))
            if pending:
                manifest.update(status="awaiting_aime_review", pending_reviews=pending, finished_at=now())
                write_json(manifest, root / "manifest.json")
                return
        if stage in ("report", "experiment"):
            command(root, "report", gpus="")
        manifest.update(status="complete" if stage in ("report", "experiment") else "stage_complete", finished_at=now())
    except Exception as error:
        manifest.update(status="stopped", reason=str(error), stopped_at=now())
        write_json(manifest, root / "manifest.json")
        raise
    write_json(manifest, root / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--train-gpu", choices=("6",), default="6")
    parser.add_argument("--inference-gpus", choices=("6,7",), default="6,7")
    parser.add_argument("--stage", choices=("prepare", "qualify", "train", "export", "evaluate", "report", "experiment"), default="experiment")
    args = parser.parse_args()
    root = args.run.resolve()
    if not root.exists():
        if args.source_run is None:
            parser.error("--source-run is required for a new RUN")
        initialize(root, args.source_run)
    elif args.source_run is not None and read_json(root / "manifest.json")["source_run"] != str(args.source_run.resolve()):
        parser.error("--source-run differs from the fixed initialization")
    if args.stage != "prepare":
        execute(root, args.stage)
