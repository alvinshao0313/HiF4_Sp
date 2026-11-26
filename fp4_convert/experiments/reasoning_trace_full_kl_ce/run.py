"""Finite qualified two-candidate pipeline on GPU 6/7, without automatic retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .artifacts import initialize, now, read_json, reuse_student_cache, verify, write_json
from .config import CANDIDATES, MODULE, REPO


def start(root, module, *arguments, gpus):
    argv = [sys.executable, "-u", "-m", MODULE + "." + module, "--run", str(root), *arguments]
    ledger = root / "commands.jsonl"
    number = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    logfile = root / "logs" / f"{number:03d}_{module}_{gpus.replace(',', '_')}.log"
    logfile.parent.mkdir(exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"}
    if module == "training":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with logfile.open("x") as log:
        child = subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
    record = {"time": now(), "command": argv, "pid": child.pid, "gpus": gpus, "log": str(logfile), "status": "started",
              "allocator": env.get("PYTORCH_CUDA_ALLOC_CONF")}
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


def join(root, jobs):
    errors = []
    for job in jobs:
        try:
            finish(root, job)
        except Exception as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError("; ".join(errors))


def resources(root, *, inference=False, training=False):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"], text=True)
    observed = {}
    for line in raw.splitlines():
        i, total, free = [int(v.strip()) for v in line.split(",")]
        if i not in (6, 7):
            continue
        observed[i] = {"total_MiB": total, "free_MiB": free}
        objective = "full_kl" if i == 6 else "hard_ce"
        q = root / "qualification/training" / objective / "manifest.json"
        needed = total * .9 + 1024 if inference else 24 * 1024
        if training and q.exists():
            needed = read_json(q)["peak_reserved"] / 2**20 + 2048
        if free < needed:
            raise RuntimeError(f"GPU {i} has {free} MiB free; needs {needed:.0f}; existing processes remain untouched")
    write_json({"time": now(), "gpus": observed}, root / "resources_latest.json")


def qualify(root):
    complete = root / "qualification/complete.json"
    if complete.exists():
        if not read_json(complete)["passed"] or read_json(complete)["source_hashes"] != read_json(root / "source_hashes.json"):
            raise ValueError("qualification source mismatch")
        return
    command(root, "validation", gpus="")
    resources(root, inference=True)
    command(root, "export", "--objective", "full_kl", "--qualification", "--zero-step", gpus="6")
    command(root, "qualify", "--objective", "full_kl", "--qualification", "--zero-step", gpus="6,7")
    if not (root / "cache/student/manifest.json").exists():
        reuse_student_cache(root)
    resources(root)
    jobs = []
    if not (root / "cache/teacher_hidden/manifest.json").exists():
        jobs.append(start(root, "cache", gpus="6"))
    jobs.append(start(root, "training", "--objective", "hard_ce", "--qualification", gpus="7"))
    join(root, jobs)
    command(root, "training", "--objective", "full_kl", "--qualification", gpus="6")
    for objective in CANDIDATES:
        checkpoint = root / "qualification/training" / objective / "formal/last_updated.pt"
        resources(root, inference=True)
        command(root, "qualify", "--objective", objective, "--qualification", "--checkpoint", str(checkpoint), gpus="6,7")
    write_json({"passed": True, "time": now(), "source_hashes": read_json(root / "source_hashes.json")}, complete)


def execute(root, stage):
    root, m = verify(root)
    m.update(status="running", stage=stage, pid=os.getpid(), training_gpus={"full_kl": 6, "hard_ce": 7}, inference_gpus=[6, 7])
    m.pop("finished_at", None)
    write_json(m, root / "manifest.json")
    try:
        if stage in ("qualify", "experiment"):
            qualify(root)
        if stage in ("train", "experiment"):
            q = read_json(root / "qualification/complete.json")
            if not q["passed"] or q["source_hashes"] != read_json(root / "source_hashes.json"):
                raise ValueError("applicable completed qualification is required")
            resources(root, training=True)
            join(root, [start(root, "training", "--objective", objective, gpus=gpu)
                        for objective, gpu in (("full_kl", "6"), ("hard_ce", "7"))])
        if stage in ("development", "experiment"):
            for objective in CANDIDATES:
                resources(root, inference=True)
                command(root, "development", "--objective", objective, gpus="6,7")
                resources(root, inference=True)
                command(root, "qualify", "--objective", objective, gpus="6,7")
        if stage in ("evaluate", "experiment"):
            pending = []
            for objective, candidate in CANDIDATES.items():
                selection = read_json(root / "development" / objective / "selection.json")
                if not selection["complete"]:
                    raise ValueError("development/checkpoint decision must be fixed before formal tests")
                for task in ("arc", "mmlu_pro", "lcb", "gsm8k", "aime25"):
                    resources(root, inference=True)
                    code = command(root, "evaluation", "--objective", objective, "--task", task, gpus="6,7")
                    if code == 21:
                        pending.append(str(root / "evaluation" / candidate / task / "corrected_responses.json"))
            if pending:
                m.update(status="awaiting_aime_review", pending_reviews=pending, finished_at=now())
                write_json(m, root / "manifest.json")
                return
        if stage in ("report", "experiment"):
            command(root, "report", gpus="")
        m.update(status="complete" if stage in ("report", "experiment") else "stage_complete", finished_at=now())
    except Exception as error:
        m.update(status="stopped", reason=str(error), stopped_at=now())
        write_json(m, root / "manifest.json")
        raise
    write_json(m, root / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--cache-run", type=Path)
    parser.add_argument("--selection-policy", choices=("validation", "development"), default="validation")
    parser.add_argument("--stage", choices=("prepare", "qualify", "train", "development", "evaluate", "report", "experiment"), default="experiment")
    args = parser.parse_args()
    root = args.run.resolve()
    if not root.exists():
        if args.source_run is None or args.cache_run is None:
            parser.error("a new RUN requires --source-run and --cache-run")
        initialize(root, args.source_run, args.cache_run, args.selection_policy)
    else:
        _, m = verify(root)
        if ((args.source_run is not None and str(args.source_run.resolve()) != m["source_run"])
                or (args.cache_run is not None and str(args.cache_run.resolve()) != m["cache_run"])
                or args.selection_policy != m["selection_policy"]):
            parser.error("entry arguments differ from the fixed RUN")
    if args.stage != "prepare":
        execute(root, args.stage)
