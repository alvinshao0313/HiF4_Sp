"""Qualified sequential background execution, without retries or monitoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .artifacts import initialize, now, read_json, verify, write_json
from .config import CANDIDATE, MODULE, REPO


def command(root, module, *arguments, gpus):
    argv = [sys.executable, "-u", "-m", MODULE + "." + module, "--run", str(root), *arguments]
    ledger = root / "commands.jsonl"
    number = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    logfile = root / "logs" / f"{number:03d}_{module}.log"
    logfile.parent.mkdir(exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
           "TOKENIZERS_PARALLELISM": "false"}
    with logfile.open("x") as log:
        child = subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
    record = {"time": now(), "command": argv, "pid": child.pid, "gpus": gpus, "log": str(logfile), "status": "started"}
    with ledger.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    code = child.wait()
    with ledger.open("a") as stream:
        stream.write(json.dumps({**record, "time": now(), "status": "exited", "exit_code": code}) + "\n")
    if code and not (module == "evaluation" and "aime25" in arguments and code == 21):
        raise RuntimeError(f"{module} exited {code}: {logfile}")
    return code


def qualify(root, train_gpu, inference_gpus):
    path = root / "qualification/complete.json"
    if path.exists():
        prior = read_json(path)
        if not prior["passed"] or prior["source_hashes"] != read_json(root / "source_hashes.json"):
            raise ValueError("qualification evidence is not applicable")
        return
    command(root, "tests", gpus="")
    command(root, "training", "--qualification", gpus=train_gpu)
    for extra in (("--zero-step",), ()):
        command(root, "export", "--qualification", *extra, gpus="")
        command(root, "qualify", "--qualification", *extra, gpus=inference_gpus)
    write_json({"passed": True, "time": now(), "source_hashes": read_json(root / "source_hashes.json")}, path)


def execute(root, stage, train_gpu, inference_gpus):
    root, manifest = verify(root)
    if "," in train_gpu or len(set(inference_gpus.split(","))) != 2:
        raise ValueError("training requires one GPU; inference requires two distinct GPUs")
    manifest.update(status="running", stage=stage, pid=os.getpid(), train_gpu=train_gpu, inference_gpus=inference_gpus)
    write_json(manifest, root / "manifest.json")
    try:
        if stage in ("qualify", "experiment"):
            qualify(root, train_gpu, inference_gpus)
        if stage in ("train", "experiment"):
            q = read_json(root / "qualification/complete.json")
            if not q["passed"] or q["source_hashes"] != read_json(root / "source_hashes.json"):
                raise ValueError("qualified source and data are required")
            command(root, "training", gpus=train_gpu)
        if stage in ("export", "experiment"):
            command(root, "export", gpus="")
            command(root, "qualify", gpus=inference_gpus)
        if stage in ("evaluate", "experiment"):
            pending = []
            for task in ("arc", "mmlu_pro", "lcb", "gsm8k", "aime25"):
                code = command(root, "evaluation", "--task", task, gpus=inference_gpus)
                if code == 21:
                    pending.append(str(root / "evaluation" / CANDIDATE / task / "corrected_responses.json"))
            if pending:
                write_json({"pending": pending}, root / "evaluation/pending_reviews.json")
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
    parser.add_argument("--train-gpu", default="6")
    parser.add_argument("--inference-gpus", default="6,7")
    parser.add_argument("--stage", choices=("prepare", "qualify", "train", "export", "evaluate", "report", "experiment"), default="experiment")
    args = parser.parse_args()
    root = args.run.resolve()
    if not root.exists():
        if args.source_run is None:
            parser.error("--source-run is required for a new RUN")
        initialize(root, args.source_run)
    elif args.source_run is not None:
        if read_json(root / "manifest.json")["source_run"] != str(args.source_run.resolve()):
            parser.error("--source-run differs from the fixed initialization")
    if args.stage != "prepare":
        execute(root, args.stage, args.train_gpu, args.inference_gpus)
