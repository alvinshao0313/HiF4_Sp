"""Sequential, fail-fast experiment execution; no retry or polling daemon."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .artifacts import bind_sources, now, read_json, validate_root, write_json
from .config import MODELS, MODULE, REPO


def start_command(root, module, *arguments, gpus):
    argv = [sys.executable, "-u", "-m", MODULE + "." + module, "--run", str(root), *arguments]
    logdir = root / "logs"
    logdir.mkdir(exist_ok=True)
    ledger = root / "commands.jsonl"
    count = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    logfile = logdir / f"{count:03d}_{module}.log"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                   "TOKENIZERS_PARALLELISM": "false"}
    with logfile.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=environment)
        entry = {"time": now(), "command": argv, "pid": child.pid, "gpu_visibility": gpus,
                 "log": str(logfile), "status": "started"}
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
    return child, entry


def finish_command(root, running):
    child, entry = running
    code = child.wait()
    ledger = root / "commands.jsonl"
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({**entry, "time": now(), "status": "exited", "exit_code": code}) + "\n")
    return code


def command(root, module, *arguments, gpus, allow_aime_review=False):
    running = start_command(root, module, *arguments, gpus=gpus)
    code = finish_command(root, running)
    if code and not (allow_aime_review and code == 21):
        raise RuntimeError(f"{module} exited {code}; see {running[1]['log']}")
    return code


def train_pair(root, train_gpu, second_train_gpu):
    running = [start_command(root, "training", "--candidate", name, gpus=gpu)
               for name, gpu in zip(MODELS[:2], (train_gpu, second_train_gpu))]
    codes = [finish_command(root, item) for item in running]
    if any(codes):
        raise RuntimeError(f"A/B training exit codes {codes}; see {[item[1]['log'] for item in running]}")


def execute(root, *, stage, train_gpu, inference_gpus, second_train_gpu=None):
    root = validate_root(root)
    bind_sources(root)
    if len(inference_gpus.split(",")) != 2 or len(set(inference_gpus.split(","))) != 2:
        raise ValueError("inference requires two distinct explicitly selected GPUs")
    if len(train_gpu.split(",")) != 1:
        raise ValueError("layerwise training uses one explicitly selected GPU")
    if second_train_gpu is not None and ("," in second_train_gpu or second_train_gpu == train_gpu):
        raise ValueError("parallel A/B training requires two distinct single GPUs")
    # The caller chooses the available devices. No automatic occupation, retries,
    # or job cancellation is performed by this sequential runner.
    manifest = read_json(root / "manifest.json")
    manifest.update(status="running", stage=stage, pid=os.getpid(), train_gpu=train_gpu,
                    second_train_gpu=second_train_gpu, inference_gpus=inference_gpus)
    write_json(manifest, root / "manifest.json")
    pipeline = stage in ("all", "experiment", "continue")
    try:
        if stage in ("prepare", "all"):
            command(root, "prepare", gpus="")
        if stage in ("generate", "all", "experiment"):
            command(root, "generate", gpus=inference_gpus)
        if stage == "generate" or pipeline:
            from .prepare import require_isolation
            from .training import samples_for
            require_isolation(root)
            samples_for(root, MODELS[1])
            reference_manifest = root / "judging/reference_manifest.json"
            if not reference_manifest.exists():
                from .sources import prepare_references
                prepare_references(root)
            if not read_json(reference_manifest)["complete"]:
                raise ValueError("complete pinned code references are required")
            command(root, "judging", gpus="")
        if stage == "qualify" or pipeline:
            command(root, "tests", gpus="")
            for name in MODELS[:2]:
                path = root / "qualification" / name / "manifest.json"
                if path.exists():
                    from .training import samples_for
                    prior = read_json(path)
                    _, _, hashes = samples_for(root, name)
                    if (not prior["complete"] or not prior["qualification"] or prior["completed_layers"] != [0, 1]
                            or prior["calibration_sha256"] != hashes or prior["source_hashes"] != read_json(root / "source_hashes.json")):
                        raise ValueError("existing qualification is not applicable to the current data and code")
                    print(json.dumps({"event": "reuse_completed_qualification", "candidate": name, "manifest": str(path)}), flush=True)
                else:
                    command(root, "training", "--candidate", name, "--qualification", gpus=train_gpu)
            command(root, "router", "--qualification", "--candidate", MODELS[0], gpus=train_gpu)
            for name in (MODELS[0], MODELS[2]):
                command(root, "export", "--candidate", name, "--qualification", gpus=train_gpu)
                command(root, "qualify", "--candidate", name, "--qualification", gpus=inference_gpus)
            write_json({"passed": True, "source_hashes": read_json(root / "source_hashes.json")}, root / "qualification/complete.json")
            from .cleanup import diagnostic_shards
            diagnostic_shards(root)
        if stage == "train" or pipeline:
            qualification = read_json(root / "qualification/complete.json")
            if not qualification["passed"] or qualification["source_hashes"] != read_json(root / "source_hashes.json"):
                raise ValueError("qualified implementation is required before formal training")
            if second_train_gpu is not None:
                train_pair(root, train_gpu, second_train_gpu)
            else:
                for name in MODELS[:2]:
                    command(root, "training", "--candidate", name, gpus=train_gpu)
            for name in MODELS[:2]:
                command(root, "export", "--candidate", name, gpus=train_gpu)
                command(root, "qualify", "--candidate", name, gpus=inference_gpus)
        if stage == "development" or pipeline:
            for name in MODELS[:2]:
                command(root, "development", "--candidate", name, gpus=inference_gpus)
            command(root, "development", "--select", gpus="")
        if stage == "router" or pipeline:
            command(root, "router", gpus=train_gpu)
            command(root, "export", "--candidate", MODELS[2], gpus=train_gpu)
            command(root, "qualify", "--candidate", MODELS[2], gpus=inference_gpus)
        if stage == "evaluate" or pipeline:
            pending_reviews = []
            for name in MODELS:
                for task in ("arc", "mmlu_pro", "lcb", "gsm8k", "aime25"):
                    code = command(root, "evaluation", "--candidate", name, "--task", task,
                                   gpus=inference_gpus, allow_aime_review=task == "aime25")
                    if code == 21:
                        pending_reviews.append(str(root / "evaluation" / name / "aime25/corrected_responses.json"))
            if pending_reviews:
                write_json({"pending": pending_reviews}, root / "evaluation/pending_reviews.json")
                manifest.update(status="awaiting_aime_review", pending_reviews=pending_reviews, stopped_at=now())
                write_json(manifest, root / "manifest.json")
                return
            command(root, "report", gpus="")
    except Exception as error:
        manifest.update(status="stopped", stopped_at=now(), reason=str(error))
        write_json(manifest, root / "manifest.json")
        raise
    manifest.update(status="complete" if pipeline else "stage_complete", finished_at=now())
    write_json(manifest, root / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "qualify", "train", "development", "router", "evaluate", "all", "experiment", "continue"), required=True,
                        help="experiment runs all post-preparation stages sequentially from an already audited RUN")
    parser.add_argument("--train-gpu", required=True)
    parser.add_argument("--second-train-gpu", help="run B concurrently with A on this distinct single GPU")
    parser.add_argument("--inference-gpus", required=True)
    args = parser.parse_args()
    execute(args.run, stage=args.stage, train_gpu=args.train_gpu, inference_gpus=args.inference_gpus,
            second_train_gpu=args.second_train_gpu)
