"""Repeat the qualified LCB/AIME Ban runs without replacing their first round."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading

from .additional_main import install_datasets
from .additional_report import TASKS, summarize
from .artifact import load_artifact, model_identity, sha256, write_json
from .evaluate import REPO, code_identity, gpu_pair

NAMES = ("lcb", "aime25")


def now():
    return datetime.now(timezone.utc).isoformat()


def repeat_command(original, target, dataset_manifest):
    command = list(original)
    command[0] = sys.executable
    for flag, value in (("--dataset_manifest", dataset_manifest),
                        ("--output_dir", Path(target) / "evaluation"),
                        ("--ban_stats_dir", Path(target) / "routing")):
        command[command.index(flag) + 1] = str(value)
    if "--max_samples" in command:
        raise ValueError("repeat source must be a full LCB/AIME evaluation")
    return command


def pool_scores(rounds):
    if not rounds or len({row["round"] for row in rounds}) != len(rounds):
        raise ValueError("repeat aggregation requires unique, completed rounds")
    values = [row["score"] for row in rounds]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("invalid repeat score")
    return {"rounds": sorted(rounds, key=lambda row: row["round"]), "completed_rounds": len(values),
            "mean_score": math.fsum(values) / len(values), "min_score": min(values),
            "max_score": max(values), "population_std": statistics.pstdev(values),
            "aggregation": "All completed rounds, including the original; same questions and protocol, not independent test sets."}


class Repeats:
    def __init__(self, args):
        self.args, self.root = args, Path(args.output).resolve()
        self.source = Path(args.source).resolve(strict=True)
        self.lock = threading.Lock()
        self.state = {"status": "preparing", "controller_pid": os.getpid(), "started_at": now(),
                      "phases": {}, "errors": []}

    def save(self):
        write_json(self.root / "status.json", self.state)

    def prepare(self):
        args = self.args
        if args.additional_rounds < 1:
            raise ValueError("additional_rounds must be positive")
        devices = [device for pair in args.gpu_pairs for device in pair.split(",")]
        if len(set(devices)) != len(devices):
            raise ValueError("GPU pairs must be disjoint")
        if self.root.exists():
            raise FileExistsError(f"repeat output must be new: {self.root}")
        status = json.loads((self.source / "status.json").read_text())
        manifest = json.loads((self.source / "manifest.json").read_text())
        for name in NAMES:
            for stage in ("short", "formal"):
                phase = status["phases"][f"{name}/{stage}"]
                if phase["status"] != "complete" or phase["exit_code"] != 0:
                    raise ValueError(f"source qualification is incomplete: {name}/{stage}")
        changed = [path for path, digest in manifest["code"]["source_sha256"].items()
                   if sha256(REPO / path) != digest]
        if changed:
            raise ValueError(f"qualified source changed; reassess applicable validation: {changed}")
        model = Path(manifest["model_path"])
        load_artifact(manifest["artifact_path"], model_identity(model, model / "hif4_runtime_spec.pt"))
        if sha256(manifest["artifact_path"]) != manifest["artifact_sha256"]:
            raise ValueError("source calibration changed")
        install_datasets(self.source / "datasets.json")
        self.audit = {name: json.loads((self.source / f"{name}_input_audit.json").read_text()) for name in NAMES}
        self.original = {name: summarize(self.source / name / "formal", name, self.audit[name]) for name in NAMES}
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "logs").mkdir()
        (self.root / "datasets.json").write_bytes((self.source / "datasets.json").read_bytes())
        commands = json.loads((self.source / "commands.json").read_text())
        self.jobs = []
        for round_index in range(2, args.additional_rounds + 2):
            for name in NAMES:
                target = self.root / f"round_{round_index:02d}" / name
                self.jobs.append({"round": round_index, "name": name, "target": str(target),
                                  "command": repeat_command(commands[name]["formal"], target, self.root / "datasets.json")})
        write_json(self.root / "commands.json", self.jobs)
        write_json(self.root / "manifest.json", {"created_at": now(), "command": sys.argv,
                   "source": str(self.source), "source_manifest_sha256": sha256(self.source / "manifest.json"),
                   "additional_rounds_per_task": args.additional_rounds, "total_rounds_including_original": args.additional_rounds + 1,
                   "gpu_pairs": args.gpu_pairs, "code": code_identity(), "seed": 1234,
                   "repeat_policy": "Fresh processes and prediction caches; all source inference and scoring settings unchanged.",
                   "qualification": "Reuses the identical source code/model/artifact and each task's completed short and full evaluation.",
                   "original_reports": {name: str(self.source / name / "formal/report.json") for name in NAMES}})
        self.reports = {name: [{"round": 1, "score": report["score"],
                               "report": str(self.source / name / "formal/report.json")}]
                        for name, report in self.original.items()}
        self.state["status"] = "running"
        self.save()

    def run_job(self, job, pair):
        label = f"round_{job['round']:02d}/{job['name']}"
        target = Path(job["target"])
        target.mkdir(parents=True, exist_ok=False)
        log_path = self.root / "logs" / f"round_{job['round']:02d}_{job['name']}.log"
        record = {"status": "starting", "gpu_pair": pair, "started_at": now(),
                  "command": job["command"], "log": str(log_path)}
        with self.lock:
            self.state["phases"][label] = record
            self.save()
        try:
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=pair, OMP_NUM_THREADS="4",
                               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
            with log_path.open("w", encoding="utf-8") as log:
                child = subprocess.Popen(job["command"], cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                with self.lock:
                    record.update(status="running", pid=child.pid)
                    self.save()
                code = child.wait()
            if code:
                raise RuntimeError(f"{label} exited {code}; see {log_path}")
            result = summarize(target, job["name"], self.audit[job["name"]])
            original = self.original[job["name"]]
            for key in ("actual_input_tokens_hash", "count", "generations_per_question", "metric"):
                if result[key] != original[key]:
                    raise ValueError(f"repeat differs from original {key}")
            protocol = lambda row: {key: value for key, value in row["protocol"].items() if key != "model_name"}
            if protocol(result) != protocol(original):
                raise ValueError("repeat inference protocol differs from original")
            write_json(target / "report.json", result)
            with self.lock:
                self.reports[job["name"]].append({"round": job["round"], "score": result["score"],
                                                  "report": str(target / "report.json")})
                record.update(status="complete", exit_code=0, report=str(target / "report.json"))
        except Exception as error:
            with self.lock:
                record.update(status="failed", exit_code=child.returncode if "child" in locals() else None,
                              error=str(error))
                self.state["errors"].append({"phase": label, "error": str(error)})
        finally:
            with self.lock:
                record["ended_at"] = now()
                self.save()

    def lane(self, index):
        for job in self.jobs[index::len(self.args.gpu_pairs)]:
            self.run_job(job, self.args.gpu_pairs[index])

    def run(self):
        self.prepare()
        with ThreadPoolExecutor(max_workers=len(self.args.gpu_pairs)) as pool:
            list(pool.map(self.lane, range(len(self.args.gpu_pairs))))
        self.state.update(status="failed" if self.state["errors"] else "complete", ended_at=now())
        self.save()
        write_json(self.root / "report.json", {"status": self.state["status"], "errors": self.state["errors"],
                   "requested_total_rounds": self.args.additional_rounds + 1,
                   "tasks": {name: pool_scores(rows) for name, rows in self.reports.items()}})
        return 1 if self.state["errors"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--additional_rounds", required=True, type=int)
    parser.add_argument("--gpu_pairs", required=True, nargs="+", type=gpu_pair)
    return Repeats(parser.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
