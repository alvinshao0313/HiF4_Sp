"""Run one prepared main.py evaluation and record its exit status and coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from collections import Counter
from datetime import datetime, timezone


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def run(root: Path, phase: str) -> None:
    repo = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "manifest.json").read_text())
    phase_dir = root / phase
    status_path = phase_dir / "status.json"
    if status_path.exists():
        raise FileExistsError(f"Existing evaluation status: {status_path}")
    if phase == "formal":
        validation_phase = manifest.get("validation_phase", "smoke")
        validation = json.loads((root / validation_phase / "status.json").read_text())
        if validation["status"] != "completed":
            raise ValueError(f"GPU {validation_phase} has not completed successfully")
    for filename, expected in manifest["source_sha256"].items():
        if hashlib.sha256((repo / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared source changed: {filename}")
    dataset = Path(manifest["dataset"]["local_path"]) / "train.jsonl"
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != manifest["dataset"]["sha256"]:
        raise ValueError("Prepared GPQA data changed")
    env = dict(os.environ)
    env.update(manifest["environment"])
    status = {"phase": phase, "status": "running", "controller_pid": os.getpid(),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "command": manifest["commands"][phase], "log": str(phase_dir / "main.log")}
    try:
        with (phase_dir / "main.log").open("x") as log:
            proc = subprocess.Popen(status["command"], cwd=repo, env=env,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            status["pid"] = proc.pid
            write_json(status_path, status)
            write_json(root / "status.json", status)
            status["returncode"] = proc.wait()
        if status["returncode"]:
            raise RuntimeError(f"main.py returned {status['returncode']}; see {status['log']}")
        results = list((phase_dir / "evaluation").rglob("results_*.json"))
        details = list((phase_dir / "evaluation").rglob("details_*.json"))
        if len(results) != 1 or len(details) != 1:
            raise ValueError(f"Expected one result/details file, got {len(results)}/{len(details)}")
        payload = json.loads(results[0].read_text())
        records = json.loads(details[0].read_text())
        command = status["command"]
        expected_count = (int(command[command.index("--max_samples") + 1])
                          if "--max_samples" in command else manifest["dataset"]["expected_questions"])
        if len(records) != expected_count:
            raise ValueError(f"Expected {expected_count} questions, got {len(records)}")
        scores, parsing = [], Counter()
        for record in records:
            evidence = record["doc"]["specific"]["gpqa_scoring"]
            if len(evidence["scores"]) != 10:
                raise ValueError("Incomplete per-question generation scores")
            scores.extend(evidence["scores"])
            parsing.update(evidence["parse_status"])
        accuracy = sum(scores) / len(scores)
        saved = [values["gpqa_accuracy_avg10"] for values in payload["results"].values()
                 if "gpqa_accuracy_avg10" in values]
        if not saved or not all(math.isclose(score, accuracy, abs_tol=1e-12) for score in saved):
            raise ValueError("Per-generation scores disagree with the saved accuracy")
        caches = list((phase_dir / "model").rglob("GENERATIVE.parquet"))
        if not caches:
            raise FileNotFoundError("Missing raw generation cache")
        if phase == "probe":
            import pyarrow.parquet as pq

            cached = [row for cache in caches for row in pq.read_table(cache).to_pylist()]
            lengths = [len(tokens) for row in cached for tokens in row["sample"]["output_tokens"]]
            limit = int(command[command.index("--max_new_tokens") + 1])
            if len(cached) != expected_count or len(lengths) != len(scores):
                raise ValueError("Probe raw cache coverage disagrees with scoring details")
            if not all(0 < size <= limit for size in lengths) or max(lengths) != limit:
                raise ValueError("Probe did not cover the configured decode steps")
            status.update(probe_generated_tokens=sum(lengths), probe_max_decode_steps=max(lengths),
                          raw_cache_coverage_verified=True)
        status.update(status="completed", completed_questions=len(records),
                      completed_generations=len(scores), correct_generations=sum(scores),
                      accuracy=accuracy, parse_status=dict(parsing),
                      accuracy_is_benchmark=phase == "formal",
                      needs_answer_review=phase != "probe" and any(k != "ok" for k in parsing),
                      results_file=str(results[0]), details_file=str(details[0]),
                      raw_caches=[str(p) for p in caches])
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, status)
        write_json(root / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "probe", "formal"), required=True)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.phase)
