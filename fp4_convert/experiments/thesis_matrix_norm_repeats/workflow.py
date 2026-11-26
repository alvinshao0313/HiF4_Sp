"""Finite TP2 sequence: qualify, then fill the four missing rounds of both models."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from ..ban_trace_calibration.processes import require_free, join_session
from ..downstream_accuracy_recovery.reporting import validate_metrics
from ..thesis_format_conversion_supplement.artifacts import process_record, read_json, run_root, write_json
from ..thesis_format_conversion_supplement.config import REASONING, REPO, TASKS
from ..thesis_format_conversion_supplement.report import collect
from .common import MODULE, ROLES, prepare, verify


def execute(root, action):
    root = run_root(root)
    verify(root)
    pair = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if pair != "4,5":
        raise ValueError("matrix_norm repeats are allocated GPU4,5 only")
    require_free(pair)
    if action == "formal" and not read_json(root / "qualification/complete.json")["complete"]:
        raise ValueError("TP2 qualification incomplete")
    commands = []
    if action == "qualify":
        stages = [("ours_a20", "mmlu_pro", 1), ("b20", "arc", 1)]
    else:
        rows = read_json(root / "summary.json")["rows"]
        stages = [(role, task, repetition) for repetition in range(1, 6)
                  for task in REASONING + ("arc",) for role in ROLES
                  if any(str(repetition) not in rows[role]["round_scores"][key]
                         for key in (("arc_easy", "arc_challenge") if task == "arc" else (task,)))]
    for role, task, repetition in stages:
        command = [sys.executable, "-u", "-m", MODULE + ".evaluation", "--run", str(root),
                   "--candidate", role, "--task", task, "--repetition", str(repetition)]
        if action == "qualify":
            command.append("--smoke")
        commands.append(command)
    directory = root / "jobs" / action
    directory.mkdir(parents=True, exist_ok=False)
    state = {**process_record(sys.argv), "status": "running", "stages": stages,
             "commands": commands, "completed": []}
    write_json(state, directory / "status.json")
    for index, command in enumerate(commands):
        require_free(pair)
        verify(root)
        log = directory / f"{index:03d}.log"
        started = time.monotonic()
        with log.open("w") as stream:
            child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state["current"] = {"index": index, "pid": child.pid, "session_id": child.pid,
                                "command": command, "log": str(log)}
            write_json(state, directory / "status.json")
            code = child.wait()
        state["completed"].append({**state["current"], "exit_code": code,
                                   "seconds": time.monotonic() - started, "children": join_session(child.pid)})
        state["status"] = "failed" if code else "running"
        write_json(state, directory / "status.json")
        if code:
            raise RuntimeError(f"evaluation exit={code}; see {log}")
    if action == "qualify":
        reason = root / "qualification/evaluation/ours_a20/round_01/mmlu_pro"
        arc = root / "qualification/evaluation/b20/round_01/arc"
        scores = validate_metrics(read_json(arc / "eval/arc/metrics.json"), "arc", smoke=True, model_kind="hif4")
        samples = read_json(arc / "arc_samples.json")
        if set(samples) != {"arc_easy", "arc_challenge"} or any(len(v) != 2 for v in samples.values()):
            raise ValueError("ARC short check lacks all four sample records")
        write_json({"complete": True, "gpu_pair": pair, "reasoning": read_json(reason / "verified_scores.json"),
                    "arc_scores": scores, "arc_samples_per_task": 2,
                    "scope": "real A20 reasoning and B20 ARC, full TP2/context/KV settings; historical full-model evidence reused",
                    "formal_tasks": TASKS, "formal_counts_unchanged": True}, root / "qualification/complete.json")
    else:
        rows = collect(root)
        if any(rows[role]["missing"] for role in ROLES):
            raise ValueError("finished jobs do not establish five complete rounds for both models")
        write_json({"complete": True, "aggregation": "five complete runs arithmetic mean",
                    "rows": {role: rows[role] for role in ROLES},
                    "calibration_boundary": "A20 S1K matches GPTQ; B20 uses self-generated traces"},
                   root / "matrix_norm_mean5.json")
    verify(root)
    state.update(status="complete", exit_code=0)
    write_json(state, directory / "status.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("prepare", "qualify", "formal"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires an audited --source run")
        prepare(args.run, args.source)
    else:
        execute(args.run, args.action)
