"""Serial TP8 paired evaluation; each task has an isolated prediction-cache path."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from Native_NVFP4_HiF4_Linear_Puncture.experiments.downstream_accuracy_recovery.evaluate import check_model
from Native_NVFP4_HiF4_Linear_Puncture.experiments.downstream_accuracy_recovery.protocol import MODULE, TASK_DIRS, protocol_for
from Native_NVFP4_HiF4_Linear_Puncture.experiments.downstream_accuracy_recovery.reporting import audit_prompts, summarize, validate_metrics
from Native_NVFP4_HiF4_Linear_Puncture.experiments.downstream_accuracy_recovery.run import REPO, source_hashes
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import sha256, write_json
from .config import source_snapshot

TP = 8
SMOKE_TASKS = ("arc", "gsm8k")


def now():
    return datetime.now(timezone.utc).isoformat()


def hashes():
    return {**source_hashes(), str(Path(__file__).relative_to(REPO)): sha256(__file__)}


def view(root, phase, name, task):
    return root / "models" / phase / name / task


def prepare(run_root):
    root = run_root / "downstream"
    root.mkdir(parents=True, exist_ok=False)
    sources = {"native_nvfp4": (source_snapshot(), "native_nvfp4"),
               "group_top_mass": (run_root / "group_top_mass/model", "hif4")}
    models = {}
    for name, (source, kind) in sources.items():
        _, metadata = check_model(source, kind)
        config = json.loads((source / "config.json").read_text())
        if (config["num_hidden_layers"], config["hidden_size"]) != (94, 4096):
            raise ValueError("paired models must both be the full 235B geometry")
        models[name] = {"source": str(source), "kind": kind, "metadata": metadata}
        for phase in ("smoke", "formal"):
            for task in TASK_DIRS:
                target = view(root, phase, name, task)
                target.mkdir(parents=True)
                for file in source.iterdir():
                    if file.is_file():
                        (target / file.name).symlink_to(file)
    audit_prompts({name: view(root, "formal", name, "gsm8k") for name in models}, root / "prompt_audit.json")
    write_json({"created_at": now(), "models": models, "tensor_parallel_size": TP,
                "protocols": {task: protocol_for(task, tensor_parallel_size=TP) for task in TASK_DIRS},
                "smoke_tasks": list(SMOKE_TASKS), "formal_tasks": list(TASK_DIRS),
                "source_sha256": hashes(), "threshold_percentage_points": 0.5,
                "objective": "mean(max(0, native_pct - hif4_pct)) <= 0.5; all five full tasks"},
               root / "manifest.json")


def run(run_root, *, smoke):
    root = run_root / "downstream"
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["source_sha256"] != hashes():
        raise ValueError("evaluation source changed since preparation")
    audit = json.loads((root / "prompt_audit.json").read_text())
    if not audit.get("complete"):
        raise ValueError("full prompt audit is incomplete")
    if not smoke:
        if json.loads((root / "smoke/status.json").read_text())["status"] != "complete":
            raise ValueError("real TP8 short evaluation must complete before formal evaluation")
        for name, model in manifest["models"].items():
            for task in SMOKE_TASKS:
                result = json.loads((root / "smoke" / name / "eval" / TASK_DIRS[task] / "metrics.json").read_text())
                validate_metrics(result, task, smoke=True, model_kind=model["kind"], tensor_parallel_size=TP)
    phase = "smoke" if smoke else "formal"
    output = root / phase
    output.mkdir(exist_ok=False)
    state = {"status": "running", "started_at": now(), "pid": os.getpid(), "tasks": {}}
    write_json(state, output / "status.json")
    try:
        for name, model in manifest["models"].items():
            for task in SMOKE_TASKS if smoke else TASK_DIRS:
                target = output / name
                target.mkdir(exist_ok=True)
                command = [sys.executable, "-u", "-m", MODULE + ".evaluate", "--model_dir",
                           str(view(root, phase, name, task)), "--model_kind", model["kind"],
                           "--output_dir", str(target), "--task", task, "--tensor_parallel_size", str(TP)]
                if smoke:
                    command.append("--smoke")
                key = name + "/" + task
                row = {"status": "running", "started_at": now(), "command": command,
                       "gpus": "0,1,2,3,4,5,6,7", "log": str(target / (task + ".log"))}
                state["tasks"][key] = row
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": row["gpus"], "OMP_NUM_THREADS": "8",
                       "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "7200",
                       "HF_ENDPOINT": "https://huggingface.co"}
                with Path(row["log"]).open("w") as log:
                    child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    row["pid"] = child.pid
                    write_json(state, output / "status.json")
                    code = child.wait()
                row.update(exit_code=code, finished_at=now(), status="complete" if code == 0 else "failed")
                write_json(state, output / "status.json")
                if code != 0:
                    raise RuntimeError(f"{key} failed with exit {code}; see {row['log']}")
                result = json.loads((target / "eval" / TASK_DIRS[task] / "metrics.json").read_text())
                reference = None if task == "arc" else audit["models"][name]["tasks"][task]
                validate_metrics(result, task, smoke=smoke, prompt_reference=reference,
                                 model_kind=model["kind"], tensor_parallel_size=TP)
                print(json.dumps({"event": "task_complete", "task": key, "phase": phase}), flush=True)
        if not smoke:
            summarize(root)
        state["status"] = "complete"
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_at"] = now()
        write_json(state, output / "status.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("prepare", "smoke", "formal", "report"))
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.phase == "prepare":
        prepare(root)
    elif args.phase == "report":
        summarize(root / "downstream")
    else:
        run(root, smoke=args.phase == "smoke")
