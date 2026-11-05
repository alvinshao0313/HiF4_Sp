"""Prepare and run a paired five-benchmark native NVFP4 / HiF4 comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock

from .evaluate import check_model
from .protocol import MODULE, TASK_DIRS, protocol_for
from .reporting import audit_prompts, summarize, validate_metrics

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "Native_NVFP4_HiF4_Linear_Puncture/results"
SOURCES = {
    "native_nvfp4": {
        "source": "/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3",
        "kind": "native_nvfp4",
    },
    "group_top_mass": {
        "source": str(RESULTS / "non_equivalent_reconstruction/full48_20260915T084300Z/group_top_mass/model"),
        "kind": "hif4",
    },
}
# Each worker owns one TP2 pair. No GPU reservation or background retry process.
FORMAL_JOBS = (
    ("0,1", (("native_nvfp4", "arc"), ("native_nvfp4", "gsm8k"), ("native_nvfp4", "lcb"))),
    ("2,3", (("group_top_mass", "arc"), ("group_top_mass", "gsm8k"), ("group_top_mass", "lcb"))),
    ("4,5", (("native_nvfp4", "mmlu_pro"),)),
    ("6,7", (("group_top_mass", "mmlu_pro"),)),
)
SMOKE_JOBS = (
    ("0,1", (("native_nvfp4", "arc"),)),
    ("2,3", (("native_nvfp4", "gsm8k"),)),
    ("4,5", (("group_top_mass", "gsm8k"),)),
    ("6,7", (("group_top_mass", "arc"),)),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [REPO / path for path in (
        "main.py",
        "3rdparty/vllm/vllm/model_executor/models/qwen3_moe.py",
        "3rdparty/lighteval/src/lighteval/models/vllm/vllm_model.py",
        "Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/evaluation/vllm_runner.py",
        "Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/evaluation/lm_eval_vllm.py",
        "Native_NVFP4_HiF4_Linear_Puncture/experiments/residual_lora_compensation/arc_runtime.py",
    )]
    return {str(path.relative_to(REPO)): digest(path) for path in paths}


def model_view(root, phase, name, task):
    return root / "models" / phase / name / task


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for name, source in SOURCES.items():
        path = Path(source["source"])
        _, checked = check_model(path, source["kind"])
        metadata[name] = {**source, "metadata": checked}
        for phase in ("smoke", "formal"):
            for task in TASK_DIRS:
                view = model_view(root, phase, name, task)
                view.mkdir(parents=True)
                for file in path.iterdir():
                    if file.is_file():
                        (view / file.name).symlink_to(file)
    audit_prompts({name: model_view(root, "formal", name, "gsm8k") for name in SOURCES},
                  root / "prompt_audit.json")
    previous = RESULTS / "residual_lora_compensation/20260922T012547Z/downstream_validation/summary.json"
    recorded = json.loads(previous.read_text())["source_sha256"]
    reused = {}
    for path in ("main.py", "3rdparty/vllm/vllm/model_executor/models/qwen3_moe.py",
                 "Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/evaluation/vllm_runner.py",
                 "Native_NVFP4_HiF4_Linear_Puncture/experiments/residual_lora_compensation/arc_runtime.py"):
        if digest(REPO / path) != recorded[path]:
            raise ValueError(f"previous task-path verification no longer applies: {path}")
        reused[path] = recorded[path]
    manifest = {
        "created_at": now(), "models": metadata,
        "objective": {"definition": "mean(max(0, 100 * (native_score - candidate_score)))",
                      "threshold_percentage_points": 0.5, "equal_weight_tasks": 5,
                      "mmlu_pro_scope": "full_test_12032"},
        "protocols": {task: protocol_for(task) for task in TASK_DIRS},
        "formal_jobs": FORMAL_JOBS, "smoke_jobs": SMOKE_JOBS,
        "source_sha256": source_hashes(),
        "prior_path_evidence": {"summary": str(previous), "unchanged_sources": reused,
                                "scope": "HiF4 ARC and MMLU/LCB generation, scoring, save and shutdown paths; new native/GSM entry paths require current smoke"},
    }
    write_json(manifest, root / "manifest.json")
    print(json.dumps({"prepared": str(root), "objective": manifest["objective"]}), flush=True)


def run(root, *, smoke):
    manifest = json.loads((root / "manifest.json").read_text())
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError("runtime source changed after preparation; review and prepare a new run")
    audit = json.loads((root / "prompt_audit.json").read_text())
    if audit.get("complete") is not True:
        raise ValueError("prompt audit incomplete")
    if not smoke:
        previous = json.loads((root / "smoke/status.json").read_text())
        if previous["status"] != "complete":
            raise ValueError("current native / HiF4 short tests have not completed")
        for _, jobs in SMOKE_JOBS:
            for name, task in jobs:
                result = json.loads((root / f"smoke/{name}/eval/{TASK_DIRS[task]}/metrics.json").read_text())
                validate_metrics(result, task, smoke=True, model_kind=manifest["models"][name]["kind"])
    phase = "smoke" if smoke else "formal"
    output = root / phase
    output.mkdir(exist_ok=False)
    state = {"status": "running", "pid": os.getpid(), "started_at": now(), "tasks": {}}
    lock = Lock()

    def save():
        write_json(state, output / "status.json")

    save()

    def worker(gpus, jobs):
        for name, task in jobs:
            key = f"{name}/{task}"
            target = output / name
            target.mkdir(exist_ok=True)
            command = [sys.executable, "-m", MODULE + ".evaluate", "--model_dir",
                       str(model_view(root, phase, name, task)), "--model_kind",
                       manifest["models"][name]["kind"], "--output_dir", str(target), "--task", task]
            if smoke:
                command.append("--smoke")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus,
                   "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "8",
                   "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "7200", "HF_ENDPOINT": "https://huggingface.co"}
            row = {"status": "running", "command": command, "gpus": gpus,
                   "started_at": now(), "log": str(target / f"{task}.log")}
            with lock:
                state["tasks"][key] = row
                save()
            try:
                with (target / f"{task}.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL)
                    with lock:
                        row["pid"] = process.pid
                        save()
                    returncode = process.wait()
                row["returncode"] = returncode
                if returncode != 0:
                    raise RuntimeError(f"{key} exited {returncode}; see {row['log']}")
                result = json.loads((target / f"eval/{TASK_DIRS[task]}/metrics.json").read_text())
                reference = None if task == "arc" else audit["models"][name]["tasks"][task]
                scores = validate_metrics(result, task, smoke=smoke, prompt_reference=reference,
                                          model_kind=manifest["models"][name]["kind"])
                with lock:
                    row.update(status="complete", finished_at=now(), scores=scores)
                    save()
            except Exception as exc:
                with lock:
                    row.update(status="failed", finished_at=now(), error=str(exc))
                    save()
                raise

    schedule = SMOKE_JOBS if smoke else FORMAL_JOBS
    errors = []
    with ThreadPoolExecutor(max_workers=len(schedule)) as executor:
        futures = [executor.submit(worker, gpus, jobs) for gpus, jobs in schedule]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
    expected = sum(len(jobs) for _, jobs in schedule)
    try:
        if errors or len(state["tasks"]) != expected or any(row["status"] != "complete" for row in state["tasks"].values()):
            raise RuntimeError("; ".join(errors) or "incomplete task coverage")
        if not smoke:
            summarize(root)
        state.update(status="complete", finished_at=now())
    except Exception as exc:
        state.update(status="failed", finished_at=now(), error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "run", "summarize"))
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    if args.action == "prepare":
        prepare(root)
    elif args.action == "summarize":
        summarize(root)
    else:
        run(root, smoke=args.action == "smoke")
