"""Four finite TP2 queues; qualification precedes formal evaluation, failures stop the affected queue."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

from ..qwen3_mxfp4.artifacts import check_hashes
from ..qwen3_mxfp4.workflow import verify_worker_release
from ..qwen3_mxfp4_rotation_eval.workflow import available
from .common import DEVICES, HERE, PROBE_QUESTIONS, REPO, TASKS, VARIANTS, model_view, now, read, save
from .report import compare, runtime_check, summarize


def command(root, variant, phase, task):
    target = root / variant / phase / task
    result = [sys.executable, "-u", str(HERE / "entry.py"), "--model_path", str(target / "model"),
        "--datasets", TASKS[task]["tag"], "--custom_tasks", str(HERE / "tasks.py"),
        "--dataset_manifest", str(root / "datasets.json"), "--tensor_parallel_size", "2",
        "--max_model_length", "40960", "--max_new_tokens", "64" if phase == "probe" else str(TASKS[task]["max_new_tokens"]),
        "--temperature", "0.6", "--top_p", "0.95", "--top_k", "20", "--min_p", "0",
        "--gpu_memory_utilization", "0.55", "--batch_size", "256", "--max_num_batched_tokens", "8192",
        "--kv_cache_dtype", "bfloat16", "--kv_cache_memory_bytes", "60129542144",
        "--linear_backend", "emulation", "--moe_backend", "emulation", "--enforce_eager",
        "--output_dir", str(target / "evaluation")]
    if phase == "probe":
        result += ["--max_samples", str(PROBE_QUESTIONS)]
    return result


def lane(root, index, phase, manifest):
    variant, devices = VARIANTS[index], DEVICES[index]
    state_path = root / f"{phase}_lane{index}.json"
    if state_path.exists():
        raise FileExistsError(f"This one-shot lane already started: {state_path}")
    state = {"status": "running", "pid": os.getpid(), "variant": variant, "devices": devices,
             "started_at": now(), "stages": {}}
    save(state_path, state)
    try:
        for task in TASKS:
            check_hashes(manifest["source_sha256"])
            available(devices)
            target = root / variant / phase / task
            target.mkdir(parents=True, exist_ok=False)
            model_view(Path(manifest["models"][variant]["checkpoint"]), target / "model")
            cmd = command(root, variant, phase, task)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices, TOKENIZERS_PARALLELISM="false",
                       HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="8",
                       VLLM_LOGGING_LEVEL="INFO", QWEN3_MXFP4_AUDIT_DIR=str(target / "audit"))
            record = {"status": "starting", "command": cmd, "gpus": devices, "started_at": now(),
                      "log": str(target / "main.log"), "output": str(target)}
            state["current_task"] = task
            state["stages"][task] = record
            save(state_path, state)
            save(target / "status.json", record)
            with (target / "main.log").open("x", encoding="utf-8") as stream:
                child = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                record.update(status="running", pid=child.pid)
                save(target / "status.json", record)
                save(state_path, state)
                code = child.wait()
            record["exit_code"] = code
            if code:
                raise RuntimeError(f"{variant}/{task} exited {code}: {target / 'main.log'}")
            runtime_check(root, variant, target, task, phase == "probe")
            verify_worker_release(target / "audit")
            count = PROBE_QUESTIONS if phase == "probe" else TASKS[task]["questions"]
            report = summarize(root, target, task, count, phase == "probe")
            record.update(status="completed", finished_at=now(), generations=report["generations"],
                          summary=str(target / "summary.json"))
            save(target / "status.json", record)
            save(state_path, state)
            print(f"Completed {phase}/{variant}/{task}: {report['generations']} responses", flush=True)
        state.update(status="completed", exit_code=0)
    except Exception as exc:
        state.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
        if "record" in locals():
            record.update(status="failed", error=state["error"], finished_at=now())
            save(target / "status.json", record)
        raise
    finally:
        state["finished_at"] = now()
        save(state_path, state)


def run(root, mode):
    root = Path(root).resolve()
    phase = "probe" if mode == "qualify" else "formal"
    exit_path = root / f"{mode}_exit.json"
    if exit_path.exists() or any((root / f"{phase}_lane{i}.json").exists() for i in range(4)):
        raise FileExistsError("Stage already started; no automatic restart")
    manifest = read(root / "manifest.json")
    check_hashes(manifest["source_sha256"])
    if mode == "formal":
        if read(root / "qualify_exit.json")["returncode"] != 0:
            raise ValueError("Formal evaluation requires all eight entry probes to pass")
        for i in range(4):
            if read(root / f"probe_lane{i}.json")["status"] != "completed":
                raise ValueError("Candidate qualification incomplete")
    result = {"returncode": 1, "pid": os.getpid(), "started_at": now()}
    try:
        for devices in DEVICES:
            available(devices)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(lane, root, i, phase, manifest) for i in range(4)]
            for future in futures:
                future.result()
        if mode == "formal":
            compare(root)
        result["returncode"] = 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["finished_at"] = now()
        save(exit_path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "formal"))
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    run(args.root, args.mode)
