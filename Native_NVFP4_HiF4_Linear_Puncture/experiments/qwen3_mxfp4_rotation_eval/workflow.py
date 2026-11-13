"""Explicit qualification and finite formal runs; no retries or protocol changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading

from .common import HERE, MODELS, REPO, TASKS, file_hash, model_view, now, read, save
from .report import compare, reasoning, runtime_check

PACKAGE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_rotation_eval"


def devices(value):
    selected = value.split(",")
    if len(selected) != 4 or len(set(selected)) != 4 or any(not item.isdigit() for item in selected):
        raise argparse.ArgumentTypeError("Specify exactly four different GPU indices")
    return value


def available(selected):
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
                         check=True, text=True, capture_output=True).stdout
    records = {fields[0]: fields[1:] for line in gpu.splitlines() if (fields := [item.strip() for item in line.split(",")])}
    active = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                            check=True, text=True, capture_output=True).stdout
    for index in selected.split(","):
        uuid, memory = records[index]
        if uuid in active or int(memory) > 1024:
            raise RuntimeError(f"GPU {index} is occupied; not starting a new task. GPU processes: {active.strip()}")


def command(root, variant, phase, task, samples):
    target = root / variant / phase / task
    model = target / "model"
    if task == "arc":
        result = [sys.executable, "-u", "-m", PACKAGE + ".arc", "--model", str(model),
                  "--output", str(target / "evaluation"), "--dataset_manifest", str(root / "arc_datasets.json"),
                  "--rounds", "2" if phase == "probe" else "5"]
        if phase == "probe":
            result += ["--limit", "65"]
        return result
    result = [sys.executable, "-u", str(HERE / "entry.py"), "--model_path", str(model),
              "--datasets", TASKS[task]["tag"], "--custom_tasks", str(HERE / "tasks.py"),
              "--dataset_manifest", str(root / "datasets.json"), "--tensor_parallel_size", "4",
              "--max_model_length", "40960", "--max_new_tokens", "64" if phase == "probe" else str(TASKS[task]["max_new_tokens"]),
              "--temperature", "0.6", "--top_p", "0.95", "--top_k", "20", "--min_p", "0",
              "--gpu_memory_utilization", "0.55", "--batch_size", "256", "--max_num_batched_tokens", "8192",
              "--kv_cache_dtype", "bfloat16", "--kv_cache_memory_bytes", "34359738368",
              "--linear_backend", "emulation", "--moe_backend", "emulation", "--enforce_eager",
              "--output_dir", str(target / "evaluation")]
    if phase == "probe":
        result += ["--max_samples", "26"]
    elif task == "mmlu_pro" and samples != 12032:
        result += ["--max_samples", str(samples)]
    return result


class Workflow:
    def __init__(self, args):
        self.args, self.root = args, Path(args.root).resolve()
        self.gpus = {"fht32": args.h32_gpus, "r16x2": args.r16x2_gpus}
        left, right = [set(value.split(",")) for value in self.gpus.values()]
        if left & right and left != right:
            raise ValueError("GPU groups must be identical (sequential) or disjoint (parallel)")
        self.parallel = not bool(left & right)
        self.lock, self.failed = threading.Lock(), threading.Event()
        self.phase = "probe" if args.mode == "qualify" else "formal"
        self.path = self.root / f"{args.mode}_status.json"
        if self.path.exists():
            raise FileExistsError(f"This one-shot stage has already started: {self.path}")
        self.state = {"status": "starting", "pid": os.getpid(), "started_at": now(), "stages": {}}

    def update(self):
        save(self.path, self.state)

    def stage(self, variant, task):
        from ..qwen3_mxfp4.workflow import verify_worker_release

        if self.failed.is_set():
            return
        target = self.root / variant / self.phase / task
        target.mkdir(parents=True, exist_ok=False)
        model_view(MODELS[variant], target / "model")
        cmd = command(self.root, variant, self.phase, task, self.args.mmlu_samples)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpus[variant], TOKENIZERS_PARALLELISM="false",
                   HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="8",
                   QWEN3_MXFP4_AUDIT_DIR=str(target / "audit"))
        record = {"status": "starting", "command": cmd, "gpus": self.gpus[variant],
                  "log": str(target / "main.log"), "started_at": now()}
        key = f"{variant}/{task}"
        with self.lock:
            self.state["stages"][key] = record
            self.update()
        try:
            available(self.gpus[variant])
            with (target / "main.log").open("w", encoding="utf-8") as stream:
                child = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                with self.lock:
                    record.update(status="running", pid=child.pid)
                    self.update()
                code = child.wait()
            if code:
                raise RuntimeError(f"{key} exited {code}; see {target / 'main.log'}")
            audit = runtime_check(variant, target / "audit")
            save(target / "runtime_validation.json", audit)
            verify_worker_release(target / "audit")
            if task != "arc":
                count = 26 if self.phase == "probe" else (175 if task == "lcb" else self.args.mmlu_samples)
                reasoning(self.root, target, task, count, self.phase == "probe")
            else:
                if read(target / "evaluation/summary.json")["status"] != "completed":
                    raise ValueError("ARC did not finish both tasks and all repeats")
            with self.lock:
                record.update(status="completed", exit_code=0, finished_at=now())
                self.update()
            print(f"Completed {self.phase}/{key}", flush=True)
        except Exception as error:
            self.failed.set()
            with self.lock:
                record.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=now(),
                              exit_code=child.returncode if "child" in locals() else None)
                self.update()
            raise

    def lane(self, variant):
        for task in ("arc", "lcb", "mmlu_pro"):
            self.stage(variant, task)

    def run(self):
        manifest = read(self.root / "manifest.json")
        for path, expected in manifest["source_sha256"].items():
            if file_hash(path) != expected:
                raise ValueError(f"Prepared dependency changed: {path}")
        execution = {"gpus": self.gpus, "mmlu_samples": self.args.mmlu_samples,
                     "reasoning_generations": 10, "arc_rounds": 5}
        if self.args.mode == "qualify":
            if (self.root / "execution.json").exists():
                raise FileExistsError("Execution configuration already exists")
            save(self.root / "execution.json", execution)
        else:
            if read(self.root / "execution.json") != execution:
                raise ValueError("Formal run differs from the qualified execution configuration")
            qualification = read(self.root / "qualify_status.json")
            if (qualification["status"] != "completed" or len(qualification["stages"]) != 6
                    or any(row["status"] != "completed" for row in qualification["stages"].values())):
                raise ValueError("Both candidates must pass all three real TP4 entry checks")
        self.state["status"] = "running"
        self.update()
        try:
            if self.parallel:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(self.lane, self.gpus))
            else:
                for variant in self.gpus:
                    self.lane(variant)
            if self.failed.is_set():
                raise RuntimeError("A stage failed; remaining stages were skipped")
            if self.args.mode == "formal":
                compare(self.root)
            self.state.update(status="completed", finished_at=now())
        except Exception as error:
            self.state.update(status="failed", error=str(error), finished_at=now())
            raise
        finally:
            self.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "formal"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--h32-gpus", required=True, type=devices)
    parser.add_argument("--r16x2-gpus", required=True, type=devices)
    parser.add_argument("--mmlu-samples", type=int, choices=(300, 12032), required=True)
    Workflow(parser.parse_args()).run()
