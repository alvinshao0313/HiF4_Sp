"""Run the requested Ban-only tasks, each short test followed by its evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from .additional_main import install_datasets
from .additional_report import TASKS, summarize
from .artifact import load_artifact, model_identity, sha256, write_json
from .evaluate import REPO, MODULE, code_identity, gpu_pair
from .evaluation_report import _sha256


def now():
    return datetime.now(timezone.utc).isoformat()


class Supplement:
    def __init__(self, args):
        self.args, self.root = args, Path(args.output).resolve()
        self.lock = threading.Lock()
        self.state = {"status": "preparing", "controller_pid": os.getpid(), "started_at": now(),
                      "phases": {}, "errors": []}

    def save(self):
        write_json(self.root / "status.json", self.state)

    def command(self, name, *, short):
        definition = TASKS[name]
        target = self.root / name / ("short" if short else "formal")
        argv = [sys.executable, "-u", "-m", MODULE + ".additional_main",
                "--dataset_manifest", str(self.root / "datasets.json"),
                "--model_path", self.args.model_path, "--hif4_runtime_spec", self.args.hif4_runtime_spec,
                "--ban_artifact_path", self.args.artifact_path, "--expert_pruning_method", "Ban",
                "--ban_lambda", "0.7", "--ban_k_min", "5", "--ban_execution", "compact",
                "--ban_stats_dir", str(target / "routing"), "--output_dir", str(target / "evaluation"),
                "--datasets", definition["task"], "--tensor_parallel_size", "2", "--enforce_eager",
                "--batch_size", "128", "--max_num_batched_tokens", "4096", "--max_model_length", "40960",
                "--max_new_tokens", str(definition["max_new_tokens"]), "--gpu_memory_utilization", "0.9",
                "--kv_cache_dtype", "bfloat16", "--moe_backend", "triton", "--fake_act_quant", "none",
                "--temperature", "0.6", "--top_p", "0.95", "--top_k", "20", "--min_p", "0"]
        if short or name == "mmlu_pro":
            argv += ["--max_samples", str(definition["short_count"] if short else definition["count"])]
        return argv

    def prepare(self):
        from datasets import Dataset
        from transformers import AutoTokenizer
        from lighteval.tasks.registry import Registry
        from lighteval.tasks.prompt_manager import PromptManager

        args = self.args
        for key in ("model_path", "hif4_runtime_spec", "artifact_path"):
            setattr(args, key, str(Path(getattr(args, key)).resolve(strict=True)))
        if self.root.exists():
            raise FileExistsError(f"output must be new: {self.root}")
        devices = [device for pair in args.gpu_pairs for device in pair.split(",")]
        if len(devices) != len(set(devices)):
            raise ValueError("supplemental task GPU pairs must be disjoint")
        identity = model_identity(args.model_path, args.hif4_runtime_spec)
        artifact = load_artifact(args.artifact_path, identity)
        if artifact.get("phase") != "formal" or artifact.get("num_samples") != 512:
            raise ValueError("supplement requires the completed 512-sample formal calibration")
        cache = Path("/home/shaoyuantian/.cache/huggingface/datasets")
        mmlu = REPO / "Native_NVFP4_HiF4_Linear_Puncture/results/module_design_ablation/formal_20261008T074812Z/mmlu_revision_recovery_20261014"
        manifest = json.loads((mmlu / "dataset_manifest.json").read_text())
        routes = {"TIGER-Lab/MMLU-Pro": {"revision": manifest["revision"], "splits": {}},
                  "lighteval/code_generation_lite": {"splits": {}},
                  "yentinglin/aime_2025": {"splits": {}}}
        for split in ("test", "validation"):
            path = mmlu / "data" / f"{split}.arrow"
            if sha256(path) != manifest["splits"][split]["sha256"]:
                raise ValueError("canonical MMLU-Pro data changed")
            routes["TIGER-Lab/MMLU-Pro"]["splits"][split] = {"path": str(path), **manifest["splits"][split]}
        for repo, split, path, count in (
            ("lighteval/code_generation_lite", "test", cache / "lighteval___code_generation_lite/v6/0.0.0/89e5fc5c2a8e748f50e95bc7235fab2372d49bfa/code_generation_lite-test.arrow", 175),
            ("yentinglin/aime_2025", "train", cache / "yentinglin___aime_2025/default/0.0.0/6f71d77b0b89b9dabe07ab466c51df33f514df7f/aime_2025-train.arrow", 30)):
            if len(Dataset.from_file(str(path))) != count:
                raise ValueError(f"unexpected task source count: {repo}")
            routes[repo]["splits"][split] = {"path": str(path), "sha256": sha256(path), "count": count}
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "logs").mkdir()
        write_json(self.root / "datasets.json", routes)
        install_datasets(self.root / "datasets.json")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        manager = PromptManager(use_chat_template=True, tokenizer=tokenizer, enable_thinking=None)
        tasks = Registry(tasks=",".join(row["task"] for row in TASKS.values()),
                         custom_tasks=str(REPO / "tasks/custom_tasks.py")).load_tasks()
        self.audits = {}
        for name, definition in TASKS.items():
            task = tasks[definition["task"]]
            docs = task.get_docs(max_samples=definition["count"])
            if len(docs) != definition["count"] or any(doc.num_samples != definition["generations"] for doc in docs):
                raise ValueError(f"actual task coverage or generation count differs: {name}")
            documents = []
            for doc in docs:
                prompt = manager.prepare_prompt(doc)
                ids = tokenizer.encode(prompt, add_special_tokens=True)
                budget = 40960 - definition["max_new_tokens"]
                documents.append({"id": str(doc.id), "prompt_sha256": _sha256(prompt),
                                  "gold_sha256": _sha256(doc.get_golds()),
                                  "input_tokens_sha256": _sha256(ids[-budget:]),
                                  "full_input_tokens": len(ids)})
            audit = {**definition, "dataset_path": task.dataset_path,
                     "dataset_revision": task.dataset_revision, "documents": documents}
            self.audits[name] = audit
            write_json(self.root / f"{name}_input_audit.json", audit)
        self.gpus = dict(zip(TASKS, args.gpu_pairs))
        write_json(self.root / "manifest.json", {"created_at": now(), "command": sys.argv,
                   "model_identity": identity, "model_path": args.model_path,
                   "artifact_path": args.artifact_path, "artifact_sha256": sha256(args.artifact_path),
                   "ban": {"lambda": 0.7, "k_min": 5, "execution": "compact", "test_set_tuning": False},
                   "tasks": TASKS, "gpu_pairs": self.gpus, "seed": 1234, "enable_thinking": None,
                   "enable_prefix_caching": False, "code": code_identity(),
                   "extra_baselines": False, "mmlu_sample_policy": "project default max_samples=300",
                   "aime_score": "existing avg@n:n=5; formal raw-response score retained"})
        write_json(self.root / "commands.json", {name: {"short": self.command(name, short=True),
                   "formal": self.command(name, short=False)} for name in TASKS})
        self.state["status"] = "running"
        self.save()

    def phase(self, name, *, short):
        stage = "short" if short else "formal"
        label, command = f"{name}/{stage}", self.command(name, short=short)
        target = self.root / name / stage
        target.mkdir(parents=True, exist_ok=False)
        log_path = self.root / "logs" / f"{name}_{stage}.log"
        record = {"status": "starting", "command": command, "gpus": self.gpus[name],
                  "started_at": now(), "log": str(log_path)}
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=self.gpus[name], OMP_NUM_THREADS="4",
                           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        with self.lock:
            self.state["phases"][label] = record
            self.save()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                with self.lock:
                    record.update(status="running", pid=child.pid)
                    self.save()
                code = child.wait()
            record["exit_code"] = code
            if code:
                raise RuntimeError(f"{label} exited {code}: {log_path}")
            result = summarize(target, name, self.audits[name], short=short)
            write_json(target / "report.json", result)
            record.update(status="complete", report=str(target / "report.json"))
        except Exception as error:
            record.update(status="failed", error=str(error))
            raise
        finally:
            with self.lock:
                record["ended_at"] = now()
                self.save()

    def task(self, name):
        try:
            self.phase(name, short=True)
            self.phase(name, short=False)
        except Exception as error:
            with self.lock:
                self.state["errors"].append({"task": name, "error": str(error)})
                self.save()

    def run(self):
        self.prepare()
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(self.task, TASKS))
        self.state.update(status="failed" if self.state["errors"] else "complete", ended_at=now())
        self.save()
        reports = {name: json.loads((self.root / name / "formal/report.json").read_text())
                   for name in TASKS if (self.root / name / "formal/report.json").exists()}
        write_json(self.root / "report.json", {"status": self.state["status"], "tasks": reports,
                   "errors": self.state["errors"], "baseline_note": "Ban-only supplement; no same-round top-8 comparison was run."})
        return 1 if self.state["errors"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "model_path", "hif4_runtime_spec", "artifact_path"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--gpu_pairs", nargs=3, type=gpu_pair, default=["0,1", "2,3", "4,5"])
    return Supplement(parser.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
