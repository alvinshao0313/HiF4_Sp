"""Run finite, paired GSM8K top8/top6/Ban evaluations in isolated output views."""
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
import threading
import time

from .artifact import load_artifact, model_identity, write_json


REPO = Path(__file__).resolve().parents[3]
MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.ban"
PROTOCOL = {
    "datasets": "gsm8k", "tensor_parallel_size": 2, "enforce_eager": True,
    "batch_size": 128, "max_num_batched_tokens": 4096, "max_model_length": 40960,
    "max_new_tokens": 32768, "gpu_memory_utilization": 0.9, "kv_cache_dtype": "bfloat16",
    "moe_backend": "triton", "temperature": 0.6, "top_p": 0.95, "top_k": 20,
    "min_p": 0, "fake_act_quant": "none",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_pair(value):
    devices = value.split(",")
    if len(devices) != 2 or any(not device or device.strip() != device for device in devices) or len(set(devices)) != 2:
        raise argparse.ArgumentTypeError("provide exactly two distinct CUDA devices, e.g. 4,5")
    return value


def model_view(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file():
            (destination / path.name).symlink_to(path.resolve())
    return destination


def code_identity():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=REPO)
    sources = [*Path(__file__).parent.glob("*.py"), REPO / "main.py",
               REPO / "3rdparty/vllm/vllm/model_executor/layers/quantization/hif4_runtime.py"]
    return {"git_head": head, "tracked_diff_against_head_sha256": hashlib.sha256(diff).hexdigest(),
            "source_sha256": {str(path.relative_to(REPO)): file_sha256(path) for path in sorted(sources)},
            "version_note": "git_head is the checkout base; source hashes include the uncommitted Ban implementation."}


class Workflow:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output).resolve()
        self.lock, self.evaluation_lock = threading.Lock(), threading.Lock()
        shared = set(args.calibration_gpus.split(",")) & set(args.evaluation_gpus.split(","))
        self.calibration_lock = self.evaluation_lock if shared else threading.Lock()
        self.state = {"status": "preparing", "controller_pid": os.getpid(), "started_at": now(),
                      "short_test": args.short_test, "phases": {}, "errors": []}

    def save_state(self):
        write_json(self.root / "status.json", self.state)

    def run_phase(self, name, command, gpus):
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=gpus, PYTHONUNBUFFERED="1")
        recorded = {key: environment.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "PYTHONUNBUFFERED", "HIF4_RUNTIME_SPEC_PATH", "PYTHONPATH",
            "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")}
        phase = {"name": name, "status": "starting", "command": command, "cwd": str(REPO),
                 "environment": recorded, "started_at": now(), "log": str(self.root / "logs" / f"{name}.log")}
        record = self.root / "phases" / f"{name}.json"
        started = time.monotonic()
        with self.lock:
            self.state["phases"][name] = phase
            write_json(record, phase)
            self.save_state()
        try:
            with Path(phase["log"]).open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                with self.lock:
                    phase.update(status="running", pid=child.pid)
                    write_json(record, phase)
                    self.save_state()
                exit_code = child.wait()
            if exit_code:
                raise RuntimeError(f"{name} exited with code {exit_code}; see {phase['log']}")
            phase.update(status="complete", exit_code=exit_code)
        except Exception as error:
            phase.update(status="failed", error=str(error))
            if "exit_code" not in phase:
                phase["exit_code"] = child.returncode if "child" in locals() else None
            raise
        finally:
            with self.lock:
                phase.update(ended_at=now(), elapsed_seconds=time.monotonic() - started)
                write_json(record, phase)
                self.save_state()

    def record_error(self, name, error):
        with self.lock:
            self.state["errors"].append({"phase": name, "error": str(error)})
            self.save_state()

    def evaluation_command(self, name, artifact=None, lam=None):
        command = [sys.executable, "-u", "-m", MODULE + ".evaluation_main",
                   "--gsm8k_arrow", self.args.gsm8k_test_arrow, "--gsm8k_sha256", self.data["gsm8k_test"]["sha256"],
                   "--model_path", str(self.views[name]), "--hif4_runtime_spec", self.args.hif4_runtime_spec,
                   "--output_dir", str(self.root / "evaluation" / name)]
        for key, value in PROTOCOL.items():
            command.append("--" + key)
            if not isinstance(value, bool):
                command.append(str(value))
        if self.args.short_test:
            command += ["--max_samples", "2"]
        if name == "Ban":
            command += ["--expert_pruning_method", "Ban", "--ban_artifact_path", str(artifact),
                        "--ban_lambda", str(lam), "--ban_k_min", "5", "--ban_execution", "compact",
                        "--ban_stats_dir", str(self.root / "routing" / "Ban")]
        else:
            command += ["--expert_pruning_method", "none"]
            if name == "top6":
                command += ["--num_experts_per_tok", "6"]
        return command

    def evaluate(self, name, artifact=None, lam=None):
        gpus = self.args.calibration_gpus if name == "Ban" else self.args.evaluation_gpus
        resource_lock = self.calibration_lock if name == "Ban" else self.evaluation_lock
        with resource_lock:
            self.run_phase(name, self.evaluation_command(name, artifact, lam), gpus)

    def baseline_lane(self):
        # A failed baseline must not cancel the independent fixed-top6 observation.
        for name in ("top8", "top6"):
            try:
                self.evaluate(name)
            except Exception as error:
                self.record_error(name, error)

    def ban_lane(self):
        try:
            if self.args.short_test:
                self.evaluate("Ban", self.args.diagnostic_artifact, 0.7)
                return
            artifact = self.root / "calibration" / "c4_ban.pt"
            # If the supplied pairs overlap, reuse the evaluation lock for both
            # phases; the same finite workflow then runs without GPU contention.
            with self.calibration_lock:
                self.run_phase("calibration", [sys.executable, "-u", "-m", MODULE + ".calibrate",
                               "--model_path", self.args.model_path, "--hif4_runtime_spec", self.args.hif4_runtime_spec,
                               "--c4_arrow", *self.args.c4_arrow, "--output", str(artifact.parent),
                               "--nsamples", "512", "--seqlen", "2048", "--seed", "0",
                               "--tensor_parallel_size", "2", "--phase", "formal"], self.args.calibration_gpus)
                self.run_phase("budget", [sys.executable, "-u", "-m", MODULE + ".budget",
                               "--model_path", self.args.model_path, "--hif4_runtime_spec", self.args.hif4_runtime_spec,
                               "--artifact_path", str(artifact), "--train_arrow", self.args.gsm8k_train_arrow,
                               "--output", str(self.root / "budget")], self.args.calibration_gpus)
            budget = json.loads((self.root / "budget" / "budget.json").read_text(encoding="utf-8"))
            if budget["budget_matched"] is not True:
                raise RuntimeError("training-only budget probe did not match average six experts")
            self.evaluate("Ban", artifact, budget["chosen_lambda"])
        except Exception as error:
            self.record_error("Ban_lane", error)

    def prepare(self):
        args = self.args
        for key in ("model_path", "hif4_runtime_spec", "gsm8k_test_arrow", "gsm8k_train_arrow"):
            setattr(args, key, str(Path(getattr(args, key)).resolve(strict=True)))
        args.c4_arrow = [str(Path(path).resolve(strict=True)) for path in args.c4_arrow]
        if self.root.exists():
            raise FileExistsError(f"output must be new: {self.root}")
        identity = model_identity(args.model_path, args.hif4_runtime_spec)
        if args.short_test:
            if not args.diagnostic_artifact:
                raise ValueError("--short_test requires --diagnostic_artifact")
            args.diagnostic_artifact = str(Path(args.diagnostic_artifact).resolve(strict=True))
            load_artifact(args.diagnostic_artifact, identity)
        elif args.diagnostic_artifact:
            raise ValueError("--diagnostic_artifact is only accepted with --short_test")
        from datasets import Dataset
        for key, count in (("gsm8k_test_arrow", 1319), ("gsm8k_train_arrow", 7473)):
            if len(Dataset.from_file(getattr(args, key))) != count:
                raise ValueError(f"{key} does not contain the full expected {count} records")
        self.data = {"c4": [{"path": path, "sha256": file_sha256(path)} for path in args.c4_arrow],
                     "gsm8k_test": {"path": args.gsm8k_test_arrow, "sha256": file_sha256(args.gsm8k_test_arrow)},
                     "gsm8k_train": {"path": args.gsm8k_train_arrow, "sha256": file_sha256(args.gsm8k_train_arrow)}}
        # main.py places a top-k override below .moe_override/<basename>_k6.
        # A unique basename keeps that unmodified helper's files and cache isolated.
        suffix = hashlib.sha256(str(self.root).encode()).hexdigest()[:12]
        unique = f"top6_{self.root.name}_{suffix}"
        override = REPO / ".moe_override" / f"{unique}_k6"
        if override.exists():
            raise FileExistsError(f"top6 override cache must be new: {override}")
        self.root.mkdir(parents=True, exist_ok=False)
        for name in ("logs", "phases", "evaluation", "routing"):
            (self.root / name).mkdir()
        source = Path(args.model_path)
        self.views = {name: model_view(source, self.root / "models" / name / unique if name == "top6"
                                       else self.root / "models" / name) for name in ("top8", "top6", "Ban")}
        manifest = {"created_at": now(), "command": sys.argv, "controller_pid": os.getpid(),
                    "short_test": args.short_test, "expected_samples": 2 if args.short_test else 1319,
                    "protocol": {**PROTOCOL, "seed": 1234, "enable_thinking": None,
                                 "generations_per_doc": 1, "enable_prefix_caching": False},
                    "ban": {"k_min": 5, "target_average_experts": 6.0, "budget_tolerance": 0.10,
                            "lambda_selection": "GSM8K train prompts only; no accuracy or test-set tuning"},
                    "calibration": {"nsamples": 512, "seqlen": 2048, "seed": 0, "phase": "formal"},
                    "gpu_pairs": {"calibration": args.calibration_gpus, "evaluation": args.evaluation_gpus},
                    "resource_serialization": "Overlapping GPU pairs share a lock; disjoint lanes may run concurrently.",
                    "activation_quantization": "HiF4 sidecar activation QDQ remains enabled; fake_act_quant=none disables additional generic quantization.",
                    "model_path": args.model_path, "hif4_runtime_spec": args.hif4_runtime_spec,
                    "model_identity": identity, "model_views": {key: str(value) for key, value in self.views.items()},
                    "top6_override_path": str(override), "data": self.data, "code": code_identity(),
                    "python": {"executable": sys.executable, "version": sys.version}}
        if args.short_test:
            manifest.update(calibration={"executed": False, "diagnostic_artifact": args.diagnostic_artifact,
                                         "artifact_sha256": file_sha256(args.diagnostic_artifact)},
                            ban={"lambda": 0.7, "k_min": 5, "budget_matched": False},
                            retention_evaluated=False)
        write_json(self.root / "manifest.json", manifest)
        self.state["status"] = "running"
        self.save_state()

    def finish(self):
        from .evaluation_report import aggregate_routing, compare_evaluations, summarize_evaluation
        summaries = {}
        for name in ("top8", "top6", "Ban"):
            if self.state["phases"].get(name, {}).get("status") != "complete":
                continue
            try:
                summaries[name] = summarize_evaluation(self.root / "evaluation" / name,
                                                       expected_count=2 if self.args.short_test else 1319)
                if name == "Ban":
                    summaries[name]["routing"] = aggregate_routing(self.root / "routing" / "Ban")
            except Exception as error:
                self.record_error(f"report/{name}", error)
        report = {"short_test": self.args.short_test, "evaluations": summaries,
                  "retention_evaluated": False, "completed_at": now()}
        try:
            if set(summaries) != {"top8", "top6", "Ban"}:
                raise RuntimeError("all three completed and validated evaluations are required")
            if self.args.short_test:
                self.validate_short_test(summaries)
                report.update(input_alignment=True, retention_evaluated=False,
                              inference_limit="Two-sample execution validation; no 98% accuracy-retention conclusion.")
            else:
                report["comparison"] = compare_evaluations(summaries)
                report["retention_evaluated"] = True
        except Exception as error:
            self.record_error("report", error)
        report.update(status="failed" if self.state["errors"] else "complete", errors=self.state["errors"])
        write_json(self.root / "report.json", report)
        self.state.update(status=report["status"], ended_at=now(), report=str(self.root / "report.json"))
        self.save_state()
        return 1 if self.state["errors"] else 0

    def validate_short_test(self, summaries):
        import pyarrow.parquet as pq
        reference = summaries["top8"]
        keys = ("id", "prompt_sha256", "gold_sha256", "input_tokens_sha256")
        inputs = [{key: sample[key] for key in keys} for sample in reference["samples"]]
        for name, summary in summaries.items():
            if ([{key: sample[key] for key in keys} for sample in summary["samples"]] != inputs
                    or summary["protocol"] != reference["protocol"]
                    or summary["lighteval_input_tokens_hash"] != reference["lighteval_input_tokens_hash"]):
                raise ValueError(f"short-test actual inputs/protocol differ: {name}")
            rows = pq.read_table(summary["prediction_cache"], columns=["sample_id", "sample.text",
                                 "sample.output_tokens", "sample.truncated_tokens_count"]).to_pylist()
            outputs = []
            for row in rows:
                texts, tokens = row["text"], row["output_tokens"]
                if (len(texts) != 1 or not texts[0].strip() or len(tokens) != 1 or not tokens[0]
                        or row["truncated_tokens_count"] != 0 or len(tokens[0]) >= PROTOCOL["max_new_tokens"]):
                    raise ValueError(f"short-test output is empty/truncated or reached generation limit: {name}")
                outputs.append({"sample_id": row["sample_id"], "output_tokens": len(tokens[0]),
                                "text_sha256": hashlib.sha256(texts[0].encode()).hexdigest(),
                                "truncated_tokens_count": row["truncated_tokens_count"]})
            summary["execution_outputs"] = outputs
            observations = []
            paths = sorted((self.root / "evaluation" / name / "generation_observations").glob("batch_*.json"))
            for path in paths:
                observations.extend(json.loads(path.read_text(encoding="utf-8")))
            if len(observations) != 2:
                raise ValueError(f"short-test must observe exactly two generated requests: {name}")
            prompt_hashes = []
            for observation in observations:
                actual = observation["outputs"]
                if len(actual) != 1 or actual[0]["finish_reason"] != "stop" or actual[0]["output_tokens"] <= 0:
                    raise ValueError(f"short-test generation did not stop normally: {name}")
                prompt_hashes.append(hashlib.sha256(json.dumps(observation["prompt_token_ids"],
                                     ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
            if sorted(prompt_hashes) != sorted(sample["input_tokens_sha256"] for sample in summary["samples"]):
                raise ValueError(f"observed generation prompts differ from cached evaluation tokens: {name}")
            summary["generation_observations"] = {"files": [str(path) for path in paths],
                                                   "requests": len(observations),
                                                   "outputs": [row["outputs"] for row in observations]}

    def run(self):
        self.prepare()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.baseline_lane), pool.submit(self.ban_lane)]
            for future in futures:
                future.result()
        return self.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "model_path", "hif4_runtime_spec", "gsm8k_test_arrow", "gsm8k_train_arrow"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--c4_arrow", nargs="+", required=True)
    parser.add_argument("--calibration_gpus", type=gpu_pair, default="4,5")
    parser.add_argument("--evaluation_gpus", type=gpu_pair, default="6,7")
    parser.add_argument("--short_test", action="store_true")
    parser.add_argument("--diagnostic_artifact")
    return Workflow(parser.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
