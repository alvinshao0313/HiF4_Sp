"""Prepare the fixed TP8, BF16, thinking GPQA ten-sample protocol."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

from .download import save, sha256


def prepare(root, sequential_rounds=None):
    root = Path(root).resolve()
    repo = Path(__file__).resolve().parents[2]
    if (root / "manifest.json").exists():
        raise FileExistsError("Prepared manifest already exists")
    if json.loads((root / "download_status.json").read_text())["status"] != "completed":
        raise ValueError("Official model verification is incomplete")
    snapshot = Path(json.loads((root / "download_manifest.json").read_text())["snapshot"])
    prior = repo / "Native_NVFP4_HiF4_Linear_Puncture/results/nvfp4_gpqa/native_tp4_fixedkv_20261109T081227Z/manifest.json"
    data = json.loads(prior.read_text())["dataset"]
    source = Path(data["local_path"]) / "train.jsonl"
    if sha256(source) != data["sha256"]:
        raise ValueError("Existing GPQA data digest differs")
    directory = root / "data/gpqa_diamond"
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, directory / "train.jsonl")
    shutil.copy2(source.parent / "README.md", directory / "README.md")
    env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "GPQA_DATASET_PATH": str(directory),
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "VLLM_LOGGING_LEVEL": "INFO", "PYTHONPATH": f"{repo / '3rdparty/lighteval/src'}:{repo}"}
    os.environ.update(env)
    from transformers import AutoTokenizer
    from lighteval.models.model_input import GenerationParameters
    from lighteval.models.vllm.vllm_model import VLLMModelConfig
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.prompt_manager import PromptManager
    from lighteval.tasks.registry import Registry
    from vllm import SamplingParams
    from .task import TASKS_TABLE

    n = 1 if sequential_rounds else 10
    task_name = "gpqa_diamond_qwen_bf16_single" if sequential_rounds else "gpqa_diamond_qwen_bf16_avg10"
    registered = Registry(tasks=f"{task_name}|0",
                          custom_tasks=str(repo / "tools/qwen3_bf16_gpqa/task.py"))
    tasks = registered.load_tasks()
    if len(tasks) != 1:
        raise ValueError("Custom task registration failed")
    random.seed(1234)
    task = next(iter(tasks.values()))
    docs = task.get_docs()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    manager = PromptManager(True, tokenizer)
    explicit = PromptManager(True, tokenizer, enable_thinking=True)
    disabled = PromptManager(True, tokenizer, enable_thinking=False)
    protocol = []
    for doc in docs:
        prompt = manager.prepare_prompt(doc)
        tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if prompt != explicit.prepare_prompt(doc) or prompt == disabled.prepare_prompt(doc) or doc.num_samples != n:
            raise ValueError("Thinking template or sample count differs")
        protocol.append({"id": doc.id, "question_sha256": doc.specific["question_sha256"],
                         "gold": doc.get_golds()[0], "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                         "input_tokens_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                         "input_tokens": len(tokens)})
    if len(protocol) != 198 or len({row["question_sha256"] for row in protocol}) != 198:
        raise ValueError("Expected 198 unique GPQA Diamond questions")
    if max(row["input_tokens"] for row in protocol) + 32768 > 40960:
        raise ValueError("Full output budget would truncate an input")
    generation = GenerationParameters(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_new_tokens=32768)
    sampling = SamplingParams(**generation.to_vllm_dict(), n=n)
    VLLMModelConfig(model_name=str(snapshot), tensor_parallel_size=8, max_model_length=40960,
                    max_num_seqs=128, max_num_batched_tokens=4096, kv_cache_memory_bytes=20 * 1024**3,
                    kv_cache_dtype="bfloat16", enforce_eager=True, generation_parameters=generation)
    phases = {"probe": {"questions": 13, "max_new_tokens": 64, "kind": "probe", "seed": 1234},
              "formal": {"questions": 198, "max_new_tokens": 32768, "kind": "formal", "seed": 1234}}
    if sequential_rounds:
        phases = {"probe": {"questions": 129, "max_new_tokens": 64, "kind": "probe", "seed": 1234},
                  "probe_round_2": {"questions": 1, "max_new_tokens": 64, "kind": "probe", "seed": 1235}}
        phases.update({f"round_{i + 1}": {"questions": 198, "max_new_tokens": 32768,
                                        "kind": "formal", "seed": 1234 + i}
                       for i in range(sequential_rounds)})
    commands = {}
    for phase, config in phases.items():
        directory = root / phase
        model = directory / "model"
        model.mkdir(parents=True)
        for path in snapshot.iterdir():
            if path.is_file() and not path.name.startswith("."):
                (model / path.name).symlink_to(path)
        command = [sys.executable, "-u", str(repo / "main.py"), "--model_path", str(model),
                   "--datasets", "gpqa_diamond_qwen_bf16_avg10|0", "--custom_tasks", str(repo / "tools/qwen3_bf16_gpqa/task.py"),
                   "--tensor_parallel_size", "8", "--max_model_length", "40960", "--max_new_tokens", str(config["max_new_tokens"]),
                   "--temperature", "0.6", "--top_p", "0.95", "--top_k", "20", "--min_p", "0",
                   "--gpu_memory_utilization", "0.45", "--batch_size", "128", "--max_num_batched_tokens", "4096",
                   "--kv_cache_dtype", "bfloat16", "--kv_cache_memory_bytes", str(20 * 1024**3), "--enforce_eager",
                   "--output_dir", str(directory / "evaluation")]
        if config["kind"] != "formal":
            command += ["--max_samples", str(config["questions"])]
        if sequential_rounds:
            command = [sys.executable, "-u", "-m", "tools.qwen3_bf16_gpqa.round_main",
                       "--run_dir", str(root), "--phase", phase]
        commands[phase] = command
    paths = list((repo / "tools/qwen3_bf16_gpqa").glob("*.py")) + [repo / name for name in (
        "main.py", "3rdparty/lighteval/src/lighteval/models/vllm/vllm_model.py",
        "3rdparty/lighteval/src/lighteval/tasks/prompt_manager.py", "3rdparty/lighteval/src/lighteval/pipeline.py",
        "3rdparty/vllm/vllm/model_executor/models/qwen3_moe.py",
        "Native_NVFP4_HiF4_Linear_Puncture/experiments/nvfp4_gpqa/task.py",
        "Native_NVFP4_HiF4_Linear_Puncture/experiments/qwen3_mxfp4/review_answers.py")]
    hashes = {str(path.relative_to(repo)): sha256(path) for path in paths}
    for path in paths:
        target = root / "source" / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    save(root / "protocol_inputs.json", protocol)
    save(root / "manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "model": "Qwen/Qwen3-30B-A3B",
        "snapshot": str(snapshot), "model_verification": str(root / "model_verification.json"),
        "dataset": {**data, "file": str(root / "data/gpqa_diamond/train.jsonl"), "local_path": str(root / "data/gpqa_diamond"),
                    "source_record": str(prior)}, "environment": env, "phases": phases, "commands": commands,
        "seed": 1234, "prompt_seed": 1234, "thinking": True, "dtype": "bfloat16", "quantization": None,
        "sequential_rounds": sequential_rounds,
        "qualification_phases": [name for name, config in phases.items() if config["kind"] == "probe"],
        "formal_phases": [name for name, config in phases.items() if config["kind"] == "formal"],
        "task_name": task_name, "metric_name": "gpqa_accuracy_single" if sequential_rounds else "gpqa_accuracy_avg10",
        "engine": {"tensor_parallel_size": 8, "max_model_length": 40960, "max_num_seqs": 128,
                   "max_num_batched_tokens": 4096, "kv_cache_dtype": "bfloat16",
                   "kv_cache_memory_bytes": 20 * 1024**3, "enforce_eager": True},
        "sampling": {"n": sampling.n, "temperature": sampling.temperature, "top_p": sampling.top_p,
                     "top_k": sampling.top_k, "min_p": sampling.min_p, "max_tokens": sampling.max_tokens},
        "scoring": "Reuse completed-final-submission parser; ambiguous final submissions await blind review; keep strict JSON diagnostic scores",
        "source_sha256": hashes, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm", "datasets", "huggingface-hub")},
        "cpu_checks": {"unique_questions": 198, "requests_per_question": n, "registered_task": True,
                       "thinking_default_equals_true": True, "max_prompt_tokens": max(row["input_tokens"] for row in protocol),
                       "full_output_budget_without_truncation": True, "backend_sampling_schema": True},
        "official_sources": ["https://huggingface.co/Qwen/Qwen3-30B-A3B#best-practices",
                             "https://arxiv.org/html/2505.09388v1#S4.SS6",
                             "https://modelscope.cn/models/Qwen/Qwen3-30B-A3B"],
        "sampling_difference_from_report": ("User requested 3 sequential full-dataset rounds, n=1, seeds 1234/1235/1236; not the report's n=10"
                                             if sequential_rounds else "None: user updated request to 10 samples per question"),
    })
    (root / "working_tree.diff").write_bytes(subprocess.check_output(["git", "diff", "--", "main.py", "3rdparty/lighteval/src/lighteval/models/vllm/vllm_model.py"], cwd=repo))
    print(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--sequential_rounds", type=int, choices=(3,))
    args = parser.parse_args()
    prepare(args.run_dir, args.sequential_rounds)
