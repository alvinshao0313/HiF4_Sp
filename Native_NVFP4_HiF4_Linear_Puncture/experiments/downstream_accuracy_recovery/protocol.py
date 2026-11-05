"""Shared evaluation definitions, independent of candidate training recipes."""
from __future__ import annotations

import json
from pathlib import Path

MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.downstream_accuracy_recovery"
TASK_DIRS = {"arc": "arc", "mmlu_pro": "mmlu_pro", "lcb": "livecodebench", "gsm8k": "gsm8k"}
TASKS = tuple(TASK_DIRS)
ARC_COUNTS = {"arc_easy": 2376, "arc_challenge": 1172}
REASONING_TASKS = {
    "mmlu_pro": {"task_name": "mmlu_pro|0", "metric": "extractive_match", "count": 12032,
                 "max_samples": None, "max_new_tokens": 32768},
    "lcb": {"task_name": "lcb:codegeneration_v6|0", "metric": "codegen_pass@1:16", "count": 175,
            "max_samples": None, "max_new_tokens": 38912},
    "gsm8k": {"task_name": "gsm8k|0", "metric": "extractive_match", "count": 1319,
              "max_samples": None, "max_new_tokens": 32768},
}


def protocol_for(task, *, smoke=False, tensor_parallel_size=2):
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    if type(tensor_parallel_size) is not int or tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be a positive integer")
    common = {"tensor_parallel_size": tensor_parallel_size, "kv_cache_dtype": "bfloat16", "enforce_eager": True,
              "max_num_batched_tokens": 4096, "smoke_only": smoke}
    if task == "arc":
        return {**common, "backend": "lm_eval_vllm", "seed": 42, "num_fewshot": 0,
                "metric": "acc,none", "max_model_length": 4096,
                "sample_counts": {k: 2 if smoke else n for k, n in ARC_COUNTS.items()}}
    definition = REASONING_TASKS[task]
    return {**common, **definition, "count": 1 if smoke else definition["count"],
            "max_samples": 1 if smoke else definition["max_samples"],
            "backend": "vllm_lighteval", "seed": 1234, "num_fewshot": 0,
            "enable_thinking": True, "generations_per_doc": 1,
            "max_model_length": 40960, "batch_size_max_num_seqs": 128,
            "gpu_memory_utilization": 0.9, "temperature": 0.6, "top_p": 0.95,
            "top_k": 20, "min_p": 0.0}


def reasoning_details(result, task):
    task_name = REASONING_TASKS[task]["task_name"]
    root = Path(result["results_file"]).parent.parent / "details"
    files = list(root.rglob(f"details_{task_name}_*.json"))
    if len(files) != 1:
        raise ValueError(f"expected exactly one {task_name} details file under {root}, got {len(files)}")
    return files[0], json.loads(files[0].read_text(encoding="utf-8"))
