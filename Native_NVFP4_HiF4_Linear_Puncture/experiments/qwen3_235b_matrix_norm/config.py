"""The approved recipe, independent of the older E4/Group transfer."""
from pathlib import Path

from ..downstream_accuracy_recovery.protocol import protocol_for
from ..qwen3_235b_transfer.config import source_snapshot
from ..reasoning_trace_calibration.config import GENERATION, SOURCES

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "Native_NVFP4_HiF4_Linear_Puncture"
MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_235b_matrix_norm"
NATIVE = source_snapshot()
PROJECTION_SOURCE = ROOT / "results/reasoning_trace_calibration/continue_20261016T135141Z/data_sources"
TASKS = ("gsm8k", "mmlu_pro", "lcb", "aime25")
MODELS = ("native_nvfp4", "matrix_norm")
COUNTS = {"train": 64, "val": 16}  # Per domain.
ENGINE = {"tensor_parallel_size": 8, "max_model_len": 40960,
          "max_num_batched_tokens": 4096, "max_num_seqs": 128,
          "gpu_memory_utilization": .9, "enforce_eager": True,
          "kv_cache_dtype": "bfloat16"}
TRAINING = {"epochs": 20, "batch_size": 4, "learning_rate": 3e-4, "seed": 42,
            "optimizer": "AdamW", "weight_decay": 0., "scheduler": "cosine",
            "selection": "validation_nmse", "group_size": 64,
            "loss": "block_delta_nmse", "router_trainable": False, "router_kl_weight": 0.,
            "world_size": 4, "saved_tensors": "cpu", "pin_memory": False}


def task_protocol(task, *, smoke=False):
    if task not in TASKS:
        raise ValueError(task)
    if task != "aime25":
        return protocol_for(task, smoke=smoke, tensor_parallel_size=8)
    return {**protocol_for("lcb", smoke=smoke, tensor_parallel_size=8),
            "task_name": "aime25_avg5|0", "metric": "avg@n:n=5",
            "count": 1 if smoke else 30, "generations_per_doc": 5}


PROTOCOL = {"source_snapshot": str(NATIVE), "revision": NATIVE.name,
            "training": TRAINING, "generation": GENERATION, "engine": ENGINE,
            "sources": SOURCES, "calibration": {"train": 128, "val": 32,
                "math_fraction": .5, "seed": 42, "full_sequences": True,
                "correctness_filter": False},
            "evaluation": {task: task_protocol(task) for task in TASKS},
            "aime": {"rounds": 1, "raw_and_final_submission_scores": True}}
