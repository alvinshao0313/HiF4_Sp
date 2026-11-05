"""The approved experiment; old Ban entrypoints retain their defaults."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from ..ban.evaluate import REPO, code_identity, file_sha256
from ..ban.artifact import model_identity, write_json

MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration"
RESULTS = REPO / "Native_NVFP4_HiF4_Linear_Puncture/results"
MODEL = RESULTS / "aime25_repeats/matrix_norm_20261015T081958Z/export"
TRACES = RESULTS / "reasoning_trace_calibration/continue_20261016T135141Z"
C4 = RESULTS / "ban/gsm8k_8to6_20261024T022233Z/calibration/c4_ban.pt"
DATASETS = RESULTS / "ban/supplement_20261025T150926Z/datasets.json"
ARMS = ("top8", "c4", "trace")
TASKS = {
    "lcb": {"task": "lcb:codegeneration_v6|0", "metric": "codegen_pass@1:16",
            "count": 175, "rounds": 3, "max_new_tokens": 38912},
    "mmlu_pro": {"task": "mmlu_pro|0", "metric": "extractive_match",
                 "count": 12032, "rounds": 1, "max_new_tokens": 32768},
}
INFERENCE = {"tensor_parallel_size": 2, "enforce_eager": True, "batch_size": 128,
             "max_num_batched_tokens": 4096, "max_model_length": 40960,
             "gpu_memory_utilization": 0.9, "kv_cache_dtype": "bfloat16",
             "moe_backend": "triton", "fake_act_quant": "none", "temperature": 0.6,
             "top_p": 0.95, "top_k": 20, "min_p": 0}
TARGET, TOLERANCE, K_MIN = 6.07, 0.05, 5


def budget_config(target=TARGET):
    if not math.isfinite(target) or not K_MIN <= target <= 8:
        raise ValueError("target experts must be finite and between k_min=5 and top_k=8")
    return {"target": target, "tolerance": TOLERANCE, "k_min": K_MIN, "max_bisection_steps": 6}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_identity():
    result = code_identity()
    paths = [*Path(__file__).parent.rglob("*.py"),
             REPO / "3rdparty/lighteval/src/lighteval/models/vllm/vllm_model.py",
             REPO / "3rdparty/lighteval/src/lighteval/tasks/lighteval_task.py",
             REPO / "3rdparty/lighteval/src/lighteval/tasks/prompt_manager.py"]
    result["source_sha256"].update({str(p.relative_to(REPO)): file_sha256(p) for p in paths})
    return result


def verify(root):
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    for name, digest in manifest["code"]["source_sha256"].items():
        if file_sha256(REPO / name) != digest:
            raise ValueError(f"experiment source changed after preparation: {name}")
    for name, digest in manifest["input_sha256"].items():
        if file_sha256(name) != digest:
            raise ValueError(f"experiment input changed after preparation: {name}")
    if model_identity(manifest["model"], manifest["spec"]) != manifest["model_identity"]:
        raise ValueError("model/sidecar identity changed")
    return manifest


def model_view(source, target):
    """Link model files only; predictions live in new, per-invocation directories."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    names = {"config.json", "generation_config.json", "model.safetensors.index.json",
             "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json",
             "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
             "hif4_runtime_spec.pt", "ablation_export.json"}
    for path in Path(source).iterdir():
        if path.is_file() and (path.name in names or path.suffix == ".safetensors"):
            (target / path.name).symlink_to(path.resolve())
    return target


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Status is read while the controller is writing it.
    temporary = path.with_suffix(path.suffix + ".writing")
    write_json(temporary, value)
    temporary.replace(path)
