"""Small provenance helpers shared only by this experiment."""
import json
import os
from pathlib import Path
import subprocess

from ..reasoning_trace_calibration.artifacts import atomic_save, load, read_json, sha256, state_digest, write_json, digest
from .config import REPO, ROOT


def run_root(value):
    root = Path(value).resolve()
    parent = ROOT / "results/thesis_format_conversion_supplement"
    if not root.is_relative_to(parent) or root == parent:
        raise ValueError(f"run must be a new child of {parent}")
    return root


def source_hashes():
    package = Path(__file__).parent
    paths = list(package.glob("*.py"))
    for name in ("e2e_diag_reconstruction/core", "non_equivalent_reconstruction", "module_design_ablation"):
        paths += list((package.parent / name).glob("*.py"))
    paths += [REPO / name for name in (
        "HiFloat4/hif4gptq/gptq/gptq_utils.py", "main.py",
        "fp4_convert/src/formats.py",
        "3rdparty/vllm/vllm/model_executor/layers/quantization/hif4_runtime.py",
        "3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/hif4_emulation_moe.py",
        "3rdparty/lighteval/src/lighteval/models/vllm/vllm_model.py")]
    paths += list((REPO / "HiFloat4/hif4_scale_threshold_optimization/src").glob("*.py"))
    return {str(p.relative_to(REPO)): sha256(p) for p in sorted(set(paths))}


def freeze_sources(root):
    value = source_hashes()
    path = root / "source_hashes.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("frozen run sources changed; inspect before starting a new stage")
    write_json(value, path)
    return value


def verify_sources(root):
    if read_json(root / "source_hashes.json") != source_hashes():
        raise ValueError("run source changed")


def process_record(command):
    return {"pid": os.getpid(), "session_id": os.getsid(0), "command": command,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
