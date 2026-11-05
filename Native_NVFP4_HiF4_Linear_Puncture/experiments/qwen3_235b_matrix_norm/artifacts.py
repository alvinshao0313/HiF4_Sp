"""Bind a new RUN to its real inputs and qualified implementation."""
from datetime import datetime, timezone
import json
from pathlib import Path

from ..non_equivalent_reconstruction.artifact import atomic_save, load, sha256, write_json
from ..reasoning_trace_calibration.artifacts import digest, state_digest
from .config import NATIVE, PROTOCOL, REPO


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    package = Path(__file__).parent
    paths = list(package.glob("*.py"))
    for directory in ("module_design_ablation", "non_equivalent_reconstruction",
                      "e2e_diag_reconstruction/core", "e2e_diag_reconstruction/data",
                      "e2e_diag_reconstruction/training", "e2e_diag_reconstruction/evaluation",
                      "downstream_accuracy_recovery", "reasoning_trace_calibration",
                      "qwen3_235b_transfer"):
        paths.extend((package.parent / directory).rglob("*.py"))
    paths += [REPO / "main.py", REPO / "tasks/aime.py", REPO / "tasks/custom_tasks.py"]
    for directory in ("Native_NVFP4_HiF4_Linear_Puncture/src",
                      "HiFloat4/hif4_scale_threshold_optimization/src"):
        paths.extend((REPO / directory).glob("*.py"))
    for name in ("3rdparty/vllm/vllm/model_executor/layers/quantization/hif4_runtime.py",
                 "3rdparty/vllm/vllm/model_executor/layers/quantization/hif4_transform_triton.py",
                 "3rdparty/vllm/vllm/model_executor/layers/quantization/hif4_triton.py",
                 "3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py",
                 "3rdparty/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py",
                 "3rdparty/vllm/vllm/model_executor/layers/linear.py",
                 "3rdparty/vllm/vllm/model_executor/models/qwen3_moe.py",
                 "3rdparty/lighteval/src/lighteval/tasks/lighteval_task.py",
                 "3rdparty/lighteval/src/lighteval/models/vllm/vllm_model.py",
                 "3rdparty/lighteval/src/lighteval/models/model_output.py",
                 "3rdparty/lighteval/src/lighteval/metrics/metrics.py",
                 "3rdparty/lighteval/src/lighteval/metrics/metrics_sample.py"):
        paths.append(REPO / name)
    return {str(p.relative_to(REPO)): sha256(p) for p in sorted(set(paths))}


def validate_root(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("kind") != "qwen3_235b_matrix_norm" or manifest.get("protocol_sha256") != digest(PROTOCOL):
        raise ValueError("RUN does not match the approved 235B matrix_norm protocol")
    return root


def verify_sources(root):
    expected = read_json(Path(root) / "source_hashes.json")
    actual = source_hashes()
    if expected != actual:
        changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
        raise ValueError(f"qualified implementation changed: {changed}")


def verify_inputs(root):
    root = validate_root(root)
    manifest = read_json(root / "inputs.json")
    for filename, expected in manifest["files"].items():
        if sha256(root / filename) != expected:
            raise ValueError(f"prepared input changed: {filename}")
    for filename, expected in manifest["native_assets"].items():
        if sha256(NATIVE / filename) != expected:
            raise ValueError(f"native metadata changed: {filename}")
    for name, entry in manifest["native_shards"].items():
        path = NATIVE / name
        # Source blobs are content-addressed; retain identity and stat evidence
        # without re-reading 130 GB before every subprocess.
        stat = path.stat()
        if {"resolved": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != entry:
            raise ValueError(f"native shard identity changed: {name}")
    return root
