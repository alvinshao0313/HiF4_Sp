"""Small artifact helpers; hashes cover actual bytes, not tensor values."""
import hashlib
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
PACKAGE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4"


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_record(tensor):
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(memoryview(raw)).hexdigest(),
            "bytes": tensor.numel() * tensor.element_size()}


def code_hashes():
    paths = list(Path(__file__).parent.glob("*.py")) + [
        REPO / "Native_NVFP4_HiF4_Linear_Puncture/experiments/diag_gradient/r64_transform.py",
        REPO / "3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py",
        REPO / "3rdparty/vllm/vllm/model_executor/layers/quantization/modelopt.py",
        REPO / "3rdparty/vllm/vllm/model_executor/models/qwen3_moe.py",
        REPO / "3rdparty/vllm/vllm/model_executor/layers/fused_moe/layer.py",
    ]
    return {str(path.relative_to(REPO)): file_hash(path) for path in sorted(paths)}


def check_hashes(expected):
    for name, digest in expected.items():
        if file_hash(REPO / name) != digest:
            raise ValueError(f"Prepared source changed: {name}")
