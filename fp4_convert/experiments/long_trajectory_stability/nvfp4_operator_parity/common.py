"""Shared helpers for E0 NVFP4 frozen-input operator parity (diagnostic only)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[4]
SMOKE_ROOT = (
    REPO_ROOT
    / "Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_smoke"
)
RESULT_ROOT = (
    REPO_ROOT
    / "Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/e0_nvfp4_operator_parity"
)
DEFAULT_LAYERS = (0, 1, 2, 8, 16, 24, 32, 40, 47)
FOCUS_PROBES = (
    ("n159_c94220492", 10, False),
    ("n346_c216392826", 12, False),
    ("n346_c216392826", 24, False),
)
CONTROL_PROBES = (
    ("n159_c94220492", 224, False),
    ("n159_c94220492", 416, False),
    ("n346_c216392826", 224, False),
    ("n346_c216392826", 416, False),
)
# Same frozen-input stress only; not TP1/TP2 shared-prefix evidence.
POST_DIV_CONTROL = (("n346_c216392826", 35, True),)


def all_probes() -> tuple[tuple[str, int, bool], ...]:
    return FOCUS_PROBES + CONTROL_PROBES + POST_DIV_CONTROL


def probe_bucket(prompt_key: str, decode_index: int, post_tp: bool) -> str:
    if post_tp:
        return "post_tp_divergence_control_only"
    for key, dec, _ in FOCUS_PROBES:
        if key == prompt_key and dec == decode_index:
            return "focus_low_margin"
    return "uniform_control"


def tensor_checksum(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous()
    if arr.dtype == torch.bfloat16:
        arr = arr.view(torch.uint16)
    elif arr.dtype == torch.float16:
        arr = arr.view(torch.uint16)
    elif arr.dtype == torch.float32:
        arr = arr.view(torch.int32)
    data = arr.numpy().tobytes()
    return hashlib.sha1(data).hexdigest()[:16]


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = (a - b).abs()
    denom = torch.linalg.vector_norm(a).clamp_min(1e-12)
    rel_l2 = float((torch.linalg.vector_norm(a - b) / denom).item())
    cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    exact = float((a == b).float().mean().item()) if a.numel() else 1.0
    scale = torch.maximum(a.abs() * (2**-7), torch.full_like(a, 1e-8))
    ulp_like = float((diff / scale).mean().item())
    sign_flip = float(((a * b) < 0).float().mean().item())
    return {
        "exact_fraction": exact,
        "max_abs": float(diff.max().item()) if a.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if a.numel() else 0.0,
        "rel_l2": rel_l2,
        "cosine": cos,
        "sign_flip_fraction": sign_flip,
        "bf16_ulp_like_mean": ulp_like,
        "numel": int(a.numel()),
    }


def channel_topk_abs(diff: torch.Tensor, k: int = 20) -> list[dict]:
    # diff: [..., out]
    flat = diff.detach().float()
    if flat.ndim == 1:
        per = flat.abs()
    else:
        per = flat.reshape(-1, flat.shape[-1]).abs().max(dim=0).values
    kk = min(k, per.numel())
    vals, idx = torch.topk(per, kk)
    return [{"channel": int(i), "max_abs": float(v)} for i, v in zip(idx.tolist(), vals.tolist())]


def load_raw_linear_parts(snapshot: Path, prefix: str) -> dict[str, torch.Tensor]:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    keys = {
        suffix: f"{prefix}.{suffix}"
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    }
    by_shard: dict[str, list[str]] = {}
    for key in keys.values():
        by_shard.setdefault(index["weight_map"][key], []).append(key)
    out: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as handle:
            for key in shard_keys:
                out[key] = handle.get_tensor(key)
    return {suffix: out[keys[suffix]] for suffix in keys}


def scalar_f32(t: torch.Tensor) -> float:
    if t.numel() != 1:
        raise ValueError(f"expected scalar, got {tuple(t.shape)}")
    return float(t.reshape(()).float().item())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_cuda_lut(device: torch.device) -> None:
    if device.type != "cuda":
        return
    from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils

    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = (
        nvfp4_emulation_utils.kE2M1ToFloat_handle.val.to(device)
    )
