"""Sidecar-aware HiF4 runtime helpers for unquantized vLLM layers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.quantization import hif4_fake

_SIDECAR_CACHE: dict[str, dict[str, Any]] = {}
_TRACE_SEEN: set[tuple[str, str]] = set()


def trace_hif4_runtime_event(event: str, **fields: Any) -> None:
    trace_path = os.environ.get("HIF4_RUNTIME_TRACE_JSONL")
    if not trace_path:
        return
    key = (event, str(fields.get("path") or fields.get("prefix") or ""))
    if key in _TRACE_SEEN:
        return
    _TRACE_SEEN.add(key)
    import json
    import time

    record = {"event": event, "pid": os.getpid(), "time": time.time(), **fields}
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _sidecar_path_from_config(path: str | os.PathLike[str] | None = None) -> str | None:
    if path not in (None, ""):
        return os.path.abspath(os.fspath(path))
    cfg = get_current_vllm_config_or_none()
    additional = (getattr(cfg, "additional_config", {}) if cfg is not None else {}) or {}
    cfg_path = additional.get("hif4_runtime_spec_path")
    env_path = os.environ.get("HIF4_RUNTIME_SPEC_PATH")
    selected = cfg_path if cfg_path not in (None, "") else env_path
    return None if selected in (None, "") else os.path.abspath(str(selected))


def load_hif4_runtime_spec(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = os.path.abspath(os.fspath(path))
    if resolved not in _SIDECAR_CACHE:
        if not Path(resolved).is_file():
            raise FileNotFoundError(f"HiF4 runtime sidecar not found: {resolved}")
        _SIDECAR_CACHE[resolved] = torch.load(resolved, map_location="cpu", weights_only=False)
        trace_hif4_runtime_event(
            "sidecar_load",
            path=resolved,
            variant=_SIDECAR_CACHE[resolved].get("variant"),
        )
    return _SIDECAR_CACHE[resolved]


def current_hif4_runtime_spec(
    path: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    resolved = _sidecar_path_from_config(path)
    if resolved is None:
        return None
    return load_hif4_runtime_spec(resolved)


def apply_dense_hif4_runtime(
    prefix: str,
    x: torch.Tensor,
    path: str | os.PathLike[str] | None = None,
) -> torch.Tensor:
    spec = current_hif4_runtime_spec(path)
    if spec is None:
        return x
    if prefix == "lm_head" or prefix.endswith(".lm_head"):
        return x
    variant = spec.get("variant")
    if variant in {"direct", "fusable", "r64", "fusable_r64"}:
        trace_hif4_runtime_event("dense_apply", prefix=prefix, variant=variant)
        return hif4_fake.hif4_fake_quantize_hifx4(x)
    if variant == "online":
        # Correctness-first placeholder for dense attention online.  Materializer
        # stores transformed weights; online activation DIAG is read from this
        # sidecar by later MoE/attention reference paths.
        trace_hif4_runtime_event("dense_apply", prefix=prefix, variant=variant)
        return hif4_fake.hif4_fake_quantize_hifx4(x)
    raise ValueError(f"unsupported hif4 runtime variant={variant!r} for {prefix}")
