"""Sidecar-aware HiF4 runtime helpers for unquantized vLLM layers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.quantization import hif4_fake

_SIDECAR_CACHE: dict[str, dict[str, Any]] = {}
_DEVICE_SCALE_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_CURRENT_LAYER_BY_DEVICE: dict[str, int] = {}
_TRACE_SEEN: set[tuple[str, str]] = set()
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_HIF4_RUNTIME_ABI_VERSION = 3


def _shard_last_dim_for_tp(scale: torch.Tensor, expected_cols: int) -> torch.Tensor:
    """Slice Online D to the local TP shard of a row-parallel input dim.

    o_proj / expert-down activations are sharded on the last dim under TP>1, but
    the sidecar stores the full-width D. Match the local activation width.
    """
    full = int(scale.shape[-1])
    if full == expected_cols:
        return scale
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    tp_size = int(get_tensor_model_parallel_world_size())
    tp_rank = int(get_tensor_model_parallel_rank())
    if tp_size <= 1 or full != expected_cols * tp_size:
        raise ValueError(
            "Online D last-dim incompatible with TP shard: "
            f"full={full} expected_cols={expected_cols} tp_size={tp_size}"
        )
    return scale.narrow(-1, tp_rank * expected_cols, expected_cols).contiguous()


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
        loaded = torch.load(resolved, map_location="cpu", weights_only=False)
        algorithm = loaded.get("algorithm_variant", loaded.get("variant"))
        variant = loaded.get("variant")
        requires_current_abi = algorithm == "online" or variant in {
            "r64",
            "fusable",
            "fusable_r64",
        }
        if requires_current_abi:
            got_abi = int(loaded.get("runtime_abi_version", -1))
            if got_abi != _HIF4_RUNTIME_ABI_VERSION:
                raise RuntimeError(
                    "obsolete HiF4 runtime sidecar; E2/E3-E7 require the optimized "
                    f"runtime ABI {_HIF4_RUNTIME_ABI_VERSION}, got {got_abi} at {resolved}. "
                    "Re-materialize before evaluation."
                )
        _SIDECAR_CACHE[resolved] = loaded
        trace_hif4_runtime_event(
            "sidecar_load", path=resolved, variant=loaded.get("variant")
        )
    return _SIDECAR_CACHE[resolved]


def current_hif4_runtime_spec(
    path: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    resolved = _sidecar_path_from_config(path)
    if resolved is None:
        return None
    return load_hif4_runtime_spec(resolved)


def layer_index_from_prefix(prefix: str) -> int:
    match = _LAYER_RE.search(prefix)
    if match is None:
        raise ValueError(f"cannot resolve transformer layer index from prefix={prefix!r}")
    return int(match.group(1))


def algorithm_variant(spec: dict[str, Any]) -> str | None:
    return spec.get("algorithm_variant", spec.get("variant"))


def set_current_layer(device: torch.device, layer_idx: int) -> None:
    _CURRENT_LAYER_BY_DEVICE[str(device)] = int(layer_idx)


def current_layer(device: torch.device) -> int:
    key = str(device)
    if key not in _CURRENT_LAYER_BY_DEVICE:
        raise RuntimeError(f"Online HiF4 MoE has no current decoder layer for {device}")
    return _CURRENT_LAYER_BY_DEVICE[key]


def _online_scale(
    spec: dict[str, Any],
    resolved_path: str,
    layer_idx: int,
    name: str,
    device: torch.device,
    *,
    expected_cols: int | None = None,
) -> torch.Tensor:
    scales = spec.get("online_activation_scale") or {}
    layer_scales = scales.get(str(layer_idx), scales.get(layer_idx))
    if layer_scales is None:
        raise KeyError(f"Online HiF4 sidecar has no scales for layer {layer_idx}")
    value = layer_scales.get(name)
    if value is None:
        raise KeyError(f"Online HiF4 sidecar layer {layer_idx} has no {name}")
    key = (resolved_path, str(device), layer_idx, name, expected_cols)
    cached = _DEVICE_SCALE_CACHE.get(key)
    if cached is None:
        cached = value.to(device=device, dtype=torch.float32).contiguous()
        if expected_cols is not None:
            cached = _shard_last_dim_for_tp(cached, int(expected_cols))
        _DEVICE_SCALE_CACHE[key] = cached
    return cached


def online_moe_scales_by_layer(
    layer_idx: int,
    device: torch.device,
    path: str | os.PathLike[str] | None = None,
    *,
    hidden_cols: int | None = None,
    down_cols: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    resolved = _sidecar_path_from_config(path)
    if resolved is None:
        raise RuntimeError("Online HiF4 MoE requires hif4_runtime_spec_path")
    spec = load_hif4_runtime_spec(resolved)
    if algorithm_variant(spec) != "online":
        raise ValueError(f"expected Online HiF4 sidecar, got {algorithm_variant(spec)!r}")
    return (
        _online_scale(
            spec, resolved, int(layer_idx), "d_gate", device, expected_cols=hidden_cols
        ),
        _online_scale(
            spec, resolved, int(layer_idx), "d_up", device, expected_cols=hidden_cols
        ),
        _online_scale(
            spec, resolved, int(layer_idx), "d_down", device, expected_cols=down_cols
        ),
        spec,
    )


def online_moe_scales(
    prefix: str,
    device: torch.device,
    path: str | os.PathLike[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    return online_moe_scales_by_layer(
        layer_index_from_prefix(prefix), device, path
    )


def apply_online_qkv_hif4_runtime(
    prefix: str,
    x: torch.Tensor,
    path: str | os.PathLike[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    resolved = _sidecar_path_from_config(path)
    if resolved is None:
        raise RuntimeError("Online HiF4 QKV requires hif4_runtime_spec_path")
    spec = load_hif4_runtime_spec(resolved)
    if algorithm_variant(spec) != "online":
        raise ValueError(f"expected Online HiF4 sidecar, got {algorithm_variant(spec)!r}")
    layer_idx = layer_index_from_prefix(prefix)
    set_current_layer(x.device, layer_idx)
    from vllm.model_executor.layers.quantization.hif4_transform_triton import (
        hif4_online_dense_qdq_triton,
    )

    kwargs = {
        "use_r64": bool(spec.get("use_r64", False)),
        "rot_order": str(spec.get("rot_order", "diag_then_rot")),
    }
    return tuple(
        hif4_online_dense_qdq_triton(
            x,
            _online_scale(
                spec,
                resolved,
                layer_idx,
                name,
                x.device,
                expected_cols=int(x.shape[-1]),
            ),
            **kwargs,
        )
        for name in ("d_q", "d_k", "d_v")
    )  # type: ignore[return-value]


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
    if prefix.endswith(".mlp.gate") or prefix.endswith(".shared_expert_gate"):
        # Router/gating linears stay BF16 by experiment contract.
        return x
    dispatch_variant = spec.get("variant")
    variant = algorithm_variant(spec)
    use_r64 = bool(spec.get("use_r64", False)) or dispatch_variant in {"r64", "fusable_r64"}
    if variant in {"direct", "fusable", "r64", "fusable_r64"}:
        trace_hif4_runtime_event("dense_apply", prefix=prefix, variant=variant)
        if not use_r64:
            return hif4_fake.hif4_fake_quantize_hifx4(x)
        from vllm.model_executor.layers.quantization.hif4_transform_triton import (
            hif4_r64_quantize_hifx4_triton,
        )

        head_dim = int(spec.get("head_dim", 128)) if prefix.endswith(".o_proj") else None
        return hif4_r64_quantize_hifx4_triton(x, head_dim=head_dim)
    if variant == "online":
        if prefix.endswith(".qkv_proj"):
            raise RuntimeError("Online QKV must use the branch-specific Q/K/V runtime")
        layer_idx = layer_index_from_prefix(prefix)
        name = "d_o" if prefix.endswith(".o_proj") else None
        if name is None:
            raise ValueError(f"unsupported Online dense HiF4 prefix={prefix!r}")
        resolved = _sidecar_path_from_config(path)
        assert resolved is not None
        from vllm.model_executor.layers.quantization.hif4_transform_triton import (
            hif4_online_dense_qdq_triton,
        )

        return hif4_online_dense_qdq_triton(
            x,
            _online_scale(
                spec,
                resolved,
                layer_idx,
                name,
                x.device,
                expected_cols=int(x.shape[-1]),
            ),
            use_r64=bool(spec.get("use_r64", False)),
            rot_order=str(spec.get("rot_order", "diag_then_rot")),
            head_dim=int(spec.get("head_dim", 128)),
        )
    raise ValueError(f"unsupported hif4 runtime variant={variant!r} for {prefix}")
