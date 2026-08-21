"""Full-model native NVFP4 semantic runtime (rotate → QDQ → GEMM on all 252 Linears)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import (
    enumerate_target_prefixes,
    load_packed_linear_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.config import TARGET_PROJECTIONS
from Native_NVFP4_HiF4_Linear_Puncture.src.native_nvfp4 import (
    decode_weight_scale_uint8,
    dequantize_packed_weight,
    qdq_nvfp4_post_rotation,
    PackedNVFP4LinearState,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.rotation import apply_block_rotation


ObserverFn = Callable[[str, torch.Tensor], None]


@dataclass
class NativeCheckpointIndex:
    snapshot_path: Path
    weight_map: dict[str, str]
    target_prefixes: list[str]
    config: Any


class NativeNVFP4SemanticLinear(nn.Module):
    """Source Linear: block rotation → optional record → NVFP4 A4 QDQ → BF16 GEMM."""

    def __init__(
        self,
        base_linear: nn.Linear,
        module_name: str,
        input_global_scale: torch.Tensor,
        rotation_matrix: torch.Tensor,
        rotation_group_size: int = 16,
        activation_group_size: int = 16,
    ) -> None:
        super().__init__()
        self.module_name = module_name
        self.rotation_group_size = int(rotation_group_size)
        self.activation_group_size = int(activation_group_size)
        self.weight = nn.Parameter(base_linear.weight.detach(), requires_grad=False)
        if base_linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(base_linear.bias.detach(), requires_grad=False)
        self.register_buffer(
            "input_global_scale",
            input_global_scale.detach().reshape(()).to(torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "rotation_matrix",
            rotation_matrix.detach().to(torch.bfloat16),
            persistent=True,
        )
        self._observer: ObserverFn | None = None
        self._recorder = None  # Callable[[Tensor], None] | None
        self._call_count = 0

    def enable_observer(self, fn: ObserverFn) -> None:
        self._observer = fn

    def disable_observer(self) -> None:
        self._observer = None

    def set_recorder(self, fn) -> None:
        """Record ``X_rot`` only; ``fn(x_rot)`` must not alter tensors in-place."""
        self._recorder = fn

    def clear_recorder(self) -> None:
        self._recorder = None

    def _maybe_record(self, x_rot: torch.Tensor) -> None:
        if self._recorder is not None:
            self._recorder(x_rot)
        if self._observer is not None:
            self._observer(self.module_name, x_rot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._call_count += 1
        x_rot = apply_block_rotation(
            x, self.rotation_matrix, self.rotation_group_size
        )
        self._maybe_record(x_rot)
        a_n = qdq_nvfp4_post_rotation(x_rot, self.input_global_scale)
        return F.linear(
            a_n.to(dtype=self.weight.dtype),
            self.weight,
            self.bias,
        )


def _load_tensor(snapshot: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard = snapshot / weight_map[key]
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _is_target_param(name: str, target_set: set[str]) -> bool:
    if not name.endswith((".weight", ".bias")):
        return False
    parent = name.rsplit(".", 1)[0]
    return parent in target_set


def load_native_nvfp4_semantic_model(
    snapshot_path: Path | str,
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rotation_group_size: int = 16,
    activation_group_size: int = 16,
) -> tuple[nn.Module, NativeCheckpointIndex]:
    snapshot = Path(snapshot_path)
    device = torch.device(device)

    cfg_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]

    # In-memory only: strip quantization loader config.
    hf_config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    object.__setattr__(hf_config, "quantization_config", None)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=dtype)

    num_layers = int(raw_cfg["num_hidden_layers"])
    target_prefixes = enumerate_target_prefixes(
        weight_map, num_layers=num_layers, projections=TARGET_PROJECTIONS
    )
    expected = num_layers * len(TARGET_PROJECTIONS)
    if len(target_prefixes) != expected:
        raise RuntimeError(
            f"expected {expected} target Linears, found {len(target_prefixes)}"
        )
    target_set = set(target_prefixes)

    # Materialize non-target parameters from shards.
    for name, _param in list(model.named_parameters()):
        if _is_target_param(name, target_set):
            continue
        if name in weight_map:
            tensor = _load_tensor(snapshot, weight_map, name)
        elif name == "lm_head.weight" and "model.embed_tokens.weight" in weight_map:
            tensor = _load_tensor(snapshot, weight_map, "model.embed_tokens.weight")
        else:
            raise KeyError(f"missing non-target parameter in checkpoint: {name}")
        set_module_tensor_to_device(
            model, name, device=str(device), value=tensor.to(dtype), dtype=dtype
        )

    for name, _buf in list(model.named_buffers()):
        if name in weight_map:
            tensor = _load_tensor(snapshot, weight_map, name)
            set_module_tensor_to_device(
                model, name, device=str(device), value=tensor, dtype=tensor.dtype
            )

    # Dequant + wrap each target Linear.
    wrapped = 0
    for prefix in target_prefixes:
        packed = load_packed_linear_state(snapshot, weight_map, prefix)
        state = PackedNVFP4LinearState(
            module_name=prefix,
            weight_packed=packed["weight_packed"],  # type: ignore[arg-type]
            weight_scale=packed["weight_scale"],  # type: ignore[arg-type]
            weight_global_scale=packed["weight_global_scale"].to(torch.float32),  # type: ignore[union-attr]
            input_global_scale=packed["input_global_scale"].to(torch.float32),  # type: ignore[union-attr]
            rotation_matrix=packed["rotation_matrix"].to(torch.bfloat16),  # type: ignore[union-attr]
            bias=packed["bias"].to(dtype) if packed["bias"] is not None else None,
        )
        if state.weight_scale.dtype == torch.uint8:
            _ = decode_weight_scale_uint8(state.weight_scale)

        w_n = dequantize_packed_weight(state, dtype=dtype, group_size=16)
        parent_name, leaf = prefix.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        old = getattr(parent, leaf)
        if not isinstance(old, nn.Linear):
            raise TypeError(f"{prefix} is not nn.Linear, got {type(old)}")

        linear = nn.Linear(
            w_n.shape[1],
            w_n.shape[0],
            bias=state.bias is not None,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(w_n.to(device=device, dtype=dtype))
            if state.bias is not None:
                linear.bias.copy_(state.bias.to(device=device, dtype=dtype))

        wrapped_mod = NativeNVFP4SemanticLinear(
            linear,
            module_name=prefix,
            input_global_scale=state.input_global_scale.to(device),
            rotation_matrix=state.rotation_matrix.to(device),
            rotation_group_size=rotation_group_size,
            activation_group_size=activation_group_size,
        )
        setattr(parent, leaf, wrapped_mod)
        wrapped += 1

    if wrapped != expected:
        raise RuntimeError(f"wrapped={wrapped} != expected={expected}")

    model.eval()

    ckpt_index = NativeCheckpointIndex(
        snapshot_path=snapshot,
        weight_map=weight_map,
        target_prefixes=target_prefixes,
        config=hf_config,
    )
    return model, ckpt_index


def iter_semantic_linears(model: nn.Module) -> list[NativeNVFP4SemanticLinear]:
    return [m for _, m in model.named_modules() if isinstance(m, NativeNVFP4SemanticLinear)]


def enable_observers(
    model: nn.Module,
    module_names: set[str] | None,
    fn: ObserverFn,
) -> None:
    for m in iter_semantic_linears(model):
        if module_names is None or m.module_name in module_names:
            m.enable_observer(fn)


def disable_all_observers(model: nn.Module) -> None:
    for m in iter_semantic_linears(model):
        m.disable_observer()


def count_wrapped_targets(model: nn.Module) -> int:
    return len(iter_semantic_linears(model))
