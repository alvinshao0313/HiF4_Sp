"""把目标 Linear 换成 HiF4FrozenBLinear（QAD 专用）。"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

_QAD_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _QAD_DIR.parent
_SCALE_TUNING_DIR = _REPO_ROOT / "ScaleTuning"
_CHUANCI_DIR = _REPO_ROOT / "ChuanCi"
for _p in (_QAD_DIR, _SCALE_TUNING_DIR, _CHUANCI_DIR):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from nvfp4_hif4_torch import HiF4Config  # noqa: E402

from hif4_frozen_b import HiF4FrozenBLinear, collect_hif4_frozen_b_linears  # noqa: E402

__all__ = [
    "wrap_model_for_qad_frozen_b",
    "freeze_non_s0_parameters",
    "collect_hif4_frozen_b_linears",
    "parse_csv_set",
    "parse_layer_indices",
]

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def parse_csv_set(raw: str | None) -> set[str]:
    if raw is None or not str(raw).strip():
        return set()
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def parse_layer_indices(raw: str | None) -> set[int] | None:
    if raw is None or not str(raw).strip():
        return None
    indices: set[int] = set()
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        indices.add(int(token))
    return indices


def _module_leaf_name(qualified_name: str) -> str:
    return qualified_name.rsplit(".", 1)[-1]


def _extract_layer_index(qualified_name: str) -> int | None:
    match = _LAYER_INDEX_RE.search(qualified_name)
    if match is None:
        return None
    return int(match.group(1))


def _name_matches(
    qualified_name: str,
    *,
    module_names: set[str],
    layer_indices: set[int] | None,
) -> bool:
    if module_names and _module_leaf_name(qualified_name) not in module_names:
        return False
    if layer_indices is not None:
        layer_idx = _extract_layer_index(qualified_name)
        if layer_idx is None or layer_idx not in layer_indices:
            return False
    return True


def _iter_named_linears(model: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def _get_parent_and_attr(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    if "." not in qualified_name:
        return model, qualified_name
    parent_name, attr = qualified_name.rsplit(".", 1)
    return model.get_submodule(parent_name), attr


def freeze_non_s0_parameters(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.endswith("s0_continuous") and param.requires_grad:
            continue
        param.requires_grad = False


def _resolve_init_linear(init_weight_model: nn.Module, qualified_name: str) -> nn.Linear:
    try:
        module = init_weight_model.get_submodule(qualified_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"pseudo-quant init model missing module: {qualified_name}"
        ) from exc
    if not isinstance(module, nn.Linear):
        raise RuntimeError(
            f"pseudo-quant init module {qualified_name} is {type(module).__name__}, expected nn.Linear"
        )
    return module


def wrap_model_for_qad_frozen_b(
    model: nn.Module,
    *,
    target_modules: Sequence[str] | set[str],
    tune_modules: Sequence[str] | set[str] | None = None,
    target_layers: set[int] | None = None,
    tune_layers: set[int] | None = None,
    config: HiF4Config = HiF4Config(),
    init_weight_model: nn.Module | None = None,
) -> list[str]:
    """把目标 Linear 换成 HiF4FrozenBLinear。

    Args:
        init_weight_model: 可选。结构与 model 对齐的伪量化 HF 模型（建议 CPU）。
            若提供，则用其对应 Linear.weight 做 HiF4，初始化 frozen_b + s0；
            teacher_weight 仍来自 model（BF16）。
    """
    target_module_set = {str(x).strip() for x in target_modules if str(x).strip()}
    if not target_module_set:
        raise ValueError("target_modules must be non-empty")

    if tune_modules is None:
        tune_module_set = set(target_module_set)
    else:
        tune_module_set = {str(x).strip() for x in tune_modules if str(x).strip()}
        if not tune_module_set:
            raise ValueError("tune_modules is empty; pass None to default to target_modules")
        unknown = tune_module_set - target_module_set
        if unknown:
            raise ValueError(f"tune_modules must be a subset of target_modules, unknown={sorted(unknown)}")

    if tune_layers is not None and target_layers is not None:
        if not tune_layers.issubset(target_layers):
            raise ValueError("tune_layers must be a subset of target_layers")

    replacements: list[tuple[str, nn.Linear, bool]] = []
    for name, linear in _iter_named_linears(model):
        if not _name_matches(name, module_names=target_module_set, layer_indices=target_layers):
            continue
        trainable = _name_matches(name, module_names=tune_module_set, layer_indices=tune_layers)
        replacements.append((name, linear, trainable))

    if not replacements:
        raise ValueError(
            "No Linear modules matched target_modules/target_layers. "
            f"target_modules={sorted(target_module_set)} target_layers={target_layers}"
        )

    wrapped_names: list[str] = []
    for name, linear, trainable in replacements:
        parent, attr = _get_parent_and_attr(model, name)
        device = linear.weight.device
        init_weight = None
        if init_weight_model is not None:
            src = _resolve_init_linear(init_weight_model, name)
            if tuple(src.weight.shape) != tuple(linear.weight.shape):
                raise RuntimeError(
                    f"pseudo-quant weight shape mismatch at {name}: "
                    f"{tuple(src.weight.shape)} vs {tuple(linear.weight.shape)}"
                )
            init_weight = src.weight.data
        # 只搬 device：s0_continuous 必须保持 float32，禁止随 weight 铸成 bf16
        # （bf16 S0 反传易 Inf → clip_grad_norm_ 出 NaN → 下一步 round_e6m2 炸）
        wrapped = HiF4FrozenBLinear(
            linear.weight.data,
            None if linear.bias is None else linear.bias.data,
            init_weight=init_weight,
            trainable_s0=trainable,
            config=config,
        )
        wrapped = wrapped.to(device=device)
        if wrapped.s0_continuous.dtype != torch.float32:
            raise RuntimeError(
                f"s0_continuous must be float32 after wrap, got {wrapped.s0_continuous.dtype} ({name})"
            )
        setattr(parent, attr, wrapped)
        wrapped_names.append(name)
        # 释放原 Linear 引用；teacher_weight 已接管 storage（未 clone）
        del linear

    freeze_non_s0_parameters(model)
    return wrapped_names
