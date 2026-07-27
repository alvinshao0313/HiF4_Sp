"""把模型中的 Linear 替换为 HiF4ScaleLinear，并控制哪些 S0 可训。"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import torch
from torch import nn

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHUANCI_DIR = _REPO_ROOT / "ChuanCi"
if str(_CHUANCI_DIR) not in sys.path:
    sys.path.insert(0, str(_CHUANCI_DIR))

from nvfp4_hif4_torch import HiF4Config  # noqa: E402

from hif4_scale_linear import HiF4ScaleLinear  # noqa: E402

__all__ = [
    "collect_hif4_scale_linears",
    "freeze_non_s0_parameters",
    "iter_named_linears",
    "parse_csv_set",
    "parse_layer_indices",
    "wrap_model_for_scale_tuning",
]

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def parse_csv_set(raw: str | None) -> set[str]:
    if raw is None or not str(raw).strip():
        return set()
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def parse_layer_indices(raw: str | None) -> set[int] | None:
    """解析层号列表；None/空字符串表示不限制层号。"""
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


def iter_named_linears(model: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def _get_parent_and_attr(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    if "." not in qualified_name:
        return model, qualified_name
    parent_name, attr = qualified_name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    return parent, attr


def wrap_model_for_scale_tuning(
    model: nn.Module,
    *,
    target_modules: Sequence[str] | set[str],
    tune_modules: Sequence[str] | set[str] | None = None,
    target_layers: set[int] | None = None,
    tune_layers: set[int] | None = None,
    config: HiF4Config = HiF4Config(),
) -> list[str]:
    """替换匹配的 nn.Linear 为 HiF4ScaleLinear。

    - target_*: 替换为 HiF4ScaleLinear（student 路径走 HiF4）
    - tune_*: 这些模块的 s0_continuous.requires_grad=True；默认同 target

    Returns:
        被替换模块的 qualified name 列表。
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
    for name, linear in iter_named_linears(model):
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
        wrapped = HiF4ScaleLinear(
            linear.weight.data,
            None if linear.bias is None else linear.bias.data,
            trainable_s0=trainable,
            config=config,
        )
        setattr(parent, attr, wrapped)
        wrapped_names.append(name)

    freeze_non_s0_parameters(model)
    return wrapped_names


def freeze_non_s0_parameters(model: nn.Module) -> None:
    """冻结除可训 s0_continuous 以外的全部参数。"""
    for name, param in model.named_parameters():
        if name.endswith("s0_continuous") and param.requires_grad:
            continue
        param.requires_grad = False


def collect_hif4_scale_linears(model: nn.Module) -> list[tuple[str, HiF4ScaleLinear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, HiF4ScaleLinear)]
