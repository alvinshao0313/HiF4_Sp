"""Discover SwiGLU MLP triples and apply channel permutations offline."""

from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.nn as nn

from .config import MLPLayerSpec

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _get_by_name(model: nn.Module, name: str) -> nn.Module:
    mod: nn.Module = model
    for part in name.split("."):
        if not hasattr(mod, part):
            raise KeyError(f"Module path not found: {name}")
        mod = getattr(mod, part)
    return mod


def discover_swiglu_mlps(model: nn.Module) -> list[MLPLayerSpec]:
    """Find complete gate/up/down SwiGLU MLPs, ordered by layer index."""
    named = dict(model.named_modules())
    mlp_prefixes: set[str] = set()

    for name, module in named.items():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith(".gate_proj") or name.endswith(".up_proj") or name.endswith(".down_proj"):
            prefix = name.rsplit(".", 1)[0]
            mlp_prefixes.add(prefix)

    specs: list[MLPLayerSpec] = []
    for prefix in sorted(mlp_prefixes):
        gate_name = f"{prefix}.gate_proj"
        up_name = f"{prefix}.up_proj"
        down_name = f"{prefix}.down_proj"
        missing = [n for n in (gate_name, up_name, down_name) if n not in named]
        if missing:
            raise ValueError(
                f"Incomplete SwiGLU MLP at {prefix!r}: missing {missing}. "
                "Partial triples are not supported."
            )
        gate = named[gate_name]
        up = named[up_name]
        down = named[down_name]
        if not all(isinstance(m, nn.Linear) for m in (gate, up, down)):
            raise TypeError(f"Expected nn.Linear for gate/up/down under {prefix}")

        if gate.weight.shape != up.weight.shape:
            raise ValueError(
                f"gate/up shape mismatch at {prefix}: "
                f"{tuple(gate.weight.shape)} vs {tuple(up.weight.shape)}"
            )
        if gate.weight.ndim != 2:
            raise ValueError(f"Expected 2D weights at {prefix}")
        d_ff, d_model = gate.weight.shape
        if down.weight.shape != (d_model, d_ff):
            raise ValueError(
                f"down_proj shape mismatch at {prefix}: "
                f"expected {(d_model, d_ff)}, got {tuple(down.weight.shape)}"
            )
        if d_ff % 64 != 0:
            raise ValueError(
                f"intermediate_size {d_ff} at {prefix} is not divisible by 64"
            )

        m = _LAYER_INDEX_RE.search(prefix)
        layer_index = int(m.group(1)) if m else len(specs)
        specs.append(
            MLPLayerSpec(
                name=prefix,
                layer_index=layer_index,
                gate_name=gate_name,
                up_name=up_name,
                down_name=down_name,
                intermediate_size=d_ff,
            )
        )

    specs.sort(key=lambda s: (s.layer_index, s.name))
    return specs


def get_mlp_modules(
    model: nn.Module,
    spec: MLPLayerSpec,
) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    """Return (gate, up, down) Linear modules for a layer spec."""
    gate = _get_by_name(model, spec.gate_name)
    up = _get_by_name(model, spec.up_name)
    down = _get_by_name(model, spec.down_name)
    if not all(isinstance(m, nn.Linear) for m in (gate, up, down)):
        raise TypeError(f"Expected nn.Linear for {spec.name}")
    return gate, up, down  # type: ignore[return-value]


def validate_permutation(perm: torch.Tensor, size: int) -> torch.Tensor:
    """Validate 1-D permutation covering 0..size-1 without duplicates."""
    if not isinstance(perm, torch.Tensor):
        raise TypeError(f"perm must be a Tensor, got {type(perm)}")
    if perm.ndim != 1:
        raise ValueError(f"perm must be 1-D, got shape {tuple(perm.shape)}")
    if perm.numel() != size:
        raise ValueError(f"perm length {perm.numel()} != size {size}")
    perm_long = perm.detach().to(dtype=torch.long, device="cpu").contiguous()
    if perm_long.min().item() < 0 or perm_long.max().item() >= size:
        raise ValueError(f"perm values must be in [0, {size})")
    unique = torch.unique(perm_long)
    if unique.numel() != size:
        raise ValueError("perm must cover each index in 0..size-1 exactly once")
    return perm_long


@torch.no_grad()
def apply_mlp_permutation_(
    gate: nn.Linear,
    up: nn.Linear,
    down: nn.Linear,
    perm: torch.Tensor,
) -> None:
    """In-place rewrite weight contents; Parameter object/dtype/device unchanged."""
    d_ff = gate.weight.shape[0]
    if up.weight.shape != gate.weight.shape:
        raise ValueError("gate/up weight shape mismatch")
    if down.weight.shape[1] != d_ff:
        raise ValueError(
            f"down in_features {down.weight.shape[1]} != gate out_features {d_ff}"
        )
    perm_long = validate_permutation(perm, d_ff)
    device = gate.weight.device
    perm_dev = perm_long.to(device=device)

    gate_w_id = id(gate.weight)
    up_w_id = id(up.weight)
    down_w_id = id(down.weight)

    gate.weight.data.copy_(gate.weight.data[perm_dev, :].clone())
    up.weight.data.copy_(up.weight.data[perm_dev, :].clone())
    down.weight.data.copy_(down.weight.data[:, perm_dev].clone())

    if gate.bias is not None:
        gate.bias.data.copy_(gate.bias.data[perm_dev].clone())
    if up.bias is not None:
        up.bias.data.copy_(up.bias.data[perm_dev].clone())

    if id(gate.weight) != gate_w_id or id(up.weight) != up_w_id or id(down.weight) != down_w_id:
        raise RuntimeError("Parameter object identity changed during permutation apply")


@torch.no_grad()
def apply_permutations_from_dict(
    model: nn.Module,
    permutations: dict[str, torch.Tensor],
    layer_specs: Iterable[MLPLayerSpec] | None = None,
) -> None:
    """Apply a {mlp_name: perm} mapping to the model in place."""
    specs = list(layer_specs) if layer_specs is not None else discover_swiglu_mlps(model)
    by_name = {s.name: s for s in specs}
    for name, perm in permutations.items():
        if name not in by_name:
            raise KeyError(f"Permutation key {name!r} not found among discovered MLPs")
        gate, up, down = get_mlp_modules(model, by_name[name])
        apply_mlp_permutation_(gate, up, down, perm)


@torch.no_grad()
def apply_permutations_from_file(model: nn.Module, path: str) -> None:
    """Load permutations.pt and apply offline."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    apply_permutations_from_dict(model, obj)
