"""Model hooks: activation collection and configurable-threshold Linear wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import HiF4QuantConfig, quantize_hif4

LINEAR_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "gate_proj",
    "down_proj",
)


def module_type_of(name: str) -> str | None:
    for s in LINEAR_SUFFIXES:
        if name.endswith(s) or f".{s}" in name:
            # Prefer exact suffix match
            if name.endswith(s):
                return s
    return None


def iter_target_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    out: list[tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and module_type_of(name) is not None:
            if mod.weight.shape[-1] % 64 == 0 or mod.weight.shape[0] % 64 == 0:
                out.append((name, mod))
    return out


@dataclass
class ActivationStore:
    """Per-layer input activations on CPU (float16), plus weight column energy."""

    inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    weight_col_energy: dict[str, torch.Tensor] = field(default_factory=dict)


class HiF4ActQuantLinear(nn.Module):
    """Linear with online HiF4 activation fake-quant using fixed (d,t8,t4).

    No candidate search at runtime.
    """

    def __init__(
        self,
        linear: nn.Linear,
        config: HiF4QuantConfig,
        *,
        quantize_input: bool = True,
        group_dim: int = -1,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.config = config
        self.quantize_input = quantize_input
        self.group_dim = group_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_input:
            # Quantize along last dim; require divisible by 64.
            if x.shape[-1] % 64 != 0:
                raise ValueError(
                    f"HiF4ActQuantLinear requires last dim % 64 == 0, got {x.shape[-1]}"
                )
            cfg = HiF4QuantConfig(
                group_size=self.config.group_size,
                group_dim=self.group_dim,
                s0_divisor=self.config.s0_divisor,
                e8_threshold=self.config.e8_threshold,
                e4_threshold=self.config.e4_threshold,
                s0_mode=self.config.s0_mode,
            )
            xq = quantize_hif4(x, config=cfg).reconstruction.to(dtype=x.dtype)
        else:
            xq = x
        return F.linear(xq, self.weight, self.bias)


def replace_linears_with_act_quant(
    model: nn.Module,
    param_map: dict[str, HiF4QuantConfig],
    *,
    default: HiF4QuantConfig | None = None,
) -> list[str]:
    """Replace target Linears with HiF4ActQuantLinear. Returns replaced names."""
    replaced: list[str] = []
    for name, mod in list(iter_target_linears(model)):
        cfg = param_map.get(name, default)
        if cfg is None:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapped = HiF4ActQuantLinear(mod, cfg)
        setattr(parent, child, wrapped)
        replaced.append(name)
    return replaced


@torch.no_grad()
def collect_layer_inputs(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    max_rows: int,
    seed: int = 0,
    layer_filter: list[str] | None = None,
    max_rows_per_batch: int | None = None,
) -> ActivationStore:
    """Collect input activations for target linears (shared token sampling)."""
    targets = iter_target_linears(model)
    if layer_filter is not None:
        allow = set(layer_filter)
        targets = [(n, m) for n, m in targets if n in allow]
    store = ActivationStore()
    for name, mod in targets:
        w = mod.weight.detach().float()
        # Column energy for diagonal output-MSE approx: ||W[:,j]||^2 for input dim j
        # F.linear uses y = x @ W.T, so input dim j contributes via W[out, j]
        store.weight_col_energy[name] = (w * w).sum(dim=0).cpu()

    buffers: dict[str, list[torch.Tensor]] = {n: [] for n, _ in targets}
    counts: dict[str, int] = {n: 0 for n, _ in targets}
    hooks = []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    batch_idx: dict[str, torch.Tensor | None] = {"idx": None}

    def make_hook(layer_name: str, in_features: int):
        def hook(_m, inputs):
            if counts[layer_name] >= max_rows:
                return
            x = inputs[0]
            flat = x.detach().reshape(-1, in_features)
            idx = batch_idx["idx"]
            if idx is None:
                return
            use = idx[idx < flat.shape[0]]
            if use.numel() == 0:
                return
            rem = max_rows - counts[layer_name]
            if use.numel() > rem:
                use = use[:rem]
            rows = flat.index_select(0, use.to(flat.device)).to(device="cpu", dtype=torch.float16)
            buffers[layer_name].append(rows)
            counts[layer_name] += rows.shape[0]

        return hook

    try:
        for name, mod in targets:
            hooks.append(mod.register_forward_pre_hook(make_hook(name, mod.in_features)))
        was_training = model.training
        model.eval()
        for batch in batches:
            if all(c >= max_rows for c in counts.values()):
                break
            # Prepare shared indices from sequence length estimate
            input_ids = batch.get("input_ids")
            if input_ids is None:
                raise KeyError("batch must contain input_ids")
            # Approximate flat token count after reshape of [B,S,H] -> we sample up to B*S
            bsz, seq = input_ids.shape[:2]
            nflat = bsz * seq
            per_batch_cap = max_rows_per_batch if max_rows_per_batch is not None else max_rows
            perm = torch.randperm(nflat, generator=gen)[: min(nflat, per_batch_cap)]
            batch_idx["idx"] = perm
            batch_dev = {
                k: (v.to(device=device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            model(**batch_dev)
        model.train(was_training)
    finally:
        for h in hooks:
            h.remove()

    for name, parts in buffers.items():
        if parts:
            store.inputs[name] = torch.cat(parts, dim=0)[:max_rows]
    return store
