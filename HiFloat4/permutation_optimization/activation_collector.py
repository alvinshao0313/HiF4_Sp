"""Collect down_proj input activations with a single model forward pass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn

from .config import MLPLayerSpec
from .model_permutation import get_mlp_modules


@dataclass
class ActivationCache:
    by_layer: dict[str, torch.Tensor] = field(default_factory=dict)  # down inputs [rows, d_ff]
    mlp_input_by_layer: dict[str, torch.Tensor] = field(default_factory=dict)  # X [rows, d_model]


@torch.no_grad()
def collect_down_inputs(
    model: nn.Module,
    layer_specs: list[MLPLayerSpec],
    calibration_batches: Iterable[dict[str, torch.Tensor]],
    input_device: torch.device,
    max_rows: int,
    seed: int,
    max_rows_per_batch: int | None = None,
) -> ActivationCache:
    """One forward: sample the same valid tokens for every down_proj input.

    Hooks are always removed, including on exceptions.

    Also collects each MLP's input ``X`` (via gate_proj pre-hook) at the same
    sampled token positions, aligned 1:1 with the down inputs.

    ``max_rows_per_batch`` caps tokens taken from each calibration batch so that
    long sequences (e.g. full s1k) cannot monopolize the entire ``max_rows``
    budget from the first sample.
    """
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    if max_rows_per_batch is not None and max_rows_per_batch <= 0:
        raise ValueError(f"max_rows_per_batch must be positive, got {max_rows_per_batch}")
    if not layer_specs:
        return ActivationCache()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    buffers: dict[str, list[torch.Tensor]] = {spec.name: [] for spec in layer_specs}
    counts: dict[str, int] = {spec.name: 0 for spec in layer_specs}
    x_buffers: dict[str, list[torch.Tensor]] = {spec.name: [] for spec in layer_specs}
    x_counts: dict[str, int] = {spec.name: 0 for spec in layer_specs}
    hooks: list = []

    batch_state: dict[str, torch.Tensor | None] = {"flat_idx": None}

    def _make_hook(layer_name: str, d_ff: int):
        def hook(_module: nn.Module, inputs: tuple) -> None:
            if counts[layer_name] >= max_rows:
                return
            if not inputs:
                raise RuntimeError(f"down_proj hook for {layer_name} received empty inputs")
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Expected Tensor input for {layer_name}, got {type(x)}")
            if x.shape[-1] != d_ff:
                raise ValueError(
                    f"{layer_name}: expected last dim {d_ff}, got {x.shape[-1]}"
                )
            flat = x.detach().reshape(-1, d_ff)
            idx = batch_state["flat_idx"]
            if idx is None:
                raise RuntimeError("Token sample indices were not prepared for this batch")
            n = flat.shape[0]
            use = idx[idx < n]
            if use.numel() == 0:
                return
            remaining = max_rows - counts[layer_name]
            if use.numel() > remaining:
                use = use[:remaining]
            rows = flat.index_select(0, use.to(device=flat.device)).to(
                device="cpu", dtype=torch.float16
            )
            buffers[layer_name].append(rows)
            counts[layer_name] += rows.shape[0]

        return hook

    def _make_x_hook(layer_name: str):
        def hook(_module: nn.Module, inputs: tuple) -> None:
            if x_counts[layer_name] >= max_rows:
                return
            if not inputs:
                raise RuntimeError(f"gate_proj hook for {layer_name} received empty inputs")
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Expected Tensor input for {layer_name}, got {type(x)}")
            d_model = x.shape[-1]
            flat = x.detach().reshape(-1, d_model)
            idx = batch_state["flat_idx"]
            if idx is None:
                raise RuntimeError("Token sample indices were not prepared for this batch")
            n = flat.shape[0]
            use = idx[idx < n]
            if use.numel() == 0:
                return
            remaining = max_rows - x_counts[layer_name]
            if use.numel() > remaining:
                use = use[:remaining]
            rows = flat.index_select(0, use.to(device=flat.device)).to(
                device="cpu", dtype=torch.float16
            )
            x_buffers[layer_name].append(rows)
            x_counts[layer_name] += rows.shape[0]

        return hook

    try:
        for spec in layer_specs:
            gate, _, down = get_mlp_modules(model, spec)
            hooks.append(
                down.register_forward_pre_hook(_make_hook(spec.name, spec.intermediate_size))
            )
            hooks.append(gate.register_forward_pre_hook(_make_x_hook(spec.name)))

        was_training = model.training
        model.eval()

        for batch in calibration_batches:
            if all(c >= max_rows for c in counts.values()) and all(
                c >= max_rows for c in x_counts.values()
            ):
                break
            batch_dev = {
                k: (v.to(device=input_device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            input_ids = batch_dev.get("input_ids")
            if input_ids is None:
                raise KeyError("calibration batch must contain 'input_ids'")
            attn = batch_dev.get("attention_mask")
            bsz, seqlen = input_ids.shape[0], input_ids.shape[1]
            n_tok = bsz * seqlen
            if attn is not None:
                valid = attn.reshape(-1).to(dtype=torch.bool, device="cpu")
                valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
            else:
                valid_idx = torch.arange(n_tok, dtype=torch.long)

            if valid_idx.numel() == 0:
                batch_state["flat_idx"] = torch.zeros(0, dtype=torch.long)
            else:
                need = max(max_rows - min(counts.values()), 1)
                if max_rows_per_batch is not None:
                    need = min(need, max_rows_per_batch)
                n_sample = min(need, valid_idx.numel())
                perm = torch.randperm(valid_idx.numel(), generator=gen)
                batch_state["flat_idx"] = valid_idx[perm[:n_sample]]

            forward_kwargs = {
                k: v
                for k, v in batch_dev.items()
                if k in ("input_ids", "attention_mask", "position_ids")
            }
            model(**forward_kwargs)

        if was_training:
            model.train()
    finally:
        for h in hooks:
            h.remove()

    by_layer: dict[str, torch.Tensor] = {}
    mlp_input_by_layer: dict[str, torch.Tensor] = {}
    for spec in layer_specs:
        parts = buffers[spec.name]
        if not parts:
            raise RuntimeError(f"No activations collected for {spec.name}")
        cat = torch.cat(parts, dim=0)
        if cat.shape[0] > max_rows:
            cat = cat[:max_rows]
        by_layer[spec.name] = cat.contiguous()

        x_parts = x_buffers[spec.name]
        if not x_parts:
            raise RuntimeError(f"No MLP input activations collected for {spec.name}")
        x_cat = torch.cat(x_parts, dim=0)
        if x_cat.shape[0] > max_rows:
            x_cat = x_cat[:max_rows]
        mlp_input_by_layer[spec.name] = x_cat.contiguous()
    return ActivationCache(by_layer=by_layer, mlp_input_by_layer=mlp_input_by_layer)
