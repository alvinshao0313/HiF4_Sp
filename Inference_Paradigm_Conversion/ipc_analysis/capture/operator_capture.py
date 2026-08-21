"""MLP / Attention stage capture hooks (eager path)."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from Inference_Paradigm_Conversion.ipc_analysis.capture.activation_capture import (
    CapturedTensor,
)


@torch.no_grad()
def capture_mlp_stages(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    phase: str,
    *,
    sample_id: str,
    layer_filter: Callable[[int], bool] | None = None,
) -> list[CapturedTensor]:
    """Capture gate/up/down inputs and SiLU(gate)*up intermediate when available."""
    captured: list[CapturedTensor] = []
    handles = []

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if ".mlp." not in name:
            continue
        # layer idx
        parts = name.split(".")
        layer_idx = -1
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_filter is not None and not layer_filter(layer_idx):
            continue

        def make_hook(n=name, li=layer_idx):
            def hook(_m, inputs):
                x = inputs[0].detach().to(torch.bfloat16)
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                captured.append(
                    CapturedTensor(
                        sample_id=sample_id,
                        phase=phase,
                        token_positions=list(range(min(256, x.shape[0]))),
                        layer_idx=li,
                        module_name=n,
                        stage="mlp_linear_input",
                        tensor=x[:256].cpu(),
                    )
                )

            return hook

        handles.append(mod.register_forward_pre_hook(make_hook()))

    try:
        model(**batch)
    finally:
        for h in handles:
            h.remove()
    return captured


@torch.no_grad()
def capture_attention_stages(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    phase: str,
    *,
    sample_id: str,
    layer_filter: Callable[[int], bool] | None = None,
) -> list[CapturedTensor]:
    """Capture q/k/v/o Linear inputs for attention projections."""
    captured: list[CapturedTensor] = []
    handles = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if ".self_attn." not in name and ".linear_attn." not in name:
            continue
        parts = name.split(".")
        layer_idx = -1
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_filter is not None and not layer_filter(layer_idx):
            continue

        def make_hook(n=name, li=layer_idx):
            def hook(_m, inputs):
                x = inputs[0].detach().to(torch.bfloat16)
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                attn_kind = "linear_attn" if ".linear_attn." in n else "full_attention"
                captured.append(
                    CapturedTensor(
                        sample_id=sample_id,
                        phase=phase,
                        token_positions=list(range(min(256, x.shape[0]))),
                        layer_idx=li,
                        module_name=n,
                        stage="attn_linear_input",
                        tensor=x[:256].cpu(),
                        extras={"attention_kind": attn_kind},
                    )
                )

            return hook

        handles.append(mod.register_forward_pre_hook(make_hook()))

    try:
        model(**batch)
    finally:
        for h in handles:
            h.remove()
    return captured
