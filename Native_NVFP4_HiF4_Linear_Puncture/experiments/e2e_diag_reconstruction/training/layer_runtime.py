"""Qwen3 single-layer replay and progressive CPU BF16 hidden cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator,
    build_validation_batches,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    CalibrationSample,
)


class LayerCallAbort(Exception):
    """Stop the full-model forward after capturing the first decoder-layer call."""


@dataclass
class PreparedLayerCall:
    args: tuple
    kwargs: dict


def capture_qwen3_pre_layer_call(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> PreparedLayerCall:
    captured: dict[str, object] = {}

    def hook(_module, args, kwargs):
        captured["args"] = args
        captured["kwargs"] = dict(kwargs)
        raise LayerCallAbort()

    handle = model.model.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        model.eval()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except LayerCallAbort:
        pass
    finally:
        handle.remove()
    if "args" not in captured:
        raise RuntimeError("failed to capture Qwen3 layer-0 inputs")
    return PreparedLayerCall(args=captured["args"], kwargs=captured["kwargs"])  # type: ignore[arg-type]


def hidden_from_prepared(prepared: PreparedLayerCall) -> torch.Tensor:
    if prepared.args:
        return prepared.args[0]
    if "hidden_states" in prepared.kwargs:
        return prepared.kwargs["hidden_states"]
    raise RuntimeError("captured layer call has no hidden_states")


def run_decoder_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    prepared_call: PreparedLayerCall,
) -> torch.Tensor:
    args = list(prepared_call.args)
    kwargs = dict(prepared_call.kwargs)
    if args:
        args[0] = hidden_states
        kwargs.pop("hidden_states", None)
    else:
        kwargs["hidden_states"] = hidden_states
    out = layer(*args, **kwargs)
    if isinstance(out, tuple):
        return out[0]
    return out


class ProgressiveHiddenCache:
    def __init__(self) -> None:
        self._hidden: dict[str, torch.Tensor] = {}

    def store(self, sample_id: str, hidden: torch.Tensor, length: int) -> None:
        if hidden.ndim != 2:
            raise ValueError(f"hidden must be [T,H], got {tuple(hidden.shape)}")
        if length <= 0 or length > int(hidden.shape[0]):
            raise ValueError(f"invalid length={length} for T={hidden.shape[0]}")
        self._hidden[sample_id] = (
            hidden[:length].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        )

    def get(self, sample_id: str) -> torch.Tensor:
        return self._hidden[sample_id]

    def has(self, sample_id: str) -> bool:
        return sample_id in self._hidden

    def clear(self) -> None:
        self._hidden.clear()

    def assemble(
        self,
        sample_ids: list[str],
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hs = [self._hidden[sid] for sid in sample_ids]
        lengths = torch.tensor([int(h.shape[0]) for h in hs], dtype=torch.long)
        tmax = int(lengths.max().item())
        hidden = int(hs[0].shape[1])
        out = torch.zeros(len(hs), tmax, hidden, dtype=torch.bfloat16, device=device)
        for i, h in enumerate(hs):
            out[i, : h.shape[0]] = h.to(device=device)
        return out, lengths


class TeacherTargetCache:
    def __init__(self) -> None:
        self.delta: dict[str, torch.Tensor] = {}
        self.attn: dict[str, torch.Tensor] = {}
        self.mlp: dict[str, torch.Tensor] = {}
        self.output: dict[str, torch.Tensor] = {}
        self.linear_in: dict[str, dict[str, torch.Tensor]] = {}
        self.linear_out: dict[str, dict[str, torch.Tensor]] = {}

    def clear(self) -> None:
        self.delta.clear()
        self.attn.clear()
        self.mlp.clear()
        self.output.clear()
        self.linear_in.clear()
        self.linear_out.clear()


def build_initial_hidden_cache(
    model: nn.Module,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    device: torch.device,
    batch_size: int,
) -> ProgressiveHiddenCache:
    cache = ProgressiveHiddenCache()
    for batch in build_validation_batches(samples, batch_size):
        packed = collator(batch)
        prepared = capture_qwen3_pre_layer_call(
            model,
            packed["input_ids"].to(device),
            packed["attention_mask"].to(device),
        )
        hidden = hidden_from_prepared(prepared)
        lengths = packed["lengths"]
        for i, sample in enumerate(batch):
            n = int(lengths[i].item())
            cache.store(sample.sample_id, hidden[i, :n], n)
    return cache


def propagate_native_layer(
    *,
    model: nn.Module,
    layer: nn.Module,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    device: torch.device,
    batch_size: int,
) -> ProgressiveHiddenCache:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.semantic_hif4 import (
        set_layer_runtime_mode,
    )

    set_layer_runtime_mode(layer, "native_nvfp4")
    layer.eval()
    nxt = ProgressiveHiddenCache()
    for batch in build_validation_batches(samples, batch_size):
        packed = collator(batch)
        prepared = capture_qwen3_pre_layer_call(
            model,
            packed["input_ids"].to(device),
            packed["attention_mask"].to(device),
        )
        sample_ids = [sample.sample_id for sample in batch]
        hidden, _ = x_cache.assemble(sample_ids, device)
        with torch.no_grad():
            y = run_decoder_layer(layer, hidden, prepared)
        for i, sample in enumerate(batch):
            n = int(packed["lengths"][i].item())
            nxt.store(sample.sample_id, y[i, :n], n)
    return nxt
