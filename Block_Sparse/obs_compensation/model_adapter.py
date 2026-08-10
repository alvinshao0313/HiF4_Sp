from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from obs_compensation.calibration import CalibrationSample


class _StopForward(Exception):
    """Private exception used to halt the full-model forward after layer-0 capture."""


@dataclass
class CapturedLayerInputs:
    hidden_states: list[torch.Tensor] = field(default_factory=list)
    layer_kwargs: list[dict[str, Any]] = field(default_factory=list)


def _clone_nested_to_cpu(value: Any, context: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, tuple):
        return tuple(_clone_nested_to_cpu(v, context) for v in value)
    if isinstance(value, list):
        return [_clone_nested_to_cpu(v, context) for v in value]
    if isinstance(value, dict):
        return {k: _clone_nested_to_cpu(v, context) for k, v in value.items()}
    raise TypeError(
        f"{context}: unsupported nested value type {type(value).__name__}"
    )


def _move_nested_to_device(value: Any, device: torch.device) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_move_nested_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _move_nested_to_device(v, device) for k, v in value.items()}
    raise TypeError(
        f"unsupported nested value type for device move: {type(value).__name__}"
    )


def _filter_kwargs_for_callable(
    callable_obj: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    ):
        return dict(kwargs)
    allowed = set(signature.parameters)
    return {k: v for k, v in kwargs.items() if k in allowed}


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidates = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        candidates.append(model.model.layers)
    if hasattr(model, "layers"):
        candidates.append(model.layers)
    for layers in candidates:
        if isinstance(layers, nn.ModuleList):
            return layers
    raise RuntimeError(
        "Cannot locate decoder layers as nn.ModuleList on model.model.layers "
        "or model.layers"
    )


@torch.no_grad()
def capture_first_decoder_layer_inputs(
    model: nn.Module,
    samples: list[CalibrationSample],
) -> CapturedLayerInputs:
    layers = get_decoder_layers(model)
    if len(layers) < 1:
        raise RuntimeError("model has no decoder layers")
    captured = CapturedLayerInputs()

    def capture_hook(module, args, kwargs):
        del module
        if args:
            hidden = args[0]
        elif "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]
        else:
            raise RuntimeError("layer-0 hook missing hidden_states")
        saved_kwargs = dict(kwargs)
        saved_kwargs.pop("hidden_states", None)
        captured.hidden_states.append(
            _clone_nested_to_cpu(hidden, "layer0.hidden_states")
        )
        captured.layer_kwargs.append(
            _clone_nested_to_cpu(saved_kwargs, "layer0.kwargs")
        )
        raise _StopForward()

    handle = layers[0].register_forward_pre_hook(capture_hook, with_kwargs=True)
    try:
        input_device = next(model.parameters()).device
        if hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                input_device = emb.weight.device
        for sample in samples:
            try:
                model(
                    input_ids=sample.input_ids.to(input_device),
                    attention_mask=sample.attention_mask.to(input_device),
                    use_cache=False,
                )
            except _StopForward:
                pass
            else:
                raise RuntimeError(
                    "expected _StopForward from layer-0 capture hook, but forward completed"
                )
    finally:
        handle.remove()

    if len(captured.hidden_states) != len(samples):
        raise RuntimeError(
            f"captured {len(captured.hidden_states)} inputs for {len(samples)} samples"
        )
    return captured


def prepare_layer_kwargs(
    model: nn.Module,
    layer: nn.Module,
    hidden_states: torch.Tensor,
    base_layer_kwargs: dict[str, Any],
) -> dict[str, Any]:
    device = hidden_states.device
    kwargs = _move_nested_to_device(base_layer_kwargs, device)
    model_type = getattr(getattr(model, "config", None), "model_type", None)

    if model_type == "qwen3_5_text":
        layer_type = getattr(layer, "layer_type", None)
        seq_len = int(hidden_states.shape[1])
        batch = int(hidden_states.shape[0])
        if layer_type == "linear_attention":
            kwargs["attention_mask"] = None
        elif layer_type == "full_attention":
            min_val = torch.finfo(hidden_states.dtype).min
            causal = torch.zeros(
                (batch, 1, seq_len, seq_len),
                dtype=hidden_states.dtype,
                device=device,
            )
            upper = torch.triu(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
                diagonal=1,
            )
            causal = causal.masked_fill(upper, min_val)
            kwargs["attention_mask"] = causal
        else:
            raise RuntimeError(
                f"Unsupported qwen3_5_text layer_type={layer_type!r}"
            )
        kwargs["use_cache"] = False
    else:
        if "use_cache" in kwargs:
            kwargs["use_cache"] = False

    return _filter_kwargs_for_callable(layer.forward, kwargs)


def run_decoder_layer(
    model: nn.Module,
    layer: nn.Module,
    hidden_states: torch.Tensor,
    base_layer_kwargs: dict[str, Any],
) -> torch.Tensor:
    kwargs = prepare_layer_kwargs(
        model=model,
        layer=layer,
        hidden_states=hidden_states,
        base_layer_kwargs=base_layer_kwargs,
    )
    output = layer(hidden_states, **kwargs)
    if isinstance(output, tuple):
        if not output:
            raise RuntimeError("decoder layer returned empty tuple")
        hidden_out = output[0]
    elif isinstance(output, torch.Tensor):
        hidden_out = output
    elif hasattr(output, "last_hidden_state"):
        hidden_out = output.last_hidden_state
    else:
        raise TypeError(
            f"Unsupported decoder layer output type: {type(output).__name__}"
        )
    if not isinstance(hidden_out, torch.Tensor):
        raise TypeError("decoder layer output hidden state is not a Tensor")
    if hidden_out.shape != hidden_states.shape:
        raise ValueError(
            f"layer output shape {tuple(hidden_out.shape)} != "
            f"input shape {tuple(hidden_states.shape)}"
        )
    return hidden_out
