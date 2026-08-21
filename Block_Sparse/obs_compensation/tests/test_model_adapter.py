from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from obs_compensation.calibration import CalibrationSample, make_calibration_sample
from obs_compensation.model_adapter import (
    _StopForward,
    _clone_nested_to_cpu,
    _move_nested_to_device,
    capture_first_decoder_layer_inputs,
    get_decoder_layers,
    prepare_layer_kwargs,
    run_decoder_layer,
)
from obs_compensation.tests.helpers import TinyCausalLM, TinyDecoderLayer


def test_get_decoder_layers_locations():
    model = TinyCausalLM()
    layers = get_decoder_layers(model)
    assert isinstance(layers, nn.ModuleList)
    assert len(layers) == 2

    wrapped = SimpleNamespace(model=SimpleNamespace(layers=model.layers), config=model.config)
    # wrap as module-like with attributes used by get_decoder_layers
    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = model.layers

    layers2 = get_decoder_layers(Wrap())
    assert len(layers2) == 2

    with pytest.raises(RuntimeError, match="Cannot locate"):
        get_decoder_layers(nn.Linear(2, 2))


def test_nested_clone_and_move():
    t = torch.randn(2, 2)
    nested = {"a": t, "b": [1, (True, t)], "c": None}
    cloned = _clone_nested_to_cpu(nested, "ctx")
    t[0, 0] = 123
    assert cloned["a"][0, 0].item() != 123
    moved = _move_nested_to_device(cloned, torch.device("cpu"))
    assert moved["b"][0] == 1
    with pytest.raises(TypeError):
        _clone_nested_to_cpu(object(), "ctx")


def test_capture_first_layer_inputs():
    model = TinyCausalLM()
    samples = [
        make_calibration_sample(torch.tensor([[1, 2, 3, 4]], dtype=torch.long)),
        make_calibration_sample(torch.tensor([[5, 6, 7, 8]], dtype=torch.long)),
    ]
    captured = capture_first_decoder_layer_inputs(model, samples)
    assert len(captured.hidden_states) == 2
    assert len(captured.layer_kwargs) == 2
    assert all(h.device.type == "cpu" for h in captured.hidden_states)
    # no leftover hooks
    assert capture_first_decoder_layer_inputs.__name__
    layers = get_decoder_layers(model)
    assert layers[0]._forward_pre_hooks == {}


def test_prepare_kwargs_qwen_masks():
    class Layer(nn.Module):
        def __init__(self, layer_type):
            super().__init__()
            self.layer_type = layer_type

        def forward(self, hidden_states, attention_mask=None, use_cache=True, **kwargs):
            del kwargs
            return hidden_states

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_5_text")

    model = Model()
    hidden = torch.zeros(1, 4, 8, dtype=torch.float32)
    linear_layer = Layer("linear_attention")
    kwargs = prepare_layer_kwargs(model, linear_layer, hidden, {"attention_mask": torch.ones(1, 4)})
    assert kwargs["attention_mask"] is None
    assert kwargs["use_cache"] is False

    full_layer = Layer("full_attention")
    kwargs = prepare_layer_kwargs(model, full_layer, hidden, {})
    mask = kwargs["attention_mask"]
    assert mask.shape == (1, 1, 4, 4)
    assert torch.all(mask.diagonal(dim1=-2, dim2=-1) == 0)
    min_val = torch.finfo(hidden.dtype).min
    upper = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.all(mask[0, 0][upper] == min_val)
    assert torch.all(mask[0, 0][~upper] == 0)

    bad = Layer("other")
    with pytest.raises(RuntimeError, match="Unsupported qwen3_5_text"):
        prepare_layer_kwargs(model, bad, hidden, {})


def test_capture_first_layer_inputs_disables_grad():
    grad_states: list[bool] = []

    class RecordingEmbedding(nn.Embedding):
        def forward(self, input_ids):
            grad_states.append(torch.is_grad_enabled())
            return super().forward(input_ids)

    model = TinyCausalLM()
    replacement = RecordingEmbedding(
        model.embed_tokens.num_embeddings,
        model.embed_tokens.embedding_dim,
    )
    replacement.load_state_dict(model.embed_tokens.state_dict())
    model.embed_tokens = replacement
    samples = [
        make_calibration_sample(torch.tensor([[1, 2, 3, 4]], dtype=torch.long)),
        make_calibration_sample(torch.tensor([[5, 6, 7, 8]], dtype=torch.long)),
    ]

    with torch.enable_grad():
        captured = capture_first_decoder_layer_inputs(model, samples)

    assert grad_states == [False, False]
    assert all(not hidden.requires_grad for hidden in captured.hidden_states)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_run_decoder_layer_output_shapes():
    model = TinyCausalLM()
    layer = model.layers[0]
    hidden = torch.randn(1, 3, 4)
    out = run_decoder_layer(model, layer, hidden, {})
    assert out.shape == hidden.shape
