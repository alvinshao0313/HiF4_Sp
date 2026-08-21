from __future__ import annotations

import torch
import torch.nn as nn

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    sample_from_ids_and_mask,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.layer_runtime import (
    ProgressiveHiddenCache,
    build_initial_hidden_cache,
    capture_qwen3_pre_layer_call,
    hidden_from_prepared,
    propagate_native_layer,
    run_decoder_layer,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.metrics import relative_l2


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden: int, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        return hidden_states * self.scale


class TinyInner(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, hidden)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(torch.arange(16 * hidden).float().reshape(16, hidden))
        self.layers = nn.ModuleList(
            [TinyDecoderLayer(hidden, 2.0), TinyDecoderLayer(hidden, 3.0)]
        )
        self.norm = nn.Identity()

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        h = self.embed_tokens(input_ids)
        pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask, position_ids=pos)
        return self.norm(h)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyInner()

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, use_cache=use_cache)


def test_capture_and_replay_matches_layer1_input():
    model = TinyLM()
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mask = torch.ones_like(ids)
    captured = {}

    def hook(_m, args, kwargs):
        captured["h"] = args[0].detach().clone()

    h0 = model.model.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    h1 = model.model.layers[1].register_forward_pre_hook(hook, with_kwargs=True)
    # capture layer0 then layer1 by running full model twice with separate hooks
    h0.remove()
    h1.remove()

    layer0_in = {}
    layer1_in = {}

    def hook0(_m, args, kwargs):
        layer0_in["h"] = args[0].detach().clone()

    def hook1(_m, args, kwargs):
        layer1_in["h"] = args[0].detach().clone()

    a = model.model.layers[0].register_forward_pre_hook(hook0, with_kwargs=True)
    b = model.model.layers[1].register_forward_pre_hook(hook1, with_kwargs=True)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, use_cache=False)
    a.remove()
    b.remove()

    prepared = capture_qwen3_pre_layer_call(model, ids, mask)
    x0 = hidden_from_prepared(prepared)
    assert torch.equal(x0, layer0_in["h"])
    y0 = run_decoder_layer(model.model.layers[0], x0, prepared)
    assert relative_l2(y0.float(), layer1_in["h"].float()) < 1e-6


def test_progressive_cache_stores_real_length_only():
    cache = ProgressiveHiddenCache()
    h = torch.randn(5, 8, dtype=torch.bfloat16)
    cache.store("s0", h, 3)
    got = cache.get("s0")
    assert got.shape == (3, 8)
    assert got.device.type == "cpu"
    padded, lengths = cache.assemble(["s0"], "cpu")
    assert padded.shape == (1, 3, 8)
    assert int(lengths[0]) == 3


def test_build_initial_hidden_cache_splits_by_length():
    model = TinyLM()
    s0 = sample_from_ids_and_mask("a", 0, torch.tensor([1, 2, 3]), torch.ones(3, dtype=torch.long), {})
    s1 = sample_from_ids_and_mask("b", 1, torch.tensor([4, 5]), torch.ones(2, dtype=torch.long), {})
    collator = DynamicCalibrationCollator(pad_token_id=0)
    cache = build_initial_hidden_cache(model, [s0, s1], collator, torch.device("cpu"), batch_size=2)
    assert cache.get("a").shape[0] == 3
    assert cache.get("b").shape[0] == 2


def test_teacher_next_hidden_differs_from_student_and_shares_current_input():
    model = TinyLM()
    sample = sample_from_ids_and_mask(
        "a", 0, torch.tensor([1, 2, 3, 4]), torch.ones(4, dtype=torch.long), {}
    )
    collator = DynamicCalibrationCollator(pad_token_id=0)
    device = torch.device("cpu")
    x0 = build_initial_hidden_cache(model, [sample], collator, device, batch_size=1)
    current = x0.get("a").clone()
    native_next = propagate_native_layer(
        model=model,
        layer=model.model.layers[0],
        samples=[sample],
        collator=collator,
        x_cache=x0,
        device=device,
        batch_size=1,
    )
    assert torch.equal(x0.get("a"), current)
    with torch.no_grad():
        model.model.layers[0].scale.fill_(5.0)
    packed = collator([sample])
    prepared = capture_qwen3_pre_layer_call(
        model, packed["input_ids"], packed["attention_mask"]
    )
    hidden, _ = x0.assemble(["a"], device)
    student_y = run_decoder_layer(model.model.layers[0], hidden, prepared)
    student_next = student_y[0, :4].detach().cpu().float()
    teacher_next = native_next.get("a").float()
    assert not torch.allclose(teacher_next, student_next)
    assert torch.allclose(teacher_next, current.float() * 2.0)
    assert torch.allclose(student_next, current.float() * 5.0)
