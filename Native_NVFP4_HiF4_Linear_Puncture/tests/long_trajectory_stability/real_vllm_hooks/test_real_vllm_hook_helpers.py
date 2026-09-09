from __future__ import annotations

from pathlib import Path

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks import forced_trajectory as ft
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.hook_spec import (
    build_probe_map,
    predictor_abs_position,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.hook_state import (
    HookCaptureState,
)


def test_predictor_position_mapping() -> None:
    assert predictor_abs_position(346, 24) == 369
    assert build_probe_map(346, [{"decode_index": 12}, {"decode_index": 24}]) == {
        357: 12,
        369: 24,
    }


def test_hook_state_lifecycle(tmp_path: Path) -> None:
    state = HookCaptureState("E0", 0, 2, str(tmp_path))
    state.begin_sample("sample", 100, {109: 10})
    assert state.sample_key == "sample"
    assert state.probe_decode_indices == {10}
    with pytest.raises(RuntimeError):
        state.begin_sample("other", 100, {})
    state.records.append({"x": 1})
    state.clear_sample()
    assert state.sample_key is None
    assert state.records == []
    assert state.probe_decode_indices == set()


def test_forced_processor_keeps_only_target_and_captures_raw_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ft, "get_tensor_model_parallel_rank", lambda: 0)
    proc = ft._ForcedTrajectoryRequestProcessor(
        [2, 1],
        sample_key="s",
        variant="E0",
        probe_decode_indices={0},
        logits_root=str(tmp_path),
    )
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = proc([], logits)
    assert torch.isneginf(out[[0, 1, 3]]).all()
    assert out[2].item() == 3.0
    saved = torch.load(tmp_path / "E0/s/rank0_decode0.pt", weights_only=False)
    assert torch.equal(saved["logits"], torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_forced_processor_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ft, "get_tensor_model_parallel_rank", lambda: 0)
    proc = ft._ForcedTrajectoryRequestProcessor(
        [2], sample_key=None, variant=None, probe_decode_indices=set(), logits_root=None
    )
    with pytest.raises(RuntimeError, match="exhausted"):
        proc([2], torch.zeros(4))


def test_worker_hook_prefill_and_decode_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch.nn as nn
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks import worker_hooks as wh

    monkeypatch.setattr(wh, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(wh, "get_tensor_model_parallel_world_size", lambda: 2)

    class Norm(nn.Module):
        def forward(self, x, residual=None):
            if residual is None:
                return x + 1
            updated = x + residual
            return updated + 1, updated

    class TupleLinear(nn.Module):
        def forward(self, x):
            return x + 1, None

    class AttnCore(nn.Module):
        def forward(self, x):
            return x + 1

    class SelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = AttnCore()
            self.o_proj = TupleLinear()

        def forward(self, positions, hidden_states):
            return self.o_proj(self.attn(hidden_states))[0]

    class Gate(nn.Module):
        def forward(self, x):
            return x[..., :4], None

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = Gate()

        def forward(self, x):
            self.gate(x)
            return x + 1

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = Norm()
            self.self_attn = SelfAttn()
            self.post_attention_layernorm = Norm()
            self.mlp = Mlp()

        def forward(self, positions, hidden_states, residual):
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.self_attn(positions, hidden_states)
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer() for _ in range(48)])
            self.norm = Norm()

        def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            hidden = inputs_embeds
            residual = None
            for layer in self.layers:
                hidden, residual = layer(positions, hidden, residual)
            hidden, residual = self.norm(hidden, residual)
            return hidden

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    model = Outer()
    install = wh.install_hooks(model, "E0", str(tmp_path))
    assert install["num_layers"] == 48
    assert install["position_hook_module"] == "Outer"
    wh.begin_sample(model, "s", 4, {3: 0, 4: 1})
    # Outer module must be invoked so the position pre-hook runs (mirrors real
    # Qwen3MoeForCausalLM.__call__; inner @support_torch_compile bypasses hooks).
    model(None, torch.tensor([0, 1, 2, 3]), None, torch.ones(4, 8))
    model(None, torch.tensor([4]), None, torch.ones(1, 8))
    flushed = wh.flush_sample(model)
    assert flushed["prefill_done"] is True
    payload = torch.load(tmp_path / "E0/s/rank0.pt", weights_only=False)
    assert {row["decode_index"] for row in payload["records"]} == {0, 1}
    assert {row["boundary"] for row in payload["records"]} >= {
        "layer_in",
        "input_norm",
        "attention_core",
        "o_proj",
        "post_attn_norm",
        "router_logits",
        "moe_out",
        "layer_out",
        "final_norm",
    }
    wh.remove_hooks(model)
