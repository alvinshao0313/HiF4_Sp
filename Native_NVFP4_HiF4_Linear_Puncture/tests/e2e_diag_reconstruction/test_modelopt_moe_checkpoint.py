from __future__ import annotations

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    load_qwen3_moe_layer_state,
    release_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import QWEN3_30B_A3B_NVFP4
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot


@pytest.mark.parametrize("layer_idx", (0, 23, 47))
def test_lazy_layer_state_contract(layer_idx: int):
    snapshot = resolve_local_snapshot(QWEN3_30B_A3B_NVFP4)
    state = load_qwen3_moe_layer_state(snapshot, layer_idx, "cpu")
    assert state.layer_idx == layer_idx
    assert tuple(state.attention["q_proj"].shape) == (4096, 2048)
    assert tuple(state.attention["k_proj"].shape) == (512, 2048)
    assert tuple(state.attention["v_proj"].shape) == (512, 2048)
    assert tuple(state.attention["o_proj"].shape) == (2048, 4096)
    assert len(state.experts) == 128
    assert tuple(state.experts[0].gate_proj.shape) == (768, 2048)
    assert tuple(state.experts[127].up_proj.shape) == (768, 2048)
    assert tuple(state.experts[127].down_proj.shape) == (2048, 768)
    assert torch.equal(state.experts[0].gate_metadata.input_global_scale_inv, state.experts[127].gate_metadata.input_global_scale_inv)
    release_qwen3_moe_layer_state(state)
    assert not state.attention
    assert not state.attention_metadata
    assert not state.experts
    assert state.router_weight.numel() == 0
