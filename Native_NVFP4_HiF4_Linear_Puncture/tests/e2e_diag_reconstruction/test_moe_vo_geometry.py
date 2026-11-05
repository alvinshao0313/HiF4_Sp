"""GQA VO compensation uses the actual checkpoint's query-head geometry."""
from __future__ import annotations

import io
import json
import math

import pytest
import torch
from safetensors import safe_open
from transformers.models.qwen3_moe.modeling_qwen3_moe import repeat_kv

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import (
    QWEN3_30B_A3B_NVFP4, QWEN3_235B_A22B_NVFP4, Qwen3MoeModelSpec,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    build_moe_diag_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_transforms import (
    apply_r64_no_cross_head, fusable_weight_transform_no_h,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot


@pytest.mark.parametrize("model", [QWEN3_30B_A3B_NVFP4, QWEN3_235B_A22B_NVFP4])
def test_real_checkpoint_vo_repeat_gradient_reload_and_compensation(model):
    snapshot = resolve_local_snapshot(model, tensor_prefixes=("model.layers.0.self_attn.",))
    spec = Qwen3MoeModelSpec.from_snapshot(snapshot)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    shapes = {}
    for proj in ("q_proj", "v_proj", "o_proj"):
        key = f"model.layers.0.self_attn.{proj}.weight"
        with safe_open(str(snapshot / index[key]), framework="pt", device="cpu") as handle:
            shapes[proj] = handle.get_slice(key).get_shape()
    q_width = spec.num_attention_heads * spec.head_dim
    kv_width = spec.num_key_value_heads * spec.head_dim
    assert shapes["q_proj"] == [q_width, spec.hidden_size // 2]
    assert shapes["v_proj"] == [kv_width, spec.hidden_size // 2]
    assert shapes["o_proj"] == [spec.hidden_size, q_width // 2]

    diag = build_moe_diag_state(spec, "fusable")
    with torch.no_grad():
        diag.z_vo.copy_(torch.linspace(-0.8, 0.7, kv_width))
    repeats = spec.num_attention_heads // spec.num_key_value_heads
    expanded = diag.d_vo_expanded()
    expected = repeat_kv(diag.d_vo().reshape(1, spec.num_key_value_heads, 1, spec.head_dim), repeats)
    torch.testing.assert_close(expanded, expected.reshape(-1), rtol=0, atol=0)
    assert expanded.numel() == q_width
    expanded.sum().backward()
    torch.testing.assert_close(diag.z_vo.grad, repeats * math.log(2) * diag.d_vo(), rtol=1e-6, atol=1e-6)

    saved = io.BytesIO()
    torch.save(diag.snapshot(), saved)
    saved.seek(0)
    restored = build_moe_diag_state(spec, "fusable")
    restored.load_snapshot(torch.load(saved, weights_only=True))
    assert set(restored.snapshot()) == {"z_qkv", "z_vo", "z_gu", "z_ud"}
    torch.testing.assert_close(restored.d_vo_expanded(), expanded, rtol=0, atol=0)

    # Verify the algebra across GQA value replication, attention and the full
    # O weight geometry. This is the pre-quantization equivalence that DIAG
    # must preserve; real quantized training is covered by the GPU short run.
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        value = torch.randn(1, spec.num_key_value_heads, 3, spec.head_dim, generator=generator)
        probabilities = torch.randn(1, spec.num_attention_heads, 2, 3, generator=generator).softmax(-1)
        scale = restored.d_vo().reshape(1, spec.num_key_value_heads, 1, spec.head_dim)
        baseline = (probabilities @ repeat_kv(value, repeats)).transpose(1, 2).reshape(1, 2, q_width)
        scaled = (probabilities @ repeat_kv(value * scale, repeats)).transpose(1, 2).reshape_as(baseline)
        torch.testing.assert_close(scaled, baseline * restored.d_vo_expanded(), rtol=2e-5, atol=2e-6)
        weight = torch.randn(spec.hidden_size, q_width, generator=generator) / math.sqrt(q_width)
        folded = fusable_weight_transform_no_h(
            weight, restored.d_vo_expanded(), torch.ones(spec.hidden_size),
            use_r64=True, head_dim=spec.head_dim,
        )
        actual = apply_r64_no_cross_head(scaled, head_dim=spec.head_dim) @ folded.T
        torch.testing.assert_close(actual, baseline @ weight.T, rtol=2e-4, atol=2e-5)
