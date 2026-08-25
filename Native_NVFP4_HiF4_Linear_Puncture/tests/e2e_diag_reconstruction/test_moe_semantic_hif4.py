from __future__ import annotations

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core import moe_semantic_hif4 as moe_mod
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    MoEExpertMasterState,
    MoELayerMasterState,
    NativeNvfp4LinearMetadata,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import (
    Qwen3MoeModelSpec,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    MoEFusableDiagState,
    MoEOnlineDiagState,
    StudentQwen3MoELayerRuntime,
    StudentStepCache,
    build_moe_diag_state,
    forward_student_attention_proj,
    forward_student_expert_proj,
    forward_student_routed_moe,
    summarize_expert_coverage,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_transforms import (
    apply_r64_no_cross_head,
    fusable_weight_transform_no_h,
    online_weight_transform_no_h,
)


def _identity_hif4(monkeypatch) -> None:
    monkeypatch.setattr(moe_mod, "qdq_hif4_ste_bf16", lambda x: x.to(torch.float32))
    monkeypatch.setattr(moe_mod, "qdq_hif4_direct", lambda x, output_dtype: x.to(torch.float32))


def _spec(hidden: int, experts: int, top_k: int, intermediate: int, heads: int = 1, kv_heads: int = 1, head_dim: int | None = None) -> Qwen3MoeModelSpec:
    hd = head_dim if head_dim is not None else hidden // heads
    return Qwen3MoeModelSpec(
        source_model="synthetic",
        architecture="Qwen3MoeForCausalLM",
        model_type="qwen3_moe",
        hidden_size=hidden,
        num_layers=1,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=hd,
        num_experts=experts,
        top_k=top_k,
        moe_intermediate_size=intermediate,
        decoder_sparse_step=1,
        mlp_only_layers=(),
        norm_topk_prob=True,
    )


def _meta() -> NativeNvfp4LinearMetadata:
    return NativeNvfp4LinearMetadata(torch.tensor(1.0), torch.tensor(1.0))


def _synthetic_state(
    *,
    hidden: int = 64,
    experts: int = 3,
    top_k: int = 2,
    intermediate: int = 32,
    heads: int = 1,
    kv_heads: int = 1,
    head_dim: int | None = None,
) -> MoELayerMasterState:
    torch.manual_seed(123)
    spec = _spec(hidden, experts, top_k, intermediate, heads, kv_heads, head_dim)
    q_out = spec.num_attention_heads * spec.head_dim
    kv_out = spec.num_key_value_heads * spec.head_dim
    attention = {
        "q_proj": torch.randn(q_out, hidden) * 0.02,
        "k_proj": torch.randn(kv_out, hidden) * 0.02,
        "v_proj": torch.randn(kv_out, hidden) * 0.02,
        "o_proj": torch.randn(hidden, q_out) * 0.02,
    }
    expert_states = [
        MoEExpertMasterState(
            gate_proj=torch.randn(intermediate, hidden) * 0.02,
            up_proj=torch.randn(intermediate, hidden) * 0.02,
            down_proj=torch.randn(hidden, intermediate) * 0.02,
            gate_metadata=_meta(),
            up_metadata=_meta(),
            down_metadata=_meta(),
        )
        for _ in range(experts)
    ]
    return MoELayerMasterState(
        layer_idx=0,
        spec=spec,
        input_layernorm_weight=torch.ones(hidden, dtype=torch.bfloat16),
        post_attention_layernorm_weight=torch.ones(hidden, dtype=torch.bfloat16),
        q_norm_weight=torch.ones(spec.head_dim, dtype=torch.bfloat16),
        k_norm_weight=torch.ones(spec.head_dim, dtype=torch.bfloat16),
        router_weight=torch.zeros(experts, hidden, dtype=torch.bfloat16),
        attention=attention,
        attention_metadata={name: _meta() for name in attention},
        experts=expert_states,
    )


def test_moe_diag_state_shapes_identity_snapshot_and_clamp():
    spec = _spec(2048, 128, 8, 768, heads=32, kv_heads=4, head_dim=128)
    fusable = build_moe_diag_state(spec, "fusable")
    assert isinstance(fusable, MoEFusableDiagState)
    assert fusable.z_qkv.shape == (2048,)
    assert fusable.z_vo.shape == (512,)
    assert fusable.z_gu.shape == (2048,)
    assert fusable.z_ud.shape == (128, 768)
    assert all(p.requires_grad for p in fusable.parameters())
    snap = fusable.snapshot()
    assert all(torch.equal(v, torch.zeros_like(v)) for v in snap.values())
    with torch.no_grad():
        fusable.z_qkv.fill_(9.0)
    fusable.clamp_log2_((-4.0, 4.0))
    assert float(fusable.z_qkv.max()) == 4.0
    fusable.load_snapshot(snap)
    assert torch.equal(fusable.z_qkv, torch.zeros_like(fusable.z_qkv))

    online = build_moe_diag_state(spec, "online")
    assert isinstance(online, MoEOnlineDiagState)
    assert online.z_q.shape == (2048,)
    assert online.z_o.shape == (4096,)
    assert online.z_gate.shape == (128, 2048)
    assert online.z_down.shape == (128, 768)
    assert not (set(fusable.snapshot()) & {"z_q", "z_gate", "z_down"})


def test_student_attention_proj_fusable_and_online_match_no_quant_math(monkeypatch):
    _identity_hif4(monkeypatch)
    torch.manual_seed(0)
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    weight = torch.randn(32, 64)
    cache = StudentStepCache.new()
    fusable = MoEFusableDiagState(hidden_size=64, num_experts=2, moe_intermediate_size=32, num_key_value_heads=1, head_dim=64)
    with torch.no_grad():
        fusable.z_qkv.copy_(torch.linspace(-0.2, 0.2, 64))
    got = forward_student_attention_proj(
        "q_proj", x, weight, fusable, use_r64=True, rot_order="diag_then_rot", step_cache=cache
    )
    d = fusable.d_qkv()
    expected = apply_r64_no_cross_head(x.float() * d) @ fusable_weight_transform_no_h(
        weight, d, torch.ones(32), use_r64=True
    ).T
    torch.testing.assert_close(got.float(), expected, rtol=3e-3, atol=2e-2)
    assert cache.weight_qdq_calls_by_proj["q_proj"] == 1

    cache = StudentStepCache.new()
    online = MoEOnlineDiagState(hidden_size=64, num_experts=2, moe_intermediate_size=32, o_input_size=64)
    with torch.no_grad():
        online.z_q.copy_(torch.linspace(-0.2, 0.2, 64))
    got = forward_student_attention_proj(
        "q_proj", x, weight, online, use_r64=True, rot_order="rot_then_diag", step_cache=cache
    )
    d = online.d_for("q_proj")
    expected = (apply_r64_no_cross_head(x.float()) * d) @ online_weight_transform_no_h(
        weight, d, use_r64=True, rot_order="rot_then_diag"
    ).T
    torch.testing.assert_close(got.float(), expected, rtol=3e-3, atol=2e-2)


def test_routed_expert_student_qdq_once_for_repeated_expert_tokens(monkeypatch):
    _identity_hif4(monkeypatch)
    state = _synthetic_state()
    diag_state = MoEFusableDiagState(hidden_size=64, num_experts=3, moe_intermediate_size=32, num_key_value_heads=1, head_dim=64)
    # Router logits tie, so torch.topk selects the same two experts for every token.
    x = torch.randn(5, 64, dtype=torch.bfloat16)
    cache = StudentStepCache.new()
    out = forward_student_routed_moe(
        x,
        state,
        diag_state,
        use_r64=False,
        rot_order="diag_then_rot",
        step_cache=cache,
    )
    assert out.output.shape == x.shape
    assert int((out.per_expert_routed_token_count > 0).sum().item()) == 2
    assert cache.weight_qdq_calls_by_proj == {
        "gate_proj": 2,
        "up_proj": 2,
        "down_proj": 2,
    }
    cache.clear()
    assert cache.transformed_weight_qdq == {}


def test_student_expert_proj_fusable_and_online_match_no_quant_math(monkeypatch):
    _identity_hif4(monkeypatch)
    torch.manual_seed(4)
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    weight = torch.randn(32, 64)
    fusable = MoEFusableDiagState(hidden_size=64, num_experts=2, moe_intermediate_size=32, num_key_value_heads=1, head_dim=64)
    with torch.no_grad():
        fusable.z_gu.copy_(torch.linspace(-0.2, 0.2, 64))
    cache = StudentStepCache.new()
    got = forward_student_expert_proj(
        "gate_proj", 0, x, weight, fusable, use_r64=True, rot_order="diag_then_rot", step_cache=cache
    )
    d = fusable.d_gu()
    expected = apply_r64_no_cross_head(x.float() * d) @ fusable_weight_transform_no_h(
        weight, d, torch.ones(32), use_r64=True
    ).T
    torch.testing.assert_close(got.float(), expected, rtol=3e-3, atol=2e-2)

    online = MoEOnlineDiagState(hidden_size=64, num_experts=2, moe_intermediate_size=32, o_input_size=64)
    with torch.no_grad():
        online.z_gate[1].copy_(torch.linspace(-0.2, 0.2, 64))
    cache = StudentStepCache.new()
    got = forward_student_expert_proj(
        "gate_proj", 1, x, weight, online, use_r64=True, rot_order="diag_then_rot", step_cache=cache
    )
    d = online.d_for("gate_proj", 1)
    expected = apply_r64_no_cross_head(x.float() * d) @ online_weight_transform_no_h(
        weight, d, use_r64=True, rot_order="diag_then_rot"
    ).T
    torch.testing.assert_close(got.float(), expected, rtol=3e-3, atol=2e-2)


def test_full_current_layer_student_forward_smoke_and_coverage(monkeypatch):
    _identity_hif4(monkeypatch)
    state = _synthetic_state(
        hidden=2048,
        experts=2,
        top_k=1,
        intermediate=768,
        heads=32,
        kv_heads=4,
        head_dim=128,
    )
    diag_state = MoEFusableDiagState()
    runtime = StudentQwen3MoELayerRuntime(
        state,
        diag_state,
        use_r64=False,
        rot_order="diag_then_rot",
    ).eval()
    x = torch.randn(1, 2, 2048, dtype=torch.bfloat16)
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        hidden_size=2048,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=768,
        norm_topk_prob=True,
    )
    rope = Qwen3MoeRotaryEmbedding(config)
    positions = torch.tensor([[0, 1]])
    cache = StudentStepCache.new()
    with torch.no_grad():
        hidden_after_attention, router_input = runtime.forward_to_router_input(
            x,
            attention_mask=None,
            position_embeddings=rope(x, positions),
            step_cache=StudentStepCache.new(),
            use_ste=True,
        )
        out = runtime(
            x,
            attention_mask=None,
            position_embeddings=rope(x, positions),
            step_cache=cache,
            use_ste=True,
        )
    assert out.output.shape == x.shape
    expected_router_input = moe_mod._rms_norm(
        hidden_after_attention,
        state.post_attention_layernorm_weight,
        runtime.rms_norm_eps,
    )
    torch.testing.assert_close(router_input, expected_router_input)
    stats = summarize_expert_coverage(
        train_counts=out.per_expert_routed_token_count,
        val_counts=torch.zeros_like(out.per_expert_routed_token_count),
        step_cache=cache,
    )
    assert stats.weight_qdq_calls_by_proj["q_proj"] == 1
    assert stats.max_routed_tokens >= stats.min_routed_tokens
    assert set(stats.never_routed_experts).issubset({0, 1})


def test_router_rollback_only_resets_d_gu_and_preserves_candidate():
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_trainer import (
        _adopt_candidate_snapshot,
    )

    candidate = {
        "z_qkv": torch.full((4,), 0.1),
        "z_vo": torch.full((2,), 0.2),
        "z_gu": torch.full((4,), 0.3),
        "z_ud": torch.full((2, 3), 0.4),
    }
    adopted = _adopt_candidate_snapshot(
        candidate,
        loss_rollback_applied=False,
        router_rollback_applied=True,
    )
    torch.testing.assert_close(adopted["z_qkv"], candidate["z_qkv"])
    torch.testing.assert_close(adopted["z_vo"], candidate["z_vo"])
    torch.testing.assert_close(adopted["z_ud"], candidate["z_ud"])
    assert torch.equal(adopted["z_gu"], torch.zeros_like(candidate["z_gu"]))
    assert torch.equal(candidate["z_gu"], torch.full_like(candidate["z_gu"], 0.3))

    full_rollback = _adopt_candidate_snapshot(
        candidate,
        loss_rollback_applied=True,
        router_rollback_applied=False,
    )
    assert all(torch.equal(v, torch.zeros_like(v)) for v in full_rollback.values())
