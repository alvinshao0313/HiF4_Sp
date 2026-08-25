from __future__ import annotations

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core import moe_semantic_hif4 as moe_mod
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    MoEExpertMasterState,
    MoELayerMasterState,
    NativeNvfp4LinearMetadata,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_fold import (
    fold_fusable_moe_layer_state,
    router_alignment_kl,
    router_compensation_topk_gate,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import (
    Qwen3MoeModelSpec,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    MoEFusableDiagState,
    StudentQwen3MoELayerRuntime,
    StudentStepCache,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_transforms import (
    transform_router_weight,
)


def _identity_hif4(monkeypatch) -> None:
    monkeypatch.setattr(moe_mod, "qdq_hif4_ste_bf16", lambda x: x.to(torch.float32))
    monkeypatch.setattr(moe_mod, "qdq_hif4_direct", lambda x, output_dtype: x.to(torch.float32))


def _meta() -> NativeNvfp4LinearMetadata:
    return NativeNvfp4LinearMetadata(torch.tensor(1.0), torch.tensor(1.0))


def _state() -> MoELayerMasterState:
    torch.manual_seed(8)
    spec = Qwen3MoeModelSpec(
        source_model="synthetic",
        architecture="Qwen3MoeForCausalLM",
        model_type="qwen3_moe",
        hidden_size=2048,
        num_layers=1,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=2,
        top_k=1,
        moe_intermediate_size=768,
        decoder_sparse_step=1,
        mlp_only_layers=(),
        norm_topk_prob=True,
    )
    attention = {
        "q_proj": torch.randn(4096, 2048) * 0.005,
        "k_proj": torch.randn(512, 2048) * 0.005,
        "v_proj": torch.randn(512, 2048) * 0.005,
        "o_proj": torch.randn(2048, 4096) * 0.005,
    }
    experts = [
        MoEExpertMasterState(
            gate_proj=torch.randn(768, 2048) * 0.005,
            up_proj=torch.randn(768, 2048) * 0.005,
            down_proj=torch.randn(2048, 768) * 0.005,
            gate_metadata=_meta(),
            up_metadata=_meta(),
            down_metadata=_meta(),
        )
        for _ in range(2)
    ]
    router = torch.zeros(2, 2048, dtype=torch.bfloat16)
    router[0, 0] = 1
    router[1, 1] = 1
    return MoELayerMasterState(
        layer_idx=0,
        spec=spec,
        input_layernorm_weight=torch.ones(2048, dtype=torch.bfloat16),
        post_attention_layernorm_weight=torch.ones(2048, dtype=torch.bfloat16),
        q_norm_weight=torch.ones(128, dtype=torch.bfloat16),
        k_norm_weight=torch.ones(128, dtype=torch.bfloat16),
        router_weight=router,
        attention=attention,
        attention_metadata={k: _meta() for k in attention},
        experts=experts,
    )


def test_router_fp32_algebra_and_bf16_topk_gate():
    torch.manual_seed(0)
    router = torch.randn(8, 2048, dtype=torch.float32) * 0.01
    hidden = torch.randn(16, 2048, dtype=torch.bfloat16)
    diag = MoEFusableDiagState()
    with torch.no_grad():
        diag.z_gu[::2] = 1.0
        diag.z_gu[1::2] = -1.0
    d = diag.d_gu()
    torch.testing.assert_close(
        (hidden.float() * d) @ transform_router_weight(router, d).T,
        hidden.float() @ router.T,
        rtol=2e-6,
        atol=2e-6,
    )
    stats = router_compensation_topk_gate(hidden, router.to(torch.bfloat16), diag, top_k=2)
    assert stats["topk_mismatches"] == 0
    assert stats["topk_mismatch_tokens"] == 0
    assert stats["topk_mismatch_ratio"] == 0.0
    assert stats["max_abs"] >= 0.0
    assert stats["rel_l2"] >= 0.0
    assert stats["kl"] >= 0.0


def test_router_alignment_kl_only_updates_d_gu():
    torch.manual_seed(17)
    hidden = torch.randn(32, 2048, dtype=torch.bfloat16)
    router = torch.randn(8, 2048, dtype=torch.bfloat16) * 0.02
    diag = MoEFusableDiagState(num_experts=2)
    with torch.no_grad():
        diag.z_gu.copy_(torch.linspace(-0.37, 0.41, 2048))
        diag.z_qkv.fill_(0.2)
        diag.z_vo.fill_(-0.1)
        diag.z_ud.fill_(0.3)
    loss = router_alignment_kl(hidden, router, diag, temperature=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert diag.z_gu.grad is not None
    assert torch.isfinite(diag.z_gu.grad).all()
    assert float(diag.z_gu.grad.abs().sum()) > 0.0
    assert diag.z_qkv.grad is None
    assert diag.z_vo.grad is None
    assert diag.z_ud.grad is None


def test_fusable_fold_full_layer_no_qdq_equivalence(monkeypatch):
    _identity_hif4(monkeypatch)
    state = _state()
    diag = MoEFusableDiagState()
    with torch.no_grad():
        diag.z_qkv.fill_(1.0)
        diag.z_vo.fill_(-1.0)
        diag.z_gu.fill_(1.0)
        diag.z_ud[:2].fill_(-1.0)
    folded = fold_fusable_moe_layer_state(state, diag, use_r64=False)
    identity = MoEFusableDiagState()
    before = StudentQwen3MoELayerRuntime(state, diag, use_r64=False, rot_order="diag_then_rot").eval()
    after = StudentQwen3MoELayerRuntime(folded, identity, use_r64=False, rot_order="diag_then_rot").eval()

    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

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
    x = torch.randn(1, 2, 2048, dtype=torch.bfloat16)
    rope = Qwen3MoeRotaryEmbedding(config)
    pos = torch.tensor([[0, 1]])
    with torch.no_grad():
        y_before = before(
            x,
            attention_mask=None,
            position_embeddings=rope(x, pos),
            step_cache=StudentStepCache.new(),
            use_ste=True,
        ).output
        y_after = after(
            x,
            attention_mask=None,
            position_embeddings=rope(x, pos),
            step_cache=StudentStepCache.new(),
            use_ste=True,
        ).output
    torch.testing.assert_close(y_after.float(), y_before.float(), rtol=2e-2, atol=2e-2)
