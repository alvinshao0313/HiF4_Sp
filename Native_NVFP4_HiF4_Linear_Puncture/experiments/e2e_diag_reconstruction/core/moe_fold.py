"""Fusable Qwen3-MoE DIAG fold with strict BF16 router compensation."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    MoEExpertMasterState,
    MoELayerMasterState,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    MoEFusableDiagState,
    MoEOnlineDiagState,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_transforms import (
    fusable_weight_transform_no_h,
    online_weight_transform_no_h,
    transform_router_weight,
)


def fold_fusable_moe_layer_state(
    state: MoELayerMasterState,
    diag_state: MoEFusableDiagState,
    *,
    use_r64: bool,
) -> MoELayerMasterState:
    """Return a folded current-layer master state without mutating input state."""
    ones_hidden = torch.ones(state.spec.hidden_size, dtype=torch.float32, device=diag_state.z_qkv.device)
    ones_q = torch.ones(state.spec.num_attention_heads * state.spec.head_dim, dtype=torch.float32, device=diag_state.z_qkv.device)
    ones_kv = torch.ones(state.spec.num_key_value_heads * state.spec.head_dim, dtype=torch.float32, device=diag_state.z_qkv.device)

    attention = {
        "q_proj": fusable_weight_transform_no_h(
            state.attention["q_proj"], diag_state.d_qkv(), ones_q, use_r64=use_r64
        ).detach(),
        "k_proj": fusable_weight_transform_no_h(
            state.attention["k_proj"], diag_state.d_qkv(), ones_kv, use_r64=use_r64
        ).detach(),
        "v_proj": fusable_weight_transform_no_h(
            state.attention["v_proj"], diag_state.d_qkv(), diag_state.d_vo(), use_r64=use_r64
        ).detach(),
        "o_proj": fusable_weight_transform_no_h(
            state.attention["o_proj"], diag_state.d_vo_expanded(), ones_hidden, use_r64=use_r64, head_dim=state.spec.head_dim
        ).detach(),
    }
    experts: list[MoEExpertMasterState] = []
    ones_moe = torch.ones(state.spec.moe_intermediate_size, dtype=torch.float32, device=diag_state.z_qkv.device)
    for expert_idx, expert in enumerate(state.experts):
        experts.append(
            MoEExpertMasterState(
                gate_proj=fusable_weight_transform_no_h(
                    expert.gate_proj, diag_state.d_gu(), ones_moe, use_r64=use_r64
                ).detach(),
                up_proj=fusable_weight_transform_no_h(
                    expert.up_proj, diag_state.d_gu(), diag_state.d_ud(expert_idx), use_r64=use_r64
                ).detach(),
                down_proj=fusable_weight_transform_no_h(
                    expert.down_proj, diag_state.d_ud(expert_idx), ones_hidden, use_r64=use_r64
                ).detach(),
                gate_metadata=expert.gate_metadata,
                up_metadata=expert.up_metadata,
                down_metadata=expert.down_metadata,
            )
        )

    return replace(
        state,
        input_layernorm_weight=(state.input_layernorm_weight.to(torch.float32) * diag_state.d_qkv()).to(state.input_layernorm_weight.dtype).detach(),
        post_attention_layernorm_weight=(state.post_attention_layernorm_weight.to(torch.float32) * diag_state.d_gu()).to(state.post_attention_layernorm_weight.dtype).detach(),
        router_weight=transform_router_weight(state.router_weight, diag_state.d_gu()).to(state.router_weight.dtype).detach(),
        attention=attention,
        experts=experts,
    )


def fold_online_moe_layer_state(
    state: MoELayerMasterState,
    diag_state: MoEOnlineDiagState,
    *,
    use_r64: bool,
    rot_order: str,
) -> MoELayerMasterState:
    """Fold the Online inverse transform into weights only.

    Online DIAG/R64 stays on the activation side at runtime.  RMSNorm and Router
    remain native; each projection weight receives the exact inverse transform
    used by the training semantic path.
    """
    attention = {
        "q_proj": online_weight_transform_no_h(
            state.attention["q_proj"], diag_state.d_for("q_proj"),
            use_r64=use_r64, rot_order=rot_order,
        ).detach(),
        "k_proj": online_weight_transform_no_h(
            state.attention["k_proj"], diag_state.d_for("k_proj"),
            use_r64=use_r64, rot_order=rot_order,
        ).detach(),
        "v_proj": online_weight_transform_no_h(
            state.attention["v_proj"], diag_state.d_for("v_proj"),
            use_r64=use_r64, rot_order=rot_order,
        ).detach(),
        "o_proj": online_weight_transform_no_h(
            state.attention["o_proj"], diag_state.d_for("o_proj"),
            use_r64=use_r64, rot_order=rot_order, head_dim=state.spec.head_dim,
        ).detach(),
    }
    experts: list[MoEExpertMasterState] = []
    for expert_idx, expert in enumerate(state.experts):
        experts.append(
            MoEExpertMasterState(
                gate_proj=online_weight_transform_no_h(
                    expert.gate_proj,
                    diag_state.d_for("gate_proj", expert_idx),
                    use_r64=use_r64,
                    rot_order=rot_order,
                ).detach(),
                up_proj=online_weight_transform_no_h(
                    expert.up_proj,
                    diag_state.d_for("up_proj", expert_idx),
                    use_r64=use_r64,
                    rot_order=rot_order,
                ).detach(),
                down_proj=online_weight_transform_no_h(
                    expert.down_proj,
                    diag_state.d_for("down_proj", expert_idx),
                    use_r64=use_r64,
                    rot_order=rot_order,
                ).detach(),
                gate_metadata=expert.gate_metadata,
                up_metadata=expert.up_metadata,
                down_metadata=expert.down_metadata,
            )
        )
    return replace(state, attention=attention, experts=experts)


def router_compensation_logits(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    diag_state: MoEFusableDiagState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native and folded BF16 router logits on the same pre-D_GU input."""
    d = diag_state.d_gu().to(device=hidden_states.device)
    hidden_bf16 = hidden_states.to(torch.bfloat16)
    router_bf16 = router_weight.to(device=hidden_states.device, dtype=torch.bfloat16)
    original = hidden_bf16 @ router_bf16.T
    folded_weight = transform_router_weight(router_weight, d).to(
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    folded = (hidden_bf16 * d.to(dtype=torch.bfloat16)) @ folded_weight.T
    return original, folded


def router_alignment_kl(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    diag_state: MoEFusableDiagState,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(ref || folded) with gradients restricted to D_GU.

    The router input is detached deliberately: this auxiliary objective must not
    constrain D_QKV/D_VO through the attention path. Router weights are frozen,
    so the only trainable quantity referenced by the folded branch is D_GU.
    """
    if temperature <= 0:
        raise ValueError("router alignment temperature must be > 0")
    original, folded = router_compensation_logits(hidden_states.detach(), router_weight, diag_state)
    t = float(temperature)
    target = torch.softmax(original.detach().float() / t, dim=-1)
    log_pred = torch.log_softmax(folded.float() / t, dim=-1)
    return F.kl_div(log_pred, target, reduction="batchmean") * (t * t)


def router_compensation_topk_gate(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    diag_state: MoEFusableDiagState,
    *,
    top_k: int,
) -> dict[str, float | int]:
    """Check folded BF16 router preserves the native BF16 router top-k IDs."""
    original, folded = router_compensation_logits(hidden_states, router_weight, diag_state)
    original_ids = torch.topk(torch.softmax(original.float(), dim=-1), top_k, dim=-1).indices
    folded_ids = torch.topk(torch.softmax(folded.float(), dim=-1), top_k, dim=-1).indices
    id_diff = original_ids != folded_ids
    diff = (original.float() - folded.float()).reshape(-1)
    rel_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(original.float().reshape(-1)).clamp_min(1e-12)
    mismatches = int(id_diff.sum().item())
    mismatch_tokens = int(id_diff.any(dim=-1).sum().item())
    n_tokens = int(original_ids.shape[0])
    kl = F.kl_div(
        torch.log_softmax(folded.float(), dim=-1),
        torch.softmax(original.float(), dim=-1),
        reduction="batchmean",
    )
    return {
        "topk_mismatches": mismatches,
        "topk_mismatch_tokens": mismatch_tokens,
        "topk_mismatch_ratio": float(mismatch_tokens / max(n_tokens, 1)),
        "max_abs": float(diff.abs().max().item()),
        "rel_l2": float(rel_l2.item()),
        "kl": float(kl.item()),
    }
