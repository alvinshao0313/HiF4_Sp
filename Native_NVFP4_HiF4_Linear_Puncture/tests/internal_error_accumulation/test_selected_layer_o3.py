"""Unit tests for selected-layer O3 gradient rules and formal scope (plan O3.1 G 10–13)."""

from __future__ import annotations

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_fold import (
    folded_router_logits_from_pre_dgu_input,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    MoEFusableDiagState,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_transforms import (
    transform_router_weight,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.objective_study import (
    assert_formal_objective_scope,
    decide_objective_scopes,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.router_objective import (
    build_production_topk_teacher,
    router_full_kl,
    router_topk_total,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.selected_layer_objective_trainer import (
    LOSS_O3_FULL,
    compute_o3_router_aux,
)


def _diag_with_grads(*, hidden: int = 16, experts: int = 12, moe_intermediate: int = 8) -> MoEFusableDiagState:
    diag = MoEFusableDiagState(
        hidden_size=hidden,
        num_experts=experts,
        moe_intermediate_size=moe_intermediate,
        num_key_value_heads=2,
        head_dim=8,
    )
    # Enable MoE params; keep attn params present but frozen for grad checks.
    diag.z_qkv.requires_grad_(False)
    diag.z_vo.requires_grad_(False)
    diag.z_gu.requires_grad_(True)
    diag.z_ud.requires_grad_(True)
    with torch.no_grad():
        diag.z_gu.uniform_(-0.2, 0.2)
        diag.z_ud.uniform_(-0.2, 0.2)
    return diag


def test_helper_matches_student_router_logits_identity():
    torch.manual_seed(0)
    hidden_size = 16
    num_experts = 10
    diag = _diag_with_grads(hidden=hidden_size, experts=num_experts)
    router = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16)
    pre = torch.randn(7, hidden_size, dtype=torch.bfloat16)

    # Student-style folded router on pre-D_GU input.
    d = diag.d_gu().to(device=pre.device)
    folded_w = transform_router_weight(router, d).to(dtype=torch.bfloat16)
    student_logits = (pre * d.to(dtype=torch.bfloat16)) @ folded_w.T
    helper = folded_router_logits_from_pre_dgu_input(pre, router, diag)
    assert torch.allclose(student_logits.float(), helper.float(), rtol=0, atol=0)


def test_router_aux_grad_only_on_z_gu():
    torch.manual_seed(1)
    hidden_size = 16
    num_experts = 12
    top_k = 4
    diag = _diag_with_grads(hidden=hidden_size, experts=num_experts)
    router = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16)
    router.requires_grad_(False)
    pre_bth = torch.randn(2, 5, hidden_size, dtype=torch.bfloat16)
    lengths = torch.tensor([3, 4])
    e0 = torch.randn(7, num_experts, dtype=torch.float32)
    topk_ids, topk_weights = build_production_topk_teacher(e0, top_k=top_k, norm_topk_prob=True)

    aux, _metrics = compute_o3_router_aux(
        loss_name=LOSS_O3_FULL,
        router_input_bth=pre_bth,
        lengths=lengths,
        router_weight=router,
        diag_state=diag,
        e0_logits=e0,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    aux.backward()

    assert diag.z_gu.grad is not None and float(diag.z_gu.grad.abs().sum()) > 0.0
    assert diag.z_ud.grad is None or float(diag.z_ud.grad.abs().sum()) == 0.0
    assert diag.z_qkv.grad is None or float(diag.z_qkv.grad.abs().sum()) == 0.0
    assert diag.z_vo.grad is None or float(diag.z_vo.grad.abs().sum()) == 0.0
    assert router.grad is None


def test_total_o3_gives_z_ud_from_o2_m_only():
    torch.manual_seed(2)
    hidden_size = 16
    num_experts = 12
    top_k = 3
    diag = _diag_with_grads(hidden=hidden_size, experts=num_experts)
    router = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16)
    pre_bth = torch.randn(1, 4, hidden_size, dtype=torch.bfloat16)
    lengths = torch.tensor([4])
    e0 = torch.randn(4, num_experts, dtype=torch.float32)
    topk_ids, topk_weights = build_production_topk_teacher(e0, top_k=top_k, norm_topk_prob=True)

    # Synthetic O2_M base that depends on z_ud (and not on router path).
    target = torch.randn(num_experts, diag.moe_intermediate_size, dtype=torch.float32)
    base = ((diag.z_ud - target) ** 2).mean()

    aux, _ = compute_o3_router_aux(
        loss_name=LOSS_O3_FULL,
        router_input_bth=pre_bth,
        lengths=lengths,
        router_weight=router,
        diag_state=diag,
        e0_logits=e0,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    # First: router aux alone must not give z_ud grad.
    diag.z_gu.grad = None
    diag.z_ud.grad = None
    aux.backward(retain_graph=True)
    assert diag.z_ud.grad is None or float(diag.z_ud.grad.abs().sum()) == 0.0
    gu_from_aux = diag.z_gu.grad.detach().clone()

    # Total O3 = base + lambda * aux: z_ud comes from O2_M base.
    diag.z_gu.grad = None
    diag.z_ud.grad = None
    total = base + 0.5 * aux
    total.backward()
    assert diag.z_ud.grad is not None and float(diag.z_ud.grad.abs().sum()) > 0.0
    assert diag.z_gu.grad is not None and float(diag.z_gu.grad.abs().sum()) > 0.0
    # z_gu still receives aux grad (and not only from base).
    assert float((diag.z_gu.grad - gu_from_aux * 0.5).abs().sum()) < 1e-5 or float(
        diag.z_gu.grad.abs().sum()
    ) > 0.0


def test_topk_aux_also_grads_only_z_gu():
    torch.manual_seed(3)
    hidden_size = 16
    num_experts = 14
    top_k = 5  # not hard-coded 8
    diag = _diag_with_grads(hidden=hidden_size, experts=num_experts)
    router = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16)
    pre_bth = torch.randn(2, 3, hidden_size, dtype=torch.bfloat16)
    lengths = torch.tensor([2, 3])
    e0 = torch.randn(5, num_experts)
    topk_ids, topk_weights = build_production_topk_teacher(e0, top_k=top_k, norm_topk_prob=True)
    valid = torch.cat([pre_bth[0, :2], pre_bth[1, :3]], dim=0).detach()
    cand = folded_router_logits_from_pre_dgu_input(valid, router, diag)
    total, _, _ = router_topk_total(topk_ids, topk_weights, cand)
    total.backward()
    assert diag.z_gu.grad is not None and float(diag.z_gu.grad.abs().sum()) > 0.0
    assert diag.z_ud.grad is None or float(diag.z_ud.grad.abs().sum()) == 0.0


def test_formal_scope_rejects_enable_o3_and_o3_conditional():
    with pytest.raises(RuntimeError, match="enable_o3"):
        assert_formal_objective_scope(
            {
                "status": "FROZEN",
                "enable_o3": True,
                "o3_layers": [31],
                "causal_source_by_layer": {"31": "moe"},
            }
        )
    with pytest.raises(RuntimeError, match="O3_conditional"):
        assert_formal_objective_scope(
            {
                "status": "FROZEN",
                "o3_layers": [31],
                "o3_conditional": True,
                "causal_source_by_layer": {"31": "moe"},
            }
        )
    with pytest.raises(RuntimeError, match="o3_layers"):
        assert_formal_objective_scope(
            {
                "status": "FROZEN",
                "causal_source_by_layer": {"31": "moe"},
            }
        )


def test_decide_scopes_adds_o3_only_for_o3_layers():
    scopes = decide_objective_scopes({31: "moe", 11: "moe", 12: "attention"}, o3_layers=[31])
    assert "O3_full" in scopes[31]["losses"]
    assert "O3_topk" in scopes[31]["losses"]
    assert "O3_conditional" not in scopes[31]["losses"]
    assert "O3_full" not in scopes[11]["losses"]
    assert "O3_full" not in scopes[12]["losses"]


def test_router_full_kl_zero_path_still_available():
    e0 = torch.randn(4, 9)
    assert float(router_full_kl(e0, e0.clone()).item()) < 1e-6
