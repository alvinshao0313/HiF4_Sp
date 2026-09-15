"""Unit tests for O3 router math (plan O3.1 G items 1–7 + helper identity)."""

from __future__ import annotations

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_fold import (
    folded_router_logits_from_pre_dgu_input,
    router_compensation_logits,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    MoEFusableDiagState,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.router_objective import (
    build_production_topk_teacher,
    router_full_kl,
    router_topk_support_hinge,
    router_topk_total,
    router_topk_weight_kl,
)


def test_router_full_kl_zero_when_candidate_equals_e0():
    torch.manual_seed(0)
    e0 = torch.randn(16, 32, dtype=torch.float32)
    kl = router_full_kl(e0, e0.clone())
    assert float(kl.item()) < 1e-6


def test_topk_weight_kl_and_hinge_zero_when_candidate_equals_e0():
    torch.manual_seed(1)
    e0 = torch.randn(12, 24, dtype=torch.float32)
    topk_ids, topk_weights = build_production_topk_teacher(e0, top_k=5, norm_topk_prob=True)
    weight_kl = router_topk_weight_kl(topk_ids, topk_weights, e0)
    hinge = router_topk_support_hinge(topk_ids, e0)
    total, w2, h2 = router_topk_total(topk_ids, topk_weights, e0)
    assert float(weight_kl.item()) < 1e-6
    assert float(hinge.item()) == 0.0
    assert float(total.item()) < 1e-6
    assert float(w2.item()) < 1e-6
    assert float(h2.item()) == 0.0


def test_relative_logits_inside_e0_topk_raise_weight_kl():
    torch.manual_seed(2)
    e0 = torch.randn(8, 20, dtype=torch.float32)
    top_k = 4
    topk_ids, topk_weights = build_production_topk_teacher(e0, top_k=top_k, norm_topk_prob=True)
    candidate = e0.clone()
    # Perturb only relative logits inside the teacher Top-k set (same support).
    for i in range(candidate.shape[0]):
        ids = topk_ids[i]
        candidate[i, ids[0]] = candidate[i, ids[0]] + 2.5
        candidate[i, ids[1]] = candidate[i, ids[1]] - 1.5
    weight_kl = router_topk_weight_kl(topk_ids, topk_weights, candidate)
    assert float(weight_kl.item()) > 0.0


def test_outside_expert_above_min_s0_raises_hinge():
    torch.manual_seed(3)
    e0 = torch.randn(6, 16, dtype=torch.float32)
    top_k = 3
    topk_ids, _ = build_production_topk_teacher(e0, top_k=top_k, norm_topk_prob=True)
    candidate = e0.clone()
    for i in range(candidate.shape[0]):
        s0 = set(int(x) for x in topk_ids[i].tolist())
        outside = next(j for j in range(candidate.shape[1]) if j not in s0)
        min_in = float(candidate[i, topk_ids[i]].min().item())
        candidate[i, outside] = min_in + 1.0
    hinge = router_topk_support_hinge(topk_ids, candidate)
    assert float(hinge.item()) > 0.0


def test_outside_expert_below_boundary_hinge_zero():
    torch.manual_seed(4)
    e0 = torch.randn(6, 16, dtype=torch.float32)
    top_k = 3
    topk_ids, _ = build_production_topk_teacher(e0, top_k=top_k, norm_topk_prob=True)
    candidate = e0.clone()
    for i in range(candidate.shape[0]):
        s0 = set(int(x) for x in topk_ids[i].tolist())
        outside = next(j for j in range(candidate.shape[1]) if j not in s0)
        min_in = float(candidate[i, topk_ids[i]].min().item())
        # Keep outside strictly below the support boundary.
        candidate[i, outside] = min_in - 0.5
        for j in range(candidate.shape[1]):
            if j not in s0 and j != outside:
                candidate[i, j] = min_in - 1.0
    hinge = router_topk_support_hinge(topk_ids, candidate)
    assert float(hinge.item()) == 0.0


def test_production_softmax_topk_norm_matches_teacher():
    torch.manual_seed(5)
    e0 = torch.randn(10, 32, dtype=torch.float32)
    top_k = 8
    topk_ids, topk_weights = build_production_topk_teacher(e0, top_k=top_k, norm_topk_prob=True)
    probs = torch.softmax(e0.float(), dim=-1)
    ref_w, ref_ids = torch.topk(probs, top_k, dim=-1)
    ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)
    assert torch.equal(topk_ids, ref_ids)
    torch.testing.assert_close(topk_weights, ref_w, rtol=0.0, atol=0.0)


def test_synthetic_k_not_hardcoded_eight():
    torch.manual_seed(6)
    e0 = torch.randn(5, 40, dtype=torch.float32)
    for k in (3, 5, 11):
        topk_ids, topk_weights = build_production_topk_teacher(e0, top_k=k, norm_topk_prob=True)
        assert topk_ids.shape == (5, k)
        assert topk_weights.shape == (5, k)
        torch.testing.assert_close(
            topk_weights.sum(dim=-1),
            torch.ones(5, dtype=torch.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        weight_kl = router_topk_weight_kl(topk_ids, topk_weights, e0)
        hinge = router_topk_support_hinge(topk_ids, e0)
        assert float(weight_kl.item()) < 1e-6
        assert float(hinge.item()) == 0.0


def test_folded_helper_matches_router_compensation_folded_branch():
    torch.manual_seed(7)
    hidden = torch.randn(16, 2048, dtype=torch.bfloat16)
    router = torch.randn(128, 2048, dtype=torch.bfloat16) * 0.02
    diag = MoEFusableDiagState(num_experts=2)
    with torch.no_grad():
        diag.z_gu.copy_(torch.linspace(-0.4, 0.35, 2048))
    _, folded = router_compensation_logits(hidden, router, diag)
    helper = folded_router_logits_from_pre_dgu_input(hidden, router, diag)
    assert helper.dtype == torch.bfloat16
    assert torch.equal(helper, folded)
