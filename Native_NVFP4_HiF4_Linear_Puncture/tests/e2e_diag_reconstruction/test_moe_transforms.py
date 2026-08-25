from __future__ import annotations

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_transforms import (
    apply_r64_no_cross_head,
    expand_vo_scale_qwen3_moe,
    fusable_weight_transform_no_h,
    online_weight_transform_no_h,
    transform_router_weight,
)


def _d(n: int) -> torch.Tensor:
    return torch.exp2(torch.linspace(-1.0, 1.0, n))


def test_online_no_h_fp32_equivalence_for_qkv_and_o():
    torch.manual_seed(0)
    x = torch.randn(3, 128)
    w = torch.randn(96, 128)
    d = _d(128)
    for order in ("diag_then_rot", "rot_then_diag"):
        wt = online_weight_transform_no_h(w, d, use_r64=True, rot_order=order, head_dim=128)
        xt = apply_r64_no_cross_head(x * d, head_dim=128) if order == "diag_then_rot" else apply_r64_no_cross_head(x, head_dim=128) * d
        torch.testing.assert_close(xt @ wt.T, x @ w.T, rtol=2e-5, atol=2e-5)


def test_fusable_expert_swiglu_and_router_equivalence():
    torch.manual_seed(1)
    x = torch.randn(5, 128)
    w_gate, w_up, w_down = torch.randn(64, 128), torch.randn(64, 128), torch.randn(128, 64)
    d_gu, d_ud = _d(128), _d(64)
    gate_t = fusable_weight_transform_no_h(w_gate, d_gu, torch.ones(64), use_r64=True)
    up_t = fusable_weight_transform_no_h(w_up, d_gu, d_ud, use_r64=True)
    down_t = fusable_weight_transform_no_h(w_down, d_ud, torch.ones(128), use_r64=True)
    x_t = apply_r64_no_cross_head(x * d_gu)
    gate = x_t @ gate_t.T
    up = x_t @ up_t.T
    swiglu = torch.nn.functional.silu(gate) * up
    out = apply_r64_no_cross_head(swiglu) @ down_t.T
    reference = (torch.nn.functional.silu(x @ w_gate.T) * (x @ w_up.T)) @ w_down.T
    torch.testing.assert_close(out, reference, rtol=3e-4, atol=1e-3)

    router = torch.randn(7, 128)
    torch.testing.assert_close((x * d_gu) @ transform_router_weight(router, d_gu).T, x @ router.T, rtol=2e-6, atol=2e-6)


def test_gqa_expand_and_r64_does_not_cross_heads():
    d_vo = _d(512)
    expanded = expand_vo_scale_qwen3_moe(d_vo)
    assert expanded.shape == (4096,)
    assert torch.equal(expanded[:128], d_vo[:128])
    assert torch.equal(expanded[7 * 128 : 8 * 128], d_vo[:128])
    assert torch.equal(expanded[8 * 128 : 9 * 128], d_vo[128:256])
    for head in range(32):
        x = torch.zeros(1, 4096)
        x[0, head * 128 : (head + 1) * 128] = 1.0
        rotated = apply_r64_no_cross_head(x, head_dim=128)
        outside = torch.cat((rotated[:, : head * 128], rotated[:, (head + 1) * 128 :]), dim=1)
        assert torch.count_nonzero(outside) == 0
