from __future__ import annotations

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.attention_propagation import (
    full_attention_propagation,
    kl_js_on_valid_support,
    top_attended_flip_rate,
)


def test_kl_only_on_valid_support():
    torch.manual_seed(0)
    logits_s = torch.randn(2, 4, 8, 8)
    logits_t = logits_s + 0.1 * torch.randn_like(logits_s)
    valid = torch.tril(torch.ones(8, 8, dtype=torch.bool)).view(1, 1, 8, 8).expand_as(logits_s)
    out = kl_js_on_valid_support(logits_s, logits_t, valid)
    assert out["kl_st"] >= 0 and out["js"] >= 0
    flip = top_attended_flip_rate(logits_s, logits_t, valid)
    assert 0.0 <= flip <= 1.0


def test_full_attention_propagation_runs():
    torch.manual_seed(1)
    t, c, h, kv, d = 8, 64, 4, 2, 16
    x = torch.randn(1, t, c)
    wq = torch.randn(h * d, c)
    wk = torch.randn(kv * d, c)
    wv = torch.randn(kv * d, c)
    wo = torch.randn(c, h * d)
    # small HiF4-like perturbation
    out = full_attention_propagation(
        x,
        wq, wq + 0.01 * torch.randn_like(wq),
        wk, wk + 0.01 * torch.randn_like(wk),
        wv, wv + 0.01 * torch.randn_like(wv),
        wo, wo + 0.01 * torch.randn_like(wo),
        num_heads=h,
        num_kv_heads=kv,
        head_dim=d,
    )
    assert out["attention_kind"] == "full_attention"
    assert "attn_logits" in out["stage_metrics"]
