"""Tests for zero-runtime-op folding used by HiF4 equivalent scaling."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.analysis.hif4_equivalent_scaling import (
    expand_gqa_o_scale,
    fold_input_columns,
    fold_output_rows_inverse,
    fold_rmsnorm_weight,
    validate_gqa_tied_scale,
)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(var + eps)
    return y * weight.float()


def test_rmsnorm_qkv_input_folding_exact_fp32():
    torch.manual_seed(10)
    x = torch.randn(3, 5, 64)
    gamma = torch.rand(64) + 0.5
    d = torch.exp2(torch.linspace(-0.75, 0.75, 64))
    q = torch.randn(48, 64)
    k = torch.randn(24, 64)
    v = torch.randn(24, 64)

    n = _rmsnorm(x, gamma)
    refs = [n @ w.T for w in (q, k, v)]

    gamma_f = fold_rmsnorm_weight(gamma, d)
    folded = [fold_input_columns(w, d) for w in (q, k, v)]
    nf = _rmsnorm(x, gamma_f)
    outs = [nf @ w.T for w in folded]

    for ref, out in zip(refs, outs, strict=True):
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=2e-6)


def test_down_folding_only_uses_up_branch_and_is_exact():
    torch.manual_seed(11)
    x = torch.randn(7, 32, dtype=torch.float64)
    wg = torch.randn(48, 32, dtype=torch.float64)
    wu = torch.randn(48, 32, dtype=torch.float64)
    wd = torch.randn(32, 48, dtype=torch.float64)
    d = torch.exp2(torch.linspace(-0.8, 0.8, 48, dtype=torch.float64))

    g = x @ wg.T
    u = x @ wu.T
    y_ref = (F.silu(g) * u) @ wd.T

    wu_f, _ = fold_output_rows_inverse(wu, d)
    wd_f = fold_input_columns(wd, d)
    y_fold = (F.silu(x @ wg.T) * (x @ wu_f.T)) @ wd_f.T
    torch.testing.assert_close(y_fold, y_ref, rtol=1e-10, atol=1e-10)


def test_wrongly_scaling_gate_rows_is_not_equivalent():
    torch.manual_seed(12)
    x = torch.randn(7, 32)
    wg = torch.randn(48, 32)
    wu = torch.randn(48, 32)
    wd = torch.randn(32, 48)
    d = torch.exp2(torch.linspace(-0.8, 0.8, 48))

    y_ref = (F.silu(x @ wg.T) * (x @ wu.T)) @ wd.T
    wg_wrong, _ = fold_output_rows_inverse(wg, d)
    wd_f = fold_input_columns(wd, d)
    y_wrong = (F.silu(x @ wg_wrong.T) * (x @ wu.T)) @ wd_f.T
    assert not torch.allclose(y_wrong, y_ref, rtol=1e-4, atol=1e-5)


def test_output_row_bias_is_scaled_together():
    torch.manual_seed(13)
    x = torch.randn(4, 16)
    w = torch.randn(8, 16)
    b = torch.randn(8)
    d = torch.exp2(torch.linspace(-0.5, 0.5, 8))
    w_f, b_f = fold_output_rows_inverse(w, d, b)
    y_ref = (x @ w.T + b) / d
    y_fold = x @ w_f.T + b_f
    torch.testing.assert_close(y_fold, y_ref, rtol=1e-5, atol=1e-6)


def test_gqa_vo_folding_exact_for_attention_value_aggregation():
    torch.manual_seed(14)
    batch = 2
    q_heads = 4
    kv_heads = 2
    head_dim = 3
    tokens = 5
    repeat = q_heads // kv_heads

    v = torch.randn(batch, kv_heads, tokens, head_dim)
    # Per-query-head row-stochastic attention matrix [B,Hq,Tout,Tin].
    p = torch.softmax(torch.randn(batch, q_heads, 4, tokens), dim=-1)
    v_rep = v.repeat_interleave(repeat, dim=1)
    attn_out = torch.einsum("bhqt,bhtd->bhqd", p, v_rep)
    flat = attn_out.transpose(1, 2).reshape(batch, 4, q_heads * head_dim)
    wo = torch.randn(7, q_heads * head_dim)
    y_ref = flat @ wo.T

    d_unique = torch.exp2(torch.linspace(-0.5, 0.5, kv_heads * head_dim)).reshape(kv_heads, head_dim)
    v_scaled = v / d_unique.view(1, kv_heads, 1, head_dim)
    v_scaled_rep = v_scaled.repeat_interleave(repeat, dim=1)
    attn_scaled = torch.einsum("bhqt,bhtd->bhqd", p, v_scaled_rep)
    flat_scaled = attn_scaled.transpose(1, 2).reshape(batch, 4, q_heads * head_dim)
    d_expanded = expand_gqa_o_scale(
        d_unique,
        num_attention_heads=q_heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
    )
    wo_f = fold_input_columns(wo, d_expanded)
    y_fold = flat_scaled @ wo_f.T
    torch.testing.assert_close(y_fold, y_ref, rtol=1e-5, atol=3e-6)


def test_gqa_validator_rejects_free_untied_scale():
    d = torch.ones(4, 2)
    d[1, 0] = 1.25
    try:
        validate_gqa_tied_scale(
            d.reshape(-1),
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
        )
    except ValueError as exc:
        assert "GQA" in str(exc)
    else:
        raise AssertionError("untied query-head scale must be rejected")


def test_composed_v_weight_supports_left_and_right_scaling():
    torch.manual_seed(15)
    wv = torch.randn(12, 16)
    d_attn = torch.exp2(torch.linspace(-0.5, 0.5, 16))
    d_o_unique = torch.exp2(torch.linspace(-0.25, 0.25, 12))
    composed = (wv * d_attn.unsqueeze(0)) / d_o_unique.unsqueeze(1)
    x = torch.randn(6, 16)
    y_ref = (x @ wv.T) / d_o_unique
    y_fold = (x / d_attn) @ composed.T
    torch.testing.assert_close(y_fold, y_ref, rtol=1e-5, atol=1e-6)


def test_composed_up_weight_supports_left_and_right_scaling():
    torch.manual_seed(16)
    wu = torch.randn(24, 16)
    d_mlp = torch.exp2(torch.linspace(-0.5, 0.5, 16))
    d_down = torch.exp2(torch.linspace(-0.25, 0.25, 24))
    composed = (wu * d_mlp.unsqueeze(0)) / d_down.unsqueeze(1)
    x = torch.randn(6, 16)
    y_ref = (x @ wu.T) / d_down
    y_fold = (x / d_mlp) @ composed.T
    torch.testing.assert_close(y_fold, y_ref, rtol=1e-5, atol=1e-6)
