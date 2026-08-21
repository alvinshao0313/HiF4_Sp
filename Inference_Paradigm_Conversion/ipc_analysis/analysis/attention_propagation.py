"""T1–T6: full-attention error propagation metrics (Qwen3 has no linear_attn)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import compute_pair_metrics


def _valid_attn_probs(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Softmax over valid positions only; returns renormed probs on valid support.

    logits/valid_mask: [..., S] ; valid_mask True = allowed key position.
    """
    tiny = torch.finfo(torch.float32).tiny
    neg = torch.finfo(torch.float32).min
    masked = logits.float().masked_fill(~valid_mask, neg)
    # metric copy only
    probs = torch.softmax(masked, dim=-1)
    probs = probs * valid_mask.float()
    z = probs.sum(dim=-1, keepdim=True).clamp_min(tiny)
    return probs / z


def kl_js_on_valid_support(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float]:
    """KL/JS only on causal/padding-valid attention support (metric copies)."""
    tiny = torch.finfo(torch.float32).tiny
    p = _valid_attn_probs(logits_s, valid_mask)
    q = _valid_attn_probs(logits_t, valid_mask)
    # only valid positions enter log
    log_p = torch.where(valid_mask, (p.clamp_min(tiny)).log(), torch.zeros_like(p))
    log_q = torch.where(valid_mask, (q.clamp_min(tiny)).log(), torch.zeros_like(q))
    kl_pq = (p * (log_p - log_q)).sum(dim=-1)
    kl_qp = (q * (log_q - log_p)).sum(dim=-1)
    m = 0.5 * (p + q)
    log_m = torch.where(valid_mask, (m.clamp_min(tiny)).log(), torch.zeros_like(m))
    js = 0.5 * (p * (log_p - log_m)).sum(dim=-1) + 0.5 * (q * (log_q - log_m)).sum(dim=-1)
    return {
        "kl_st": float(kl_pq.mean().item()),
        "kl_ts": float(kl_qp.mean().item()),
        "js": float(js.mean().item()),
    }


def entropy_on_valid(logits: torch.Tensor, valid_mask: torch.Tensor) -> float:
    tiny = torch.finfo(torch.float32).tiny
    p = _valid_attn_probs(logits, valid_mask)
    log_p = torch.where(valid_mask, (p.clamp_min(tiny)).log(), torch.zeros_like(p))
    h = -(p * log_p).sum(dim=-1)
    return float(h.mean().item())


def top_attended_flip_rate(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """Fraction of queries whose argmax key differs (valid support only)."""
    neg = torch.finfo(torch.float32).min
    s = logits_s.float().masked_fill(~valid_mask, neg)
    t = logits_t.float().masked_fill(~valid_mask, neg)
    # exclude fully-masked queries
    has = valid_mask.any(dim=-1)
    if not bool(has.any()):
        return 0.0
    flip = (s.argmax(dim=-1) != t.argmax(dim=-1)) & has
    return float(flip.float().sum().item() / has.float().sum().item())


def local_gain(delta_in: torch.Tensor, delta_out: torch.Tensor) -> dict[str, Any]:
    nin = float(torch.linalg.vector_norm(delta_in.float()).item())
    nout = float(torch.linalg.vector_norm(delta_out.float()).item())
    if nin == 0.0:
        return {"gain": 0.0, "gain_status": "zero_input_error", "norm_in": nin, "norm_out": nout}
    return {"gain": nout / nin, "gain_status": "ok", "norm_in": nin, "norm_out": nout}


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Minimal RoPE for [B, H, T, D] with cos/sin [T, D/2] or broadcastable."""
    # x: [..., T, D]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    while cos.ndim < x1.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack((out1, out2), dim=-1).flatten(-2)
    return out


@torch.no_grad()
def full_attention_propagation(
    attn_input: torch.Tensor,
    w_q_n: torch.Tensor,
    w_q_h: torch.Tensor,
    w_k_n: torch.Tensor,
    w_k_h: torch.Tensor,
    w_v_n: torch.Tensor,
    w_v_h: torch.Tensor,
    w_o_n: torch.Tensor,
    w_o_h: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Frozen-input full attention path comparing source vs HiF4-converted Q/K/V/O."""
    x = attn_input.float()
    if x.ndim == 2:
        x = x.unsqueeze(0)  # [1, T, C]
    bsz, tlen, _ = x.shape

    def proj(w):
        return F.linear(x, w.float())

    q_n, q_h = proj(w_q_n), proj(w_q_h)
    k_n, k_h = proj(w_k_n), proj(w_k_h)
    v_n, v_h = proj(w_v_n), proj(w_v_h)

    def split_heads(t, n_heads):
        return t.view(bsz, tlen, n_heads, head_dim).transpose(1, 2)

    qn = split_heads(q_n, num_heads)
    qh = split_heads(q_h, num_heads)
    kn = split_heads(k_n, num_kv_heads)
    kh = split_heads(k_h, num_kv_heads)
    vn = split_heads(v_n, num_kv_heads)
    vh = split_heads(v_h, num_kv_heads)

    # Expand KV for GQA
    if num_heads != num_kv_heads:
        rep = num_heads // num_kv_heads
        kn = kn.repeat_interleave(rep, dim=1)
        kh = kh.repeat_interleave(rep, dim=1)
        vn = vn.repeat_interleave(rep, dim=1)
        vh = vh.repeat_interleave(rep, dim=1)

    # Synthetic RoPE angles (fixed seed geometry) — same for S/T
    half = head_dim // 2
    pos = torch.arange(tlen, device=x.device, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    freqs = torch.outer(pos, inv)
    cos, sin = freqs.cos(), freqs.sin()
    qn_r, qh_r = apply_rope(qn, cos, sin), apply_rope(qh, cos, sin)
    kn_r, kh_r = apply_rope(kn, cos, sin), apply_rope(kh, cos, sin)

    scale = 1.0 / math.sqrt(head_dim)
    logits_n = torch.matmul(qn_r, kn_r.transpose(-2, -1)) * scale
    logits_h = torch.matmul(qh_r, kh_r.transpose(-2, -1)) * scale

    # causal mask [T, T]
    causal = torch.tril(torch.ones(tlen, tlen, device=x.device, dtype=torch.bool))
    valid = causal.view(1, 1, tlen, tlen).expand_as(logits_n)

    probs_n = _valid_attn_probs(logits_n, valid)
    probs_h = _valid_attn_probs(logits_h, valid)
    av_n = torch.matmul(probs_n, vn)
    av_h = torch.matmul(probs_h, vh)
    # merge heads
    av_n_m = av_n.transpose(1, 2).contiguous().view(bsz, tlen, num_heads * head_dim)
    av_h_m = av_h.transpose(1, 2).contiguous().view(bsz, tlen, num_heads * head_dim)
    o_n = F.linear(av_n_m, w_o_n.float())
    o_h = F.linear(av_h_m, w_o_h.float())
    res_n = x + o_n
    res_h = x + o_h

    kl = kl_js_on_valid_support(logits_n, logits_h, valid)
    flip = top_attended_flip_rate(logits_n, logits_h, valid)
    ent_n = entropy_on_valid(logits_n, valid)
    ent_h = entropy_on_valid(logits_h, valid)

    stages = {
        "q_proj": (q_n, q_h),
        "k_proj": (k_n, k_h),
        "v_proj": (v_n, v_h),
        "q_after_rope": (qn_r, qh_r),
        "k_after_rope": (kn_r, kh_r),
        "attn_logits": (logits_n, logits_h),
        "av_output": (av_n_m, av_h_m),
        "o_proj_output": (o_n, o_h),
        "residual_output": (res_n, res_h),
    }
    metrics = {k: compute_pair_metrics(a, b) for k, (a, b) in stages.items()}

    # RoPE gain / logits gain from q/k
    rope_gain_q = local_gain(qh - qn, qh_r - qn_r)
    logits_gain = local_gain(
        torch.cat([(qh_r - qn_r).reshape(-1), (kh_r - kn_r).reshape(-1)]),
        (logits_h - logits_n).reshape(-1),
    )

    return {
        "attention_kind": "full_attention",
        "stage_metrics": metrics,
        "kl_js": kl,
        "top_attended_flip_rate": flip,
        "entropy_source": ent_n,
        "entropy_target": ent_h,
        "entropy_change": ent_h - ent_n,
        "rope_gain_q": rope_gain_q,
        "logits_gain_from_qk": logits_gain,
        "hypothesis_id": "H6-Attention",
        "evidence_class": "controlled_causal_evidence",
    }


def detect_linear_attn_modules(model: torch.nn.Module) -> list[str]:
    """Return linear_attn module names if present; empty for standard Qwen3-8B."""
    names = []
    for name, _ in model.named_modules():
        if ".linear_attn." in name or name.endswith("linear_attn"):
            names.append(name)
    return names
