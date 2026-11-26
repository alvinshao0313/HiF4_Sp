from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def hidden_metrics(reference: torch.Tensor, current: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cur = current.float()
    diff = cur - ref
    return {
        "hidden_rel_l2": float(diff.norm().item() / max(ref.norm().item(), 1e-30)),
        "hidden_cosine": float(F.cosine_similarity(ref[None], cur[None], dim=-1).item()),
        "hidden_max_abs": float(diff.abs().max().item()),
    }


def _kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p_log = F.log_softmax(p_logits.float(), dim=-1)
    q_log = F.log_softmax(q_logits.float(), dim=-1)
    p = p_log.exp()
    return float((p * (p_log - q_log)).sum().item())


def router_metrics(reference: torch.Tensor, current: torch.Tensor, top_k: int) -> dict[str, float | int | bool]:
    ref = reference.float()
    cur = current.float()
    ref_idx = torch.topk(ref, k=top_k, dim=-1).indices
    cur_idx = torch.topk(cur, k=top_k, dim=-1).indices
    overlap = len(set(ref_idx.tolist()) & set(cur_idx.tolist()))
    ref_sorted = torch.topk(ref, k=min(top_k + 1, ref.numel()), dim=-1).values
    cur_sorted = torch.topk(cur, k=min(top_k + 1, cur.numel()), dim=-1).values
    ref_margin = float(ref_sorted[top_k - 1].item() - ref_sorted[top_k].item()) if ref_sorted.numel() > top_k else float("nan")
    cur_margin = float(cur_sorted[top_k - 1].item() - cur_sorted[top_k].item()) if cur_sorted.numel() > top_k else float("nan")
    return {
        "router_kl_e0_to_variant": _kl_from_logits(ref, cur),
        "router_topk_overlap": overlap / float(top_k),
        "router_topk_exact": bool(torch.equal(torch.sort(ref_idx).values, torch.sort(cur_idx).values)),
        "router_ref_boundary_margin": ref_margin,
        "router_variant_boundary_margin": cur_margin,
    }


def logit_metrics(reference: torch.Tensor, current: torch.Tensor, target_token_id: int) -> dict[str, float | int | bool]:
    ref = reference.float()
    cur = current.float()
    ref_centered = ref - ref.mean()
    cur_centered = cur - cur.mean()
    ref_top2 = torch.topk(ref, k=2)
    cur_top2 = torch.topk(cur, k=2)
    ref_logp = F.log_softmax(ref, dim=-1)
    cur_logp = F.log_softmax(cur, dim=-1)
    log_mix = torch.logaddexp(ref_logp, cur_logp) - math.log(2.0)
    ref_p = ref_logp.exp()
    cur_p = cur_logp.exp()
    js = 0.5 * (ref_p * (ref_logp - log_mix)).sum() + 0.5 * (cur_p * (cur_logp - log_mix)).sum()
    target = int(target_token_id)
    target_ref = ref[target]
    target_cur = cur[target]
    return {
        "logit_kl_e0_to_variant": _kl_from_logits(ref, cur),
        "logit_js": float(js.item()),
        "logit_centered_cosine": float(F.cosine_similarity(ref_centered[None], cur_centered[None], dim=-1).item()),
        "top1_agreement": bool(int(ref_top2.indices[0]) == int(cur_top2.indices[0])),
        "e0_top1_token": int(ref_top2.indices[0]),
        "variant_top1_token": int(cur_top2.indices[0]),
        "e0_top1_top2_margin": float((ref_top2.values[0] - ref_top2.values[1]).item()),
        "variant_top1_top2_margin": float((cur_top2.values[0] - cur_top2.values[1]).item()),
        "target_token_id": target,
        "e0_target_nll": float(-ref_logp[target].item()),
        "variant_target_nll": float(-cur_logp[target].item()),
        "target_nll_delta": float((ref_logp[target] - cur_logp[target]).item()),
        "e0_target_rank": int((ref > target_ref).sum().item()) + 1,
        "variant_target_rank": int((cur > target_cur).sum().item()) + 1,
    }
