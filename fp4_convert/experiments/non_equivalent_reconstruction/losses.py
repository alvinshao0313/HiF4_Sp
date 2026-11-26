"""Teacher top-k terms with full-expert normalization, never a top-k softmax."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def router_loss_per_token(student_logits, teacher_logits, *, kind="top_mass", top_k=8, temperature=1.0):
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("router logits must have matching [...,experts] shapes")
    if not math.isfinite(temperature) or temperature <= 0 or top_k <= 0:
        raise ValueError("temperature and top_k must be positive")
    if kind not in {"top_partial", "top_mass"}:
        raise ValueError(kind)
    s = student_logits.float() / temperature
    t = teacher_logits.detach().float() / temperature
    k = min(top_k, t.shape[-1])
    ids = t.topk(k, dim=-1, sorted=False).indices
    log_p = F.log_softmax(s, dim=-1)
    log_q = F.log_softmax(t, dim=-1)
    lq = log_q.gather(-1, ids)
    lp = log_p.gather(-1, ids)
    loss = (lq.exp() * (lq - lp)).sum(-1)
    if kind == "top_mass" and k < t.shape[-1]:
        inside = torch.zeros_like(t, dtype=torch.bool).scatter_(-1, ids, True)
        log_q_tail = torch.logsumexp(log_q.masked_fill(inside, -torch.inf), dim=-1)
        log_p_tail = torch.logsumexp(log_p.masked_fill(inside, -torch.inf), dim=-1)
        loss = loss + log_q_tail.exp() * (log_q_tail - log_p_tail)
    return loss * temperature**2


def masked_router_loss(student_logits, teacher_logits, mask, **kwargs):
    if tuple(mask.shape) != tuple(student_logits.shape[:-1]):
        raise ValueError("mask must match token dimensions")
    valid = mask.bool()
    if not valid.any():
        raise ValueError("router loss requires valid tokens")
    return router_loss_per_token(student_logits[valid], teacher_logits[valid], **kwargs).mean()


@torch.no_grad()
def router_metrics(student_logits, teacher_logits, mask, *, top_k=8):
    s = student_logits[mask.bool()].float()
    t = teacher_logits[mask.bool()].float()
    if s.shape[0] == 0:
        raise ValueError("router metrics require valid tokens")
    k = min(top_k, s.shape[-1])
    ids = t.topk(k, dim=-1).indices
    student_ids = s.topk(k, dim=-1).indices
    same = (ids.sort(-1).values == student_ids.sort(-1).values).all(-1)
    overlap = (ids.unsqueeze(-1) == student_ids.unsqueeze(-2)).any(-1).float().mean(-1)
    outside = torch.ones_like(s, dtype=torch.bool).scatter_(-1, ids, False)
    return {
        "topk_set_match": same.float().mean().item(),
        "topk_overlap": overlap.mean().item(),
        "student_outside_mass": s.softmax(-1).masked_fill(~outside, 0).sum(-1).mean().item(),
        "teacher_outside_mass": t.softmax(-1).masked_fill(~outside, 0).sum(-1).mean().item(),
    }
