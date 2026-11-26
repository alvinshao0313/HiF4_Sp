"""Pure math Router objectives for formal O3_full / O3_topk (plan O3.1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_production_topk_teacher(
    e0_logits: torch.Tensor,
    *,
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Production path: softmax(float32) -> topk -> optional Top-k renormalize.

    Returns:
        topk_ids: [N, k] int64
        topk_weights: [N, k] float32 (renormalized when norm_topk_prob is True)
    """
    if e0_logits.ndim != 2:
        raise ValueError(f"e0_logits must be 2D [N,E], got shape={tuple(e0_logits.shape)}")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if top_k > e0_logits.shape[-1]:
        raise ValueError(f"top_k={top_k} exceeds num_experts={e0_logits.shape[-1]}")
    if not norm_topk_prob:
        raise ValueError("norm_topk_prob must be True for production Top-k teacher")

    probs = torch.softmax(e0_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, int(top_k), dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids, topk_weights


def router_full_kl(e0_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    """KL(softmax(E0) || softmax(candidate)) at T=1.0, mean over rows."""
    if e0_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"shape mismatch: e0_logits={tuple(e0_logits.shape)} "
            f"candidate_logits={tuple(candidate_logits.shape)}"
        )
    if e0_logits.ndim != 2:
        raise ValueError(f"logits must be 2D [N,E], got ndim={e0_logits.ndim}")
    target = torch.softmax(e0_logits.float(), dim=-1)
    log_pred = torch.log_softmax(candidate_logits.float(), dim=-1)
    return F.kl_div(log_pred, target, reduction="batchmean")


def router_topk_weight_kl(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """KL(E0 Top-k weights || candidate softmax on the same E0 Top-k set)."""
    if topk_ids.ndim != 2 or topk_weights.ndim != 2 or candidate_logits.ndim != 2:
        raise ValueError("topk_ids/topk_weights/candidate_logits must all be 2D")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            f"topk_ids shape {tuple(topk_ids.shape)} != topk_weights {tuple(topk_weights.shape)}"
        )
    if topk_ids.shape[0] != candidate_logits.shape[0]:
        raise ValueError(
            f"N mismatch: topk rows={topk_ids.shape[0]} candidate rows={candidate_logits.shape[0]}"
        )
    cand_on_s0 = candidate_logits.float().gather(-1, topk_ids.long())
    log_pred = torch.log_softmax(cand_on_s0, dim=-1)
    teacher = topk_weights.float()
    return F.kl_div(log_pred, teacher, reduction="batchmean")


def router_topk_support_hinge(
    topk_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """mean(ReLU(max_out - min_in)) with zero margin on E0 teacher Top-k set."""
    if topk_ids.ndim != 2 or candidate_logits.ndim != 2:
        raise ValueError("topk_ids and candidate_logits must be 2D")
    if topk_ids.shape[0] != candidate_logits.shape[0]:
        raise ValueError(
            f"N mismatch: topk rows={topk_ids.shape[0]} candidate rows={candidate_logits.shape[0]}"
        )
    n, e = candidate_logits.shape
    k = topk_ids.shape[-1]
    if k >= e:
        raise ValueError(f"support hinge requires outside experts: k={k}, E={e}")

    cand = candidate_logits.float()
    ids = topk_ids.long()
    min_in = cand.gather(-1, ids).min(dim=-1).values

    inside = torch.zeros(n, e, dtype=torch.bool, device=cand.device)
    inside.scatter_(-1, ids, True)
    outside = cand.masked_fill(inside, float("-inf"))
    max_out = outside.max(dim=-1).values
    return F.relu(max_out - min_in).mean()


def router_topk_total(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, weight_kl, support_hinge) with support coeff fixed at 1.0."""
    weight_kl = router_topk_weight_kl(topk_ids, topk_weights, candidate_logits)
    support_hinge = router_topk_support_hinge(topk_ids, candidate_logits)
    total = weight_kl + support_hinge
    return total, weight_kl, support_hinge


def topk_id_match_ratio(
    teacher_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """Fraction of rows whose candidate Top-k ID set equals the teacher set."""
    k = int(teacher_ids.shape[-1])
    cand_ids = torch.topk(candidate_logits.float(), k, dim=-1).indices
    teacher_sorted = torch.sort(teacher_ids.long(), dim=-1).values
    cand_sorted = torch.sort(cand_ids, dim=-1).values
    return (teacher_sorted == cand_sorted).all(dim=-1).float().mean()


def topk_overlap_mean(
    teacher_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """Mean |intersection(teacher Top-k, candidate Top-k)| / k."""
    k = int(teacher_ids.shape[-1])
    cand_ids = torch.topk(candidate_logits.float(), k, dim=-1).indices
    n, e = candidate_logits.shape
    teacher_oh = torch.zeros(n, e, dtype=torch.float32, device=candidate_logits.device)
    cand_oh = torch.zeros(n, e, dtype=torch.float32, device=candidate_logits.device)
    teacher_oh.scatter_(-1, teacher_ids.long(), 1.0)
    cand_oh.scatter_(-1, cand_ids, 1.0)
    overlap = (teacher_oh * cand_oh).sum(dim=-1) / float(k)
    return overlap.mean()


def topk_margin_mean(
    topk_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """Mean(min_in - max_out) on candidate logits using teacher Top-k set."""
    n, e = candidate_logits.shape
    k = topk_ids.shape[-1]
    if k >= e:
        raise ValueError(f"topk_margin requires outside experts: k={k}, E={e}")
    cand = candidate_logits.float()
    ids = topk_ids.long()
    min_in = cand.gather(-1, ids).min(dim=-1).values
    inside = torch.zeros(n, e, dtype=torch.bool, device=cand.device)
    inside.scatter_(-1, ids, True)
    max_out = cand.masked_fill(inside, float("-inf")).max(dim=-1).values
    return (min_in - max_out).mean()


def outside_teacher_topk_mass(
    topk_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """Mean softmax mass on experts outside the E0 teacher Top-k set (diagnostic only)."""
    n, e = candidate_logits.shape
    probs = torch.softmax(candidate_logits.float(), dim=-1)
    inside = torch.zeros(n, e, dtype=torch.bool, device=candidate_logits.device)
    inside.scatter_(-1, topk_ids.long(), True)
    return probs.masked_fill(inside, 0.0).sum(dim=-1).mean()


def router_proxy_metrics_from_logits(
    e0_logits: torch.Tensor,
    e1_logits: torch.Tensor,
    *,
    top_k: int,
    norm_topk_prob: bool = True,
) -> dict[str, float]:
    """Scalar proxies between captured E0/E1 router logits (accepts 1D or [N,E])."""
    a = e0_logits.detach().float().reshape(-1)
    b = e1_logits.detach().float().reshape(-1)
    if a.numel() != b.numel():
        raise RuntimeError(f"router logit numel mismatch: {a.numel()} vs {b.numel()}")
    e0 = a.unsqueeze(0)
    e1 = b.unsqueeze(0)
    topk_ids, topk_weights = build_production_topk_teacher(e0, top_k=top_k, norm_topk_prob=norm_topk_prob)
    total, wkl, hinge = router_topk_total(topk_ids, topk_weights, e1)
    return {
        "router_full_kl": float(router_full_kl(e0, e1).item()),
        "router_topk_weight_kl": float(wkl.item()),
        "router_topk_support_hinge": float(hinge.item()),
        "router_topk_total": float(total.item()),
    }
