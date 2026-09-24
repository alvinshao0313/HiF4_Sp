"""Losses and diagnostics for cumulative hidden-error cancellation."""
from __future__ import annotations

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.losses import (
    masked_router_loss,
    router_metrics as _base_router_metrics,
)


def absolute_mse(output, target, valid):
    if output.shape != target.shape or valid.shape != output.shape[:-1]:
        raise ValueError("MSE boundary shapes differ")
    values = (output.float() - target.detach().float())[valid]
    if not values.numel():
        raise ValueError("empty valid MSE")
    return values.square().mean()


def batch_cosine(current_error, direction, valid, eps=1e-12):
    if current_error.shape != direction.shape or valid.shape != current_error.shape[:-1]:
        raise ValueError("direction boundary shapes differ")
    a = current_error.float()[valid].reshape(-1)
    b = direction.detach().float()[valid].reshape(-1)
    na, nb = a.norm(), b.norm()
    if not torch.isfinite(na + nb) or na <= eps or nb <= eps:
        return current_error.new_zeros((), dtype=torch.float32), False, float(na.detach()), float(nb.detach())
    cosine = (a @ b) / (na * nb)
    if not torch.isfinite(cosine):
        raise RuntimeError("nonfinite direction cosine")
    return 1. + cosine, True, float(na.detach()), float(nb.detach())


def direction_metrics(error, direction, valid):
    _, active, norm_error, norm_direction = batch_cosine(error, direction, valid)
    if not active:
        return {"active": False, "cosine": None, "error_norm": norm_error,
                "direction_norm": norm_direction}
    a = error.float()[valid].reshape(-1)
    b = direction.detach().float()[valid].reshape(-1)
    cosine = float((a @ b) / (a.norm() * b.norm()))
    return {"active": True, "cosine": cosine, "error_norm": norm_error,
            "direction_norm": norm_direction}


def decompose_errors(student_output, native_target, native_on_student_input, valid):
    c = student_output.float() - native_target.detach().float()
    p = native_on_student_input.float() - native_target.detach().float()
    q = student_output.float() - native_on_student_input.detach().float()
    if not torch.allclose(c[valid], (p + q)[valid], atol=2e-5, rtol=2e-5):
        raise RuntimeError("cumulative error decomposition failed")
    return {"cumulative": c, "propagated": p, "local": q}


def summarize_error(error, valid):
    values = error.float()[valid]
    if not torch.isfinite(values).all():
        raise RuntimeError("nonfinite hidden-error diagnostic")
    return {"mse": float(values.square().mean()), "l2": float(values.norm()),
            "tokens": int(valid.sum()), "channels": int(values.shape[-1])}


@torch.no_grad()
def router_metrics(student_logits, teacher_logits, mask, *, top_k=8):
    """Legacy router metrics plus explicit top-k margins and mismatch rate."""
    result = _base_router_metrics(student_logits, teacher_logits, mask, top_k=top_k)
    student = student_logits[mask.bool()].float()
    teacher = teacher_logits[mask.bool()].float()
    k = min(top_k, teacher.shape[-1])

    def margin(logits):
        values = logits.topk(k, dim=-1).values
        if k == logits.shape[-1]:
            return values[:, -1]
        outside = logits.masked_fill(torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, logits.topk(k, dim=-1).indices, True), -torch.inf)
        return values[:, -1] - outside.max(dim=-1).values

    result.update({
        "teacher_route_margin": float(margin(teacher).mean()),
        "student_route_margin": float(margin(student).mean()),
        "route_mismatch_rate": 1.0 - float(result["topk_set_match"]),
    })
    return result


__all__ = ["absolute_mse", "batch_cosine", "direction_metrics", "decompose_errors",
           "masked_router_loss", "router_metrics", "summarize_error"]
