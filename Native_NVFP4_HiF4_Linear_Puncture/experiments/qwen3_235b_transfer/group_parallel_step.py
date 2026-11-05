"""One original global-batch Group update with token-correct router loss."""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.distributed as dist

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.losses import masked_router_loss, router_metrics


ROUTER_METRICS = ("topk_set_match", "topk_overlap", "student_outside_mass", "teacher_outside_mass")
_GRADIENT_BUCKET_BYTES = 64 * 1024 * 1024


@torch.no_grad()
def sum_gradients(named_parameters, *, process_group=None) -> None:
    """SUM gradients in bounded buckets, preserving globally unused grad=None."""
    presence = torch.tensor([int(parameter.grad is not None) for _, parameter in named_parameters],
                            dtype=torch.int32, device=named_parameters[0][1].device)
    dist.all_reduce(presence, op=dist.ReduceOp.SUM, group=process_group)
    used = presence.tolist()
    if not any(used):
        raise RuntimeError("missing Group gradients on every rank")
    by_type = defaultdict(list)
    for (_, parameter), count in zip(named_parameters, used):
        if count == 0:
            parameter.grad = None
        else:
            by_type[(parameter.device, parameter.dtype)].append(parameter)

    def reduce_bucket(parameters):
        flat = torch.cat([
            (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1)
            for parameter in parameters
        ])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=process_group)
        if not bool(torch.isfinite(flat).all()):
            raise RuntimeError("non-finite globally summed Group gradients")
        offset = 0
        for parameter in parameters:
            end = offset + parameter.numel()
            parameter.grad = flat[offset:end].view_as(parameter)
            offset = end

    for parameters in by_type.values():
        bucket, size = [], 0
        for parameter in parameters:
            nbytes = parameter.numel() * parameter.element_size()
            if bucket and size + nbytes > _GRADIENT_BUCKET_BYTES:
                reduce_bucket(bucket)
                bucket, size = [], 0
            bucket.append(parameter)
            size += nbytes
        if bucket:
            reduce_bucket(bucket)


def group_parallel_step(student, x, target, teacher_logits, loss_mask, attention_mask,
                        snapshot, cfg, optimizer, scheduler, *, process_group=None) -> dict:
    """Retain full global padding and SUM the derivative of the global objective.

    Each rank contributes N_r / SUM(D_r) plus lambda * n_r R_r / SUM(n_r).
    R_r is the original masked router mean; no extra world-size average applies.
    The caller owns identical replicas, optimizer state, and original batch order.
    """
    if not dist.is_initialized():
        raise RuntimeError("Group sample parallelism requires an initialized process group")
    if x.ndim != 3 or x.shape[0] != 1 or target.shape != x.shape:
        raise ValueError("each Group rank needs matching [1, global_seq_len, hidden] inputs")
    if teacher_logits.shape[:2] != x.shape[:2] or loss_mask.shape != x.shape[:2] or attention_mask.shape != x.shape[:2]:
        raise ValueError("Group masks/router targets must preserve the global padded shape")
    if x.requires_grad or target.requires_grad or teacher_logits.requires_grad:
        raise ValueError("Group hidden inputs and teacher targets must be frozen")
    named_parameters = sorted((name, parameter) for name, parameter in student.named_parameters()
                              if parameter.requires_grad)
    if not named_parameters:
        raise ValueError("Group update requires trainable parameters")
    student.train()
    valid = attention_mask.bool() & loss_mask.bool()
    local_tokens = int(valid.sum())
    global_tokens = torch.tensor(local_tokens, dtype=torch.int64, device=x.device)
    dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM, group=process_group)
    if int(global_tokens) == 0:
        raise ValueError("empty effective global Group loss mask")
    optimizer.zero_grad(set_to_none=True)
    call = build_qwen3_moe_layer_call(str(snapshot), x)
    out = student(x, position_embeddings=call.position_embeddings, use_ste=True)
    pred_delta = out.output[valid].float() - x[valid].float()
    target_delta = target[valid].float() - x[valid].float()
    num = (pred_delta - target_delta).square().sum()
    den = target_delta.square().sum()
    # This original helper detaches BEFORE moe_norm, retaining gradients to the
    # trainable norm and router while keeping the auxiliary loss out of attention.
    aux_logits = student.router_aux_logits(out.pre_moe_norm)
    router = (masked_router_loss(aux_logits, teacher_logits, valid, kind=cfg.router_loss,
                                 top_k=cfg.router_top_k, temperature=cfg.router_temperature)
              if local_tokens else aux_logits.sum() * 0.0)
    totals = torch.stack((num.detach(), den.detach(), router.detach() * local_tokens))
    dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=process_group)
    if not bool(torch.isfinite(totals).all()) or float(totals[1]) <= 0:
        raise RuntimeError("non-finite Group objective or zero global reconstruction energy")
    local_objective = num / totals[1] + cfg.router_loss_weight * router * (local_tokens / global_tokens)
    global_objective = totals[0] / totals[1] + cfg.router_loss_weight * totals[2] / global_tokens
    if not bool(torch.isfinite(global_objective)):
        raise RuntimeError("non-finite global Group objective")
    local_objective.backward()
    sum_gradients(named_parameters, process_group=process_group)
    optimizer.step()
    scheduler.step()

    measures = (router_metrics(out.router_logits, teacher_logits, valid, top_k=student.base.spec.top_k)
                if local_tokens else {name: 0.0 for name in ROUTER_METRICS})
    weighted = torch.tensor([float(router.detach()) * local_tokens,
                             *(measures[name] * local_tokens for name in ROUTER_METRICS)],
                            dtype=torch.float64, device=x.device)
    dist.all_reduce(weighted, op=dist.ReduceOp.SUM, group=process_group)
    selected = out.router_logits.detach()[valid].topk(student.base.spec.top_k, dim=-1).indices
    counts = torch.bincount(selected.reshape(-1), minlength=student.base.spec.num_experts)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=process_group)
    values = weighted.tolist()
    return {
        "num": float(totals[0]), "den": float(totals[1]), "router_sum": values[0],
        "tokens": int(global_tokens), "objective": float(global_objective),
        "router_metric_sums": dict(zip(ROUTER_METRICS, values[1:])),
        "expert_token_counts": counts.cpu().tolist(),
    }
