"""One E4 update with sample parallelism and SUM-reduced DIAG gradients.

The caller owns process-group initialization, replica/optimizer initialization,
the original global batch order, and global padding before splitting samples.
This module does not wrap the runtime in DDP or install saved-tensor hooks.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import E2ETrainConfig
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    StudentQwen3MoELayerRuntime,
    StudentStepCache,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.losses import (
    finalize_reconstruction_loss,
    masked_reconstruction_components,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_layer_runtime import (
    build_qwen3_moe_layer_call,
)


def _require_process_group() -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("E4 sample parallelism requires an initialized process group")


@torch.no_grad()
def broadcast_parameters(
    module: torch.nn.Module,
    *,
    src: int = 0,
    process_group=None,
) -> None:
    """Synchronize parameters from a global source rank in the supplied group.

    Call before constructing fresh optimizers. Existing optimizer and scheduler
    state, frozen layer weights, and RNG state remain the caller's responsibility.
    All ranks must have identical parameter names, shapes, and dtypes.
    """
    _require_process_group()
    for _name, parameter in sorted(module.named_parameters()):
        dist.broadcast(parameter, src=src, group=process_group)


@torch.no_grad()
def _sum_gradients(named_parameters, *, process_group=None) -> None:
    # AdamW skips grad=None parameters. Preserve that behavior when no rank used
    # a parameter; only locally absent, globally present gradients become zero.
    presence = torch.tensor(
        [int(parameter.grad is not None) for _name, parameter in named_parameters],
        dtype=torch.int32,
        device=named_parameters[0][1].device,
    )
    dist.all_reduce(presence, op=dist.ReduceOp.SUM, group=process_group)
    for (_name, parameter), used_ranks in zip(named_parameters, presence.tolist()):
        if not used_ranks:
            parameter.grad = None
            continue
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        gradient = gradient.contiguous()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=process_group)
        parameter.grad = gradient
    if any(
        not bool(torch.isfinite(parameter.grad).all())
        for _name, parameter in named_parameters
        if parameter.grad is not None
    ):
        raise RuntimeError("non-finite E4 globally summed gradients")


def e4_parallel_step(
    runtime: StudentQwen3MoELayerRuntime,
    hidden: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    snapshot: str | Path,
    cfg: E2ETrainConfig,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    process_group=None,
) -> dict[str, float | int]:
    """Perform one original global-batch update across one sample per rank.

    Each input must be a batch-dimension slice of the original globally padded
    batch, with frozen hidden/teacher tensors and matching masks. In particular,
    attention_mask is used only for loss masking: E4's runtime call continues to
    receive None, so independently trimming padding would change its attention.

    Replicas must start with identical frozen weights, DIAG values, requires_grad
    flags, optimizer state, and scheduler state. The optimizer must contain all
    and only trainable DIAG parameters. No DDP gradient reducer may be installed.
    The caller retains batch ordering, epoch count, scheduler T_max, and RNG state.
    """
    _require_process_group()
    if cfg.diag_mode != "fusable" or cfg.recon_loss != "block_delta_nmse":
        raise ValueError("E4 parallel step requires fusable block_delta_nmse")
    if float(cfg.router_align_loss_weight) != 0.0:
        raise ValueError("E4 parallel step only supports router_align_loss_weight=0")
    if hidden.ndim != 3 or hidden.shape[0] != 1 or target.shape != hidden.shape:
        raise ValueError("each rank requires matching [1, global_seq_len, hidden] inputs")
    if loss_mask.shape != hidden.shape[:2] or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("loss and attention masks must match the local token dimensions")
    if hidden.requires_grad or target.requires_grad:
        raise ValueError("E4 hidden inputs and teacher targets must be frozen")
    named_parameters = sorted(
        (name, parameter)
        for name, parameter in runtime.diag_state.named_parameters()
        if parameter.requires_grad
    )
    if not named_parameters:
        raise ValueError("E4 parallel step requires trainable DIAG parameters")

    runtime.train()
    call = build_qwen3_moe_layer_call(str(snapshot), hidden)
    cache = StudentStepCache.new()
    optimizer.zero_grad(set_to_none=True)
    out = runtime(
        hidden,
        attention_mask=None,
        position_embeddings=call.position_embeddings,
        step_cache=cache,
        use_ste=True,
    )
    loss_mask = loss_mask.to(hidden.device)
    attention_mask = attention_mask.to(hidden.device)
    pred = out.output.float() - hidden.float()
    tgt = target.float() - hidden.float()
    num, den = masked_reconstruction_components(
        pred, tgt, loss_mask, attention_mask, cfg.recon_loss
    )
    global_components = torch.stack((num.detach(), den.detach()))
    dist.all_reduce(global_components, op=dist.ReduceOp.SUM, group=process_group)
    global_num, global_den = global_components.unbind()
    global_loss = finalize_reconstruction_loss(global_num, global_den, cfg.recon_loss)
    if not bool(torch.isfinite(global_components).all()) or not bool(torch.isfinite(global_loss)):
        raise RuntimeError("non-finite E4 global reconstruction loss")

    # With SUM-reduced gradients, local N_r / sum(D_r) gives the original
    # gradient of sum(N_r) / sum(D_r), without a world-size factor.
    loss = finalize_reconstruction_loss(num, global_den, cfg.recon_loss)
    loss.backward()
    _sum_gradients(named_parameters, process_group=process_group)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    runtime.diag_state.clamp_log2_(cfg.diag_log2_clamp)

    local_counts = torch.stack((
        (loss_mask.bool() & attention_mask.bool()).sum(),
        attention_mask.bool().sum(),
    ))
    global_counts = local_counts.clone()
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=process_group)
    return {
        "loss": float(global_loss),
        "num": float(global_num),
        "den": float(global_den),
        "local_num": float(num.detach()),
        "local_den": float(den.detach()),
        "local_effective_tokens": int(local_counts[0]),
        "global_effective_tokens": int(global_counts[0]),
        "local_attention_tokens": int(local_counts[1]),
        "global_attention_tokens": int(global_counts[1]),
        "world_size": dist.get_world_size(process_group),
    }
