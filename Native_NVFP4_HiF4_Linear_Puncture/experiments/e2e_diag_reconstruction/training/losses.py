"""Masked reconstruction losses. Single source of truth for train and validation."""

from __future__ import annotations

import torch

EPS = 1e-30
LOSS_TYPES = ("block_delta_nmse", "block_output_nmse", "mse")


def _combine_mask(loss_mask: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return (loss_mask.to(torch.bool) & attention_mask.to(torch.bool)).to(torch.float32)


def masked_reconstruction_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"unknown recon_loss={loss_type!r}")
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
    mask3 = _combine_mask(loss_mask, attention_mask)[..., None]
    num = ((pred.float() - target.float()).pow(2) * mask3).sum()
    if loss_type == "mse":
        den = mask3.sum()
    else:
        den = (target.float().pow(2) * mask3).sum()
    return num, den


def finalize_reconstruction_loss(
    total_num: torch.Tensor | float,
    total_den: torch.Tensor | float,
    loss_type: str,
) -> torch.Tensor:
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"unknown recon_loss={loss_type!r}")
    num = total_num if isinstance(total_num, torch.Tensor) else torch.tensor(float(total_num))
    den = total_den if isinstance(total_den, torch.Tensor) else torch.tensor(float(total_den))
    return num / den.clamp_min(EPS)


def layer_objective(
    *,
    y_h: torch.Tensor,
    y_n: torch.Tensor,
    x: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_type: str,
    attn_h: torch.Tensor | None = None,
    attn_n: torch.Tensor | None = None,
    mlp_h: torch.Tensor | None = None,
    mlp_n: torch.Tensor | None = None,
    attn_weight: float = 0.0,
    mlp_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if loss_type == "block_delta_nmse":
        pred = y_h.float() - x.float()
        target = y_n.float() - x.float()
    else:
        pred = y_h
        target = y_n
    num, den = masked_reconstruction_components(
        pred, target, loss_mask, attention_mask, loss_type
    )
    loss = finalize_reconstruction_loss(num, den, loss_type)
    if attn_weight > 0:
        if attn_h is None or attn_n is None:
            raise RuntimeError("attn aux weight > 0 but attn tensors are missing")
        a_num, a_den = masked_reconstruction_components(
            attn_h, attn_n, loss_mask, attention_mask, "block_output_nmse"
        )
        loss = loss + attn_weight * finalize_reconstruction_loss(
            a_num, a_den, "block_output_nmse"
        )
    if mlp_weight > 0:
        if mlp_h is None or mlp_n is None:
            raise RuntimeError("mlp aux weight > 0 but mlp tensors are missing")
        m_num, m_den = masked_reconstruction_components(
            mlp_h, mlp_n, loss_mask, attention_mask, "block_output_nmse"
        )
        loss = loss + mlp_weight * finalize_reconstruction_loss(
            m_num, m_den, "block_output_nmse"
        )
    return loss, num, den
