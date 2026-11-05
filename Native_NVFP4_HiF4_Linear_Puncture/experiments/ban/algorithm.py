"""Ban routing and calibration math from the pinned upstream implementation.

Adapted from MingZwhy/An-Empirical-Study-on-Expert-Pruning, under the MIT
license in UPSTREAM_LICENSE. This module has no vLLM or model dependencies.
"""

from __future__ import annotations

import math

import torch


SOURCE_COMMIT = "3ac1169c9a48f6cda2477975a024aadba5d02f8b"
SOURCE_URL = (
    "https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning/"
    f"tree/{SOURCE_COMMIT}"
)


def validate_parameters(lam: float, k_min: int, top_k: int) -> None:
    """Validate scalar settings; upstream clips k_min to the model top-k."""
    if not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("Ban lambda must be in [0, 1]")
    if isinstance(k_min, bool) or not isinstance(k_min, int) or k_min < 1:
        raise ValueError("Ban k_min must be an integer >= 1")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("Ban top_k must be an integer >= 1")


def top3_ratio(weights: torch.Tensor) -> torch.Tensor:
    """Return Top3/TopK mass for weights already sorted in descending order."""
    if weights.ndim < 1 or weights.shape[-1] < 1:
        raise ValueError("Ban weights must contain at least one expert")
    return weights[..., : min(3, weights.shape[-1])].sum(dim=-1) / weights.sum(
        dim=-1
    ).clamp_min(1e-20)


def ban_route(
    logits: torch.Tensor,
    top_k: int,
    layer_sensitivity: torch.Tensor,
    r_min: float,
    r_max: float,
    lam: float = 0.7,
    k_min: int = 3,
    renormalize: bool = True,
    indices_type: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed-width weights/IDs and per-token retained expert counts.

    IDs keep the complete original sorted top-k. Pruned suffix weights are
    zero; the runtime uses counts to avoid dispatching those suffixes.
    """
    validate_parameters(lam, k_min, top_k)
    if logits.ndim != 2 or top_k > logits.shape[-1]:
        raise ValueError("Ban logits must have shape [tokens, experts] with experts >= top_k")
    if layer_sensitivity.numel() != 1:
        raise ValueError("Ban layer_sensitivity must be a scalar tensor")
    if not math.isfinite(r_min) or not math.isfinite(r_max) or r_max < r_min:
        raise ValueError("Ban ratio bounds must be finite with r_min <= r_max")
    if indices_type not in (torch.int32, torch.int64):
        raise ValueError("Ban expert IDs require int32 or int64")

    weights, ids = torch.topk(
        torch.softmax(logits.float(), dim=-1), k=top_k, dim=-1, sorted=True
    )
    ratio = top3_ratio(weights)
    token_sensitivity = ((r_max - ratio) / max(r_max - r_min, 1e-6)).clamp(0.0, 1.0)
    layer_sensitivity = layer_sensitivity.to(device=weights.device, dtype=weights.dtype).clamp(
        0.0, 1.0
    )
    sensitivity = (lam * 0.5 * (layer_sensitivity + token_sensitivity)).clamp(0.0, 1.0)
    minimum = min(k_min, top_k)
    counts = torch.round(minimum + (top_k - minimum) * sensitivity).long().clamp(
        min=minimum, max=top_k
    )
    ranks = torch.arange(top_k, device=logits.device).view(1, top_k)
    weights = weights.masked_fill(ranks >= counts.view(-1, 1), 0.0)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.float(), ids.to(dtype=indices_type), counts


def reference_distribution(
    logits: torch.Tensor, topn: int = 1000
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize the native logits over their top-n vocabulary support."""
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("Ban reference logits must have a nonempty vocabulary")
    if isinstance(topn, bool) or not isinstance(topn, int) or topn < 1:
        raise ValueError("Ban topn must be an integer >= 1")
    top_logits, indices = torch.topk(logits.float(), k=min(topn, logits.shape[-1]), dim=-1)
    log_probs = torch.log_softmax(top_logits, dim=-1)
    return indices, log_probs.exp(), log_probs


def reference_kl(
    logits: torch.Tensor,
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Mean KL(native || pruned), both normalized on native top-n support."""
    indices, probs, log_probs = reference
    selected_logits = torch.gather(logits.float(), dim=-1, index=indices)
    candidate_log_probs = torch.log_softmax(selected_logits, dim=-1)
    return (probs * (log_probs - candidate_log_probs)).sum(dim=-1).mean()


def normalize_layer_kl(rawkl: torch.Tensor) -> torch.Tensor:
    """Apply upstream min-max normalization after averaging each layer KL."""
    if rawkl.ndim != 1 or rawkl.numel() == 0 or not torch.isfinite(rawkl).all():
        raise ValueError("Ban raw layer KL must be a finite nonempty vector")
    rawkl = rawkl.float()
    minimum = rawkl.min()
    return (rawkl - minimum) / (rawkl.max() - minimum).clamp_min(1e-20)
