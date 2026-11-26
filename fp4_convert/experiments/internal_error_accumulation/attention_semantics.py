"""Explicit causal masks for the IEA trainer's existing layer runtime.

The shared runtime sets is_causal=False, so None means unrestricted attention.
Keep its other callers unchanged; IEA must supply its own causal mask.
"""
from __future__ import annotations

import torch

TRAINING_PATH_VERSION = 3


def causal_attention_mask(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[1] <= 0:
        raise ValueError('expected nonempty [batch, tokens, hidden]')
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    # Samples use right padding. Every valid query can see only earlier valid
    # keys; padded query outputs are excluded by the existing length/loss mask.
    return (positions[:, None] >= positions[None, :])[None, None]
