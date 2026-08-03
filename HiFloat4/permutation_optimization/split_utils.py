"""Deterministic, auditable search/validation row splits.

Search and validation token rows must be disjoint and traceable. This module
is the single source of truth for row splitting; the old per-call
``_split_activation`` allowed X and activation to be split independently,
which silently put validation rows back into search data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RowSplit:
    """Disjoint row indices over a shared row axis (dim 0)."""

    search_idx: torch.Tensor
    validation_idx: torch.Tensor


def make_row_split(
    n_rows: int,
    validation_fraction: float,
    seed: int,
) -> RowSplit:
    """Return disjoint search/validation row indices covering all rows.

    The same ``seed`` always yields the same split; different seeds yield
    different validation sets. Indices are CPU ``torch.long``.
    """
    if not isinstance(n_rows, int) or isinstance(n_rows, bool) or n_rows < 2:
        raise ValueError(f"n_rows must be an int >= 2, got {n_rows!r}")
    if (
        not isinstance(validation_fraction, (int, float))
        or isinstance(validation_fraction, bool)
        or not (0.0 < float(validation_fraction) < 0.5)
    ):
        raise ValueError(
            f"validation_fraction must be in (0, 0.5), got {validation_fraction!r}"
        )
    n_val = max(1, int(round(n_rows * validation_fraction)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    idx = torch.randperm(n_rows, generator=generator)
    search_idx = idx[:-n_val].contiguous()
    validation_idx = idx[-n_val:].contiguous()
    return RowSplit(search_idx=search_idx, validation_idx=validation_idx)


def apply_row_split(x: torch.Tensor, split: RowSplit) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``x`` along dim 0 into (search, validation) using ``split``."""
    if x.shape[0] < int(split.search_idx.max().item()) + 1 or (
        split.validation_idx.numel() > 0
        and x.shape[0] < int(split.validation_idx.max().item()) + 1
    ):
        raise ValueError(
            f"x has {x.shape[0]} rows but split indices exceed it"
        )
    search = x.index_select(0, split.search_idx.to(device=x.device))
    validation = x.index_select(0, split.validation_idx.to(device=x.device))
    return search.contiguous(), validation.contiguous()
