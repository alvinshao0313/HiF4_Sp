"""Tests for search/validation row split utilities."""

from __future__ import annotations

import torch
import pytest

from permutation_optimization.split_utils import (
    RowSplit,
    apply_row_split,
    make_row_split,
)


def test_make_row_split_is_disjoint_and_complete():
    split = make_row_split(n_rows=512, validation_fraction=0.2, seed=42)
    search = set(split.search_idx.tolist())
    val = set(split.validation_idx.tolist())
    assert search.isdisjoint(val)
    assert search | val == set(range(512))
    assert len(val) == 102
    assert len(search) == 410


def test_make_row_split_reproducible_and_seed_sensitive():
    a = make_row_split(512, 0.2, 42)
    b = make_row_split(512, 0.2, 42)
    c = make_row_split(512, 0.2, 43)
    assert torch.equal(a.search_idx, b.search_idx)
    assert torch.equal(a.validation_idx, b.validation_idx)
    assert not torch.equal(a.validation_idx, c.validation_idx)


def test_apply_row_split_keeps_x_and_activation_aligned():
    x = torch.arange(512).view(512, 1)
    a = torch.cat([x, x + 1000], dim=1)
    split = make_row_split(512, 0.2, 42)
    x_search, x_val = apply_row_split(x, split)
    a_search, a_val = apply_row_split(a, split)
    assert torch.equal(a_search[:, 0], x_search[:, 0])
    assert torch.equal(a_val[:, 0], x_val[:, 0])


def test_make_row_split_rejects_bad_args():
    with pytest.raises(ValueError):
        make_row_split(n_rows=1, validation_fraction=0.2, seed=42)
    with pytest.raises(ValueError):
        make_row_split(n_rows=16, validation_fraction=0.0, seed=42)
    with pytest.raises(ValueError):
        make_row_split(n_rows=16, validation_fraction=0.5, seed=42)


def test_make_row_split_index_dtype_long():
    split = make_row_split(64, 0.25, 7)
    assert split.search_idx.dtype == torch.long
    assert split.validation_idx.dtype == torch.long
    assert isinstance(split, RowSplit)
