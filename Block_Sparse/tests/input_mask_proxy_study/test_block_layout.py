from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.input_mask_proxy_study.block_layout import (  # noqa: E402
    output_block_scores,
    split_activation_blocks,
    split_weight_blocks,
    stable_topk_mask,
)


def test_split_activation_shape():
    x = torch.randn(64, 128)
    xb = split_activation_blocks(x)
    assert xb.shape == (2, 2, 32, 64)


def test_split_weight_shape():
    w = torch.randn(96, 128)
    wb = split_weight_blocks(w)
    assert wb.shape == (3, 2, 32, 64)


def test_partial_sum_matches_dense():
    torch.manual_seed(0)
    x = torch.randn(64, 128, dtype=torch.float32)
    w = torch.randn(96, 128, dtype=torch.float32)
    xb = split_activation_blocks(x)
    wb = split_weight_blocks(w)
    # Reconstruct dense via partial products.
    y = torch.zeros(64, 96, dtype=torch.float32)
    a, kb, _, _ = xb.shape
    jb = wb.shape[0]
    for i in range(a):
        for j in range(jb):
            acc = torch.zeros(32, 32, dtype=torch.float32)
            for k in range(kb):
                acc = acc + xb[i, k] @ wb[j, k].T
            y[i * 32 : (i + 1) * 32, j * 32 : (j + 1) * 32] = acc
    assert torch.allclose(y, x @ w.T, atol=1e-5, rtol=1e-5)


def test_stable_topk_tie_prefers_smaller_index():
    scores = torch.tensor([[1.0, 1.0, 0.5, 1.0]], dtype=torch.float32)
    mask = stable_topk_mask(scores, keep_count=2)
    assert mask.shape == (1, 4)
    assert int(mask.sum()) == 2
    assert bool(mask[0, 0]) and bool(mask[0, 1])
    assert not bool(mask[0, 3])


def test_stable_topk_keep_count_exact():
    scores = torch.randn(5, 10)
    mask = stable_topk_mask(scores, keep_count=3)
    assert torch.all(mask.sum(dim=-1) == 3)


def test_output_block_scores_mean_square():
    y = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    # Make 2x2 blocks of 1x1 effectively by using block 1 — use proper shape
    y = torch.zeros(32, 64, dtype=torch.float32)
    y[:32, :32] = 2.0
    y[:32, 32:] = 1.0
    scores = output_block_scores(y)
    assert scores.shape == (1, 2)
    assert scores[0, 0].item() == pytest.approx(4.0)
    assert scores[0, 1].item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "fn,shape",
    [
        (lambda: split_activation_blocks(torch.randn(33, 64)), (33, 64)),
        (lambda: split_activation_blocks(torch.randn(32, 65)), (32, 65)),
        (lambda: split_weight_blocks(torch.randn(33, 64)), (33, 64)),
        (lambda: output_block_scores(torch.randn(32, 33)), (32, 33)),
    ],
)
def test_non_divisible_shapes_error(fn, shape):
    with pytest.raises(ValueError):
        fn()
