from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.block_utils import reduce_weight_gradient_to_blocks
from block_pruning.config import parse_block_size


def _naive_block_reduce(
    weight: torch.Tensor,
    grad: torch.Tensor,
    block_height: int,
    block_width: int,
) -> torch.Tensor:
    d_out, d_in = weight.shape
    n_out, n_in = d_out // block_height, d_in // block_width
    out = torch.zeros(n_out, n_in, dtype=torch.float64)
    for a in range(n_out):
        for b in range(n_in):
            w = weight[
                a * block_height : (a + 1) * block_height,
                b * block_width : (b + 1) * block_width,
            ]
            g = grad[
                a * block_height : (a + 1) * block_height,
                b * block_width : (b + 1) * block_width,
            ]
            out[a, b] = (w.double() * g.double()).sum()
    return out


def test_parse_block_size():
    assert parse_block_size("128") == (128, 128)
    assert parse_block_size(128) == (128, 128)
    assert parse_block_size("64x128") == (64, 128)
    assert parse_block_size("64X128") == (64, 128)
    try:
        parse_block_size("abc")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_block_reduce_matches_naive_square():
    torch.manual_seed(0)
    h = w = 128
    weight = torch.randn(256, 384, dtype=torch.float32)
    grad = torch.randn(256, 384, dtype=torch.float32)
    fast = reduce_weight_gradient_to_blocks(weight, grad, h, w)
    slow = _naive_block_reduce(weight, grad, h, w)
    max_err = (fast.double().cpu() - slow).abs().max().item()
    assert max_err < 1e-3, max_err


def test_block_reduce_matches_naive_rect():
    torch.manual_seed(1)
    h, w = 64, 128
    weight = torch.randn(256, 256, dtype=torch.float32)
    grad = torch.randn(256, 256, dtype=torch.float32)
    fast = reduce_weight_gradient_to_blocks(weight, grad, h, w)
    slow = _naive_block_reduce(weight, grad, h, w)
    assert fast.shape == (4, 2)
    max_err = (fast.double().cpu() - slow).abs().max().item()
    assert max_err < 1e-3, max_err


def test_block_reduce_rejects_nondivisible():
    weight = torch.randn(130, 128)
    grad = torch.randn(130, 128)
    try:
        reduce_weight_gradient_to_blocks(weight, grad, 128, 128)
        assert False, "expected ValueError"
    except ValueError:
        pass
