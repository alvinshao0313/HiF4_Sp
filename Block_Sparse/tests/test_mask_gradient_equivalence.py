from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.block_utils import reduce_weight_gradient_to_blocks


def test_explicit_block_mask_gradient_equivalence():
    """∂L/∂m_b ≈ sum_{(i,j)∈b} W_ij * ∂L/∂W_ij at m_b=1."""
    torch.manual_seed(1)
    block_height, block_width = 4, 2
    d_out, d_in = 8, 8
    linear = nn.Linear(d_in, d_out, bias=False)
    x = torch.randn(2, d_in)

    weight = linear.weight.detach().clone().requires_grad_(True)
    masks = torch.ones(
        d_out // block_height, d_in // block_width, requires_grad=True
    )
    element_mask = masks.repeat_interleave(block_height, dim=0).repeat_interleave(
        block_width, dim=1
    )
    masked_w = weight * element_mask
    y = x @ masked_w.t()
    loss = (y ** 2).sum()
    loss.backward()
    mask_grad = masks.grad.detach().clone()

    weight2 = linear.weight.detach().clone().requires_grad_(True)
    y2 = x @ weight2.t()
    loss2 = (y2 ** 2).sum()
    loss2.backward()
    reduced = reduce_weight_gradient_to_blocks(
        weight2.detach(), weight2.grad.detach(), block_height, block_width
    )

    max_err = (mask_grad.float() - reduced.float()).abs().max().item()
    assert max_err < 1e-4, max_err


def test_fisher_squares_before_accumulate():
    a = 3.0
    score_sq = torch.tensor(0.0)
    for signal in (a, -a):
        score_sq = score_sq + signal ** 2
    fisher = score_sq / 2.0
    assert abs(fisher.item() - a * a) < 1e-8
    signed_mean = (a + (-a)) / 2.0
    assert abs(signed_mean) < 1e-8
