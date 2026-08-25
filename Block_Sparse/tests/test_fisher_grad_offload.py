from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.block_utils import reduce_weight_gradient_to_blocks
from block_pruning.gradient_scorer import (
    _empty_accumulators,
    _register_fisher_grad_offload_hooks,
    offload_weight_grad_to_block_accumulators,
)
from block_pruning.mlp_registry import MLPLinearTarget, initialize_all_one_masks


def test_offload_helper_matches_direct_reduce():
    torch.manual_seed(0)
    block_h, block_w = 4, 2
    linear = nn.Linear(8, 8, bias=False)
    linear.weight.requires_grad_(True)
    x = torch.randn(3, 8)
    y = x @ linear.weight.t()
    (y**2).sum().backward()

    targets = [
        MLPLinearTarget(
            module_name="layers.0.mlp.up_proj",
            module=linear,
            layer_index=0,
            projection_type="up_proj",
        )
    ]
    masks = initialize_all_one_masks(targets, block_h, block_w)
    acc = _empty_accumulators(targets, block_h, block_w)
    offload_weight_grad_to_block_accumulators(
        weight=linear.weight,
        grad=linear.weight.grad,
        module_name=targets[0].module_name,
        block_height=block_h,
        block_width=block_w,
        current_masks=masks,
        accumulators=acc,
    )

    expected = reduce_weight_gradient_to_blocks(
        linear.weight.detach(), linear.weight.grad.detach(), block_h, block_w
    ).double()
    got = acc[targets[0].module_name]["score_signed"]
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        acc[targets[0].module_name]["score_sq"], expected.square(), atol=1e-6, rtol=1e-6
    )


def test_hook_offload_matches_post_backward_reduce_and_frees_grad():
    torch.manual_seed(2)
    block_h, block_w = 2, 2
    d_out, d_in = 4, 4

    linear_a = nn.Linear(d_in, d_out, bias=False)
    linear_b = nn.Linear(d_in, d_out, bias=False)
    with torch.no_grad():
        linear_b.weight.copy_(linear_a.weight)

    targets_a = [
        MLPLinearTarget("layers.0.mlp.gate_proj", linear_a, 0, "gate_proj"),
        MLPLinearTarget("layers.0.mlp.up_proj", linear_b, 0, "up_proj"),
    ]
    # Two modules share the same forward path via concatenated loss terms.
    # Rebuild with independent copies for baseline path.
    linear_a2 = nn.Linear(d_in, d_out, bias=False)
    linear_b2 = nn.Linear(d_in, d_out, bias=False)
    with torch.no_grad():
        linear_a2.weight.copy_(linear_a.weight)
        linear_b2.weight.copy_(linear_b.weight)
    targets_b = [
        MLPLinearTarget("layers.0.mlp.gate_proj", linear_a2, 0, "gate_proj"),
        MLPLinearTarget("layers.0.mlp.up_proj", linear_b2, 0, "up_proj"),
    ]

    x = torch.randn(5, d_in)
    masks = initialize_all_one_masks(targets_a, block_h, block_w)

    # Baseline: keep grads until after backward, then reduce.
    for t in targets_b:
        t.module.weight.requires_grad_(True)
    y = (x @ linear_a2.weight.t()) + (x @ linear_b2.weight.t())
    (y**2).sum().backward()
    baseline = _empty_accumulators(targets_b, block_h, block_w)
    for t in targets_b:
        offload_weight_grad_to_block_accumulators(
            weight=t.module.weight,
            grad=t.module.weight.grad,
            module_name=t.module_name,
            block_height=block_h,
            block_width=block_w,
            current_masks=masks,
            accumulators=baseline,
        )

    # Hook path: free grads during backward.
    for t in targets_a:
        t.module.weight.requires_grad_(True)
    hooked = _empty_accumulators(targets_a, block_h, block_w)
    seen: set[str] = set()
    handles = _register_fisher_grad_offload_hooks(
        targets_a, block_h, block_w, masks, hooked, seen
    )
    try:
        y2 = (x @ linear_a.weight.t()) + (x @ linear_b.weight.t())
        (y2**2).sum().backward()
    finally:
        for h in handles:
            h.remove()

    assert seen == {t.module_name for t in targets_a}
    for t in targets_a:
        assert t.module.weight.grad is None
        for key in ("score_sq", "score_abs", "score_signed"):
            assert torch.allclose(
                hooked[t.module_name][key],
                baseline[t.module_name][key],
                atol=1e-6,
                rtol=1e-6,
            ), key
