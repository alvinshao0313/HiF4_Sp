from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.dynamic_input_sparse.common import ratio_to_keep_count  # noqa: E402
from Block_Sparse.dynamic_input_sparse.m8_energy import (  # noqa: E402
    compute_all_output_weight_energy,
    predict_m8_input_mask,
)
from Block_Sparse.input_mask_proxy_study.block_layout import (  # noqa: E402
    split_activation_blocks,
    split_weight_blocks,
)
from Block_Sparse.input_mask_proxy_study.energy_recovery import (  # noqa: E402
    recover_input_masks_energy_unconditioned,
)
from Block_Sparse.input_mask_proxy_study.hif4_proxy import (  # noqa: E402
    build_hif4_ternary_proxy,
)


def test_static_weight_energy_parity():
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.float32)
    w_blocks = split_weight_blocks(w, block_out=32, block_k=64)
    ref = w_blocks.square().mean(dim=(-1, -2)).sum(dim=0)
    new = compute_all_output_weight_energy(w)
    assert torch.equal(ref, new)


@pytest.mark.parametrize("ratio", [0.75, 0.50, 0.25])
def test_per_token_input_mask_parity(ratio):
    torch.manual_seed(1)
    t, din, dout = 5, 256, 64
    x = torch.randn(t, din)
    w = torch.randn(dout, din)
    g_w = compute_all_output_weight_energy(w)
    mx_new = predict_m8_input_mask(x, g_w, ratio)

    xp = build_hif4_ternary_proxy(x).proxy
    # row-block size 1 -> [T,Kb,1,64]
    act = split_activation_blocks(xp, block_rows=1, block_k=64)
    keep = ratio_to_keep_count(ratio, din // 64)
    ref = recover_input_masks_energy_unconditioned(act, g_w, (keep,))
    assert torch.equal(mx_new, ref.masks_by_keep[keep])


def test_deterministic_ties():
    # Equal scores -> smaller K-block indices kept first
    t, kb = 1, 4
    scores_equal_via_x = torch.ones(t, kb * 64)
    # Craft weight energy equal and proxy energy equal by constant input
    w = torch.ones(32, kb * 64)
    g_w = compute_all_output_weight_energy(w)
    mx = predict_m8_input_mask(scores_equal_via_x, g_w, 0.5)
    # keep_count = 2 -> indices 0,1
    assert torch.equal(mx[0], torch.tensor([True, True, False, False]))


def test_fused_gate_up_parity():
    torch.manual_seed(2)
    din, dout = 256, 64
    w_gate = torch.randn(dout, din)
    w_up = torch.randn(dout, din)
    w_fused = torch.cat([w_gate, w_up], dim=0)
    g_gate = compute_all_output_weight_energy(w_gate)
    g_up = compute_all_output_weight_energy(w_up)
    g_fused = compute_all_output_weight_energy(w_fused)
    assert torch.allclose(g_fused, g_gate + g_up, atol=0, rtol=0)
    x = torch.randn(3, din)
    for ratio in (0.75, 0.5, 0.25):
        m1 = predict_m8_input_mask(x, g_gate + g_up, ratio)
        m2 = predict_m8_input_mask(x, g_fused, ratio)
        assert torch.equal(m1, m2)


def test_keep_one_fast_path():
    x = torch.randn(2, 128)
    g = compute_all_output_weight_energy(torch.randn(32, 128))
    mx = predict_m8_input_mask(x, g, 1.0)
    assert bool(mx.all())
