from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.dynamic_input_sparse.common import ratio_to_keep_count  # noqa: E402
from Block_Sparse.dynamic_input_sparse.m1_oracle import (  # noqa: E402
    predict_m1_full_output_mask,
    predict_m1_full_output_mask_multiweight,
)
from Block_Sparse.dynamic_input_sparse.m8_energy import (  # noqa: E402
    compute_all_output_weight_energy,
    predict_m8_input_mask,
)
from Block_Sparse.input_mask_proxy_study.block_layout import (  # noqa: E402
    split_activation_blocks,
    split_weight_blocks,
)
from Block_Sparse.input_mask_proxy_study.exact_recovery import (  # noqa: E402
    recover_input_masks_exact,
)


def _sse_from_mask(x: torch.Tensor, w: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
    """Per-token full-output SSE for kept K-blocks."""
    t, din = x.shape
    dout = w.shape[0]
    kb = din // 64
    x_blocks = x.reshape(t, kb, 64).to(torch.float64)
    w_by_k = w.reshape(dout, kb, 64).permute(1, 0, 2).to(torch.float64)
    p = torch.einsum("tkd,kod->tko", x_blocks, w_by_k)
    y = p.sum(dim=1)
    yhat = (p * mx.to(p.dtype).unsqueeze(-1)).sum(dim=1)
    return ((y - yhat) ** 2).sum(dim=-1)


@pytest.mark.parametrize("keep_ratio", [0.75, 0.5, 0.25])
def test_parity_with_scalar_exact_recovery(keep_ratio):
    torch.manual_seed(0)
    t, kb, dout = 3, 4, 64
    din = kb * 64
    x = torch.randn(t, din, dtype=torch.float32)
    w = torch.randn(dout, din, dtype=torch.float32)
    keep = ratio_to_keep_count(keep_ratio, kb)

    # Old API: activation row-block=1, all-ones output mask
    x_blocks = split_activation_blocks(x, block_rows=1, block_k=64)
    w_blocks = split_weight_blocks(w, block_out=32, block_k=64)
    jb = dout // 32
    my = torch.ones(t, jb, dtype=torch.bool)
    old = recover_input_masks_exact(x_blocks, w_blocks, my, (keep,))

    new = predict_m1_full_output_mask(
        x, w, keep_ratio, token_chunk_size=8, return_internal=True
    )
    assert torch.equal(new.final_mask, old.masks_by_keep[keep])
    assert torch.equal(new.greedy_mask, old.greedy_masks_by_keep[keep])
    assert torch.equal(new.removal_order, old.removal_order)
    assert torch.equal(new.swap_count, old.swap_count_by_keep[keep])
    # Identical masks => identical recomputed FP64 SSE/MSE.
    sse_new = _sse_from_mask(x, w, new.final_mask)
    sse_old = _sse_from_mask(x, w, old.masks_by_keep[keep])
    assert torch.equal(sse_new, sse_old)
    mse = sse_old / dout
    # Stored Gram-tracked MSE from the old path may drift ~1e-8 relative;
    # require tight agreement with direct recomputation.
    mse_old = old.mse_by_keep[keep].to(torch.float64)
    rel_mse = (mse - mse_old).abs() / torch.clamp(mse_old.abs(), min=1e-12)
    assert bool((rel_mse <= 1e-6).all().item())


def test_chunk_invariance():
    torch.manual_seed(1)
    x = torch.randn(5, 256)
    w = torch.randn(64, 256)
    a = predict_m1_full_output_mask(x, w, 0.5, token_chunk_size=1, return_internal=True)
    b = predict_m1_full_output_mask(x, w, 0.5, token_chunk_size=8, return_internal=True)
    assert torch.equal(a.final_mask, b.final_mask)
    assert torch.equal(a.removal_order, b.removal_order)
    assert torch.equal(a.swap_count, b.swap_count)


def test_gate_up_concatenation_parity():
    torch.manual_seed(2)
    x = torch.randn(4, 256)
    w_gate = torch.randn(64, 256)
    w_up = torch.randn(64, 256)
    w_fused = torch.cat([w_gate, w_up], dim=0)
    for ratio in (0.75, 0.5, 0.25):
        m_multi = predict_m1_full_output_mask_multiweight(x, [w_gate, w_up], ratio)
        m_fused = predict_m1_full_output_mask(x, w_fused, ratio)
        assert torch.equal(m_multi, m_fused)


def test_oracle_non_regression_and_report():
    torch.manual_seed(3)
    x = torch.randn(3, 256)
    w = torch.randn(64, 256)
    g_w = compute_all_output_weight_energy(w)
    for ratio in (0.75, 0.5, 0.25):
        internal = predict_m1_full_output_mask(
            x, w, ratio, return_internal=True
        )
        m8 = predict_m8_input_mask(x, g_w, ratio)
        rnd = torch.zeros_like(m8)
        keep = ratio_to_keep_count(ratio, 256 // 64)
        for t in range(x.shape[0]):
            idx = torch.randperm(256 // 64)[:keep]
            rnd[t, idx] = True
        sse_final = _sse_from_mask(x, w, internal.final_mask)
        sse_greedy = _sse_from_mask(x, w, internal.greedy_mask)
        sse_m8 = _sse_from_mask(x, w, m8)
        sse_rnd = _sse_from_mask(x, w, rnd)
        assert bool((sse_final <= sse_greedy + 1e-8).all().item())
        # Report only (no hard gate that M1 beats M8/random)
        print(
            f"ratio={ratio} M1_final={sse_final.tolist()} "
            f"M1_greedy={sse_greedy.tolist()} M8={sse_m8.tolist()} rnd={sse_rnd.tolist()}"
        )


def test_keep_one_fast_path():
    x = torch.randn(2, 128)
    w = torch.randn(32, 128)
    mx = predict_m1_full_output_mask(x, w, 1.0)
    assert bool(mx.all())
