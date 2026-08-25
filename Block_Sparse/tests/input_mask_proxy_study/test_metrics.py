from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.input_mask_proxy_study.metrics import (  # noqa: E402
    kendall_tau_b,
    mask_metrics,
    nrmse,
    reconstruct_real_output,
    spearman_rank,
)


def test_identical_masks():
    m = torch.tensor([[True, False, True], [False, True, True]])
    met = mask_metrics(m, m)
    assert met["overlap"] == 1.0
    assert met["iou"] == 1.0
    assert met["hamming_agreement"] == 1.0
    assert met["exact_row_match"] == 1.0


def test_disjoint_masks():
    a = torch.tensor([[True, False]])
    b = torch.tensor([[False, True]])
    met = mask_metrics(a, b)
    assert met["overlap"] == 0.0
    assert met["iou"] == 0.0


def test_ranking_perfect_and_reverse():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman_rank(x, x) == 1.0
    assert abs(spearman_rank(x, -x) - (-1.0)) < 1e-9
    assert kendall_tau_b(x, x) == 1.0
    assert abs(kendall_tau_b(x, -x) - (-1.0)) < 1e-9


def test_nrmse_and_real_reconstruct():
    torch.manual_seed(0)
    x_blocks = torch.randn(1, 2, 32, 64)
    w_blocks = torch.randn(2, 2, 32, 64)
    mask = torch.tensor([[True, False]])
    y = reconstruct_real_output(x_blocks, w_blocks, torch.ones(1, 2, dtype=torch.bool))
    y_hat = reconstruct_real_output(x_blocks, w_blocks, mask)
    # Manual NRMSE on full
    num = torch.linalg.vector_norm(y - y_hat).item()
    den = torch.linalg.vector_norm(y).item() + 1e-12
    assert abs(nrmse(y_hat, y) - num / den) < 1e-8
