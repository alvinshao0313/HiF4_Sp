from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.dynamic_input_sparse.common import (  # noqa: E402
    expand_k_block_mask,
    flatten_tokens,
    ratio_to_keep_count,
    restore_tokens,
)
from Block_Sparse.dynamic_input_sparse.masked_linear import (  # noqa: E402
    apply_input_block_mask,
)
from Block_Sparse.input_mask_proxy_study.config import (  # noqa: E402
    ratio_to_keep_count as study_ratio_to_keep_count,
)


@pytest.mark.parametrize("ratio,total", [(0.75, 40), (0.5, 40), (0.25, 40), (0.75, 144)])
def test_keep_count_matches_study(ratio, total):
    assert ratio_to_keep_count(ratio, total) == study_ratio_to_keep_count(ratio, total)


def test_mask_expansion_zeros_intended_ranges():
    mx = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    elem = expand_k_block_mask(mx, 64)
    assert elem.shape == (1, 256)
    assert bool(elem[0, 0:64].all())
    assert bool((~elem[0, 64:128]).all())
    assert bool(elem[0, 128:192].all())
    assert bool((~elem[0, 192:256]).all())


def test_flatten_restore_shapes():
    for shape in [(64,), (3, 64), (2, 5, 64)]:
        x = torch.randn(*shape)
        flat, leading = flatten_tokens(x)
        y = restore_tokens(flat, leading)
        assert y.shape == x.shape
        assert torch.equal(x, y)


def test_apply_mask_and_invalid_din():
    x = torch.randn(2, 128)
    mx = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    out = apply_input_block_mask(x, mx, 64)
    assert torch.equal(out[0, 0:64], x[0, 0:64])
    assert torch.equal(out[0, 64:128], torch.zeros(64))
    with pytest.raises(ValueError):
        apply_input_block_mask(torch.randn(2, 100), mx, 64)
