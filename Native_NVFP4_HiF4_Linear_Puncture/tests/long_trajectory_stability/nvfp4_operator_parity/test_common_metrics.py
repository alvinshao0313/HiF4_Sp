from __future__ import annotations

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.nvfp4_operator_parity.common import (
    compare_tensors,
)


def test_compare_tensors_identity_is_zero_error():
    x = torch.randn(4, 8, dtype=torch.float32)
    metrics = compare_tensors(x, x.clone())
    assert metrics["exact_fraction"] == 1.0
    assert metrics["max_abs"] == 0.0
    assert metrics["mean_abs"] == 0.0
    assert metrics["rel_l2"] == 0.0
    assert metrics["sign_flip_fraction"] == 0.0
