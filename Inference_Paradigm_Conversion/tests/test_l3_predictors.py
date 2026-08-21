from __future__ import annotations

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.l3_predictors import (
    diagonal_activation_weighted_error,
    empirical_output_error,
)


def test_diagonal_weighted_formula():
    # a: 2 tokens, K=2; dw: O=1,K=2
    a = torch.tensor([[1.0, 0.0], [1.0, 2.0]])  # E[a^2]=[1, 2]
    dw = torch.tensor([[3.0, 4.0]])  # col_sq=[9,16]
    # sum = 1*9 + 2*16 = 41
    assert abs(diagonal_activation_weighted_error(dw, a) - 41.0) < 1e-6


def test_empirical_output_error_positive():
    a = torch.randn(4, 8)
    dw = torch.randn(3, 8)
    assert empirical_output_error(dw, a) > 0
