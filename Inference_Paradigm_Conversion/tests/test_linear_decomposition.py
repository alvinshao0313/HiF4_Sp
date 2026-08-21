from __future__ import annotations

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.linear_decomposition import (
    verify_decomposition_identity,
)


def test_linear_decomposition_identity():
    torch.manual_seed(0)
    a_n = torch.randn(16, 64)
    a_h = a_n + 0.05 * torch.randn_like(a_n)
    w_n = torch.randn(32, 64)
    w_h = w_n + 0.05 * torch.randn_like(w_n)
    out = verify_decomposition_identity(a_n, a_h, w_n, w_h, atol=1e-10)
    assert out["ok"], out
