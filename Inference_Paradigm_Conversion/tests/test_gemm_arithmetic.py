from __future__ import annotations

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.gemm_arithmetic import (
    gemm_chain_p1,
    gemm_chain_p2,
)


def test_gemm_chains_run():
    torch.manual_seed(0)
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    w = torch.randn(64, 128)
    scale = torch.tensor(32.0)
    p1 = gemm_chain_p1(x, w)
    p2 = gemm_chain_p2(x, w, scale)
    assert p1["path_id"] == "P1_semantic"
    assert p2["path_id"] == "P2_matched_semantic"
    assert "nmse" in p1["output"]
