"""Tests for AX1 S0 divisor search."""

from __future__ import annotations

import pytest
import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_incremental_io import (
    assert_split_isolation,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_s0_divisor_search import (
    candidate_alphas,
    search_output_aware_group_alphas,
    search_s0_divisor_oracle,
)
from Inference_Paradigm_Conversion.ipc_analysis.capture.prompts import build_prompt_bank


def _synthetic_tensors():
    torch.manual_seed(0)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    w = torch.randn(16, 128)
    scale = torch.tensor(32.0)
    from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation

    a_n = quantize_nvfp4_activation(x, scale, output_dtype=torch.bfloat16).dequantized
    return x, a_n, w, scale


def test_candidate_alphas_count():
    alphas = candidate_alphas()
    assert alphas.numel() == 49
    assert float(alphas[0]) == pytest.approx(4.0)
    assert float(alphas[-1]) == pytest.approx(10.0)


def test_search_s0_divisor_oracle_recovery_bounds():
    x, a_n, w, _ = _synthetic_tensors()
    out = search_s0_divisor_oracle(x, a_n, w, alpha_chunk=4)
    assert 4.0 <= out["alpha_oracle_nvfp4"] <= 10.0
    assert -0.5 <= out["output_recovery"] <= 1.5
    assert -0.5 <= out["activation_recovery"] <= 1.5


def test_output_aware_group_alphas():
    x, a_n, w, _ = _synthetic_tensors()
    alphas = candidate_alphas()[:16]
    rows = search_output_aware_group_alphas(x, a_n, w, alphas, top_k=2, random_k=2, energy_k=2)
    assert len(rows) >= 1
    assert "alpha_oracle_output" in rows[0]


def test_split_isolation():
    bank = build_prompt_bank(8)
    discovery = [p for p in bank if p.split == "discovery"]
    assert_split_isolation("discovery", discovery)
    with pytest.raises(ValueError):
        assert_split_isolation("discovery", bank)
