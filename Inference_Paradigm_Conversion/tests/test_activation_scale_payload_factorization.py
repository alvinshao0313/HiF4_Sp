"""Tests for AX4 scale/payload factorization."""

from __future__ import annotations

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_scale_payload_factorization import (
    run_cross_format_factorization,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation


def test_hybrids_marked_invalid():
    torch.manual_seed(3)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    scale = torch.tensor(20.0)
    a_n = quantize_nvfp4_activation(x, scale).dequantized
    w = torch.randn(8, 128)
    rows = run_cross_format_factorization(x, a_n, w, scale)
    hybrids = [r for r in rows if r["hybrid"] in {"HN", "NH"}]
    assert len(hybrids) == 4
    assert all(r["is_valid_hardware_format"] is False for r in hybrids)
    assert all(r["purpose"] == "mechanism_probe" for r in hybrids)


def test_recovery_bounds():
    torch.manual_seed(4)
    x = torch.randn(1, 64, dtype=torch.bfloat16)
    scale = torch.tensor(8.0)
    a_n = quantize_nvfp4_activation(x, scale).dequantized
    w = torch.randn(4, 64)
    rows = run_cross_format_factorization(x, a_n, w, scale)
    for r in rows:
        if r["is_valid_hardware_format"]:
            assert -0.5 <= r["R_Y"] <= 1.5
            assert -0.5 <= r["R_A"] <= 1.5
