"""Tests for AX2 group size ablation."""

from __future__ import annotations

import pytest
import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_group_size_ablation import (
    apply_sub16_dispersion,
    run_dispersion_sweep,
    run_group_size_ablation,
    sub16_dispersion_stats,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation


def _tensors():
    torch.manual_seed(1)
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    w = torch.randn(8, 128)
    scale = torch.tensor(24.0)
    a_n = quantize_nvfp4_activation(x, scale).dequantized
    return x, a_n, w


def test_sub16_dispersion_stats_shape():
    g = torch.randn(8, 64)
    stats = sub16_dispersion_stats(g)
    assert stats["sub16_amax"].shape == (8, 4)


def test_apply_sub16_dispersion_preserves_rms():
    x = torch.randn(64, dtype=torch.float32)
    y = apply_sub16_dispersion(x, 1.0)
    r0 = x.pow(2).mean().sqrt()
    r1 = y.float().pow(2).mean().sqrt()
    assert float(r0.item()) == pytest.approx(float(r1.item()), rel=1e-3)


def test_g16_ne_g64_energy():
    x, a_n, w = _tensors()
    rows = run_group_size_ablation(x, a_n, w)
    by_gs = {int(r["group_size"]): r for r in rows}
    assert by_gs[16]["is_standard_hif4"] is False
    assert by_gs[64]["is_standard_hif4"] is True
    assert by_gs[16]["activation_error_energy"] != by_gs[64]["activation_error_energy"]


def test_dispersion_sweep_rows():
    x, a_n, w = _tensors()
    rows = run_dispersion_sweep(x, a_n, w, [0.0, 1.0])
    assert len(rows) == 6  # 3 group sizes × 2 doses
