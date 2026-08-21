from __future__ import annotations

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.weight_conversion import (
    compute_sub16_dispersion,
    convert_weight_w0,
    split_64_to_sub16_along_k,
)


def test_split_64_to_4x16_along_k():
    w = torch.arange(2 * 128, dtype=torch.float32).reshape(2, 128)
    g = split_64_to_sub16_along_k(w)
    assert g.shape == (2, 2, 4, 16)
    # first group of row0 equals w[0,0:64] split into 4x16
    assert torch.equal(g[0, 0].reshape(-1), w[0, 0:64])


def test_sub16_dispersion_uniform_vs_skewed():
    uniform = torch.ones(4, 16)
    d0 = compute_sub16_dispersion(uniform)
    assert d0.sub16_amax_ratio == 1.0
    skewed = torch.zeros(4, 16)
    skewed[0] = 8.0
    skewed[1] = 1.0
    skewed[2] = 1.0
    skewed[3] = 1.0
    d1 = compute_sub16_dispersion(skewed)
    assert d1.sub16_amax_ratio == 8.0
    assert d1.sub16_energy_share_max > 0.5


def test_convert_weight_w0_shapes():
    w = torch.randn(32, 128, dtype=torch.float32)
    out = convert_weight_w0(w, device="cpu")
    assert out["W_H_FP32"].shape == w.shape
    assert out["W_H_BF16"].dtype == torch.bfloat16
    assert "nmse" in out["E_hif4_format"]
    assert "nmse" in out["E_target_storage"]
