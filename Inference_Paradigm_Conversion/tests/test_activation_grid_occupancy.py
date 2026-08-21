"""Tests for AX3 grid occupancy and full theoretical representable sets."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_grid_occupancy import (
    analyze_grid_occupancy_row,
    build_theoretical_grid_json,
    enumerate_nvfp4_e4m3_scale_values,
    enumerate_hif4_full_internal_grid,
    enumerate_hif4_payload_grid,
    enumerate_hif4_s0_values,
    enumerate_nvfp4_full_internal_grid,
    enumerate_nvfp4_payload_grid,
    theoretical_grid_stats,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import quantize_nvfp4_activation
from Inference_Paradigm_Conversion.ipc_analysis.reporting.activation_incremental_report import (
    _fig_theory_grid,
)
from ChuanCi.nvfp4_hif4_torch import E2M1_VALUES, round_e6m2


def test_grids_nonempty():
    nv = enumerate_nvfp4_payload_grid()
    hf = enumerate_hif4_payload_grid()
    assert nv.numel() > 0
    assert hf.numel() > 0
    assert float(hf.max()) <= 1.75 + 1e-6


def test_nvfp4_full_grid_not_payload():
    nv_payload = enumerate_nvfp4_payload_grid()
    nv_full, meta = enumerate_nvfp4_full_internal_grid()
    assert nv_full.numel() > nv_payload.numel()
    assert nv_full.numel() > 16
    assert bool((nv_full == 0).any())
    assert torch.isfinite(nv_full).all()
    assert int(torch.unique(nv_full).numel()) == int(nv_full.numel())
    # sorted unique
    assert torch.all(nv_full[1:] >= nv_full[:-1])
    # sign symmetry for nonzero
    pos = nv_full[nv_full > 0]
    neg = nv_full[nv_full < 0]
    assert pos.numel() == neg.numel()
    assert torch.allclose(pos, (-neg).flip(0), rtol=0, atol=0)
    scales = enumerate_nvfp4_e4m3_scale_values()
    assert meta["scale_format"]["num_scale_values"] == int(scales.numel())


def test_hif4_full_grid_not_payload():
    hf_payload = enumerate_hif4_payload_grid()
    hf_full, meta = enumerate_hif4_full_internal_grid()
    assert hf_full.numel() > hf_payload.numel()
    assert hf_full.numel() > 16
    assert bool((hf_full == 0).any())
    assert torch.isfinite(hf_full).all()
    assert int(torch.unique(hf_full).numel()) == int(hf_full.numel())
    assert torch.all(hf_full[1:] >= hf_full[:-1])
    pos = hf_full[hf_full > 0]
    neg = hf_full[hf_full < 0]
    assert pos.numel() == neg.numel()
    assert torch.allclose(pos, (-neg).flip(0), rtol=0, atol=0)
    # S0 matches round_e6m2 codebook
    s0 = enumerate_hif4_s0_values()
    assert torch.equal(s0, round_e6m2(s0))
    assert meta["s0_format"]["num_scale_values"] == int(s0.numel())


def test_combination_spot_check():
    nv_full, _ = enumerate_nvfp4_full_internal_grid()
    scales = enumerate_nvfp4_e4m3_scale_values()
    # E4M3FN includes an exact 1.0 scale value.
    s = scales[(scales - 1.0).abs().argmin()]
    for p in (0.5, 2.0, 6.0, -1.5):
        expected = float(s.item() * p)
        assert bool(torch.any(torch.isclose(nv_full, torch.tensor(expected), rtol=0, atol=0)))

    hf_full, _ = enumerate_hif4_full_internal_grid()
    s0 = enumerate_hif4_s0_values()
    s0_pick = s0[(s0 - 1.0).abs().argmin()]
    for e8, e4, p in [(0, 0, 1.0), (1, 0, 0.5), (1, 1, 1.75), (0, 1, -1.25)]:
        expected = float(s0_pick.item() * (2 ** (e8 + e4)) * p)
        assert bool(torch.any(torch.isclose(hf_full, torch.tensor(expected), rtol=0, atol=0)))


def test_full_grid_no_global_scale_args():
    for fn in (enumerate_nvfp4_full_internal_grid, enumerate_hif4_full_internal_grid, enumerate_nvfp4_e4m3_scale_values, enumerate_hif4_s0_values):
        params = inspect.signature(fn).parameters
        forbidden = {"input_global_scale", "global_scale", "tensor_scale"}
        assert forbidden.isdisjoint(params.keys()), fn.__name__


def test_full_grid_dedup():
    nv_full, _ = enumerate_nvfp4_full_internal_grid()
    hf_full, _ = enumerate_hif4_full_internal_grid()
    assert torch.unique(nv_full).numel() == nv_full.numel()
    assert torch.unique(hf_full).numel() == hf_full.numel()


def test_plotting_smoke_full_grids():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        theory = build_theoretical_grid_json(td_path)
        fig_dir = td_path / "figures"
        fig_dir.mkdir()
        paths = _fig_theory_grid(theory, fig_dir)
        required = [
            fig_dir / "fig_ax3_full_internal_grid_hist.png",
            fig_dir / "fig_ax3_full_internal_grid_log2_hist.png",
            fig_dir / "fig_ax3_full_grid_density_near_zero.png",
        ]
        for p in required:
            assert p.is_file() and p.stat().st_size > 0, p
        assert any(Path(x).name == "fig_ax3_payload_codebook.png" for x in paths)


def test_theoretical_grid_json():
    j = build_theoretical_grid_json()
    assert "nvfp4_stats" in j
    assert j["nvfp4_stats"]["num_positive_levels"] > 0
    assert j["nvfp4_full_stats"]["num_unique_values"] > j["nvfp4_stats"]["num_positive_levels"]
    assert j["hif4_full_stats"]["num_unique_values"] > j["hif4_stats"]["num_positive_levels"]


def test_analyze_grid_occupancy_row():
    torch.manual_seed(2)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    scale = torch.tensor(16.0)
    nv = quantize_nvfp4_activation(x, scale, collect_metadata=True)
    hf = quantize_hif4_tensor(x.float(), variant="full", output_dtype=torch.bfloat16)
    w = torch.randn(4, 64)
    row = analyze_grid_occupancy_row(
        x, nv.dequantized, hf.dequantized, w, nv.metadata, hf.metadata, alpha_oracle=6.5
    )
    assert "nv_occ_zero_rate" in row
    assert "conversion_output_error" in row


def test_theoretical_stats():
    stats = theoretical_grid_stats(enumerate_hif4_payload_grid())
    assert stats["num_positive_levels"] > 0


def test_e2m1_magnitudes_match_codebook():
    assert torch.allclose(enumerate_nvfp4_payload_grid(), E2M1_VALUES)
