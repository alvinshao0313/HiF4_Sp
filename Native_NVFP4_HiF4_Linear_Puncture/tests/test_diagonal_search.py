"""Per-channel diagonal search tests (synthetic / CPU)."""

from __future__ import annotations

import inspect

import torch

from Native_NVFP4_HiF4_Linear_Puncture.src.diagonal_search import (
    DiagonalSearchConfig,
    DiagonalSearchResult,
    group_output_error_via_gram,
    search_channelwise_diagonal,
)


def _cfg(**overrides) -> DiagonalSearchConfig:
    base = dict(
        parameterization="log2",
        coarse_log2_offsets=(-1.0, -0.5, 0.0, 0.5, 1.0),
        refine_log2_offsets=(-0.25, 0.0, 0.25),
        num_coarse_sweeps=1,
        num_refine_sweeps=1,
        log2_scale_min=-4.0,
        log2_scale_max=4.0,
        search_token_rows_per_module=32,
        search_output_channels_per_module=16,
        full_calibration_rescore=True,
        eps=1.0e-8,
    )
    base.update(overrides)
    return DiagonalSearchConfig(**base)


def test_diagonal_transform_is_exact_before_quantization():
    torch.manual_seed(0)
    n, k, o = 17, 64, 11
    x = torch.randn(n, k, dtype=torch.float64)
    w = torch.randn(o, k, dtype=torch.float64)
    d = torch.exp(torch.randn(k, dtype=torch.float64) * 0.3)

    y0 = x @ w.T
    y1 = (x / d) @ (w * d).T
    rel = (y0 - y1).pow(2).sum().sqrt() / y0.pow(2).sum().sqrt()
    assert float(rel) < 1e-12


def test_search_api_has_no_alpha_and_no_validation_tensor():
    sig = inspect.signature(search_channelwise_diagonal)
    names = set(sig.parameters)
    assert "alpha" not in names
    assert "x_rot_val" not in names
    assert "a_n_val" not in names
    assert "validation" not in names
    assert "x_val" not in names
    assert {"x_rot_cal", "a_n_cal", "w_n", "config"} <= names


def test_each_channel_has_independent_log2_parameter(monkeypatch):
    import Native_NVFP4_HiF4_Linear_Puncture.src.diagonal_search as ds_mod

    monkeypatch.setattr(ds_mod, "qdq_hif4_direct", lambda t: t, raising=False)

    torch.manual_seed(1)
    k = 128
    x = torch.randn(32, k, dtype=torch.bfloat16)
    w = torch.randn(16, k, dtype=torch.bfloat16)
    result = search_channelwise_diagonal(x, x.clone(), w, _cfg())
    assert isinstance(result, DiagonalSearchResult)
    assert tuple(result.d.shape) == (k,)
    assert tuple(result.log2_d.shape) == (k,)
    # Per-channel parameterization: K independent log2 values, not a scalar alpha.
    assert result.log2_d.numel() == k


def test_group_output_error_gram_matches_explicit_matmul():
    torch.manual_seed(2)
    n, g, o = 40, 64, 25
    a_t = torch.randn(n, g, dtype=torch.float64)
    w_t = torch.randn(o, g, dtype=torch.float64)
    a_r = torch.randn(n, g, dtype=torch.float64)
    w_r = torch.randn(o, g, dtype=torch.float64)

    err_gram = group_output_error_via_gram(a_t, w_t, a_r, w_r)
    err_ref = ((a_t @ w_t.T) - (a_r @ w_r.T)).pow(2).sum()
    rel = abs(float(err_gram) - float(err_ref)) / max(float(err_ref), 1e-30)
    assert rel <= 1e-10


def test_full_calibration_worse_group_rolls_back_to_identity(monkeypatch):
    import Native_NVFP4_HiF4_Linear_Puncture.src.diagonal_search as ds_mod

    torch.manual_seed(3)
    k = 64
    x = torch.randn(20, k, dtype=torch.bfloat16)
    w = torch.randn(8, k, dtype=torch.bfloat16)

    # Any non-identity d makes reconstruction worse under this adversarial qdq.
    def adversarial_qdq(t: torch.Tensor) -> torch.Tensor:
        # Detect scaling away from original magnitude distribution.
        return t * 0.05

    monkeypatch.setattr(ds_mod, "qdq_hif4_direct", adversarial_qdq, raising=False)

    result = search_channelwise_diagonal(
        x,
        x.clone(),
        w,
        _cfg(
            coarse_log2_offsets=(-1.0, 0.0, 1.0),
            refine_log2_offsets=(0.0,),
            num_refine_sweeps=0,
            search_token_rows_per_module=8,
            search_output_channels_per_module=8,
        ),
    )
    assert torch.allclose(result.d.float(), torch.ones(k), atol=1e-6)
    assert bool((~result.group_kept_mask).all())


def test_diagonal_search_returns_finite_positive_scales(monkeypatch):
    import Native_NVFP4_HiF4_Linear_Puncture.src.diagonal_search as ds_mod

    monkeypatch.setattr(ds_mod, "qdq_hif4_direct", lambda t: t, raising=False)
    torch.manual_seed(4)
    k = 64
    x = torch.randn(16, k, dtype=torch.bfloat16)
    w = torch.randn(4, k, dtype=torch.bfloat16)
    result = search_channelwise_diagonal(
        x,
        x.clone(),
        w,
        _cfg(
            coarse_log2_offsets=(-0.5, 0.0, 0.5),
            refine_log2_offsets=(0.0,),
            num_refine_sweeps=0,
            search_token_rows_per_module=16,
            search_output_channels_per_module=4,
        ),
    )
    d = result.d.float()
    assert torch.isfinite(d).all()
    assert bool((d > 0).all())
    assert torch.isfinite(result.log2_d.float()).all()
