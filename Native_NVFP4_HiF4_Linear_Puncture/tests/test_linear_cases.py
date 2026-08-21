"""Offline linear case runner tests (mocked quant / synthetic tensors)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from Native_NVFP4_HiF4_Linear_Puncture.src import linear_cases as lc_mod
from Native_NVFP4_HiF4_Linear_Puncture.src.linear_cases import (
    HEADLINE_VARIANT_IDS,
    run_module_linear_cases,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.metrics import (
    aggregate_global_nmse,
    compute_recovery,
)


def _identity_qdq(x, *args, **kwargs):
    return x


def test_all_six_headline_variants_use_same_validation_rows(monkeypatch):
    monkeypatch.setattr(lc_mod, "qdq_nvfp4_post_rotation", _identity_qdq)
    monkeypatch.setattr(lc_mod, "qdq_mxfp8_post_rotation", _identity_qdq)
    monkeypatch.setattr(lc_mod, "qdq_hif4_direct", _identity_qdq)

    def fake_diag_val(x_rot_val, w_n, d, bias=None, **kwargs):
        return F.linear(x_rot_val / d, w_n * d, bias)

    monkeypatch.setattr(lc_mod, "diagonal_validation_output", fake_diag_val)

    torch.manual_seed(0)
    n, k, o = 9, 64, 5
    x_rot_val = torch.randn(n, k, dtype=torch.bfloat16)
    x_rot_cal = torch.randn(n, k, dtype=torch.bfloat16)
    w_n = torch.randn(o, k, dtype=torch.bfloat16)

    out = run_module_linear_cases(
        module_name="synthetic.q_proj",
        x_rot_cal=x_rot_cal,
        x_rot_val=x_rot_val,
        input_global_scale=torch.tensor(1.0),
        w_n=w_n,
        w_h_rtn=w_n.clone(),
        w_h_greedy=w_n.clone(),
        bias=None,
        d=torch.ones(k, dtype=torch.float32),
    )

    assert list(HEADLINE_VARIANT_IDS) == [
        "E1_WN_AM",
        "E2_WH_AM_RTN",
        "E3_WH_AM_GREEDY",
        "E4_WH_AH_RTN",
        "E5_WH_AH_DIAG",
        "E6_WH_AH_GREEDY",
    ]
    for vid in HEADLINE_VARIANT_IDS:
        y = out["headline"][vid]
        assert torch.is_tensor(y)
        assert y.shape[0] == n


def test_baseline_is_wn_an_not_fp(monkeypatch):
    monkeypatch.setattr(lc_mod, "qdq_nvfp4_post_rotation", lambda x, s: x * 0.5)
    monkeypatch.setattr(lc_mod, "qdq_mxfp8_post_rotation", _identity_qdq)
    monkeypatch.setattr(lc_mod, "qdq_hif4_direct", _identity_qdq)
    monkeypatch.setattr(
        lc_mod,
        "diagonal_validation_output",
        lambda x_rot_val, w_n, d, bias=None, **kw: F.linear(x_rot_val, w_n, bias),
    )

    torch.manual_seed(1)
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    w = torch.randn(3, 64, dtype=torch.bfloat16)
    out = run_module_linear_cases(
        module_name="synthetic.k_proj",
        x_rot_cal=x,
        x_rot_val=x,
        input_global_scale=torch.tensor(1.0),
        w_n=w,
        w_h_rtn=w,
        w_h_greedy=w,
        bias=None,
        d=torch.ones(64),
    )
    y_nn = out["Y_NN"]
    y_fp = F.linear(x, w, None)
    y_nn_ref = F.linear(x * 0.5, w, None)
    assert torch.allclose(y_nn.float(), y_nn_ref.float(), rtol=1e-3, atol=1e-3)
    assert not torch.allclose(y_nn.float(), y_fp.float(), rtol=1e-3, atol=1e-3)


def test_recovery_uses_error_energy_against_same_nn_reference():
    e2, e3, e4, e5, e6 = 8.0, 2.0, 10.0, 4.0, 5.0
    assert abs(float(compute_recovery(e2, e3)) - (8 - 2) / 8) < 1e-12
    assert abs(float(compute_recovery(e4, e5)) - (10 - 4) / 10) < 1e-12
    assert abs(float(compute_recovery(e4, e6)) - (10 - 5) / 10) < 1e-12


def test_global_nmse_is_energy_aggregated_not_mean_of_module_nmse():
    err = [1.0, 1.0]
    ref = [1.0, 100.0]
    g = aggregate_global_nmse(err, ref)
    mean_nmse = 0.5 * (1.0 / 1.0 + 1.0 / 100.0)
    assert abs(float(g) - (2.0 / 101.0)) < 1e-12
    assert abs(float(g) - mean_nmse) > 1e-3
