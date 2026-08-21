from __future__ import annotations

from Inference_Paradigm_Conversion.ipc_analysis.analysis.synthetic_s_suite import (
    run_s1_dispersion,
    run_s5_wa_angle_sweep,
    run_s6_mlp_product,
)
from Inference_Paradigm_Conversion.ipc_analysis.metrics.statistics import mean_ci


def test_mean_ci():
    out = mean_ci([1.0, 2.0, 3.0])
    assert abs(out["mean"] - 2.0) < 1e-9
    assert out["ci_low"] < out["mean"] < out["ci_high"]


def test_s1_runs():
    out = run_s1_dispersion(seeds=2)
    assert "curve_output_error_energy" in out
    assert len(out["rows"]) == 2 * 5


def test_s5_angle_varies():
    out = run_s5_wa_angle_sweep(seeds=2)
    assert "0" in out["mean_total_error_by_angle"]


def test_s6_cases():
    out = run_s6_mlp_product(seeds=2)
    assert "both_same" in out["mean_silu_nmse"]
