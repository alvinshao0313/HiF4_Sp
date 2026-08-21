from __future__ import annotations

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.a5_interventions import (
    intervene_single_outlier,
    run_a5_on_groups,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.linear_decomposition import (
    shapley_wa,
    verify_decomposition_identity,
)


def test_a5_rms_preserved_single_outlier():
    torch.manual_seed(0)
    g = torch.randn(64)
    rms0 = float(g.pow(2).mean().sqrt())
    g2 = intervene_single_outlier(g, 4.0)
    rms1 = float(g2.pow(2).mean().sqrt())
    assert abs(rms0 - rms1) / max(rms0, 1e-12) < 1e-5


def test_a5_runs_deterministic():
    torch.manual_seed(1)
    groups = torch.randn(4, 64)
    scale = torch.tensor(32.0)
    a = run_a5_on_groups(groups, scale, max_groups=4, seed=7)
    b = run_a5_on_groups(groups, scale, max_groups=4, seed=7)
    assert len(a) == len(b) and len(a) > 0
    assert a[0]["nmse_h_vs_n"] == b[0]["nmse_h_vs_n"]


def test_shapley_and_fp64_audit():
    torch.manual_seed(2)
    a_n = torch.randn(8, 64)
    a_h = a_n + 0.05 * torch.randn_like(a_n)
    w_n = torch.randn(32, 64)
    w_h = w_n + 0.05 * torch.randn_like(w_n)
    sh = shapley_wa(a_n, a_h, w_n, w_h)
    assert sh["shapley_residual_rel"] < 1e-10
    out = verify_decomposition_identity(a_n, a_h, w_n, w_h)
    assert out["ok"]
    assert out["fp64_audit"]["ok"]
