from __future__ import annotations

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.a2_counterfactual import (
    hif4_activation_counterfactuals,
    nvfp4_activation_counterfactuals,
    search_oracle_global_scale,
)


def test_nvfp4_a2_deterministic():
    torch.manual_seed(0)
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    scale = torch.tensor(32.0)
    a = nvfp4_activation_counterfactuals(x, scale)
    b = nvfp4_activation_counterfactuals(x, scale)
    assert [r["variant"] for r in a["variants"]] == [r["variant"] for r in b["variants"]]
    for ra, rb in zip(a["variants"], b["variants"]):
        assert abs(ra["nmse"] - rb["nmse"]) < 1e-12


def test_hif4_oracle_not_worse_than_full():
    torch.manual_seed(1)
    x = torch.randn(8, 128, dtype=torch.float32)
    out = hif4_activation_counterfactuals(x)
    by = {r["variant"]: r for r in out["variants"]}
    e_full = by["full"]["error_energy"]
    for name in ("oracle_e8", "oracle_e4", "oracle_e8_e4_joint"):
        assert by[name]["error_energy"] <= e_full + 1e-6


def test_oracle_scale_search_runs():
    torch.manual_seed(2)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    scale = torch.tensor(16.0)
    o = search_oracle_global_scale(x, scale)
    assert o["oracle_scale"] > 0
    assert "oracle_scale_search_boundary_hit" in o


def test_output_aware_rcf_present():
    torch.manual_seed(3)
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    w = torch.randn(32, 64)
    scale = torch.tensor(24.0)
    nv = nvfp4_activation_counterfactuals(x, scale, w_n=w)
    assert "R_cf_output" in nv["variants"][0]
    hf = hif4_activation_counterfactuals(x, w_n=w)
    assert "R_cf_output" in hf["variants"][0]
