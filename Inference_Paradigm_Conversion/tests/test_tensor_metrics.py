from __future__ import annotations

import math

import torch

from Inference_Paradigm_Conversion.ipc_analysis.metrics.tensor_metrics import (
    compute_group_metrics,
    compute_pair_metrics,
)


def test_pair_metrics_hand_vector():
    ref = torch.tensor([1.0, 0.0, -1.0, 2.0])
    tgt = torch.tensor([1.0, 1.0, -1.0, 2.0])
    m = compute_pair_metrics(ref, tgt)
    # error = [0,1,0,0], err_energy=1, ref_energy=1+0+1+4=6
    assert math.isclose(m["error_energy"], 1.0)
    assert math.isclose(m["reference_energy"], 6.0)
    assert math.isclose(m["nmse"], 1.0 / 6.0)
    assert math.isclose(m["mae"], 0.25)


def test_zero_reference_energy_defined():
    ref = torch.zeros(4)
    tgt = torch.ones(4)
    m = compute_pair_metrics(ref, tgt)
    assert m["reference_energy"] == 0.0
    assert m["nmse"] == 1.0e300  # +inf sentinel
    assert m["error_energy"] == 4.0


def test_group_metrics_along_k():
    ref = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    tgt = ref + 1
    groups = compute_group_metrics(ref, tgt, group_size=4, group_dim=-1)
    assert len(groups) == 4
