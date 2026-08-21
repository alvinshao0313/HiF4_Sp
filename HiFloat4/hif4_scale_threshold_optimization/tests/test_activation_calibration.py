"""Smoke tests for activation calibration path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.activation_calibration import calibrate_activations  # noqa: E402
from src.model_hooks import HiF4ActQuantLinear  # noqa: E402
from src.quantizer import HiF4QuantConfig  # noqa: E402


def test_act_linear_no_search_and_shapes():
    lin = nn.Linear(128, 64, bias=False)
    cfg = HiF4QuantConfig(s0_divisor=7.0, e8_threshold=3.9, e4_threshold=1.95)
    wrapped = HiF4ActQuantLinear(lin, cfg)
    x = torch.randn(2, 8, 128)
    y = wrapped(x)
    assert y.shape == (2, 8, 64)


def test_calibrate_tiny():
    if not torch.cuda.is_available():
        return
    inputs = {"layer.q_proj": torch.randn(64, 128)}
    energy = {"layer.q_proj": torch.ones(128)}
    out = calibrate_activations(
        inputs, energy, granularity="per_layer", device="cuda", val_fraction=0.25
    )
    assert "layer.q_proj" in out["param_map"]
    assert out["summary"]["layers"]["layer.q_proj"]["val_output_mse"] >= 0.0
