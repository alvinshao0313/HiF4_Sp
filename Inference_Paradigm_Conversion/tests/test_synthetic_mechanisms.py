from __future__ import annotations

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.mlp_propagation import (
    product_exact_decomposition,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.synthetic_mechanisms import (
    apply_dispersion_dose,
    equalize_sub16_rms,
)


def test_equalize_preserves_total_rms():
    torch.manual_seed(0)
    g = torch.randn(4, 16)
    g[0] *= 8
    g[3] *= 0.25
    rms0 = torch.sqrt((g * g).mean())
    ge = equalize_sub16_rms(g)
    rms1 = torch.sqrt((ge * ge).mean())
    assert torch.allclose(rms0, rms1, rtol=1e-5, atol=1e-6)
    sub = torch.sqrt((ge * ge).mean(dim=-1))
    # subblock RMS nearly equal
    assert float(sub.max() / sub.min()) < 1.01


def test_dispersion_dose_preserves_total_rms():
    torch.manual_seed(1)
    g = torch.randn(4, 16)
    rms0 = torch.sqrt((g * g).mean())
    gd = apply_dispersion_dose(g, 1.5)
    rms1 = torch.sqrt((gd * gd).mean())
    assert torch.allclose(rms0, rms1, rtol=1e-5, atol=1e-6)


def test_product_decomposition_identity():
    torch.manual_seed(2)
    gn = torch.randn(8, 16)
    gh = gn + 0.1 * torch.randn_like(gn)
    un = torch.randn(8, 16)
    uh = un + 0.1 * torch.randn_like(un)
    d = product_exact_decomposition(gn, gh, un, uh)
    assert d["residual_rel"] < 1e-10
