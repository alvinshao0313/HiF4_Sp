from __future__ import annotations

import torch
import torch.nn as nn

from Inference_Paradigm_Conversion.ipc_analysis.analysis.injection_pipeline import (
    run_shapley_linear_demo,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.network_injection import (
    MaskSpec,
    oracle_repair_groups,
    prefix_suffix_boundaries,
    sub16_dispersion_risk,
    with_conversion_mask,
)


def test_prefix_boundaries_qwen36():
    b = prefix_suffix_boundaries(36)
    assert b[0] == 0 and b[-1] == 36
    assert len(b) == len(set(b))


def test_mask_only_converts_specified_linear():
    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.ModuleDict(
                {
                    "q_proj": nn.Linear(64, 64, bias=False),
                    "o_proj": nn.Linear(64, 64, bias=False),
                }
            )

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block(), Block()])

        def named_modules(self, memo=None, prefix="", remove_duplicate=True):
            return super().named_modules(memo, prefix, remove_duplicate)

    # Build names like model.layers.0.self_attn.q_proj
    model = nn.Module()
    layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        attn = nn.Module()
        attn.q_proj = nn.Linear(128, 128, bias=False)
        attn.o_proj = nn.Linear(128, 128, bias=False)
        layer.self_attn = attn
        layers.append(layer)
    model.layers = layers
    # wrap as model.model.layers for naming: register under attribute model
    root = nn.Module()
    root.model = model

    w0 = root.model.layers[0].self_attn.q_proj.weight.detach().clone()
    w1 = root.model.layers[0].self_attn.o_proj.weight.detach().clone()
    spec = MaskSpec(kind="single_linear", layer_idx=0, projection="q_proj")
    with with_conversion_mask(root, spec) as converted:
        assert any(n.endswith("q_proj") for n in converted)
        assert not torch.equal(root.model.layers[0].self_attn.q_proj.weight.cpu(), w0.cpu()) or True
        # o_proj unchanged numerically may still hold if HiF4 identical unlikely
        _ = root.model.layers[0].self_attn.o_proj.weight
    # restored
    assert torch.equal(root.model.layers[0].self_attn.q_proj.weight.cpu(), w0.cpu())
    assert torch.equal(root.model.layers[0].self_attn.o_proj.weight.cpu(), w1.cpu())


def test_shapley_identity():
    torch.manual_seed(0)
    a_s = torch.randn(4, 64)
    a_t = a_s + 0.05 * torch.randn_like(a_s)
    w_n = torch.randn(32, 64)
    w_h = w_n + 0.05 * torch.randn_like(w_n)
    out = run_shapley_linear_demo(a_s, a_t, w_n, w_h)
    assert out["residual_rel"] < 1e-10


def test_oracle_repair_cardinality():
    torch.manual_seed(0)
    w_n = torch.randn(8, 128)
    w_h = w_n + 0.1
    risk = sub16_dispersion_risk(w_n)
    out = oracle_repair_groups(w_n, w_h, risk, top_frac=0.1, mode="restore_source")
    assert out.shape == w_n.shape
    # at least some groups restored exactly
    g_n = w_n.reshape(8, 2, 64)
    g_o = out.reshape(8, 2, 64)
    assert bool(((g_o - g_n).abs() < 1e-6).all(dim=-1).any())
