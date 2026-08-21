"""HiF4 weight direct / hierarchical scale-search tests."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from Native_NVFP4_HiF4_Linear_Puncture.src import weight_variants as wv_mod
from Native_NVFP4_HiF4_Linear_Puncture.src.weight_variants import (
    build_hif4_direct_weight,
    build_hif4_greedy_weight,
)


def test_hierarchical_scale_search_enables_e8_e4_enumeration(monkeypatch):
    seen = {}

    def fake_search(weight, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(reconstruction=weight.clone())

    monkeypatch.setattr(wv_mod, "search_weight_groups", fake_search)

    w = torch.randn(16, 128, dtype=torch.bfloat16)
    _ = build_hif4_greedy_weight(w)
    assert seen.get("enumerate_e8_e4") is True
    assert seen.get("budget") == "full"


def test_greedy_weight_nmse_never_uses_activation_data():
    sig = inspect.signature(build_hif4_greedy_weight)
    param_names = set(sig.parameters)
    forbidden = {
        "x",
        "x_rot",
        "activation",
        "activations",
        "a_n",
        "a_m",
        "a_h",
        "X",
        "X_rot",
    }
    assert param_names.isdisjoint(forbidden)

    w = torch.randn(8, 64, dtype=torch.bfloat16)
    out = build_hif4_greedy_weight(w)
    assert out.reconstruction.shape == w.shape


def test_direct_and_greedy_shapes_match_source_weight(monkeypatch):
    monkeypatch.setattr(
        wv_mod,
        "search_weight_groups",
        lambda weight, **kwargs: SimpleNamespace(reconstruction=weight.clone()),
    )
    monkeypatch.setattr(
        wv_mod,
        "quantize_hif4",
        lambda values, *, config=None: SimpleNamespace(reconstruction=values.clone()),
    )

    w = torch.randn(12, 192, dtype=torch.bfloat16)
    direct = build_hif4_direct_weight(w)
    greedy = build_hif4_greedy_weight(w)
    assert direct.reconstruction.shape == w.shape
    assert greedy.reconstruction.shape == w.shape


def test_weight_grouping_is_along_k_dimension(monkeypatch):
    seen = {}

    def fake_search(weight, **kwargs):
        seen["shape"] = tuple(weight.shape)
        assert weight.ndim == 2
        assert weight.shape[-1] % 64 == 0
        return SimpleNamespace(reconstruction=weight.clone())

    monkeypatch.setattr(wv_mod, "search_weight_groups", fake_search)

    w = torch.randn(4, 128, dtype=torch.bfloat16)
    _ = build_hif4_greedy_weight(w)
    assert seen["shape"] == (4, 128)
