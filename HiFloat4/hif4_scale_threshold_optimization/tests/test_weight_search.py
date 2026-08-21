"""Correctness tests for vectorized weight search."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_HIFLOAT4 = _ROOT.parent
for p in (_ROOT, _HIFLOAT4):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from src.metrics import nmse  # noqa: E402
from src.quantizer import HiF4QuantConfig, quantize_hif4  # noqa: E402
from src.weight_search import (  # noqa: E402
    brute_force_group_search_reference,
    search_weight_groups,
    standard_rtn_quantize,
)


def _rand_weight(rows: int, cols: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, cols, generator=g, dtype=torch.float32)
    flat = x.view(-1)
    flat[::128] *= 15.0
    return x


def test_vectorized_matches_bruteforce_small():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    w = _rand_weight(4, 128, seed=11).cuda()
    # Compare each 64-group
    groups = w.reshape(-1, 64)
    vec = search_weight_groups(w, budget="fast", group_chunk_size=32, device="cuda")
    vec_groups = vec.reconstruction.reshape(-1, 64)
    for i in range(groups.shape[0]):
        ref_recon, ref_err, ref_s0, ref_e8, ref_e4 = brute_force_group_search_reference(
            groups[i].cpu()
        )
        # MSE of vectorized group should match reference within tight tol
        v = vec_groups[i].cpu()
        err_v = float(((groups[i].cpu() - v) ** 2).sum().item())
        err_r = float(((groups[i].cpu() - ref_recon) ** 2).sum().item())
        assert abs(err_v - err_r) <= 1e-8 * max(1.0, err_r), (
            f"group {i}: vec_err={err_v} ref_err={err_r}"
        )
        assert torch.allclose(v, ref_recon, rtol=0, atol=1e-6)


def test_search_mse_not_worse_than_standard():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    w = _rand_weight(32, 512, seed=42).cuda()
    std = standard_rtn_quantize(w)
    searched = search_weight_groups(w, budget="fast", device="cuda")
    assert searched.nmse <= nmse(w, std) + 1e-12


def test_full_not_worse_than_fast():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    w = _rand_weight(16, 256, seed=7).cuda()
    fast = search_weight_groups(w, budget="fast", device="cuda")
    full = search_weight_groups(w, budget="full", device="cuda")
    assert full.nmse <= fast.nmse + 1e-12


def test_s0_only_mode_runs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    w = _rand_weight(8, 256, seed=3).cuda()
    out = search_weight_groups(w, budget="fast", enumerate_e8_e4=False, device="cuda")
    assert out.reconstruction.shape == w.shape
    assert out.nmse >= 0.0
