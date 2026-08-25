from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.input_mask_proxy_study.s0mean_recovery import (  # noqa: E402
    recover_input_masks_s0mean_energy,
)


def test_s0_block_reduction_exact():
    t, kb = 64, 3
    activation_s0 = torch.arange(t * kb).reshape(t, kb).float()
    weight_energy = torch.ones(1, kb)
    output_mask = torch.ones(2, 1, dtype=torch.bool)
    result = recover_input_masks_s0mean_energy(
        activation_s0, 32, weight_energy, output_mask, (1,)
    )
    reference = activation_s0.reshape(2, 32, 3).mean(dim=1)
    assert torch.equal(result.activation_statistic, reference)


def test_score_matches_explicit_loop():
    torch.manual_seed(0)
    a, jb, kb = 2, 3, 4
    activation_s0 = torch.randn(a * 32, kb)
    weight_energy = torch.randn(jb, kb).abs() + 0.1
    output_mask = torch.tensor(
        [[True, False, True], [False, True, True]], dtype=torch.bool
    )
    result = recover_input_masks_s0mean_energy(
        activation_s0, 32, weight_energy, output_mask, (2,)
    )
    scores = torch.zeros(a, kb)
    for i in range(a):
        for k in range(kb):
            a_stat = activation_s0[i * 32 : (i + 1) * 32, k].mean()
            b = sum(
                float(weight_energy[j, k]) for j in range(jb) if bool(output_mask[i, j])
            )
            scores[i, k] = a_stat * b
    assert torch.allclose(result.scores, scores, atol=1e-6)


def test_stable_tie():
    a, jb, kb = 1, 1, 4
    activation_s0 = torch.ones(32, kb)
    weight_energy = torch.ones(jb, kb)
    output_mask = torch.ones(a, jb, dtype=torch.bool)
    result = recover_input_masks_s0mean_energy(
        activation_s0, 32, weight_energy, output_mask, (2,)
    )
    assert torch.equal(
        result.masks_by_keep[2][0], torch.tensor([True, True, False, False])
    )


def test_keep_count_and_nested():
    torch.manual_seed(1)
    activation_s0 = torch.randn(32, 6)
    weight_energy = torch.randn(2, 6).abs() + 0.1
    output_mask = torch.ones(1, 2, dtype=torch.bool)
    result = recover_input_masks_s0mean_energy(
        activation_s0, 32, weight_energy, output_mask, (2, 4, 5)
    )
    assert torch.all(result.masks_by_keep[2].sum(-1) == 2)
    assert torch.all(result.masks_by_keep[2] <= result.masks_by_keep[4])
    assert torch.all(result.masks_by_keep[4] <= result.masks_by_keep[5])


def test_interface_forbids_full_activation():
    sig = inspect.signature(recover_input_masks_s0mean_energy)
    params = set(sig.parameters)
    for forbidden in ("x", "xp", "activation_blocks", "w_blocks"):
        assert forbidden not in params
