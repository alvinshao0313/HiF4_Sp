from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.input_mask_proxy_study.energy_recovery import (  # noqa: E402
    recover_input_masks_energy,
    recover_input_masks_energy_unconditioned,
)


def test_vectorized_matches_triple_loop():
    torch.manual_seed(0)
    a, jb, kb = 2, 3, 4
    x_blocks = torch.randn(a, kb, 32, 64)
    w_blocks = torch.randn(jb, kb, 32, 64)
    weight_energy = w_blocks.square().mean(dim=(-1, -2))
    output_mask = torch.tensor(
        [[True, False, True], [False, True, True]], dtype=torch.bool
    )
    result = recover_input_masks_energy(x_blocks, weight_energy, output_mask, (2,))
    scores = torch.zeros(a, kb)
    for i in range(a):
        for k in range(kb):
            ex = x_blocks[i, k].square().mean()
            acc = 0.0
            for j in range(jb):
                if bool(output_mask[i, j]):
                    acc += float(weight_energy[j, k])
            scores[i, k] = ex * acc
    assert torch.allclose(result.scores, scores, atol=1e-6)


def test_mean_square_and_keep_and_nested():
    torch.manual_seed(1)
    a, jb, kb = 1, 2, 6
    x_blocks = torch.randn(a, kb, 32, 64)
    w_blocks = torch.randn(jb, kb, 32, 64)
    weight_energy = w_blocks.square().mean(dim=(-1, -2))
    output_mask = torch.ones(a, jb, dtype=torch.bool)
    result = recover_input_masks_energy(x_blocks, weight_energy, output_mask, (2, 4, 5))
    assert torch.all(result.masks_by_keep[2].sum(-1) == 2)
    assert torch.all(result.masks_by_keep[2] <= result.masks_by_keep[4])
    assert torch.all(result.masks_by_keep[4] <= result.masks_by_keep[5])


def test_stable_tie():
    # Equal scores -> smaller indices kept
    a, jb, kb = 1, 1, 4
    x_blocks = torch.ones(a, kb, 32, 64)
    w_blocks = torch.ones(jb, kb, 32, 64)
    weight_energy = w_blocks.square().mean(dim=(-1, -2))
    output_mask = torch.ones(a, jb, dtype=torch.bool)
    result = recover_input_masks_energy(x_blocks, weight_energy, output_mask, (2,))
    assert torch.equal(result.masks_by_keep[2][0], torch.tensor([True, True, False, False]))


def test_bad_output_mask_shape():
    x_blocks = torch.randn(1, 2, 32, 64)
    weight_energy = torch.randn(2, 2)
    output_mask = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError):
        recover_input_masks_energy(x_blocks, weight_energy, output_mask, (1,))


def test_signature_consumes_weight_energy_not_w_blocks():
    sig = inspect.signature(recover_input_masks_energy)
    params = list(sig.parameters)
    assert params[:4] == [
        "activation_blocks",
        "weight_energy",
        "output_mask",
        "keep_counts",
    ]
    assert "w_blocks" not in params


def test_unconditioned_exact_score_and_stable_tie():
    a, kb = 2, 3
    activation_blocks = torch.zeros(a, kb, 32, 64)
    activation_blocks[:, 0].fill_(1.0)  # E_X=1
    activation_blocks[:, 1].fill_(2.0)  # E_X=4
    activation_blocks[:, 2].fill_(3.0)  # E_X=9
    all_w = torch.tensor([9.0, 2.0, 1.0])
    result = recover_input_masks_energy_unconditioned(
        activation_blocks, all_w, (2,)
    )
    expected = torch.tensor([[9.0, 8.0, 9.0], [9.0, 8.0, 9.0]])
    assert torch.allclose(result.scores, expected, atol=1e-6)
    # ties 9,9 at k0/k2 -> smaller index first: [0,2,1]
    assert torch.equal(result.ranking[0], torch.tensor([0, 2, 1]))
    assert torch.equal(
        result.masks_by_keep[2][0], torch.tensor([True, False, True])
    )


def test_unconditioned_nested_and_validations():
    torch.manual_seed(0)
    x = torch.randn(1, 6, 32, 64)
    all_w = torch.randn(6).abs() + 0.1
    result = recover_input_masks_energy_unconditioned(x, all_w, (2, 4, 5))
    assert torch.all(result.masks_by_keep[2].sum(-1) == 2)
    assert torch.all(result.masks_by_keep[2] <= result.masks_by_keep[4])
    assert torch.all(result.masks_by_keep[4] <= result.masks_by_keep[5])

    with pytest.raises(ValueError):
        recover_input_masks_energy_unconditioned(torch.randn(2, 3), all_w, (1,))
    with pytest.raises(ValueError):
        recover_input_masks_energy_unconditioned(x, torch.randn(1, 6), (1,))
    with pytest.raises(ValueError):
        recover_input_masks_energy_unconditioned(x, torch.randn(5), (1,))
    bad = x.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        recover_input_masks_energy_unconditioned(bad, all_w, (1,))
    bad_w = all_w.clone()
    bad_w[0] = float("inf")
    with pytest.raises(ValueError):
        recover_input_masks_energy_unconditioned(x, bad_w, (1,))
    with pytest.raises(ValueError):
        recover_input_masks_energy_unconditioned(x, all_w, ())
    with pytest.raises(ValueError):
        recover_input_masks_energy_unconditioned(x, all_w, (0,))
    with pytest.raises(ValueError):
        recover_input_masks_energy_unconditioned(x, all_w, (7,))


def test_unconditioned_signature_has_no_output_mask():
    sig = inspect.signature(recover_input_masks_energy_unconditioned)
    params = list(sig.parameters)
    assert params == [
        "activation_blocks",
        "all_output_weight_energy",
        "keep_counts",
    ]
    assert "output_mask" not in params
