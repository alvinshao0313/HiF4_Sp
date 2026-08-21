from __future__ import annotations

import pytest
import torch

from obs_compensation.hessian import HessianAccumulator, build_obs_system


def test_accumulator_matches_direct_formula():
    d = 3
    x1 = torch.randn(2, 4, d)
    x2 = torch.randn(1, 5, d)
    all_x = torch.cat(
        [x1.reshape(-1, d).float(), x2.reshape(-1, d).float()], dim=0
    )
    expected = 2.0 * all_x.t().matmul(all_x) / all_x.shape[0]
    acc = HessianAccumulator(dimension=d, device=torch.device("cpu"), context="test")
    acc.add_batch(x1)
    acc.add_batch(x2)
    snapshot = acc.finalize()
    torch.testing.assert_close(snapshot.matrix, expected)
    assert abs(snapshot.diagonal_mean - float(torch.diag(expected).mean().item())) < 1e-7
    assert snapshot.matrix.dtype == torch.float32
    assert torch.isfinite(snapshot.matrix).all()
    torch.testing.assert_close(snapshot.matrix, snapshot.matrix.t())
    assert torch.equal(snapshot.dead_columns, torch.diag(expected) == 0)


def test_accumulator_validation():
    acc = HessianAccumulator(2, torch.device("cpu"), "v")
    with pytest.raises(ValueError, match="last dim"):
        acc.add_batch(torch.randn(1, 3))
    with pytest.raises(ValueError, match="rank"):
        acc.add_batch(torch.randn(3))
    with pytest.raises(TypeError, match="floating"):
        acc.add_batch(torch.ones(1, 2, dtype=torch.long))
    with pytest.raises(RuntimeError, match="zero batches"):
        HessianAccumulator(2, torch.device("cpu"), "empty").finalize()
    with pytest.raises(ValueError, match="non-finite"):
        acc.add_batch(torch.tensor([[1.0, float("nan")]]))


def test_build_obs_system_both_directions():
    d = 3
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    acc = HessianAccumulator(d, torch.device("cpu"), "dir")
    acc.add_batch(x.unsqueeze(0))
    snapshot = acc.finalize()
    assert bool(snapshot.dead_columns[2].item())
    original = snapshot.matrix.clone()

    ltr = torch.arange(d, dtype=torch.int64)
    rtl = torch.arange(d - 1, -1, -1, dtype=torch.int64)
    ltr_system = build_obs_system(snapshot, ltr, 0.01, "ltr")
    rtl_system = build_obs_system(snapshot, rtl, 0.01, "rtl")
    torch.testing.assert_close(snapshot.matrix, original)
    assert torch.equal(ltr_system.column_order, ltr)
    assert torch.equal(rtl_system.column_order, rtl)
    inv_check = ltr_system.inverse_column_order.index_select(0, ltr_system.column_order)
    assert torch.equal(inv_check, torch.arange(d, dtype=torch.int64))

    # right-to-left equals factorizing explicitly reversed Hessian with identity order
    rev_snapshot_matrix = snapshot.matrix.index_select(0, rtl).index_select(1, rtl)
    from obs_compensation.hessian import HessianSnapshot

    rev_snap = HessianSnapshot(
        matrix=rev_snapshot_matrix,
        num_tokens=snapshot.num_tokens,
        diagonal_mean=float(torch.diag(rev_snapshot_matrix).mean().item()),
        dead_columns=torch.diag(rev_snapshot_matrix) == 0,
    )
    rtl_via_rev = build_obs_system(rev_snap, torch.arange(d, dtype=torch.int64), 0.01, "rev")
    torch.testing.assert_close(
        rtl_system.upper_inverse_cholesky, rtl_via_rev.upper_inverse_cholesky
    )

    R = ltr_system.upper_inverse_cholesky
    assert torch.allclose(torch.triu(R), R)
    # rebuild damped reordered H and check R.T @ R ~ inv(H)
    order = ltr
    h = snapshot.matrix.index_select(0, order).index_select(1, order).clone()
    diag = torch.arange(d)
    dead = torch.diag(h) == 0
    h[diag[dead], diag[dead]] = 1.0
    damp = 0.01 * float(torch.diag(h).mean().item())
    h[diag, diag] += damp
    approx_inv = R.t().matmul(R)
    torch.testing.assert_close(approx_inv, torch.linalg.inv(h), rtol=1e-4, atol=1e-5)
    assert abs(ltr_system.damp_value - damp) < 1e-8


def test_build_obs_system_rejects_bad_inputs():
    acc = HessianAccumulator(2, torch.device("cpu"), "bad")
    acc.add_batch(torch.eye(2).unsqueeze(0))
    snap = acc.finalize()
    with pytest.raises(ValueError, match="percdamp"):
        build_obs_system(snap, torch.arange(2, dtype=torch.int64), 0.0, "x")
    with pytest.raises(ValueError, match="bijective"):
        build_obs_system(snap, torch.tensor([0, 0], dtype=torch.int64), 0.01, "x")
