from __future__ import annotations

import pytest
import torch

from block_pruning.block_utils import expand_block_mask
from obs_compensation.hessian import HessianAccumulator, build_obs_system
from obs_compensation.solver import (
    build_directional_column_order,
    resolve_obs_order_policy,
    solve_fixed_mask_obs,
)


@pytest.mark.parametrize(
    "requested,mlp_permutation,resolved,gate_up,down",
    [
        ("auto", "none", "standard", "left_to_right", "left_to_right"),
        ("standard", "none", "standard", "left_to_right", "left_to_right"),
        ("auto", "wanda_shared", "permutation_aware", "left_to_right", "right_to_left"),
        ("standard", "wanda_shared", "standard", "left_to_right", "left_to_right"),
        ("permutation_aware", "wanda_shared", "permutation_aware", "left_to_right", "right_to_left"),
    ],
)
def test_resolve_obs_order_policy(
    requested, mlp_permutation, resolved, gate_up, down
):
    policy = resolve_obs_order_policy(requested, mlp_permutation)
    assert policy.requested_policy == requested
    assert policy.resolved_policy == resolved
    assert policy.gate_up_direction == gate_up
    assert policy.down_direction == down


def test_resolve_rejects_unsorted_aware_and_unknown():
    with pytest.raises(ValueError, match="requires mlp_permutation=wanda_shared"):
        resolve_obs_order_policy("permutation_aware", "none")
    with pytest.raises(ValueError, match="Unsupported obs_order_policy"):
        resolve_obs_order_policy("mask_count", "none")
    with pytest.raises(ValueError, match="Unsupported source mlp_permutation"):
        resolve_obs_order_policy("auto", "weird")


def test_directional_orders():
    assert build_directional_column_order(4, "left_to_right").tolist() == [0, 1, 2, 3]
    assert build_directional_column_order(4, "right_to_left").tolist() == [3, 2, 1, 0]
    order = build_directional_column_order(3, "left_to_right")
    assert order.dtype == torch.int64
    assert order.device.type == "cpu"
    with pytest.raises(ValueError):
        build_directional_column_order(0, "left_to_right")
    with pytest.raises(ValueError):
        build_directional_column_order(2, "middle_out")


def test_no_pruning_identity_both_directions():
    weight = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    block_mask = torch.ones((2, 2), dtype=torch.bool)
    acc = HessianAccumulator(2, torch.device("cpu"), "identity")
    acc.add_batch(torch.eye(2).unsqueeze(0))
    snapshot = acc.finalize()
    for direction in ("left_to_right", "right_to_left"):
        order = build_directional_column_order(2, direction)
        system = build_obs_system(snapshot, order, 0.01, direction)
        result = solve_fixed_mask_obs(
            weight=weight,
            block_keep_mask=block_mask,
            block_height=1,
            block_width=1,
            system=system,
            solver_block_size=2,
            context=direction,
        )
        torch.testing.assert_close(
            result.compensated_weight, weight.float(), rtol=0, atol=0
        )
        assert result.kept_delta_l2 == 0.0
        assert result.original_pruned_l2 == 0.0


def test_exact_zero_enforcement():
    weight = torch.randn(4, 4)
    block_mask = torch.ones(2, 2, dtype=torch.bool)
    block_mask[0, 1] = False
    acc = HessianAccumulator(4, torch.device("cpu"), "zero")
    acc.add_batch(torch.randn(8, 4))
    snapshot = acc.finalize()
    order = build_directional_column_order(4, "left_to_right")
    system = build_obs_system(snapshot, order, 0.01, "zero")
    result = solve_fixed_mask_obs(
        weight, block_mask, 2, 2, system, 2, "zero"
    )
    element_mask = expand_block_mask(block_mask, 2, 2)
    assert torch.count_nonzero(result.compensated_weight[~element_mask]) == 0
    assert torch.isfinite(result.compensated_weight[element_mask]).all()
    assert result.compensated_weight.shape == weight.shape


def test_diagonal_hessian_both_directions():
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    block_mask = torch.tensor([[True, False], [True, True]])
    element_mask = expand_block_mask(block_mask, 1, 1)
    acc = HessianAccumulator(2, torch.device("cpu"), "diag")
    acc.add_batch(torch.eye(2).unsqueeze(0))
    snapshot = acc.finalize()
    expected = weight.float() * element_mask.float()
    for direction in ("left_to_right", "right_to_left"):
        order = build_directional_column_order(2, direction)
        system = build_obs_system(snapshot, order, 0.01, direction)
        result = solve_fixed_mask_obs(
            weight, block_mask, 1, 1, system, 2, direction
        )
        torch.testing.assert_close(result.compensated_weight, expected)


def test_sorted_down_direction_advantage():
    x = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.1],
            [-1.0, -0.9],
            [0.5, 0.6],
        ]
    )
    weight = torch.tensor([[1.0, 1.0]])
    block_mask = torch.tensor([[True, False]])
    acc = HessianAccumulator(2, torch.device("cpu"), "sorted_down_toy")
    acc.add_batch(x.unsqueeze(0))
    snapshot = acc.finalize()
    ltr_order = build_directional_column_order(2, "left_to_right")
    rtl_order = build_directional_column_order(2, "right_to_left")
    ltr_system = build_obs_system(snapshot, ltr_order, 0.01, "ltr")
    rtl_system = build_obs_system(snapshot, rtl_order, 0.01, "rtl")
    ltr = solve_fixed_mask_obs(weight, block_mask, 1, 1, ltr_system, 2, "ltr")
    rtl = solve_fixed_mask_obs(weight, block_mask, 1, 1, rtl_system, 2, "rtl")
    dense_y = x @ weight.t()
    direct_zero_y = x @ torch.tensor([[1.0, 0.0]]).t()
    ltr_y = x @ ltr.compensated_weight.t()
    rtl_y = x @ rtl.compensated_weight.t()
    direct_zero_mse = torch.mean((direct_zero_y - dense_y) ** 2)
    ltr_mse = torch.mean((ltr_y - dense_y) ** 2)
    rtl_mse = torch.mean((rtl_y - dense_y) ** 2)
    torch.testing.assert_close(ltr_mse, direct_zero_mse, rtol=1e-6, atol=1e-7)
    assert rtl_mse < ltr_mse
    assert rtl_mse < direct_zero_mse
    assert rtl.compensated_weight[0, 1].item() == 0.0
    assert rtl.compensated_weight[0, 0].item() != weight[0, 0].item()


def test_column_coordinate_restoration():
    weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    block_mask = torch.tensor([[True, False, True, True], [True, True, False, True]])
    acc = HessianAccumulator(4, torch.device("cpu"), "restore")
    acc.add_batch(torch.randn(16, 4))
    snapshot = acc.finalize()
    order = torch.tensor([3, 2, 1, 0], dtype=torch.int64)
    system = build_obs_system(snapshot, order, 0.01, "restore")
    result = solve_fixed_mask_obs(weight, block_mask, 1, 1, system, 2, "restore")
    element_mask = expand_block_mask(block_mask, 1, 1)
    assert torch.count_nonzero(result.compensated_weight[~element_mask]) == 0
    assert torch.equal(
        system.inverse_column_order.index_select(0, system.column_order),
        torch.arange(4, dtype=torch.int64),
    )
    # reversing twice style: solve with reverse then reverse again equals ltr solve layout coords
    order2 = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    system2 = build_obs_system(snapshot, order2, 0.01, "ltr")
    result2 = solve_fixed_mask_obs(weight, block_mask, 1, 1, system2, 2, "ltr")
    assert result.compensated_weight.shape == result2.compensated_weight.shape


def test_solver_accepts_fully_pruned_block_row():
    torch.manual_seed(7)
    weight = torch.randn(4, 4)
    block_mask = torch.tensor(
        [
            [False, False],
            [True, False],
        ],
        dtype=torch.bool,
    )
    acc = HessianAccumulator(4, torch.device("cpu"), "fully_pruned_row")
    acc.add_batch(torch.randn(32, 4))
    snapshot = acc.finalize()
    order = build_directional_column_order(4, "left_to_right")
    system = build_obs_system(snapshot, order, 0.01, "fully_pruned_row")

    result = solve_fixed_mask_obs(
        weight=weight,
        block_keep_mask=block_mask,
        block_height=2,
        block_width=2,
        system=system,
        solver_block_size=2,
        context="fully_pruned_row",
    )

    element_mask = expand_block_mask(block_mask, 2, 2)
    assert torch.count_nonzero(result.compensated_weight[:2]) == 0
    assert torch.count_nonzero(result.compensated_weight[~element_mask]) == 0
    assert torch.isfinite(result.compensated_weight).all()
    assert torch.count_nonzero(result.compensated_weight[2:, :2]) > 0


def test_solver_validation_failures():
    weight = torch.randn(2, 2)
    mask = torch.ones(2, 2, dtype=torch.bool)
    acc = HessianAccumulator(2, torch.device("cpu"), "val")
    acc.add_batch(torch.eye(2).unsqueeze(0))
    snap = acc.finalize()
    system = build_obs_system(snap, torch.arange(2, dtype=torch.int64), 0.01, "val")
    with pytest.raises(ValueError, match="rank-2"):
        solve_fixed_mask_obs(torch.randn(2), mask, 1, 1, system, 1, "x")
    with pytest.raises(TypeError, match="bool"):
        solve_fixed_mask_obs(weight, torch.ones(2, 2), 1, 1, system, 1, "x")
    with pytest.raises(ValueError, match="solver_block_size"):
        solve_fixed_mask_obs(weight, mask, 1, 1, system, 0, "x")
