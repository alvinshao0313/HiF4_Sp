from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.block_utils import reduce_weight_wanda_to_blocks
from block_pruning.config import GradientBlockPruningConfig
from block_pruning.gradient_scorer import BlockScoreRecord
from block_pruning.mlp_registry import MLPLinearTarget
from block_pruning.wanda_scorer import (
    InputRMSRecord,
    collect_mlp_input_rms,
    collect_wanda_block_scores,
)


def test_reduce_weight_wanda_hand_computed():
    weight = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0],
            [5.0, -6.0, 7.0, -8.0],
            [9.0, -10.0, 11.0, -12.0],
            [13.0, -14.0, 15.0, -16.0],
        ]
    )
    input_rms = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = torch.tensor(
        [
            [
                1 * 1 + 2 * 2 + 5 * 1 + 6 * 2,
                3 * 3 + 4 * 4 + 7 * 3 + 8 * 4,
            ],
            [
                9 * 1 + 10 * 2 + 13 * 1 + 14 * 2,
                11 * 3 + 12 * 4 + 15 * 3 + 16 * 4,
            ],
        ],
        dtype=torch.float32,
    )
    got = reduce_weight_wanda_to_blocks(weight, input_rms, 2, 2)
    assert got.shape == (2, 2)
    torch.testing.assert_close(got, expected)


def test_reduce_weight_wanda_cpu_rms_with_cuda_weight():
    if not torch.cuda.is_available():
        return
    weight = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0],
            [5.0, -6.0, 7.0, -8.0],
            [9.0, -10.0, 11.0, -12.0],
            [13.0, -14.0, 15.0, -16.0],
        ],
        device="cuda",
    )
    input_rms = torch.tensor([1.0, 2.0, 3.0, 4.0])  # CPU
    expected = reduce_weight_wanda_to_blocks(weight.cpu(), input_rms, 2, 2)
    got = reduce_weight_wanda_to_blocks(weight, input_rms, 2, 2)
    assert got.device.type == "cuda"
    torch.testing.assert_close(got.cpu(), expected)


def test_reduce_weight_wanda_rejects_bad_shapes():
    weight = torch.randn(4, 4)
    try:
        reduce_weight_wanda_to_blocks(weight, torch.randn(2, 2), 2, 2)
        assert False, "expected ValueError for rank-2 input_rms"
    except ValueError as e:
        assert "rank 1" in str(e)

    try:
        reduce_weight_wanda_to_blocks(weight, torch.randn(3), 2, 2)
        assert False, "expected ValueError for channel mismatch"
    except ValueError as e:
        assert "d_in" in str(e)

    try:
        reduce_weight_wanda_to_blocks(torch.randn(5, 4), torch.randn(4), 2, 2)
        assert False, "expected ValueError for non-divisible"
    except ValueError as e:
        assert "not divisible" in str(e)


class _TinyTwoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 2, bias=False)
        self.b = nn.Linear(3, 2, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        # Deterministic activations independent of input_ids content.
        x_a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        x_b = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]])
        _ = self.a(x_a)
        _ = self.b(x_b)
        return x_a


def test_collect_mlp_input_rms_hand_computed():
    model = _TinyTwoLinear()
    targets = [
        MLPLinearTarget("layers.0.mlp.a", model.a, 0, "up_proj"),
        MLPLinearTarget("layers.0.mlp.b", model.b, 0, "down_proj"),
    ]
    batches = [
        {
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "labels": torch.zeros(1, 2, dtype=torch.long),
        }
    ]
    records = collect_mlp_input_rms(model, batches, targets)

    # a: tokens [[1,0,0],[0,2,0]] -> square_sum=[1,4,0], n=2
    assert records["layers.0.mlp.a"].num_tokens == 2
    torch.testing.assert_close(
        records["layers.0.mlp.a"].channel_square_sum,
        torch.tensor([1.0, 4.0, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        records["layers.0.mlp.a"].input_rms,
        torch.sqrt(torch.tensor([1.0, 4.0, 0.0], dtype=torch.float64) / 2),
    )

    # b: tokens [[0,0,3],[0,0,4],[0,0,0]] -> square_sum=[0,0,25], n=3
    assert records["layers.0.mlp.b"].num_tokens == 3
    torch.testing.assert_close(
        records["layers.0.mlp.b"].channel_square_sum,
        torch.tensor([0.0, 0.0, 25.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        records["layers.0.mlp.b"].input_rms,
        torch.sqrt(torch.tensor([0.0, 0.0, 25.0], dtype=torch.float64) / 3),
    )


def test_collect_mlp_input_rms_failures():
    model = _TinyTwoLinear()
    targets = [MLPLinearTarget("layers.0.mlp.a", model.a, 0, "up_proj")]
    try:
        collect_mlp_input_rms(model, [], targets)
        assert False, "expected empty batches error"
    except ValueError:
        pass

    unused = nn.Linear(3, 2, bias=False)
    bad_targets = [MLPLinearTarget("layers.0.mlp.unused", unused, 0, "up_proj")]
    batches = [
        {
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "labels": torch.zeros(1, 2, dtype=torch.long),
        }
    ]
    try:
        collect_mlp_input_rms(model, batches, bad_targets)
        assert False, "expected never-invoked error"
    except RuntimeError as e:
        assert "never invoked" in str(e)


def test_primary_score_wanda():
    score = torch.arange(4, dtype=torch.float64).reshape(2, 2)
    rec = BlockScoreRecord(
        module_name="m",
        layer_index=0,
        projection_type="up_proj",
        weight_shape=(4, 4),
        block_size="2",
        block_height=2,
        block_width=2,
        fisher=torch.zeros_like(score),
        abs_taylor=torch.zeros_like(score),
        signed_mean=torch.zeros_like(score),
        current_mask=torch.ones_like(score, dtype=torch.bool),
        wanda=score,
    )
    assert torch.equal(rec.primary_score("wanda"), score)
    rec2 = BlockScoreRecord(
        module_name="m",
        layer_index=0,
        projection_type="up_proj",
        weight_shape=(4, 4),
        block_size="2",
        block_height=2,
        block_width=2,
        fisher=torch.zeros_like(score),
        abs_taylor=torch.zeros_like(score),
        signed_mean=torch.zeros_like(score),
        current_mask=torch.ones_like(score, dtype=torch.bool),
        wanda=None,
    )
    try:
        rec2.primary_score("wanda")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Wanda score missing" in str(e)


def test_collect_wanda_block_scores_matches_formula():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.lin.weight.copy_(
                    torch.tensor(
                        [
                            [1.0, -2.0, 3.0, -4.0],
                            [5.0, -6.0, 7.0, -8.0],
                            [9.0, -10.0, 11.0, -12.0],
                            [13.0, -14.0, 15.0, -16.0],
                        ]
                    )
                )

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            return self.lin(x)

    model = M()
    targets = [MLPLinearTarget("layers.0.mlp.up_proj", model.lin, 0, "up_proj")]
    batches = [
        {
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "labels": torch.zeros(1, 2, dtype=torch.long),
        }
    ]
    cfg = GradientBlockPruningConfig(block_size="2", score_type="fisher_budget_wanda")
    records = collect_wanda_block_scores(model, batches, targets, cfg)
    expected = reduce_weight_wanda_to_blocks(
        model.lin.weight.detach(),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        2,
        2,
    ).double()
    torch.testing.assert_close(records["layers.0.mlp.up_proj"].wanda, expected)
    assert records["layers.0.mlp.up_proj"].primary_score("wanda").shape == (2, 2)


def test_wanda_accepts_precomputed_rms_without_second_forward():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4, bias=False)
            self.n_calls = 0
            with torch.no_grad():
                self.lin.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4))

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            self.n_calls += 1
            x = torch.ones(1, 4)
            return self.lin(x)

    model = M()
    targets = [MLPLinearTarget("layers.0.mlp.up_proj", model.lin, 0, "up_proj")]
    cfg = GradientBlockPruningConfig(block_size="2", score_type="fisher_budget_wanda")
    rms = {
        "layers.0.mlp.up_proj": InputRMSRecord(
            module_name="layers.0.mlp.up_proj",
            layer_index=0,
            projection_type="up_proj",
            num_tokens=1,
            channel_square_sum=torch.ones(4, dtype=torch.float64),
            input_rms=torch.ones(4, dtype=torch.float64),
        )
    }
    before = model.n_calls
    records = collect_wanda_block_scores(
        model,
        batches=None,
        targets=targets,
        config=cfg,
        input_rms_records=rms,
    )
    assert model.n_calls == before
    expected = reduce_weight_wanda_to_blocks(
        model.lin.weight.detach(), torch.ones(4), 2, 2
    ).double()
    torch.testing.assert_close(records["layers.0.mlp.up_proj"].wanda, expected)
