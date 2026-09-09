from __future__ import annotations

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.production_puncture import (
    ProductionPunctureOp, closure_metrics, error_decomposition, frozen_router,
    identity_metrics, puncture_key, router_decomposition,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.quant_features import (
    aggregate_token_features, distribution_features, feature_rows_from_payload, quant_features,
)


def test_group_features_shape_and_defined_zero_semantics():
    x = torch.arange(1, 129, dtype=torch.bfloat16)
    row = quant_features(x)
    assert row["group64_count"] == 2
    assert len(row["group64_amax_over_rms"]) == 2
    assert torch.tensor(row["subgroup16_amax"]).shape == (2, 4)
    assert row["subgroup16_imbalance_by_group64"] == [4.0, 1.6]
    assert row["hif4_same_state_qdq_rel_l2"] > 0
    assert row["hif4_qdq_scope"] == "activation_only"
    zeros = distribution_features(torch.zeros(64))
    assert zeros["rms"] == 0 and zeros["amax_over_rms"] is None
    assert zeros["kurtosis"] is None
    assert zeros["group64_amax_over_rms"] == [None]
    with pytest.raises(ValueError):
        distribution_features(torch.zeros(65))


def test_feature_boundary_and_history_are_hard_requirements():
    metadata = {"sample_key": "s", "variant": "E0", "tp_rank": 0, "layer": 2,
                "boundary": "input_norm", "role": "normalized", "decode_index": 3,
                "abs_position": 20, "tensor": torch.ones(64)}
    payload = {"variant": "E0", "world_size": 2, "records": [metadata]}
    with pytest.raises(RuntimeError, match="forced token history"):
        feature_rows_from_payload(payload, canonical_history_verified=False)
    rows = feature_rows_from_payload(payload, canonical_history_verified=True)
    token = aggregate_token_features(rows)[0]
    assert token["argmax_amax_over_rms"] == {"layer": 2, "boundary": "input_norm"}
    metadata["boundary"] = "moe_out"
    with pytest.raises(RuntimeError, match="non-feature"):
        feature_rows_from_payload(payload, canonical_history_verified=True)


@pytest.mark.parametrize("operator", ["qkv", "o_proj", "moe"])
def test_error_decomposition_closes_and_cancellation_is_preserved(operator):
    assert puncture_key("case", 2, 47, operator, 1) == ("case", 2, 47, operator, 1)
    y0 = torch.tensor([1., 2.], dtype=torch.bfloat16)
    y1 = torch.tensor([2., 1.], dtype=torch.bfloat16)
    fixed = torch.tensor([3., 0.], dtype=torch.bfloat16)
    result = error_decomposition(y0, y1, fixed)
    assert result["closure_pass"]
    assert result["e_l2"] == pytest.approx(2**0.5)
    assert result["q_l2"] == pytest.approx(8**0.5)
    assert result["cos_q_p"] == pytest.approx(-1)
    split = router_decomposition(y0, fixed, torch.tensor([2., 1.]))
    assert split["router_closure_pass"]
    assert split["q_expert_l2"] == pytest.approx(2**0.5)
    assert split["q_router_l2"] == pytest.approx(2**0.5)
    with pytest.raises(RuntimeError, match="closure failed"):
        closure_metrics(y0, y1, fixed)


def test_identity_does_not_accept_unmeasured_tolerance():
    actual = torch.tensor([1., 2.])
    repeats = [actual.clone() for _ in range(3)]
    assert identity_metrics(actual, repeats)["identity_pass"]
    wrong = actual + 0.00001
    assert not identity_metrics(wrong, repeats)["identity_pass"]
    with pytest.raises(ValueError, match="three"):
        identity_metrics(actual, repeats[:2])


def test_puncture_key_rejects_prefill_and_e4():
    with pytest.raises(ValueError, match="prefill"):
        puncture_key("s", 0, 2, "moe", 0)
    with pytest.raises(ValueError, match="permitted"):
        ProductionPunctureOp([], "x", "x", "x", "E4", canonical_history_verified=True)
    with pytest.raises(ValueError, match="identity artifacts"):
        ProductionPunctureOp([], "x", "x", "x", "E1", canonical_history_verified=True)


def test_frozen_router_hook_is_removed_after_success_and_exception():
    # This exercises only temporary PyTorch-hook lifecycle, never production
    # math or a model simulation. Real identity needs the TP2 GPU gate.
    class Gate(torch.nn.Module):
        def forward(self, x):
            return x, None
    gate = Gate()
    x, forced = torch.zeros(1, 4), torch.ones(1, 4)
    with frozen_router(gate, forced) as calls:
        assert torch.equal(gate(x)[0], forced)
    assert calls["count"] == 1 and not gate._forward_hooks
    assert torch.equal(gate(x)[0], x)
    with pytest.raises(RuntimeError, match="caller failure"):
        with frozen_router(gate, forced):
            gate(x)
            raise RuntimeError("caller failure")
    assert not gate._forward_hooks
    with pytest.raises(RuntimeError, match="never fired"):
        with frozen_router(gate, forced):
            pass
    assert not gate._forward_hooks
