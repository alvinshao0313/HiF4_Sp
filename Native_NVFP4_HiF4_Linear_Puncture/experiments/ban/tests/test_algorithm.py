"""CPU checks against the pinned upstream Ban equations and KL definition."""

import math

import pytest
import torch

from ..algorithm import (
    SOURCE_COMMIT,
    ban_route,
    normalize_layer_kl,
    reference_distribution,
    reference_kl,
    top3_ratio,
    validate_parameters,
)


def test_route_matches_independent_scalar_equations():
    generator = torch.Generator().manual_seed(17)
    # Avoid tied BF16 logits: torch.topk intentionally has no stable tie order.
    logits = torch.randn(13, 128, generator=generator)
    weights, ids, counts = ban_route(logits, 8, torch.tensor(0.43), 0.38, 0.9)
    for row, returned_weights, returned_ids, count in zip(logits, weights, ids, counts):
        scores = [math.exp(float(value) - float(row.max())) for value in row]
        total = sum(scores)
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:8]
        raw = [scores[index] / total for index in order]
        ratio = sum(raw[:3]) / sum(raw)
        token = max(0.0, min(1.0, (0.9 - ratio) / (0.9 - 0.38)))
        expected_k = max(3, min(8, round(3 + 5 * 0.7 * 0.5 * (0.43 + token))))
        expected_weights = [value / sum(raw[:expected_k]) if i < expected_k else 0.0
                            for i, value in enumerate(raw)]
        assert count.item() == expected_k
        assert returned_ids.tolist() == order
        torch.testing.assert_close(returned_weights, torch.tensor(expected_weights), rtol=2e-6, atol=1e-7)
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    assert counts.dtype == torch.int64
    assert SOURCE_COMMIT == "3ac1169c9a48f6cda2477975a024aadba5d02f8b"


@pytest.mark.parametrize("lam,expected", [(0.2, 4), (0.6, 4), (1.0, 6)])
def test_rounding_uses_ties_to_even(lam, expected):
    # r_max < the token's ratio makes s_token exactly zero; s_layer is one.
    _, _, counts = ban_route(torch.zeros(1, 8), 8, torch.tensor(1.0), 0.0, 0.2, lam=lam)
    assert counts.item() == expected


def test_zero_lambda_minimum_and_clamped_sensitivities():
    logits = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    _, _, counts = ban_route(logits, 8, torch.tensor(2.0), 0.9, 1.0, lam=0.0)
    assert counts.item() == 3
    _, _, counts = ban_route(logits, 8, torch.tensor(2.0), 0.99, 1.0, lam=1.0)
    assert counts.item() == 8
    _, _, counts = ban_route(logits, 8, torch.tensor(-1.0), 0.0, 0.1, lam=1.0)
    assert counts.item() == 3


def test_raw_and_renormalized_weights_keep_identical_routing():
    logits = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    raw, ids, counts = ban_route(logits, 6, torch.tensor(0.3), 0.4, 0.95, renormalize=False)
    normalized, normalized_ids, normalized_counts = ban_route(
        logits, 6, torch.tensor(0.3), 0.4, 0.95, indices_type=torch.int64
    )
    base, base_ids = torch.topk(logits.float().softmax(-1), 6, sorted=True)
    keep = torch.arange(6).view(1, 6) < counts.view(-1, 1)
    assert torch.equal(raw, base.masked_fill(~keep, 0.0))
    assert torch.equal(ids.long(), base_ids)
    assert torch.equal(ids.long(), normalized_ids)
    assert torch.equal(counts, normalized_counts)
    torch.testing.assert_close(normalized, raw / raw.sum(-1, keepdim=True), rtol=0, atol=0)
    torch.testing.assert_close(normalized.sum(-1), torch.ones(2))


def test_small_topk_large_k_min_and_degenerate_ratio_range():
    logits = torch.zeros(3, 5)
    weights, _, counts = ban_route(logits, 2, torch.tensor(0.5), 1.0, 1.0, k_min=3)
    assert counts.tolist() == [2, 2, 2]
    assert torch.equal(weights, torch.full((3, 2), 0.5))
    assert torch.equal(top3_ratio(weights), torch.ones(3))
    _, _, counts = ban_route(logits, 5, torch.tensor(0.5), 0.6, 0.6, k_min=1)
    assert torch.isfinite(counts).all()
    assert top3_ratio(torch.tensor([[0.4, 0.3, 0.2, 0.1]])).item() == pytest.approx(0.9)


def test_kl_direction_and_native_vocabulary_support():
    native = torch.tensor([[2.0, 1.0, -5.0, -6.0], [0.0, 3.0, 2.0, -8.0]])
    candidate = torch.tensor([[0.0, 2.0, 100.0, -20.0], [0.0, 2.0, 3.0, 90.0]])
    reference = reference_distribution(native, topn=2)
    indices, probs, log_probs = reference
    assert indices.tolist() == [[0, 1], [1, 2]]
    expected = []
    for row, support in zip(candidate.tolist(), indices.tolist()):
        # Both native supports have native probabilities softmax([2, 1]).
        p = [math.exp(1) / (math.exp(1) + 1), 1 / (math.exp(1) + 1)]
        q_raw = [math.exp(row[index]) for index in support]
        q = [value / sum(q_raw) for value in q_raw]
        expected.append(sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q)))
    assert reference_kl(candidate, reference).item() == pytest.approx(sum(expected) / 2, rel=1e-6)
    torch.testing.assert_close(probs, log_probs.exp())
    assert reference_kl(native, reference).item() == 0.0
    # Logits outside the native support have no tail bucket in upstream Ban.
    candidate[:, 3] = -999.0
    candidate[0, 2] = -999.0
    assert reference_kl(candidate, reference).item() == pytest.approx(sum(expected) / 2, rel=1e-6)
    assert reference_distribution(native, topn=1000)[0].shape == native.shape


def test_layer_kl_minmax_and_constant_values():
    torch.testing.assert_close(normalize_layer_kl(torch.tensor([0.2, 0.3, 0.6])), torch.tensor([0.0, 0.25, 1.0]))
    assert torch.equal(normalize_layer_kl(torch.ones(4)), torch.zeros(4))
    # Tiny KL differences retain upstream 1e-20 denominator semantics.
    tiny = normalize_layer_kl(torch.tensor([0.0, 1e-21]))
    torch.testing.assert_close(tiny, torch.tensor([0.0, 0.1]))


@pytest.mark.parametrize("lam,k_min,top_k", [
    (-0.1, 3, 8), (1.1, 3, 8), (float("nan"), 3, 8), (float("inf"), 3, 8),
    (0.7, 0, 8), (0.7, 1.5, 8), (0.7, True, 8), (0.7, 3, 0), (0.7, 3, 1.5),
])
def test_invalid_parameters_fail(lam, k_min, top_k):
    with pytest.raises(ValueError):
        validate_parameters(lam, k_min, top_k)


def test_invalid_shapes_and_artifact_values_fail():
    with pytest.raises(ValueError):
        ban_route(torch.zeros(1, 4), 8, torch.tensor(0.5), 0.0, 1.0)
    with pytest.raises(ValueError):
        ban_route(torch.zeros(1, 8), 8, torch.ones(2), 0.0, 1.0)
    with pytest.raises(ValueError):
        ban_route(torch.zeros(1, 8), 8, torch.tensor(0.5), 1.0, 0.0)
    with pytest.raises(ValueError):
        ban_route(torch.zeros(1, 8), 8, torch.tensor(0.5), 0.0, float("nan"))
    with pytest.raises(ValueError):
        reference_distribution(torch.zeros(1, 8), topn=0)
    with pytest.raises(ValueError):
        normalize_layer_kl(torch.tensor([0.0, float("nan")]))
    with pytest.raises(ValueError):
        normalize_layer_kl(torch.tensor([]))
