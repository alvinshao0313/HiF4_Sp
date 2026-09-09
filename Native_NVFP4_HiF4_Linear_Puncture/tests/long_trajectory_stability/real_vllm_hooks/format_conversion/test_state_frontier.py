from __future__ import annotations

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks import forced_trajectory as ft
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.state_intervention import (
    StateInterventionSession, canonical_scope, interpolate_pair, make_identity_gate,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.semantic_frontier import (
    coarse_prefix_lengths, contains_subsequence, full_budget_points,
    make_prefix_release_params, summarize_frontier,
)


def scope():
    return canonical_scope({"prompt_key": "sample", "input_ids": [10, 11], "output_ids": [12, 13]},
                           8, "layer_out", 1)


def reference(variant="E0"):
    session = StateInterventionSession(scope(), 0, variant, "capture")
    session.set_positions("sample", torch.tensor([0, 1]))
    session.apply_pair("sample", 8, "layer_out", torch.ones(2, 4), torch.full((2, 4), 2.))
    session.set_positions("sample", torch.tensor([2]))
    session.apply_pair("sample", 8, "layer_out", torch.full((1, 4), 3.), torch.full((1, 4), 4.))
    return session.finish()


def test_full_prefix_capture_and_scope_isolation():
    session = StateInterventionSession(scope(), 0, "E0", "capture")
    branch, residual = torch.ones(2, 4), torch.zeros(2, 4)
    session.set_positions("sample", torch.tensor([0, 1]))
    out = session.apply_pair("sample", 7, "layer_out", branch, residual)
    assert out[0] is branch and not session.seen
    session.apply_pair("sample", 8, "layer_out", branch, residual)
    with pytest.raises(RuntimeError, match="sample isolation"):
        session.set_positions("other", torch.tensor([2]))
    with pytest.raises(RuntimeError, match="incomplete full prefix"):
        session.finish()
    assert session.closed and not session.seen and session.reference is None
    with pytest.raises(RuntimeError, match="sample isolation"):
        session.set_positions("sample", torch.tensor([2]))


def test_full_prefix_noop_and_excluded_positions():
    ref = reference()
    assert set(ref["pairs"]) == {0, 1, 2}
    session = StateInterventionSession(scope(), 0, "E0", "noop", reference=ref)
    session.set_positions("sample", torch.tensor([0, 1, 2, 3]))
    b, r = torch.zeros(4, 4), torch.zeros(4, 4)
    patched_b, patched_r = session.apply_pair("sample", 8, "layer_out", b, r)
    assert torch.equal(patched_b[2], torch.full((4,), 3.))
    assert torch.equal(patched_r[2], torch.full((4,), 4.))
    assert torch.equal(patched_b[3], b[3]) and torch.equal(patched_r[3], r[3])
    assert not b.any() and not r.any()
    assert session.finish()["full_prefix_complete"]


def test_interpolation_alpha0_preserves_both_exact_endpoint_tensors():
    b = torch.tensor([float("inf"), 1.], dtype=torch.bfloat16)
    r = torch.tensor([4., 3.], dtype=torch.bfloat16)
    pair0, pair1 = (b, r), (torch.ones_like(b), torch.ones_like(r))
    out = interpolate_pair(pair0, pair1, 0)
    assert out[0] is b and out[1] is r
    assert interpolate_pair(pair0, pair1, 1)[0] is pair1[0]
    assert interpolate_pair((r, None), (r, None), .5)[1] is None
    with pytest.raises(RuntimeError, match="presence differs"):
        interpolate_pair((r, None), (r, r), .5)


def test_reset_and_injection_gate_fail_closed():
    with pytest.raises(RuntimeError, match="missing matching noop"):
        StateInterventionSession(scope(), 0, "E1", "reset", reference=reference())
    with pytest.raises(RuntimeError, match="missing matching alpha0"):
        StateInterventionSession(scope(), 0, "E0", "inject", reference=reference(),
                                 deviation=reference("E1"), alpha=.5)
    changed = scope() | {"canonical_output_sha256": "different"}
    with pytest.raises(RuntimeError, match="mismatch"):
        StateInterventionSession(changed, 0, "E0", "noop", reference=reference())
    with pytest.raises(RuntimeError, match="mismatch"):
        StateInterventionSession(scope(), 1, "E0", "noop", reference=reference())


def test_identity_gate_measures_noise_floor_and_requires_exact_decisions():
    base = torch.tensor([2., 1., 0.])
    rows = [{"rank": rank, "baseline": base, "repeat": base.clone(), "patched": base.clone(),
             "target": 0, "forced_exact": True, "full_prefix_complete": True} for rank in (0, 1)]
    gate = make_identity_gate(scope(), "noop", rows)
    assert gate["status"] == "PASS"
    rows[0]["patched"][1] += .0001
    assert make_identity_gate(scope(), "noop", rows)["status"] == "STATE_RESET_BLOCKED"
    rows[0]["repeat"] = rows[0]["patched"].clone()
    assert make_identity_gate(scope(), "noop", rows)["status"] == "PASS"
    rows[1]["full_prefix_complete"] = False
    assert make_identity_gate(scope(), "noop", rows)["status"] == "STATE_RESET_BLOCKED"


def test_force_release_leaves_free_logits_untouched(monkeypatch):
    monkeypatch.setattr(ft, "get_tensor_model_parallel_rank", lambda: 0)
    proc = ft._ForcedTrajectoryRequestProcessor([2], sample_key=None, variant=None,
        probe_decode_indices=set(), logits_root=None, release_after_prefix=True)
    logits = torch.tensor([1., 3., 2.])
    assert torch.isneginf(proc([], logits.clone())[[0, 1]]).all()
    assert proc([2], logits) is logits
    assert torch.equal(logits, torch.tensor([1., 3., 2.]))
    params = make_prefix_release_params([2], max_tokens=4)
    assert params.max_tokens == 4 and params.min_tokens == 1 and not params.ignore_eos
    assert params.extra_args[ft.RELEASE_AFTER_PREFIX_KEY] is True
    ft.ForcedTrajectoryLogitsProcessor.validate_params(params)


def test_frontier_preserves_nonmonotonic_accuracy_and_rejects_short_screens():
    rows = [{"sample_key": "s", "kind": "rescue", "k": k, "pass": value,
             "official_judged": True, "full_budget": True}
            for k, value in ((4, False), (12, True), (20, False), (28, True))]
    result = summarize_frontier(rows)
    assert result["status"] == "MULTI_FRONTIER" and len(result["transition_intervals"]) == 3
    assert [r["pass"] for r in result["vector"]] == [False, True, False, True]
    rows[0]["full_budget"] = False
    with pytest.raises(ValueError, match="full-budget"):
        summarize_frontier(rows)


def test_frontier_coarse_points_and_transition_neighbors():
    assert contains_subsequence([1, 2, 3, 4], [2, 3])
    assert not contains_subsequence([1, 2, 3], [3, 4])
    points = coarse_prefix_lengths(4, 100, thinking_end=30, code_start=34)
    assert {4, 12, 20, 36, 68, 29, 30, 31, 33, 34, 35} == set(points)
    rows = [{"sample_key": "s", "kind": "rescue", "k": k,
             "finished_thinking": status, "contains_code_fence": False}
            for k, status in ((4, False), (12, False), (20, True), (36, False))]
    assert full_budget_points(rows) == [12, 20, 36]
