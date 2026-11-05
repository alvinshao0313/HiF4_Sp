"""Budget, complete-sequence, and held-out isolation contracts."""
import copy

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import CalibrationSample
from Native_NVFP4_HiF4_Linear_Puncture.experiments.matrix_norm_efficiency.selection import Candidate, select


def candidates():
    return [Candidate(f"{domain}_{index}", domain, length, f"{domain}: solve the problem about topic {index}")
            for domain in ("math", "code") for index, length in enumerate((2, 4, 6, 8, 10, 12))]


@pytest.mark.parametrize("method", ["random50", "coverage50"])
def test_full_sequences_respect_domain_and_length_budgets(method):
    rows = candidates()
    before = copy.deepcopy(rows)
    ids, report = select(rows, method)
    assert rows == before
    assert ids == [r.sample_id for r in rows if r.sample_id in ids]
    assert len(ids) == len(set(ids))
    assert report["budget_tokens"] == sum(r.tokens for r in rows) // 2
    assert report["selected_tokens"] == sum(r.tokens for r in rows if r.sample_id in ids)
    assert report["selected_tokens"] <= report["budget_tokens"]
    for stratum in report["strata"]:
        assert stratum["selected_ids_by_rank"]
        assert stratum["selected_tokens"] <= stratum["budget_tokens"]
        remaining = [r.tokens for r in rows if r.sample_id in stratum["source_ids"] and r.sample_id not in ids]
        assert not remaining or min(remaining) > stratum["unused_tokens"]


def test_selectors_share_caps_and_repeat_deterministically():
    rows = candidates()
    reports = [select(rows, method)[1] for method in ("random50", "coverage50")]
    assert [(r["name"], r["budget_tokens"]) for r in reports[0]["strata"]] == [
        (r["name"], r["budget_tokens"]) for r in reports[1]["strata"]]
    for method in ("random50", "coverage50"):
        ids, report = select(rows, method)
        assert (ids, report) == select(rows, method)
        assert set(ids) == set(select(list(reversed(rows)), method)[0])


def test_full_selection_keeps_every_sequence_in_source_order():
    rows = candidates()
    ids, report = select(rows, "full")
    assert ids == [r.sample_id for r in rows]
    assert report["selected_tokens"] == report["source_tokens"]
    assert report["unused_tokens"] == 0


def test_invalid_budget_and_duplicate_ids_fail():
    with pytest.raises(ValueError, match="complete sequence"):
        select(candidates(), "random50", fraction=.01)
    with pytest.raises(ValueError, match="duplicate"):
        select(candidates() + candidates()[:1], "random50")


def test_numeric_variant_of_test_question_blocks_fixed_validation():
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.matrix_norm_efficiency.prepare import isolation_audit

    selected = [{"id": "val_1", "role": "selected", "split": "val", "question": "Find the sum of 123 and 456."}]
    tests = [{"id": "aime24_1", "role": "test", "question": "Find the sum of 321 and 654."}]
    result = isolation_audit(selected, tests)
    assert result["status"] == "blocked_test_overlap"
    assert result["test_overlap"][0]["affected_splits"] == ["val"]
    assert selected[0]["id"] == "val_1"


def test_loss_subsampling_is_not_accepted_as_whole_sequence_selection():
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.matrix_norm_efficiency.prepare import _whole_sequences

    sample = CalibrationSample("sample", 0, torch.arange(4), torch.tensor([1, 0, 1, 1]))
    with pytest.raises(ValueError, match="complete-sequence"):
        _whole_sequences([sample])
