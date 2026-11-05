import math

import pytest

from Native_NVFP4_HiF4_Linear_Puncture.experiments.downstream_accuracy_recovery.reporting import METRICS, compare_scores


def test_improvements_do_not_cancel_losses():
    native = dict.fromkeys(METRICS, 0.8)
    candidate = {**native, "MMLU-Pro": 0.7, "LCB": 0.9}
    result = compare_scores(native, candidate)
    assert result["positive_loss_pp"]["LCB"] == 0
    assert math.isclose(result["mean_positive_loss_pp"], 2.0)
    assert not result["observed_target_met"]


def test_exact_threshold_and_above_threshold():
    native = dict.fromkeys(METRICS, 0.8)
    candidate = {**native, "GSM8K": 0.775}
    assert compare_scores(native, candidate)["observed_target_met"]
    candidate["GSM8K"] = 0.774
    assert not compare_scores(native, candidate)["observed_target_met"]


def test_missing_gsm8k_is_not_an_acceptable_partial_comparison():
    native = dict.fromkeys(METRICS, 0.8)
    candidate = {key: value for key, value in native.items() if key != "GSM8K"}
    with pytest.raises(ValueError, match="all five"):
        compare_scores(native, candidate)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.01, 1.01, True])
def test_invalid_scores_rejected(invalid):
    native = dict.fromkeys(METRICS, 0.8)
    with pytest.raises(ValueError, match="finite fractions"):
        compare_scores(native, {**native, "ARC-C": invalid})
