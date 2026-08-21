"""Tests for the threshold-gate experiment result summarizer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SUMMARIZER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "qwen35_4b_perm_threshold_gate"
    / "summarize_threshold_results.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("summarize_threshold_results", _SUMMARIZER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lm_eval(arc_easy: float, arc_challenge: float, piqa: float) -> dict:
    return {
        "results": {
            "arc_easy": {"acc,none": arc_easy},
            "arc_challenge": {"acc,none": arc_challenge},
            "piqa": {"acc,none": piqa},
            "wikitext": {"word_perplexity,none": 12.0},
        }
    }


def test_fast_summary_computes_percentage_point_deltas():
    mod = _load()
    rows = mod.summarize_fast_variants(
        {
            "identity": _lm_eval(0.80, 0.50, 0.78),
            "tau_0p50": _lm_eval(0.81, 0.505, 0.782),
        },
        {"tau_0p50": {"threshold_pct": 0.5, "n_reordered": 8}},
    )
    row = next(r for r in rows if r["variant"] == "tau_0p50")
    assert row["task_deltas_pp"]["arc_easy"] == pytest.approx(1.0)
    assert row["task_deltas_pp"]["arc_challenge"] == pytest.approx(0.5)
    assert row["macro_delta_pp"] > 0.0
    assert row["n_reordered"] == 8


def test_fast_summary_ranks_by_macro_accuracy():
    mod = _load()
    rows = mod.summarize_fast_variants(
        {
            "identity": _lm_eval(0.80, 0.50, 0.78),
            "tau_0p25": _lm_eval(0.805, 0.506, 0.782),
            "tau_1p00": _lm_eval(0.804, 0.510, 0.780),
        },
        {
            "tau_0p25": {"threshold_pct": 0.25, "n_reordered": 14},
            "tau_1p00": {"threshold_pct": 1.0, "n_reordered": 5},
        },
    )
    selected = mod.select_fast_thresholds(rows, max_candidates=2)
    assert selected[0] == "tau_1p00"
    assert set(selected) == {"tau_0p25", "tau_1p00"}


def test_fast_summary_rejects_missing_identity():
    mod = _load()
    with pytest.raises(KeyError, match="identity"):
        mod.summarize_fast_variants(
            {"tau_0p50": _lm_eval(0.81, 0.51, 0.79)},
            {"tau_0p50": {"threshold_pct": 0.5, "n_reordered": 8}},
        )


def test_fast_summary_excludes_non_accuracy_metrics():
    mod = _load()
    scores = mod.extract_accuracy_scores(_lm_eval(0.80, 0.50, 0.78))
    assert set(scores) == {"arc_easy", "arc_challenge", "piqa"}
    assert "wikitext" not in scores


def test_select_fast_thresholds_enforces_eligibility():
    mod = _load()
    rows = mod.summarize_fast_variants(
        {
            "identity": _lm_eval(0.80, 0.50, 0.78),
            # macro delta negative -> ineligible
            "tau_0p00": _lm_eval(0.79, 0.49, 0.77),
            # arc_challenge drops more than 0.2pp -> ineligible
            "tau_0p25": _lm_eval(0.82, 0.495, 0.79),
            # only 1/3 tasks >= identity -> ineligible
            "tau_0p50": _lm_eval(0.81, 0.49, 0.77),
        },
        {
            "tau_0p00": {"threshold_pct": 0.0, "n_reordered": 30},
            "tau_0p25": {"threshold_pct": 0.25, "n_reordered": 14},
            "tau_0p50": {"threshold_pct": 0.5, "n_reordered": 8},
        },
    )
    assert mod.select_fast_thresholds(rows, max_candidates=2) == []
