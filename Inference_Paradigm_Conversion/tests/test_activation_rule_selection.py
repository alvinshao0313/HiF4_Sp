"""Tests for AX5 rule selection."""

from __future__ import annotations

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_rule_selection import (
    build_root_cause_ranking,
    run_rule_selection,
)


def test_skip_low_recovery():
    rows = [{"output_recovery": 0.01, "split": "discovery"}]
    out = run_rule_selection(rows)
    assert out["status"] == "skipped_due_to_low_s0_recovery"
    assert out["candidate_for_e2e"] is False


def test_rule_selection_completed():
    rows = [
        {
            "output_recovery": 0.2,
            "output_recovery_current": 0.05,
            "alpha_oracle_nvfp4": 6.5,
            "max_over_rms": 3.0,
            "projection": "q_proj",
            "split": "discovery",
        }
        for _ in range(5)
    ]
    out = run_rule_selection(rows)
    assert out["status"] == "completed"
    assert "R0_global_alpha" in out


def test_root_cause_ranking():
    ax1 = [{"output_recovery": 0.3, "projection": "q_proj"}]
    ax2 = [{"R_Y": 0.1, "is_standard_hif4": False, "group_size": 16}]
    ax4 = [{"R_Y": 0.15, "is_valid_hardware_format": False, "hybrid": "HN"}]
    ranking = build_root_cause_ranking({"ax1": ax1, "ax2": ax2, "ax4": ax4})
    assert len(ranking) >= 2
    assert ranking[0]["rank"] == 1
