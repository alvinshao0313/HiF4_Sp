"""Smoke tests for the read-only HiF4 scaling report."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from Inference_Paradigm_Conversion.ipc_analysis.reporting.hif4_scaling_report import (
    build_hif4_scaling_report,
)


def test_report_builds_from_existing_artifacts_without_model_loading():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        torch.save(
            {
                "schema_version": 1,
                "recipes": {
                    "pts_layer_c00": {"kind": "pts_layer", "granularity": 0, "alpha": 0.0, "deployable": True},
                    "phase_g64": {"kind": "phase_g64", "granularity": 64, "alpha": 0.0, "deployable": True},
                    "eq_g16_a0p5": {"kind": "equalize", "granularity": 16, "alpha": 0.5, "deployable": True},
                },
                "scales": {
                    "4": {
                        "attn_in": {
                            "pts_layer_c00": torch.ones(64),
                            "phase_g64": torch.ones(64),
                            "eq_g16_a0p5": torch.ones(64),
                        }
                    }
                },
            },
            run_dir / "candidate_scales.pt",
        )
        torch.save(
            {
                "records": {
                    "a": {
                        "domain": "attn_in",
                        "recipe_id": "eq_g16_a0p5",
                        "layer": 4,
                        "projection": "q_proj",
                        "phase": "prefill",
                        "activation_conv_error_sum": 8.0,
                        "baseline_activation_conv_error_sum": 10.0,
                        "activation_local_error_sum": 9.0,
                        "baseline_activation_local_error_sum": 10.0,
                        "activation_numel": 100,
                        "hif4_zero_count": 10,
                        "nv_nonzero_to_hif4_zero_count": 4,
                        "hif4_boundary_count": 5,
                        "dispersion_group_count": 2,
                        "before_sub16_log2_amax_range_sum": 4.0,
                        "after_sub16_log2_amax_range_sum": 2.0,
                        "before_sub8_log2_amax_range_sum": 5.0,
                        "after_sub8_log2_amax_range_sum": 3.0,
                        "before_sub4_log2_amax_range_sum": 6.0,
                        "after_sub4_log2_amax_range_sum": 4.0,
                        **{f"payload_bin_{i}_count": 10 + i for i in range(8)},
                    }
                }
            },
            run_dir / "es_eval_merged.pt",
        )
        torch.save(
            {
                "records": {
                    "b": {
                        "domain": "attn_in",
                        "recipe_id": "eq_g16_a0p5",
                        "layer": 4,
                        "projection": "q_proj",
                        "phase": "prefill",
                        "joint_conv_error_sum": 7.0,
                        "baseline_conv_error_sum": 10.0,
                        "joint_local_error_sum": 9.0,
                        "baseline_local_error_sum": 10.0,
                        "weight_local_error_sum": 4.0,
                        "baseline_weight_local_error_sum": 5.0,
                        "joint_numel": 100,
                    }
                }
            },
            run_dir / "es_full_eval_merged.pt",
        )
        (run_dir / "best_scaling_policy.json").write_text(
            json.dumps(
                {
                    "domain_recipes": {"attn_in": "eq_g16_a0p5"},
                    "representative_enabled_by_layer": {"4": ["attn_in"]},
                    "domain_diagnostics": {},
                    "block_summary": {},
                }
            ),
            encoding="utf-8",
        )
        summary = build_hif4_scaling_report(run_dir)
        assert (run_dir / "report.md").is_file()
        assert (run_dir / "summary.json").is_file()
        assert (run_dir / "es2_activation_only_candidates.csv").is_file()
        assert (run_dir / "es3_full_w4a4_candidates.csv").is_file()
        assert summary["selected_recipes"]["attn_in"] == "eq_g16_a0p5"
        assert "figures" in summary
