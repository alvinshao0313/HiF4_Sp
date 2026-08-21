"""Synthetic tests for the HiF4 scaling global-barrier experiment pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from Inference_Paradigm_Conversion.ipc_analysis.analysis.hif4_scaling_pipeline import (
    build_candidate_scales,
    merge_scaling_eval,
    merge_scaling_stats,
)


def _stats_shard(
    shard_id: int,
    *,
    sample_id: str,
    sum_sq_value: float,
    max_abs_value: float,
    phase_error_offset: float,
) -> dict:
    width = 64
    pts = 33
    err = torch.arange(pts, dtype=torch.float64).reshape(1, pts) + phase_error_offset
    return {
        "schema_version": 1,
        "split": "discovery",
        "shard_id": shard_id,
        "num_shards": 2,
        "config_sha256": "cfg",
        "sample_ids": [sample_id],
        "model_meta": {
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
        },
        "stats": {
            "4:attn_in:prefill": {
                "sum_sq": torch.full((width,), sum_sq_value, dtype=torch.float64),
                "max_abs": torch.full((width,), max_abs_value, dtype=torch.float64),
                "count": 2,
            },
            "4:attn_in:decode": {
                "sum_sq": torch.full((width,), sum_sq_value / 2.0, dtype=torch.float64),
                "max_abs": torch.full((width,), max_abs_value / 2.0, dtype=torch.float64),
                "count": 1,
            },
        },
        "phase_g64": {
            "4:attn_in:prefill": {
                "error_sum": err,
                "count": 64,
            },
            "4:attn_in:decode": {
                "error_sum": err * 0.5,
                "count": 64,
            },
        },
    }


def _config() -> dict:
    return {
        "group_size": 64,
        "representative_layers": [4],
        "run_pts_layer": True,
        "run_phase_g64": True,
        "equalization_granularities": [16, 8, 4, 1],
        "alphas": [0.0, 0.5, 1.0],
        "min_scale": 0.5,
        "max_scale": 2.0,
        "pts_log2_min": -1.0,
        "pts_log2_max": 1.0,
        "pts_points": 33,
        "run_weight_aware_balance": False,
        "weight_aware_beta": 0.5,
        "build_wide_bound_candidates": False,
    }


def test_merge_scaling_stats_merges_raw_sufficient_statistics_before_finalize():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        torch.save(
            _stats_shard(
                0,
                sample_id="a",
                sum_sq_value=4.0,
                max_abs_value=3.0,
                phase_error_offset=10.0,
            ),
            run_dir / "es_stats_shard0.pt",
        )
        torch.save(
            _stats_shard(
                1,
                sample_id="b",
                sum_sq_value=9.0,
                max_abs_value=5.0,
                phase_error_offset=20.0,
            ),
            run_dir / "es_stats_shard1.pt",
        )
        out = merge_scaling_stats(run_dir, expected_num_shards=2)
        merged = torch.load(out, map_location="cpu", weights_only=False)
        stat = merged["stats"]["4:attn_in:prefill"]
        torch.testing.assert_close(stat["sum_sq"], torch.full((64,), 13.0, dtype=torch.float64))
        torch.testing.assert_close(stat["max_abs"], torch.full((64,), 5.0, dtype=torch.float64))
        assert stat["count"] == 4
        err = merged["phase_g64"]["4:attn_in:prefill"]["error_sum"]
        expected = (
            torch.arange(33, dtype=torch.float64).reshape(1, 33) * 2.0 + 30.0
        )
        torch.testing.assert_close(err, expected)
        assert sorted(merged["sample_ids"]) == ["a", "b"]


def test_merge_scaling_stats_rejects_duplicate_samples():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        torch.save(_stats_shard(0, sample_id="same", sum_sq_value=1, max_abs_value=1, phase_error_offset=0), run_dir / "es_stats_shard0.pt")
        torch.save(_stats_shard(1, sample_id="same", sum_sq_value=1, max_abs_value=1, phase_error_offset=0), run_dir / "es_stats_shard1.pt")
        try:
            merge_scaling_stats(run_dir, expected_num_shards=2)
        except ValueError as exc:
            assert "sample" in str(exc).lower()
        else:
            raise AssertionError("duplicate sample ids must fail merge")


def test_build_candidate_scales_uses_same_deploy_scale_for_prefill_and_decode():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        # One already-merged artifact is enough for this test.
        merged = _stats_shard(0, sample_id="a", sum_sq_value=16, max_abs_value=8, phase_error_offset=0)
        merged["num_shards"] = 1
        merged["shard_id"] = None
        merged["stats"]["4:attn_in:decode"]["max_abs"] = torch.full((64,), 32.0, dtype=torch.float64)
        merged["stats"]["4:attn_in:decode"]["sum_sq"] = torch.full((64,), 1024.0, dtype=torch.float64)
        merged["stats"]["4:attn_in:decode"]["count"] = 1
        p = run_dir / "es_stats_merged.pt"
        torch.save(merged, p)
        out = build_candidate_scales(p, config=_config())
        art = torch.load(out, map_location="cpu", weights_only=False)
        assert "4" in art["scales"]
        assert "attn_in" in art["scales"]["4"]
        eq0 = art["scales"]["4"]["attn_in"]["eq_g16_a0"]
        torch.testing.assert_close(eq0, torch.ones_like(eq0), rtol=0, atol=0)
        # A candidate artifact stores one D per layer/domain/recipe, never phase-specific D.
        assert not any("prefill" in key or "decode" in key for key in art["scales"]["4"]["attn_in"])


def test_phase_g64_selects_one_global_candidate_per_group_from_merged_error():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        merged = _stats_shard(0, sample_id="a", sum_sq_value=4, max_abs_value=2, phase_error_offset=0)
        merged["num_shards"] = 1
        merged["shard_id"] = None
        # Force candidate index 7 to be globally optimal in both phases.
        for phase in ("prefill", "decode"):
            e = torch.full((1, 33), 100.0, dtype=torch.float64)
            e[0, 7] = 1.0
            merged["phase_g64"][f"4:attn_in:{phase}"]["error_sum"] = e
        p = run_dir / "es_stats_merged.pt"
        torch.save(merged, p)
        out = build_candidate_scales(p, config=_config())
        art = torch.load(out, map_location="cpu", weights_only=False)
        d = art["scales"]["4"]["attn_in"]["phase_g64"]
        grid = art["pts_grid"]
        torch.testing.assert_close(d, torch.full((64,), float(grid[7])))
        assert art["phase_g64_best_indices"]["4:attn_in"] == [7]


def _eval_shard(shard_id: int, candidate_hash: str, *, sample: str, error: float) -> dict:
    return {
        "schema_version": 1,
        "split": "discovery",
        "shard_id": shard_id,
        "num_shards": 2,
        "config_sha256": "cfg",
        "stats_merged_sha256": "stats",
        "candidate_scales_sha256": candidate_hash,
        "sample_ids": [sample],
        "records": {
            "4:q_proj:eq_g16_a0": {
                "joint_conv_error_sum": error,
                "joint_local_error_sum": error * 2,
                "joint_numel": 10,
                "baseline_conv_error_sum": 10.0,
                "baseline_local_error_sum": 20.0,
            }
        },
    }


def test_merge_scaling_eval_rejects_different_candidate_scale_hashes():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        torch.save(_eval_shard(0, "A", sample="a", error=2.0), run_dir / "es_eval_shard0.pt")
        torch.save(_eval_shard(1, "B", sample="b", error=3.0), run_dir / "es_eval_shard1.pt")
        try:
            merge_scaling_eval(run_dir, expected_num_shards=2)
        except ValueError as exc:
            assert "candidate_scales_sha256" in str(exc)
        else:
            raise AssertionError("mixed candidate scale hashes must fail")


def test_merge_scaling_eval_derives_recovery_from_global_raw_sums():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        torch.save(_eval_shard(0, "same", sample="a", error=2.0), run_dir / "es_eval_shard0.pt")
        torch.save(_eval_shard(1, "same", sample="b", error=3.0), run_dir / "es_eval_shard1.pt")
        merged = merge_scaling_eval(run_dir, expected_num_shards=2)
        rec = merged["records"]["4:q_proj:eq_g16_a0"]
        assert rec["joint_conv_error_sum"] == 5.0
        assert rec["baseline_conv_error_sum"] == 20.0
        assert abs(rec["joint_R_Y_conv"] - 0.75) < 1e-12
        assert rec["joint_numel"] == 20
