from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.input_mask_proxy_study.artifacts import (  # noqa: E402
    atomic_write_json,
    write_run_artifacts,
)
from Block_Sparse.input_mask_proxy_study.report import (  # noqa: E402
    build_aggregate_summary,
    render_report,
    select_winners,
)


def test_atomic_json(tmp_path: Path):
    p = tmp_path / "a.json"
    atomic_write_json(p, {"x": 1})
    assert json.loads(p.read_text())["x"] == 1


def test_write_artifacts_and_winners(tmp_path: Path):
    condition = []
    latency = []
    methods = [
        "full_exact_ref",
        "xproxy_exact_own_output",
        "xproxy_energy_own_output",
        "full_energy_ref_output",
        "xwproxy_exact_ref_output",
        "xwproxy_exact_own_output",
        "xproxy_s0mean_energy_own_output",
        "xproxy_energy_unconditioned_own_output",
    ]
    for mid in methods:
        condition.append(
            {
                "method_id": mid,
                "output_keep_ratio": 0.5,
                "input_keep_ratio": 0.5,
                "output_overlap_to_ref_mean": 1.0,
                "output_overlap_to_ref_median": 1.0,
                "output_overlap_to_ref_p10": 1.0,
                "output_overlap_to_ref_p90": 1.0,
                "input_overlap_to_m1_mean": 0.9 if mid != "full_exact_ref" else 1.0,
                "input_overlap_to_m1_median": 0.9 if mid != "full_exact_ref" else 1.0,
                "input_overlap_to_m1_p10": 0.9,
                "input_overlap_to_m1_p90": 0.9,
                "input_overlap_to_conditional_oracle_mean": 0.9,
                "input_overlap_to_conditional_oracle_median": 0.9,
                "real_output_nrmse_mean": 0.1,
                "real_output_nrmse_median": 0.1,
                "nrmse_regret_vs_m1_mean": 0.0,
                "nrmse_regret_vs_m1_median": 0.0,
                "joint_sparse_output_nrmse_mean": 0.2,
                "joint_sparse_output_nrmse_median": 0.2,
                "num_output_blocks_kept": 1,
                "num_input_blocks_kept": 1,
                "num_compute_blocks": 1,
                "compute_block_ratio": 0.1,
                "spearman_mean": 1.0,
                "kendall_mean": 1.0,
            }
        )
        for scope in (
            "activation_proxy_build_ms",
            "output_generation_ms",
            "input_recovery_ms",
            "online_total_ms",
        ):
            latency.append(
                {
                    "method_id": mid,
                    "output_keep_ratio": 0.5,
                    "input_keep_ratio": 0.5,
                    "timing_scope": scope,
                    "median_ms": 0.0
                    if (
                        scope == "activation_proxy_build_ms"
                        and mid in {"full_exact_ref", "full_energy_ref_output"}
                    )
                    else 1.0 + hash(mid + scope) % 5,
                    "p10_ms": 1.0,
                    "p90_ms": 2.0,
                    "repeats": 3,
                    "peak_memory_bytes": 1,
                }
            )
        if mid in {
            "xproxy_energy_own_output",
            "full_energy_ref_output",
            "xproxy_s0mean_energy_own_output",
            "xproxy_energy_unconditioned_own_output",
        }:
            latency.append(
                {
                    "method_id": mid,
                    "output_keep_ratio": 0.5,
                    "input_keep_ratio": 0.5,
                    "timing_scope": "activation_statistic_ms",
                    "median_ms": 0.5 if mid != "xproxy_s0mean_energy_own_output" else 0.1,
                    "p10_ms": 0.1,
                    "p90_ms": 1.0,
                    "repeats": 3,
                    "peak_memory_bytes": 1,
                }
            )
    for mid, scope in [
        ("xwproxy_exact_ref_output", "weight_proxy_offline_ms"),
        ("xwproxy_exact_own_output", "weight_proxy_offline_ms"),
        ("xproxy_energy_own_output", "weight_energy_offline_ms"),
        ("full_energy_ref_output", "weight_energy_offline_ms"),
        ("xproxy_s0mean_energy_own_output", "weight_energy_offline_ms"),
    ]:
        latency.append(
            {
                "method_id": mid,
                "output_keep_ratio": "",
                "input_keep_ratio": "",
                "timing_scope": scope,
                "median_ms": 3.0,
                "p10_ms": 2.0,
                "p90_ms": 4.0,
                "repeats": 3,
                "peak_memory_bytes": 1,
            }
        )

    # smoke cardinality: 8*1*4 + 4*1 + 5 = 41
    assert len([r for r in latency if r["timing_scope"] in {
        "activation_proxy_build_ms",
        "output_generation_ms",
        "input_recovery_ms",
        "online_total_ms",
    } and r["output_keep_ratio"] != ""]) == 32
    assert len([r for r in latency if r["timing_scope"] == "activation_statistic_ms"]) == 4
    assert len([r for r in latency if r["timing_scope"].endswith("_offline_ms")]) == 5
    assert len(latency) == 41

    winners = select_winners(condition, latency)
    assert winners["candidate_mask_fidelity_winner"] in methods
    m7_vs_m3 = {
        "input_mask_overlap_mean": 0.9,
        "input_mask_overlap_median": 0.9,
        "input_mask_iou_mean": 0.85,
        "real_output_nrmse_delta_mean": 0.01,
        "input_recovery_speedup": 2.0,
        "activation_statistic_speedup": 5.0,
        "online_total_speedup": 1.05,
        "s0mean_vs_xp_energy_spearman": 0.8,
        "s0mean_vs_xp_energy_pearson": 0.75,
    }
    m8_vs_m3 = {
        "input_mask_overlap_mean": 0.92,
        "input_mask_overlap_median": 0.93,
        "input_mask_iou_mean": 0.88,
        "overlap_to_m1_delta_mean": -0.02,
        "conditional_overlap_delta_mean": -0.02,
        "real_output_nrmse_delta_mean": 0.002,
        "input_recovery_speedup": 1.2,
        "by_output_keep_ratio": {},
        "by_input_keep_ratio": {},
    }
    agg = build_aggregate_summary(
        condition, latency, m7_vs_m3=m7_vs_m3, m8_vs_m3=m8_vs_m3
    )
    assert "m7_vs_m3" in agg
    assert "m8_vs_m3" in agg
    report = render_report(
        aggregate=agg,
        condition_summary=condition,
        manifest={"module_name": "x", "activation_shape": [1], "weight_shape": [1]},
    )
    assert "八种方法" in report
    assert "MY-conditioning ablation" in report
    assert "xproxy_energy_unconditioned_own_output" in report
    out = tmp_path / "run"
    write_run_artifacts(
        out,
        manifest={"ok": True},
        config={"a": 1},
        masks={"x": 1},
        per_block_metrics=[{"i": 0}],
        condition_summary=condition,
        latency_rows=latency,
        aggregate_summary=agg,
        report_md=report,
    )
    assert (out / "report.md").is_file()
    assert (out / "aggregate_summary.json").is_file()
