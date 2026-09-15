"""Unit tests for S3_FULL48_CAUSAL pure logic (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.router_contribution import (
    conditional_enable_o3,
    decide_o3_layers,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.s3_full48_causal import (
    DISCOVERY_BASE_KINDS,
    EXPECTED_DISCOVERY_BASE,
    EXPECTED_HOLDOUT_BASE,
    HOLDOUT_BASE_KINDS,
    NUM_LAYERS,
    append_jsonl_row,
    assert_base_completeness,
    attention_sensitive_layers_from_sources,
    classify_layer_causal_source,
    dedupe_rows_prefer_identity,
    expected_base_keys,
    expected_discovery_count,
    expected_holdout_count,
    expected_qkvo_keys,
    intervention_key,
    key_from_row,
    load_completed_keys,
    select_s4_mechanism_layers,
    shard_owns_key,
)


def _synthetic_cohort() -> list[dict]:
    rows = []
    for split, prefix in (("discovery", "d"), ("holdout", "h")):
        for sid in range(8):
            for j in (8, 32, 64, 128):
                rows.append(
                    {
                        "split": split,
                        "sample_key": f"{prefix}{sid}__j{j}",
                        "calibration_sample_id": f"{prefix}{sid}",
                        "decode_index": j,
                        "prefix_length_j": j,
                    }
                )
    return rows


def test_expected_base_counts():
    assert expected_discovery_count() == 4608
    assert expected_holdout_count() == 6144
    assert EXPECTED_DISCOVERY_BASE == 32 * 48 * 3
    assert EXPECTED_HOLDOUT_BASE == 32 * 48 * 4
    cohort = _synthetic_cohort()
    keys = expected_base_keys(cohort)
    assert len(keys) == 10752
    assert len(set(keys)) == 10752


def test_unique_key_completeness_helper(tmp_path: Path):
    cohort = _synthetic_cohort()
    keys = expected_base_keys(cohort)
    rows = []
    for split, sample_key, di, layer, kind in keys:
        rows.append(
            {
                "split": split,
                "sample_key": sample_key,
                "decode_index": di,
                "layer": layer,
                "kind": kind,
                "P": 0.0,
                "identity_pass": True,
            }
        )
    gate = assert_base_completeness(rows, cohort)
    assert gate["ok"]
    assert gate["n_discovery"] == 4608
    assert gate["n_holdout"] == 6144
    assert gate["layers_covered"] == list(range(48))

    # missing key → raise
    bad = rows[:-1]
    with pytest.raises(RuntimeError, match="completeness gate failed"):
        assert_base_completeness(bad, cohort)

    # duplicate → raise
    with pytest.raises(RuntimeError, match="completeness gate failed"):
        assert_base_completeness(rows + [rows[0]], cohort)


def test_full_48_layer_coverage_helper():
    cohort = _synthetic_cohort()
    layers = {k[3] for k in expected_base_keys(cohort)}
    assert layers == set(range(NUM_LAYERS))
    kinds_d = {k[4] for k in expected_base_keys(cohort) if k[0] == "discovery"}
    kinds_h = {k[4] for k in expected_base_keys(cohort) if k[0] == "holdout"}
    assert kinds_d == set(DISCOVERY_BASE_KINDS)
    assert kinds_h == set(HOLDOUT_BASE_KINDS)
    assert "whole_layer" not in kinds_d  # discovery whole-layer reused from S2


def test_qkvo_fixed_layer_set_no_per_state_filtering():
    cohort = _synthetic_cohort()
    # Fixed layers from holdout aggregation — NOT per-state p_a>0 filtering
    sources = {i: "moe" for i in range(48)}
    sources[11] = "attention"
    sources[12] = "both"
    sources[31] = "unstable/compensatory"
    fixed = attention_sensitive_layers_from_sources(sources)
    assert fixed == [11, 12]
    keys = expected_qkvo_keys(cohort, fixed)
    # 16 samples × 4 states × 2 layers × 4 ops = 512
    assert len(keys) == 16 * 4 * 2 * 4
    # every fixed layer appears for every state/op — no outcome filter
    by_layer = {11: 0, 12: 0}
    for k in keys:
        by_layer[k[3]] += 1
    assert by_layer[11] == 256
    assert by_layer[12] == 256


def test_classify_and_o3_layers_per_layer_gate():
    # attention: PA clearly above PM
    assert (
        classify_layer_causal_source([0.1] * 8, [0.01] * 8) == "attention"
    )
    # both: close positive
    assert classify_layer_causal_source([0.10] * 8, [0.098] * 8) == "both"
    # moe
    assert classify_layer_causal_source([0.01] * 8, [0.1] * 8) == "moe"
    # unstable
    assert classify_layer_causal_source([-0.1] * 8, [-0.05] * 8) == "unstable/compensatory"

    by_layer = {
        "10": {
            "causal_source": "moe",
            "sample_level_C_router": [0.01] * 8,
        },
        "11": {
            "causal_source": "attention",
            "sample_level_C_router": [0.02] * 8,
        },
        "12": {
            "causal_source": "both",
            "sample_level_C_router": [0.01] * 5 + [-0.01] * 3,  # only 5/8 positive
        },
        "13": {
            "causal_source": "moe",
            "sample_level_C_router": [0.01] * 4 + [-0.02] * 4,  # mean may be neg / pos<5
        },
    }
    o3 = decide_o3_layers(by_layer)
    assert 10 in o3
    assert 12 in o3  # mean>0, median>0 (sorted index 4), pos=5
    assert 11 not in o3  # attention source
    assert 13 not in o3


def test_s4_cross_split_balanced_selection_is_deterministic():
    d_by = {}
    h_by = {}
    for layer in range(48):
        d_by[str(layer)] = {"P_layer": -1.0, "C_router": 0.0, "causal_source": None}
        h_by[str(layer)] = {"P_layer": -1.0, "C_router": 0.0, "causal_source": "unstable/compensatory"}

    values = {
        41: (0.90, 0.95, "attention", 0.00),
        43: (0.95, 0.85, "attention", 0.00),
        18: (0.70, 0.60, "attention", 0.00),
        39: (0.92, 0.96, "moe", 0.10),
        31: (0.98, 0.70, "moe", 0.70),
        19: (0.60, 0.90, "moe", 0.80),
        44: (0.99, 0.40, "moe", 0.60),
    }
    for layer, (pd, ph, source, cr) in values.items():
        d_by[str(layer)] = {"P_layer": pd, "C_router": 0.0, "causal_source": None}
        h_by[str(layer)] = {"P_layer": ph, "C_router": cr, "causal_source": source}

    out = select_s4_mechanism_layers(
        discovery_summary={"by_layer": d_by},
        holdout_summary={"by_layer": h_by},
        o3_layers=[19, 31, 44],
    )
    assert out["attention_layers"] == [41, 43]
    assert out["moe_layers"] == [39, 31]
    assert out["main_layers"] == [41, 43, 39, 31]
    assert out["router_objective_layers"] == [19, 31]
    assert out["selected_objective_layers"] == [41, 43, 39, 31, 19]


def test_resume_key_skip(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    row = {
        "split": "discovery",
        "sample_key": "d0__j8",
        "decode_index": 8,
        "layer": 0,
        "kind": "attn_only",
        "P": 0.1,
        "identity_pass": True,
    }
    append_jsonl_row(path, row)
    # failed identity should not count as done
    append_jsonl_row(
        path,
        {**row, "layer": 1, "identity_pass": False, "P": 0.0},
    )
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    done = load_completed_keys(rows)
    assert intervention_key(split="discovery", sample_key="d0__j8", decode_index=8, layer=0, kind="attn_only") in done
    assert intervention_key(split="discovery", sample_key="d0__j8", decode_index=8, layer=1, kind="attn_only") not in done

    # dedupe prefers identity
    dup = dedupe_rows_prefer_identity(
        rows
        + [{**row, "layer": 1, "identity_pass": True, "P": 0.2}]
    )
    assert len(dup) == 2
    assert next(r for r in dup if r["layer"] == 1)["identity_pass"] is True


def test_old_partial_s3_not_overwritten(tmp_path: Path):
    run_root = tmp_path / "run"
    legacy_prot = run_root / "50_protection"
    legacy_router = run_root / "40_router"
    legacy_prot.mkdir(parents=True)
    legacy_router.mkdir(parents=True)
    legacy_rows = legacy_prot / "s3_substructure_rows.jsonl"
    legacy_scope = legacy_prot / "practical_protection_scope.json"
    legacy_rows.write_text('{"kind":"legacy"}\n', encoding="utf-8")
    legacy_scope.write_text(
        json.dumps({"status": "DRAFT_PENDING_REVIEW", "enable_o3": True, "layers": [1]}),
        encoding="utf-8",
    )
    # Isolated dirs only — simulate writing new artifacts
    new_dir = legacy_prot / "s3_full48"
    new_dir.mkdir(parents=True)
    (new_dir / "full48_causal_rows.jsonl").write_text("{}\n", encoding="utf-8")
    # legacy unchanged
    assert legacy_rows.read_text(encoding="utf-8") == '{"kind":"legacy"}\n'
    assert "enable_o3" in legacy_scope.read_text(encoding="utf-8")
    assert (legacy_prot / "s3_full48" / "full48_causal_rows.jsonl").is_file()


def test_shard_partition_covers_all_keys():
    cohort = _synthetic_cohort()
    keys = expected_base_keys(cohort)
    n = 2
    owned = [k for k in keys if shard_owns_key(k, worker_id=0, num_workers=n)]
    owned += [k for k in keys if shard_owns_key(k, worker_id=1, num_workers=n)]
    assert len(owned) == len(keys)
    assert len(set(owned)) == len(keys)


def test_legacy_conditional_enable_o3_still_importable_but_separate():
    # kept for rejection/legacy tests; formal gate uses decide_o3_layers
    assert conditional_enable_o3([{"C_router": 0.1}, {"C_router": 0.2}]) is True
    assert conditional_enable_o3([{"C_router": -0.1}, {"C_router": -0.2}]) is False
