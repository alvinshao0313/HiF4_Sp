"""Artifact-backed reporting checks; no model or GPU initialization."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xxhash

from ..evaluation_report import aggregate_routing, compare_evaluations, summarize_evaluation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def evaluation(root, correctness=(1, 1, 0), *, token_offset=0, prompt_suffix="", gold_suffix=""):
    stamp, model = "2026-10-23T01-00-00", root / "model"
    rows, cached = [], []
    for index, correct in enumerate(correctness):
        prompt = f"question {index}{prompt_suffix}"
        rows.append({"doc": {"id": str(index)}, "metric": {"extractive_match": correct},
                     "model_response": {"input": prompt, "text_post_processed": ["answer"]},
                     "gold": [f"gold {index}{gold_suffix}"]})
        cached.append({"sample_id": str(index), "sample": {"input": prompt, "input_tokens": [index + token_offset + 1]}})
    token_hash = xxhash.xxh64("".join(sorted(xxhash.xxh64(str(row["sample"]["input_tokens"])).hexdigest()
                                              for row in cached))).hexdigest()
    result = {"results": {"gsm8k|0": {"extractive_match": sum(correctness) / len(correctness),
                                     "extractive_match_stderr": 0.1}},
              "config_general": {"model_config": {"model_name": str(model), "seed": 1234,
                                                    "generation_parameters": {"temperature": 0.6}}},
              "summary_tasks": {"gsm8k|0": {"hashes": {"hash_input_tokens": token_hash}}}}
    write(root / "output" / "results" / f"results_{stamp}.json", result)
    write(root / "output" / "details" / stamp / f"details_gsm8k|0_{stamp}.json", rows)
    cache = model / "model-hash" / "gsm8k|0" / "task-hash" / "GENERATIVE.parquet"
    cache.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(cached), cache)
    return summarize_evaluation(root, expected_count=len(correctness))


def routing(root, name, token_counts, expert_counts, *, tp=2, execution="compact", dispatched=None):
    histograms = []
    for tokens, experts in zip(token_counts, expert_counts):
        bins = [0] * 9
        bins[experts] = tokens
        histograms.append(bins)
    selected = [tokens * experts for tokens, experts in zip(token_counts, expert_counts)]
    write(root / name, {"execution": execution, "tp_ranks": tp,
                        "tokens": token_counts, "selected": selected,
                        "dispatched": selected if dispatched is None else dispatched,
                        "histogram": histograms})


def test_exact_counts_pairing_and_relative_retention(tmp_path):
    baseline = evaluation(tmp_path / "top8", (1,) * 50 + (0,))
    candidate = evaluation(tmp_path / "top6", (1,) * 49 + (0, 0))
    result = compare_evaluations({"top8": baseline, "top6": candidate})["evaluations"]["top6"]
    assert result["correct_count"] == 49
    assert result["relative_retention"] == 0.98
    assert result["observed_retention_met"] is True
    assert result["drop_pp"] == pytest.approx(100 / 51)
    assert result["lost_correct"] == 1
    assert result["gained_correct"] == 0
    assert baseline["token_source"].endswith("(actual vLLM prompt_token_ids)")


def test_below_threshold_is_not_rescued_by_standard_error(tmp_path):
    baseline = evaluation(tmp_path / "top8", (1,) * 50)
    candidate = evaluation(tmp_path / "top6", (1,) * 48 + (0, 0))
    result = compare_evaluations({"top8": baseline, "top6": candidate})
    assert result["evaluations"]["top6"]["observed_retention_met"] is False


@pytest.mark.parametrize("difference", [{"token_offset": 10}, {"prompt_suffix": "changed"}, {"gold_suffix": "changed"}])
def test_rejects_unpaired_actual_inputs_or_gold(tmp_path, difference):
    base = evaluation(tmp_path / "top8")
    candidate = evaluation(tmp_path / "Ban", **difference)
    with pytest.raises(ValueError, match="do not align"):
        compare_evaluations({"top8": base, "Ban": candidate})


def test_rejects_incomplete_coverage(tmp_path):
    evaluation(tmp_path)
    with pytest.raises(ValueError, match="1319 samples"):
        summarize_evaluation(tmp_path)


def test_rejects_score_count_disagreement(tmp_path):
    summary = evaluation(tmp_path)
    path = Path(summary["results_file"])
    data = json.loads(path.read_text())
    data["results"]["gsm8k|0"]["extractive_match"] = 0.99
    write(path, data)
    with pytest.raises(ValueError, match="exact per-sample"):
        summarize_evaluation(tmp_path, expected_count=3)


def test_rejects_token_hash_not_matching_actual_cache(tmp_path):
    summary = evaluation(tmp_path)
    path = Path(summary["results_file"])
    data = json.loads(path.read_text())
    data["summary_tasks"]["gsm8k|0"]["hashes"]["hash_input_tokens"] = "stale"
    write(path, data)
    with pytest.raises(ValueError, match="actual cached input tokens"):
        summarize_evaluation(tmp_path, expected_count=3)


def test_rejects_multiple_result_rounds(tmp_path):
    summary = evaluation(tmp_path)
    write(Path(summary["results_file"]).with_name("results_other.json"), {})
    with pytest.raises(ValueError, match="exactly one main.py results"):
        summarize_evaluation(tmp_path, expected_count=3)


def test_weighted_routing_counters_do_not_sum_tp_ranks(tmp_path):
    routing(tmp_path, "routing_0000.json", [1, 1], [8, 8])
    routing(tmp_path, "routing_0001.json", [3, 3], [4, 4])
    result = aggregate_routing(tmp_path)
    assert result["average_experts"] == 5.0  # (16 + 24) / 8, not mean(8, 4).
    assert result["token_layer_count"] == 8
    assert result["selected_count"] == result["dispatched_count"] == 40
    assert result["tp_ranks"] == 2
    assert result["observed_budget_met"] is False


def test_measured_budget_and_accuracy_are_separate(tmp_path):
    routing(tmp_path / "routing", "routing_0000.json", [1, 1], [6, 6])
    baseline, candidate = evaluation(tmp_path / "top8"), evaluation(tmp_path / "Ban")
    candidate["routing"] = aggregate_routing(tmp_path / "routing")
    row = compare_evaluations({"top8": baseline, "Ban": candidate})["evaluations"]["Ban"]
    assert row["routing"]["observed_budget_met"] is True
    assert row["observed_retention_and_budget_met"] is True


def test_compact_dispatch_disagreement_is_not_hidden(tmp_path):
    routing(tmp_path, "routing_0000.json", [1, 1], [6, 6], dispatched=[8, 8])
    with pytest.raises(ValueError, match="selected experts differ"):
        aggregate_routing(tmp_path)


def test_masked_reports_actual_unpruned_dispatch(tmp_path):
    routing(tmp_path, "routing_0000.json", [1, 1], [6, 6], execution="masked", dispatched=[8, 8])
    result = aggregate_routing(tmp_path)
    assert result["average_experts"] == 6
    assert result["average_dispatched_experts"] == 8
