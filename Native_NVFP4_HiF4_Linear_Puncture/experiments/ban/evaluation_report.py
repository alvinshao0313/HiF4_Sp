"""Paired GSM8K accuracy and measured Ban routing budgets, without running models."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


TASK = "gsm8k|0"
METRIC = "extractive_match"
RETENTION_THRESHOLD = 0.98
BUDGET_TARGET = 6.0
BUDGET_TOLERANCE = 0.10


def _sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _one(paths, label):
    paths = sorted(paths)
    if len(paths) != 1:
        raise ValueError(f"expected exactly one {label}, found {len(paths)}: {paths}")
    return paths[0]


def _fraction(value, label):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{label} must be a finite fraction in [0, 1]")
    return float(value)


def _actual_tokens(model_path, details, expected_hash, task=TASK):
    # main.py's slim details omit input_tokens. The prediction cache retains
    # vLLM's actual prompt_token_ids, so re-tokenizing displayed text is unneeded.
    import pyarrow.parquet as pq
    import xxhash

    model = Path(model_path)
    cache = _one((path for path in model.rglob("GENERATIVE.parquet")
                  if path.parent.parent.name == task), f"{task} prediction cache")
    rows = pq.read_table(cache, columns=["sample_id", "sample.input", "sample.input_tokens"]).to_pylist()
    inputs, hashes = {}, []
    for row in rows:
        # Arrow projects nested struct fields into columns named input/input_tokens.
        sample_id, sample = str(row["sample_id"]), row
        if sample_id in inputs:
            raise ValueError(f"duplicate cached GSM8K sample: {sample_id}")
        tokens = sample["input_tokens"]
        if not tokens or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens):
            raise ValueError(f"invalid actual input tokens for GSM8K sample {sample_id}")
        inputs[sample_id] = sample
        hashes.append(xxhash.xxh64(str(tokens)).hexdigest())
    if set(inputs) != set(details):
        raise ValueError("GSM8K cache and details sample IDs differ")
    actual_hash = xxhash.xxh64("".join(sorted(hashes))).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("actual cached input tokens differ from results summary_tasks token hash")
    for sample_id, row in details.items():
        if inputs[sample_id]["input"] != row["model_response"]["input"]:
            raise ValueError(f"GSM8K cache/detail prompt differs for {sample_id}")
    return {sample_id: _sha256(sample["input_tokens"]) for sample_id, sample in inputs.items()}, cache


def summarize_evaluation(run_dir: Path, *, expected_count=1319) -> dict:
    """Read one completed main.py evaluation, including actual cached input IDs."""
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    root = Path(run_dir)
    results_file = _one(root.rglob("results_*.json"), "main.py results file")
    raw = _read_json(results_file)
    score = _fraction(raw["results"][TASK][METRIC], "GSM8K score")
    date_id = results_file.stem.removeprefix("results_")
    details_file = results_file.parent.parent / "details" / date_id / f"details_{TASK}_{date_id}.json"
    rows = _read_json(details_file)
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"GSM8K details must contain {expected_count} samples")
    details = {}
    for row in rows:
        sample_id = row["doc"]["id"]
        if sample_id is None or str(sample_id) in details:
            raise ValueError("missing or duplicate GSM8K sample ID")
        sample_id = str(sample_id)
        prompt, gold = row["model_response"]["input"], row["gold"]
        if not isinstance(prompt, str) or not prompt or not isinstance(gold, list) or not gold:
            raise ValueError(f"missing GSM8K prompt or gold for {sample_id}")
        generations = row["model_response"]["text_post_processed"]
        if not isinstance(generations, list) or len(generations) != 1:
            raise ValueError("GSM8K comparison requires one generation per sample")
        correct = _fraction(row["metric"][METRIC], f"sample {sample_id} correctness")
        if correct not in (0.0, 1.0):
            raise ValueError("GSM8K exact correct counts require binary per-sample scores")
        details[sample_id] = row
    correct_count = sum(int(row["metric"][METRIC]) for row in details.values())
    if not math.isclose(score, correct_count / expected_count, rel_tol=0, abs_tol=1e-12):
        raise ValueError("GSM8K aggregate score differs from exact per-sample correct count")
    config = raw["config_general"]["model_config"]
    token_hash = raw["summary_tasks"][TASK]["hashes"]["hash_input_tokens"]
    if not isinstance(token_hash, str) or not token_hash:
        raise ValueError("missing actual lighteval input-token hash")
    tokens, cache = _actual_tokens(config["model_name"], details, token_hash)
    samples = [{"id": sample_id, "correct": int(row["metric"][METRIC]),
                "prompt_sha256": _sha256(row["model_response"]["input"]),
                "gold_sha256": _sha256(row["gold"]), "input_tokens_sha256": tokens[sample_id]}
               for sample_id, row in sorted(details.items())]
    config_keys = ("generation_parameters", "seed", "tensor_parallel_size", "data_parallel_size",
                   "pipeline_parallel_size", "max_model_length", "max_num_batched_tokens",
                   "max_num_seqs", "kv_cache_dtype", "enforce_eager", "enable_thinking",
                   "add_special_tokens", "override_chat_template", "system_prompt")
    return {"task": TASK, "metric": METRIC, "count": expected_count,
            "correct_count": correct_count, "score": score,
            "standard_error": raw["results"][TASK].get(f"{METRIC}_stderr"),
            "results_file": str(results_file), "details_file": str(details_file),
            "prediction_cache": str(cache), "run_dir": str(root),
            "score_source": f"results[{TASK!r}][{METRIC!r}]",
            "correctness_source": f"details[*].metric[{METRIC!r}]",
            "prompt_source": "details[*].model_response.input",
            "gold_source": "details[*].gold",
            "token_source": "GENERATIVE.parquet: sample.input_tokens (actual vLLM prompt_token_ids)",
            "lighteval_input_tokens_hash": token_hash,
            "token_hash_source": f"summary_tasks[{TASK!r}].hashes.hash_input_tokens",
            "protocol": {key: config.get(key) for key in config_keys},
            "samples": samples}


def _nonnegative_ints(values, label):
    if not isinstance(values, list) or not values or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError(f"{label} must contain nonnegative integer counts")
    return values


def aggregate_routing(stats_dir: Path) -> dict:
    """Pool batch counters from engine.snapshot; TP ranks are already deduplicated."""
    files = sorted(Path(stats_dir).glob("routing_*.json"))
    if not files:
        raise FileNotFoundError(f"no routing_*.json under {stats_dir}")
    totals, execution, ranks = None, None, None
    for path in files:
        row = _read_json(path)
        current_execution, current_ranks = row["execution"], row["tp_ranks"]
        if current_execution not in ("compact", "masked"):
            raise ValueError("unknown Ban routing execution")
        if not isinstance(current_ranks, int) or isinstance(current_ranks, bool) or current_ranks <= 0:
            raise ValueError("invalid TP rank count")
        if totals is None:
            execution, ranks = current_execution, current_ranks
        if (current_execution, current_ranks) != (execution, ranks):
            raise ValueError("routing files mix execution modes or TP sizes")
        counts = {key: _nonnegative_ints(row[key], key) for key in ("tokens", "selected", "dispatched")}
        if len({len(values) for values in counts.values()}) != 1:
            raise ValueError("routing layer counter lengths differ")
        hist = row["histogram"]
        if len(hist) != len(counts["tokens"]):
            raise ValueError("routing histogram layer count differs")
        for index, bins in enumerate(hist):
            bins = _nonnegative_ints(bins, "histogram")
            if (sum(bins) != counts["tokens"][index]
                    or sum(k * count for k, count in enumerate(bins)) != counts["selected"][index]):
                raise ValueError("routing histogram differs from token/selection counters")
        if execution == "compact" and counts["selected"] != counts["dispatched"]:
            raise ValueError("compact Ban selected experts differ from dispatched experts")
        if totals is None:
            totals = {key: [0] * len(values) for key, values in counts.items()}
        if any(len(counts[key]) != len(totals[key]) for key in counts):
            raise ValueError("routing files have different layer counts")
        for key in totals:
            totals[key] = [a + b for a, b in zip(totals[key], counts[key])]
    if not all(totals["tokens"]):
        raise ValueError("Ban routing was not observed at every layer")
    token_count = sum(totals["tokens"])
    selected, dispatched = sum(totals["selected"]), sum(totals["dispatched"])
    average = selected / token_count
    return {"stats_dir": str(stats_dir), "files": [str(path) for path in files],
            "execution": execution, "tp_ranks": ranks,
            "tp_aggregation": "engine.snapshot checks rank equality and stores one rank; ranks are not summed",
            "layer_counters": totals, "token_layer_count": token_count,
            "selected_count": selected, "dispatched_count": dispatched,
            "average_experts": average, "average_dispatched_experts": dispatched / token_count,
            "target_average_experts": BUDGET_TARGET, "budget_tolerance": BUDGET_TOLERANCE,
            "observed_budget_met": abs(average - BUDGET_TARGET) <= BUDGET_TOLERANCE + 1e-12}


def compare_evaluations(evaluations: dict[str, dict]) -> dict:
    """Compare observed paired accuracies to top8; 98% is a relative threshold."""
    if "top8" not in evaluations or len(evaluations) < 2:
        raise ValueError("comparison requires top8 and at least one pruned evaluation")
    baseline = evaluations["top8"]
    if baseline["correct_count"] <= 0:
        raise ValueError("relative retention is undefined for a zero-score top8 baseline")
    identity_keys = ("id", "prompt_sha256", "gold_sha256", "input_tokens_sha256")
    reference = [{key: sample[key] for key in identity_keys} for sample in baseline["samples"]]
    comparisons = {}
    for name, summary in evaluations.items():
        if summary["task"] != TASK or summary["metric"] != METRIC:
            raise ValueError(f"unexpected task or scoring metric for {name}")
        if (summary["count"] != baseline["count"]
                or [{key: sample[key] for key in identity_keys} for sample in summary["samples"]] != reference
                or summary["lighteval_input_tokens_hash"] != baseline["lighteval_input_tokens_hash"]):
            raise ValueError(f"sample IDs, prompts, actual input tokens or gold do not align for {name}")
        if summary["protocol"] != baseline["protocol"]:
            raise ValueError(f"evaluation protocol differs from top8 for {name}")
        correct = summary["correct_count"]
        # Integer comparison avoids a floating-point threshold deciding a one-question boundary.
        retained = 100 * correct >= 98 * baseline["correct_count"]
        base_correct = [sample["correct"] for sample in baseline["samples"]]
        current_correct = [sample["correct"] for sample in summary["samples"]]
        comparisons[name] = {"score": summary["score"], "score_pct": 100 * summary["score"],
                             "count": summary["count"], "correct_count": correct,
                             "relative_retention": correct / baseline["correct_count"],
                             "drop_pp": 100 * (baseline["score"] - summary["score"]),
                             "retention_threshold": RETENTION_THRESHOLD,
                             "observed_retention_met": retained,
                             "lost_correct": sum(a == 1 and b == 0 for a, b in zip(base_correct, current_correct)),
                             "gained_correct": sum(a == 0 and b == 1 for a, b in zip(base_correct, current_correct))}
        if "routing" in summary:
            comparisons[name]["routing"] = summary["routing"]
            comparisons[name]["observed_retention_and_budget_met"] = (
                retained and summary["routing"]["observed_budget_met"])
    return {"baseline": "top8", "task": TASK, "metric": METRIC,
            "paired_input_alignment": True, "retention_threshold": RETENTION_THRESHOLD,
            "definition": "score_pruned / score_top8 >= 0.98; same-round GSM8K observations",
            "inference_limit": "Observed scores do not establish a 98% statistical lower bound or other-task retention.",
            "evaluations": comparisons}
