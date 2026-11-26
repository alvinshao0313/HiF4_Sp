"""Verify full raw coverage and compare all contemporaneous evaluation arms."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import statistics

import numpy as np

from ..ban.evaluation_report import _one, _sha256, aggregate_routing
from .config import ARMS, TARGET, TASKS, TOLERANCE, now, read, save


def require_coverage(ids, reference, name):
    if len(ids) != len(set(ids)) or set(ids) != set(reference):
        raise ValueError(f"incomplete, duplicate or wrong task coverage: {name}")


def protocol(config):
    result = dict(config)
    result.pop("model_name", None)
    additional = dict(result.get("additional_config") or {})
    additional.pop("hif4_runtime_spec_path", None)
    result["additional_config"] = additional
    return result


def summarize(target, name, arm, audit, *, short=False, target_experts=TARGET, budget_tolerance=TOLERANCE):
    import pyarrow.parquet as pq
    import xxhash

    target, definition = Path(target).resolve(), TASKS[name]
    expected_count = 2 if short else definition["count"]
    reference = {row["id"]: row for row in audit["documents"][:expected_count]}
    if len(reference) != expected_count:
        raise ValueError("input audit does not cover the requested full task")
    result_path = _one((target / "evaluation").rglob("results_*.json"), "task results")
    raw = read(result_path)
    tag, metric = definition["task"], definition["metric"]
    stamp = result_path.stem.removeprefix("results_")
    details_path = result_path.parent.parent / "details" / stamp / f"details_{tag}_{stamp}.json"
    rows = read(details_path)
    require_coverage([str(row["doc"]["id"]) for row in rows], reference, name)
    details, per_question = {}, {}
    for row in rows:
        key = str(row["doc"]["id"])
        expected = reference[key]
        if (_sha256(row["model_response"]["input"]) != expected["prompt_sha256"]
                or _sha256(row["gold"]) != expected["gold_sha256"]
                or len(row["model_response"]["text_post_processed"]) != 1):
            raise ValueError(f"prompt/gold/generation mismatch: {name}/{key}")
        value = row["metric"][metric]
        if value not in (0.0, 1.0):
            raise ValueError("single-generation accuracy requires binary question scores")
        details[key], per_question[key] = row, int(value)
    score = raw["results"][tag][metric]
    if not math.isclose(score, sum(per_question.values()) / expected_count, rel_tol=0, abs_tol=1e-12):
        raise ValueError("aggregate accuracy differs from question scores")
    config = raw["config_general"]["model_config"]
    model = Path(config["model_name"])
    cache = _one((p for p in model.rglob("GENERATIVE.parquet") if p.parent.parent.name == tag), "fresh prediction cache")
    cached_ids, hashes = [], []
    # A full MMLU cache can contain hundreds of millions of output tokens.
    # Stream bounded row batches instead of materializing it as Python lists.
    for batch in pq.ParquetFile(cache).iter_batches(batch_size=32, columns=[
            "sample_id", "sample.input", "sample.input_tokens", "sample.text", "sample.output_tokens"]):
        for record in batch.to_pylist():
            key, sample = str(record["sample_id"]), record["sample"]
            if key not in reference:
                raise ValueError("cached sample does not belong to the formal task")
            ids = sample["input_tokens"]
            if (not ids or _sha256(ids) != reference[key]["input_tokens_sha256"]
                    or sample["input"] != details[key]["model_response"]["input"]
                    or len(sample["text"]) != 1 or not sample["text"][0].strip()
                    or len(sample["output_tokens"]) != 1 or not sample["output_tokens"][0]):
                raise ValueError(f"raw cached inputs/responses differ: {name}/{key}")
            cached_ids.append(key)
            hashes.append(xxhash.xxh64(str(ids)).hexdigest())
    require_coverage(cached_ids, reference, name)
    token_hash = xxhash.xxh64("".join(sorted(hashes))).hexdigest()
    if token_hash != raw["summary_tasks"][tag]["hashes"]["hash_input_tokens"]:
        raise ValueError("raw cache tokens differ from the recorded task hash")
    observations = [row for p in (target / "evaluation/generation_observations").glob("batch_*.json") for row in read(p)]
    if (len(observations) != expected_count or any(len(row["outputs"]) != 1 for row in observations)
            or Counter(_sha256(row["prompt_token_ids"]) for row in observations)
            != Counter(row["input_tokens_sha256"] for row in reference.values())):
        raise ValueError("actual vLLM requests do not cover the exact task inputs")
    endings = Counter(row["outputs"][0]["finish_reason"] for row in observations)
    if set(endings) - {"stop", "length"} or any(row["outputs"][0]["output_tokens"] <= 0 for row in observations):
        raise ValueError("incomplete or empty generation requests")
    if arm == "top8":
        routing = {"average_experts": 8.0, "source": "configured unpruned top-8", "measured": False}
    else:
        routing = aggregate_routing(target / "routing")
        if routing["tp_ranks"] != 2 or len(routing["layer_counters"]["tokens"]) != 48:
            raise ValueError("formal Ban routing did not cover TP2 and all 48 layers")
        routing.update(target_average_experts=target_experts, budget_tolerance=budget_tolerance,
                       observed_budget_met=abs(routing["average_experts"] - target_experts) <= budget_tolerance,
                       budget_scope="Lambda matched on held-out validation; formal budget is observed only.")
        routing["per_layer_average_experts"] = [s / n for s, n in zip(
            routing["layer_counters"]["selected"], routing["layer_counters"]["tokens"])]
    return {"status": "complete", "arm": arm, "task": tag, "metric": metric,
            "short_test": short, "count": expected_count, "generations_per_question": 1,
            "correct_count": sum(per_question.values()), "score": score, "score_pct": 100 * score,
            "per_question": per_question, "actual_input_tokens_hash": token_hash,
            "protocol": protocol(config), "routing": routing, "finish_reasons": dict(endings),
            "results_file": str(result_path), "details_file": str(details_path), "prediction_cache": str(cache)}


def clustered_bootstrap(differences, seed=0, samples=10000):
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or not values.size:
        raise ValueError("bootstrap requires [rounds, paired questions]")
    # Each question keeps all its repeated runs together in every resample.
    question_means = values.mean(axis=0)
    generator = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, 100):
        count = min(100, samples - start)
        indices = generator.integers(0, len(question_means), size=(count, len(question_means)))
        means[start:start + count] = question_means[indices].mean(axis=1)
    return (np.quantile(means, [0.025, 0.975]) * 100).tolist()


def build_report(root):
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    state, output = read(root / "status.json"), {"created_at": now(), "budget": manifest["budget"], "tasks": {}, "errors": []}
    for name, definition in TASKS.items():
        arms = {}
        for arm in ARMS:
            rows = []
            for index in range(1, definition["rounds"] + 1):
                label = f"{arm}/{name}/round_{index:02d}"
                phase = state["phases"].get(label, {})
                if phase.get("status") != "complete" or phase.get("exit_code") != 0:
                    output["errors"].append({"phase": label, "status": phase.get("status", "not_run")})
                    continue
                rows.append(read(root / label / "report.json"))
            if rows:
                scores = [row["score"] for row in rows]
                arms[arm] = {"rounds": rows, "completed_rounds": len(rows),
                             "mean_score_pct": 100 * statistics.mean(scores),
                             "min_score_pct": 100 * min(scores), "max_score_pct": 100 * max(scores),
                             "population_std_pp": 100 * statistics.pstdev(scores),
                             "total_correct": sum(row["correct_count"] for row in rows),
                             "answer_count": sum(row["count"] for row in rows)}
                if arm != "top8":
                    arms[arm]["average_experts"] = sum(r["routing"]["selected_count"] for r in rows) / sum(
                        r["routing"]["token_layer_count"] for r in rows)
        task = {"arms": arms}
        if all(a in arms and arms[a]["completed_rounds"] == definition["rounds"] for a in ARMS):
            baseline = arms["top8"]
            if baseline["total_correct"] <= 0:
                raise ValueError("top-8 score is zero; retention is undefined")
            reference = baseline["rounds"][0]
            for group in arms.values():
                for row in group["rounds"]:
                    if (row["protocol"] != reference["protocol"] or row["actual_input_tokens_hash"] != reference["actual_input_tokens_hash"]
                            or set(row["per_question"]) != set(reference["per_question"])):
                        raise ValueError("paired evaluation protocols/inputs differ")
            for arm in ("c4", "trace"):
                group = arms[arm]
                group.update(retention_pct=100 * group["total_correct"] / baseline["total_correct"],
                             observed_98pct_retention_met=100 * group["total_correct"] >= 98 * baseline["total_correct"],
                             delta_top8_pp=group["mean_score_pct"] - baseline["mean_score_pct"])
            keys = sorted(reference["per_question"])
            differences = [[a["per_question"][key] - b["per_question"][key] for key in keys]
                           for a, b in zip(arms["trace"]["rounds"], arms["c4"]["rounds"])]
            task.update(trace_minus_c4_pp=arms["trace"]["mean_score_pct"] - arms["c4"]["mean_score_pct"],
                        trace_minus_c4_paired_95pct_ci_pp=clustered_bootstrap(differences),
                        observed_budget_difference=abs(arms["trace"]["average_experts"] - arms["c4"]["average_experts"]))
            task["equal_test_budget_interpretation_allowed"] = task["observed_budget_difference"] <= 0.10
        output["tasks"][name] = task
    output.update(status="complete" if not output["errors"] else "incomplete",
                  interpretation="Same-round scores; same-seed repeats, not independent seeds. CI resamples questions, preserving repeats; conditional on observed generations. No throughput claim.")
    from ..ban.artifact import load_artifact
    paths = {"c4": Path(manifest["c4_artifact"]), "trace": root / "calibration/formal/c4_ban.pt"}
    output["calibration"] = {}
    for arm, path in paths.items():
        if path.exists():
            artifact = load_artifact(path, manifest["model_identity"])
            output["calibration"][arm] = {"artifact": str(path), "r_min": artifact["r_min"], "r_max": artifact["r_max"],
                 "raw_layer_kl": artifact["raw_layer_kl"].tolist(), "layer_sensitivity": artifact["layer_sensitivity"].tolist()}
    save(root / "report.json", output)
    return output
