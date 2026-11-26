"""Check actual TP2 execution, request identities, and all five response scores."""
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
from statistics import mean, variance

from ..qwen3_mxfp4_rotation_eval.report import one
from .common import REFERENCE, TASKS, VARIANTS, digest, file_hash, read, save


def runtime_check(root, variant, target, task, short):
    expected = read(Path(root) / "expected_runtime.json")[variant]
    ranks = {}
    for path in (target / "audit").glob("worker-*.jsonl"):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        experts = [row for row in events if row["event"] == "loaded_experts"]
        if not experts:
            continue
        rank_set = {str(row["tp_rank"]) for row in experts}
        if len(rank_set) != 1 or len(experts) != 48:
            raise ValueError("Incomplete expert layer coverage")
        rank = rank_set.pop()
        if rank in ranks or rank not in expected["ranks"]:
            raise ValueError("Duplicate or unexpected TP rank")
        if ({row["prefix"]: row["tensors"] for row in experts} != expected["ranks"][rank]
                or any(row["variant"] != variant or row["method"] != "Qwen3ModelOptMXFP4FusedMoE" for row in experts)):
            raise ValueError("Expert methods or TP2 packed weights/scales differ")
        methods = {row["prefix"]: row["method"] for row in events if row["event"] == "non_expert_method"}
        if methods != expected["non_expert_methods"]:
            raise ValueError("Non-expert method changed")
        ranks[rank] = 48
    if set(ranks) != {"0", "1"}:
        raise ValueError("Both TP2 ranks must be present")
    reference = read(read(REFERENCE / variant / "formal/status.json")["results_file"])
    expected_config = deepcopy(reference["config_general"]["model_config"])
    expected_config.pop("model_name")
    expected_config["generation_parameters"]["max_new_tokens"] = 64 if short else TASKS[task]["max_new_tokens"]
    actual = read(one((target / "evaluation").rglob("results_*.json")))["config_general"]["model_config"]
    actual.pop("model_name")
    if actual != expected_config or actual["kv_cache_dtype"] != "bfloat16":
        changed = {key: (expected_config.get(key), actual.get(key)) for key in set(expected_config) | set(actual)
                   if expected_config.get(key) != actual.get(key)}
        raise ValueError(f"Unexpected inference configuration changes: {changed}")
    report = {"status": "passed", "ranks": ranks, "expert_slices_exact": True,
              "non_expert_methods_equal": True, "protocol": actual,
              "reference": str(REFERENCE / variant / "formal/runtime_validation.json")}
    save(target / "runtime_validation.json", report)
    return report


def summarize(root, target, task, count, short):
    import pyarrow.parquet as pq

    target = Path(target)
    expected = read(Path(root) / f"{task}_inputs.json")["documents"][:count]
    reference = {row["id"]: row for row in expected}
    raw_path = one((target / "evaluation").rglob("results_*.json"))
    details_path = one((target / "evaluation").rglob("details_*.json"))
    raw, details = read(raw_path), read(details_path)
    if len(details) != count or {str(row["doc"]["id"]) for row in details} != set(reference):
        raise ValueError("Incomplete or different question coverage")
    scores = {}
    for row in details:
        key = str(row["doc"]["id"])
        values = row["doc"]["specific"]["combination_eval_scoring"]["scores"]
        if (len(values) != 5 or any(value not in (0, 1) for value in values)
                or abs(mean(values) - row["metric"]["accuracy_avg5"]) > 1e-12
                or digest(row["model_response"]["input"]) != reference[key]["prompt_sha256"]
                or digest(row["gold"]) != reference[key]["gold_sha256"]):
            raise ValueError("Individual score, prompt, or gold differs from the frozen protocol")
        scores[key] = values
    score = mean(mean(values) for values in scores.values())
    if abs(score - raw["results"][TASKS[task]["tag"]]["accuracy_avg5"]) > 1e-12:
        raise ValueError("Aggregate differs from recorded per-response scores")
    cache = one((target / "model").rglob("GENERATIVE.parquet"))
    token_key = "probe_input_tokens_sha256" if short else "input_tokens_sha256"
    seen, output_tokens = [], 0
    for batch in pq.ParquetFile(cache).iter_batches(batch_size=8, columns=[
            "sample_id", "sample.input", "sample.input_tokens", "sample.text", "sample.output_tokens"]):
        for item in batch.to_pylist():
            key, sample = str(item["sample_id"]), item["sample"]
            if (key not in reference or digest(sample["input_tokens"]) != reference[key][token_key]
                    or digest(sample["input"]) != reference[key]["prompt_sha256"]
                    or len(sample["text"]) != 5 or len(sample["output_tokens"]) != 5
                    or any(not text.strip() for text in sample["text"])):
                raise ValueError("Raw cache has different inputs or incomplete response coverage")
            seen.append(key)
            output_tokens += sum(len(tokens) for tokens in sample["output_tokens"])
    if len(seen) != count or set(seen) != set(reference):
        raise ValueError("Raw cache question IDs differ")
    observations = [row for path in (target / "evaluation/generation_observations").glob("batch_*.json") for row in read(path)]
    if (len(observations) != count or any(len(row["outputs"]) != 5 for row in observations)
            or Counter(digest(row["prompt_token_ids"]) for row in observations)
            != Counter(row[token_key] for row in reference.values())):
        raise ValueError("Actual requests have different inputs or output counts")
    endings = Counter(item["finish_reason"] for row in observations for item in row["outputs"])
    if set(endings) - {"stop", "length"}:
        raise ValueError("Unexpected generation finish reason")
    config = dict(raw["config_general"]["model_config"])
    config.pop("model_name", None)
    counts = [int(sum(values[index] for values in scores.values())) for index in range(5)]
    rates = [100 * value / count for value in counts]
    report = {"status": "completed", "task": task, "short_test": short,
        "questions": count, "generations": count * 5, "correct": sum(counts), "accuracy": score,
        "accuracy_percent": score * 100, "correct_by_sampling_index": counts,
        "accuracy_percent_by_sampling_index": rates, "sample_variance_pp_squared": variance(rates),
        "variance_definition": "Five sampling indices over the same questions, ddof=1; not independent seeds",
        "per_question_scores": scores, "finish_reasons": dict(endings), "output_tokens": output_tokens,
        "truncation_rate": endings["length"] / (count * 5), "input_sha256": digest(expected), "protocol": config,
        "results": str(raw_path), "details": str(details_path), "cache": str(cache),
        "source_sha256": {str(path): file_hash(path) for path in (raw_path, details_path, cache)}}
    save(target / "summary.json", report)
    with (target / "sampling_accuracy.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sampling_index", "correct", "questions", "accuracy_percent"])
        writer.writerows((i + 1, n, count, rates[i]) for i, n in enumerate(counts))
    return report


def compare(root):
    root = Path(root)
    output = {"status": "completed", "metric": "mean single-response accuracy", "tasks": {}}
    for task in TASKS:
        arms = {variant: read(root / variant / "formal" / task / "summary.json") for variant in VARIANTS}
        first = arms[VARIANTS[0]]
        if any(row["input_sha256"] != first["input_sha256"] or row["protocol"] != first["protocol"] for row in arms.values()):
            raise ValueError("Comparison inputs or inference protocols differ")
        output["tasks"][task] = arms
    save(root / "comparison.json", output)
    with (root / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "variant", "mean_accuracy_percent", "sample_variance_pp_squared", *[f"sampling_{i}" for i in range(1, 6)]])
        for task, arms in output["tasks"].items():
            for variant, row in arms.items():
                writer.writerow([task, variant, row["accuracy_percent"], row["sample_variance_pp_squared"], *row["accuracy_percent_by_sampling_index"]])
    return output
