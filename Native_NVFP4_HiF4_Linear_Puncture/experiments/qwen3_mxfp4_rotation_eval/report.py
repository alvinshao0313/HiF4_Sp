"""Validate coverage and actual inputs before reporting paired mean accuracy."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean, pstdev

from .common import RUNS, TASKS, digest, read, save


def one(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one result/cache, found {len(paths)}")
    return paths[0]


def runtime_check(variant, directory):
    def canonical(root):
        output = {}
        for path in Path(root).glob("worker-*.jsonl"):
            events = [json.loads(line) for line in path.read_text().splitlines()]
            experts = [event for event in events if event["event"] == "loaded_experts"]
            if not experts:
                continue
            ranks = {row["tp_rank"] for row in experts}
            if len(ranks) != 1 or len(experts) != 48:
                raise ValueError("Missing TP4 / 48-layer expert coverage")
            rank = ranks.pop()
            if rank in output:
                raise ValueError("Duplicate worker rank")
            output[rank] = {
                "experts": {row["prefix"]: {key: row[key] for key in ("method", "variant", "tensors")}
                            for row in experts},
                "other": {row["prefix"]: row["method"] for row in events if row["event"] == "non_expert_method"},
            }
        if set(output) != {0, 1, 2, 3}:
            raise ValueError("Missing tensor-parallel worker")
        return output
    if canonical(directory) != canonical(RUNS[variant] / variant / "formal/audit"):
        raise ValueError("Runtime methods or TP-sliced packed bytes differ from the qualified checkpoint load")
    return {"status": "passed", "ranks": 4, "layers_per_rank": 48,
            "reference": str(RUNS[variant] / variant / "formal/audit")}


def reasoning(root, target, task, count, short):
    import pyarrow.parquet as pq

    target = Path(target)
    tag = TASKS[task]["tag"]
    expected = read(Path(root) / f"{task}_inputs.json")["documents"][:count]
    reference = {row["id"]: row for row in expected}
    raw_path = one((target / "evaluation").rglob("results_*.json"))
    raw = read(raw_path)
    details_path = one((target / "evaluation").rglob("details_*.json"))
    rows = read(details_path)
    if len(rows) != count or {str(row["doc"]["id"]) for row in rows} != set(reference):
        raise ValueError("Reasoning results have incomplete or different questions")
    scores = {}
    for row in rows:
        key = str(row["doc"]["id"])
        values = row["doc"]["specific"]["rotation_eval_scoring"]["scores"]
        if (len(values) != 10 or any(value not in (0, 1) for value in values)
                or abs(mean(values) - row["metric"]["accuracy_avg10"]) > 1e-12
                or digest(row["model_response"]["input"]) != reference[key]["prompt_sha256"]
                or digest(row["gold"]) != reference[key]["gold_sha256"]):
            raise ValueError("Question score, prompt or gold differs from the frozen protocol")
        scores[key] = values
    score = mean([mean(values) for values in scores.values()])
    if abs(score - raw["results"][tag]["accuracy_avg10"]) > 1e-12:
        raise ValueError("Aggregate score differs from the individual response scores")
    cache = one((target / "model").rglob("GENERATIVE.parquet"))
    seen = []
    token_key = "probe_input_tokens_sha256" if short else "input_tokens_sha256"
    for batch in pq.ParquetFile(cache).iter_batches(batch_size=8, columns=[
            "sample_id", "sample.input", "sample.input_tokens", "sample.text", "sample.output_tokens"]):
        for item in batch.to_pylist():
            key, sample = str(item["sample_id"]), item["sample"]
            if (key not in reference or digest(sample["input_tokens"]) != reference[key][token_key]
                    or digest(sample["input"]) != reference[key]["prompt_sha256"]
                    or len(sample["text"]) != 10 or len(sample["output_tokens"]) != 10
                    or any(not text.strip() for text in sample["text"])):
                raise ValueError("Raw prediction cache is incomplete or has different inputs")
            seen.append(key)
    if len(seen) != count or set(seen) != set(reference):
        raise ValueError("Raw prediction cache question coverage differs")
    observations = [row for path in (target / "evaluation/generation_observations").glob("batch_*.json") for row in read(path)]
    if (len(observations) != count or any(len(row["outputs"]) != 10 for row in observations)
            or Counter(digest(row["prompt_token_ids"]) for row in observations)
            != Counter(row[token_key] for row in reference.values())):
        raise ValueError("Actual vLLM requests have different inputs or response counts")
    endings = Counter(item["finish_reason"] for row in observations for item in row["outputs"])
    if set(endings) - {"stop", "length"}:
        raise ValueError("Unexpected generation finish reason")
    config = dict(raw["config_general"]["model_config"])
    config.pop("model_name", None)
    report = {"status": "completed", "task": task, "short_test": short, "questions": count,
              "generations": count * 10, "accuracy": score,
              "accuracy_by_sample_index": [mean(values[index] for values in scores.values()) for index in range(10)],
              "per_question_scores": scores, "finish_reasons": dict(endings),
              "truncation_rate": endings["length"] / (count * 10),
              "input_sha256": digest(expected), "protocol": config,
              "results": str(raw_path), "details": str(details_path), "cache": str(cache)}
    save(target / "summary.json", report)
    return report


def compare(root):
    root = Path(root)
    result = {"status": "completed", "metric": "mean accuracy", "comparisons": {}}
    for task in TASKS:
        arms = {variant: read(root / variant / "formal" / task / "summary.json") for variant in RUNS}
        h32, r16 = arms["fht32"], arms["r16x2"]
        if h32["input_sha256"] != r16["input_sha256"] or h32["protocol"] != r16["protocol"]:
            raise ValueError("Reasoning comparison inputs/protocols differ")
        result["comparisons"][task] = {"models": arms,
            "r16x2_minus_h32_pp": 100 * (r16["accuracy"] - h32["accuracy"])}
    arc = {variant: read(root / variant / "formal/arc/evaluation/summary.json") for variant in RUNS}
    for task in ("arc_easy", "arc_challenge"):
        all_rounds = [row["tasks"][task] for arm in arc.values() for row in arm["rounds"]]
        if len(all_rounds) != 10 or len({row["input_sha256"] for row in all_rounds}) != 1:
            raise ValueError("ARC paired/repeated input coverage differs")
        arms = {}
        for variant, report in arc.items():
            arms[variant] = {}
            for metric in ("acc", "acc_norm"):
                values = [mean(row["tasks"][task]["scores"][metric]) for row in report["rounds"]]
                arms[variant][metric] = {"mean": mean(values), "rounds": values, "population_std": pstdev(values)}
        result["comparisons"][task] = {"models": arms,
            "r16x2_minus_h32_pp": 100 * (arms["r16x2"]["acc"]["mean"] - arms["fht32"]["acc"]["mean"])}
    save(root / "comparison.json", result)
    import csv
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "h32_accuracy_pct", "r16x2_accuracy_pct", "r16x2_minus_h32_pp"])
        for task, row in result["comparisons"].items():
            key = (lambda item: item["accuracy"]) if task in TASKS else (lambda item: item["acc"]["mean"])
            writer.writerow([task, 100 * key(row["models"]["fht32"]), 100 * key(row["models"]["r16x2"]), row["r16x2_minus_h32_pp"]])
    return result
