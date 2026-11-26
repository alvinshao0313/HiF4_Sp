"""Validate the real outputs and routing of supplemental Ban evaluations."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

from .evaluation_report import _actual_tokens, _one, _sha256, aggregate_routing


TASKS = {
    "mmlu_pro": {"task": "mmlu_pro|0", "metric": "extractive_match", "count": 300,
                 "generations": 1, "max_new_tokens": 32768, "short_count": 2},
    "lcb": {"task": "lcb:codegeneration_v6|0", "metric": "codegen_pass@1:16", "count": 175,
            "generations": 1, "max_new_tokens": 38912, "short_count": 1},
    "aime25": {"task": "aime25_avg5|0", "metric": "avg@n:n=5", "count": 30,
               "generations": 5, "max_new_tokens": 38912, "short_count": 1},
}


def summarize(root, name, audit, *, short=False):
    import pyarrow.parquet as pq

    root, definition = Path(root).resolve(), TASKS[name]
    count = definition["short_count"] if short else definition["count"]
    result_path = _one((root / "evaluation").rglob("results_*.json"), "task results")
    raw = json.loads(result_path.read_text())
    tag, metric = definition["task"], definition["metric"]
    stamp = result_path.stem.removeprefix("results_")
    details_path = result_path.parent.parent / "details" / stamp / f"details_{tag}_{stamp}.json"
    rows = json.loads(details_path.read_text())
    details = {str(row["doc"]["id"]): row for row in rows}
    reference = audit["documents"][:count]
    if len(rows) != count or set(details) != {row["id"] for row in reference}:
        raise ValueError(f"incomplete or wrong task coverage: {name}")
    score = raw["results"][tag][metric]
    values = []
    for expected in reference:
        row = details[expected["id"]]
        if (_sha256(row["model_response"]["input"]) != expected["prompt_sha256"]
                or _sha256(row["gold"]) != expected["gold_sha256"]
                or len(row["model_response"]["text_post_processed"]) != definition["generations"]):
            raise ValueError(f"prompt/gold/generation count differs: {name}/{expected['id']}")
        value = row["metric"][metric]
        if not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError("invalid per-question metric")
        if name == "aime25" and not math.isclose(value * 5, round(value * 5), abs_tol=1e-10):
            raise ValueError("AIME score is not an average of five binary answers")
        values.append(value)
    if not math.isclose(math.fsum(values) / count, score, rel_tol=0, abs_tol=1e-12):
        raise ValueError("aggregate score differs from per-question metrics")
    config = raw["config_general"]["model_config"]
    actual_hash = raw["summary_tasks"][tag]["hashes"]["hash_input_tokens"]
    tokens, cache = _actual_tokens(config["model_name"], details, actual_hash, task=tag)
    for expected in reference:
        if tokens[expected["id"]] != expected["input_tokens_sha256"]:
            raise ValueError(f"actual input tokens differ from the saved input audit: {name}/{expected['id']}")
    cached = pq.read_table(cache, columns=["sample_id", "sample.text", "sample.output_tokens",
                                          "sample.truncated_tokens_count"]).to_pylist()
    if any(len(row["text"]) != definition["generations"]
           or len(row["output_tokens"]) != definition["generations"]
           or any(not text.strip() for text in row["text"]) for row in cached):
        raise ValueError("raw responses are empty or incomplete")
    observations = [row for path in (root / "evaluation/generation_observations").glob("batch_*.json")
                    for row in json.loads(path.read_text())]
    if (len(observations) != count
            or any(len(row["outputs"]) != definition["generations"] for row in observations)):
        raise ValueError("actual worker request/generation coverage differs")
    endings = Counter(value["finish_reason"] for row in observations for value in row["outputs"])
    if set(endings) - {"stop", "length"}:
        raise ValueError(f"unexpected generation finish reason: {endings}")
    routing = aggregate_routing(root / "routing")
    return {"status": "complete", "task": tag, "metric": metric, "count": count,
            "generations_per_question": definition["generations"], "score": score,
            "score_pct": 100 * score, "short_test": short,
            "results_file": str(result_path), "details_file": str(details_path),
            "prediction_cache": str(cache), "actual_input_tokens_hash": actual_hash,
            "finish_reasons": dict(endings), "routing": routing,
            "per_question": {key: row["metric"][metric] for key, row in details.items()},
            "protocol": config,
            "scoring_note": "Official existing scorer; AIME uses avg@n:n=5 over raw responses, not pass@5."}
