"""Raw coverage and final-submission scoring for the remaining Ban tasks."""
from collections import Counter
import math
from pathlib import Path

from ..ban.evaluation_report import _one, _sha256, aggregate_routing
from ..reasoning_trace_calibration.scoring import aime_submission
from .config import INFERENCE, read, save
from .report import protocol, require_coverage

TASKS = {
    "gsm8k": {"task": "gsm8k|0", "metric": "extractive_match", "count": 1319,
              "generations": 1, "max_new_tokens": 32768},
    "aime24": {"task": "aime24_avg5|0", "metric": "avg@n:n=5", "count": 30,
               "generations": 5, "max_new_tokens": 38912},
    "aime25": {"task": "aime25_avg5|0", "metric": "avg@n:n=5", "count": 30,
               "generations": 5, "max_new_tokens": 38912},
}


def summarize(target, name, audit, *, arm, short=False, historical=None):
    import pyarrow.parquet as pq
    import xxhash

    target = Path(target)
    definition = TASKS[name]
    count = 1 if short else definition["count"]
    reference = {r["id"]: r for r in audit["documents"][:count]}
    result_path = Path(historical["results_file"]) if historical else _one(
        (target / "evaluation").rglob("results_*.json"), "task results")
    raw = read(result_path)
    stamp = result_path.stem.removeprefix("results_")
    details_path = result_path.parent.parent / "details" / stamp / f"details_{definition['task']}_{stamp}.json"
    rows = read(details_path)
    require_coverage([str(r["doc"]["id"]) for r in rows], reference, name)
    details, formal = {}, {}
    for row in rows:
        key = str(row["doc"]["id"])
        assert _sha256(row["model_response"]["input"]) == reference[key]["prompt_sha256"]
        assert _sha256(row["gold"]) == reference[key]["gold_sha256"]
        assert len(row["model_response"]["text_post_processed"]) == definition["generations"]
        value = row["metric"][definition["metric"]]
        assert math.isfinite(value) and 0 <= value <= 1
        assert math.isclose(value * definition["generations"], round(value * definition["generations"]), abs_tol=1e-10)
        details[key], formal[key] = row, value
    score = raw["results"][definition["task"]][definition["metric"]]
    assert math.isclose(math.fsum(formal.values()) / count, score, abs_tol=1e-12)
    config = raw["config_general"]["model_config"]
    for key, value in {"tensor_parallel_size": 2, "max_num_seqs": 128,
                       "max_num_batched_tokens": 4096, "max_model_length": 40960,
                       "kv_cache_dtype": "bfloat16", "enforce_eager": True, "seed": 1234}.items():
        assert config[key] == value, f"inference configuration differs: {key}"
    generation = config["generation_parameters"]
    assert generation["max_new_tokens"] == (64 if short else definition["max_new_tokens"])
    for key in ("temperature", "top_p", "top_k", "min_p"):
        assert generation[key] == INFERENCE[key]
    cache = Path(historical["prediction_cache"]) if historical else _one(
        (p for p in Path(config["model_name"]).rglob("GENERATIVE.parquet")
         if p.parent.parent.name == definition["task"]), "isolated raw cache")
    reviews_path = target / "reviews.json"
    reviews = read(reviews_path) if reviews_path.exists() else {}
    cached_ids, hashes, responses, unresolved = [], [], [], []
    for batch in pq.ParquetFile(cache).iter_batches(batch_size=8, columns=[
            "sample_id", "sample.input", "sample.input_tokens", "sample.text", "sample.output_tokens"]):
        for row in batch.to_pylist():
            key, sample = str(row["sample_id"]), row["sample"]
            assert sample["input"] == details[key]["model_response"]["input"]
            assert _sha256(sample["input_tokens"]) == reference[key]["input_tokens_sha256"]
            assert len(sample["text"]) == len(sample["output_tokens"]) == definition["generations"]
            assert all(text.strip() for text in sample["text"]) and all(sample["output_tokens"])
            cached_ids.append(key)
            hashes.append(xxhash.xxh64(str(sample["input_tokens"])).hexdigest())
            if name.startswith("aime") and not short:
                for index, text in enumerate(sample["text"], 1):
                    record = {"id": key, "response_index": index, "raw_sha256": _sha256(text),
                              "gold": str(details[key]["gold"][0]), **aime_submission(text, details[key]["gold"][0])}
                    if record["review"]:
                        review = reviews.get(f"{key}:{index}")
                        if review is None:
                            unresolved.append({**record, "final_tail": text[-6000:]})
                        else:
                            assert review["raw_sha256"] == record["raw_sha256"]
                            assert type(review["score"]) is int and review["score"] in (0, 1) and review["reason"].strip()
                            record.update(score=review["score"], review=False, decision=review["reason"])
                    responses.append(record)
    require_coverage(cached_ids, reference, name)
    token_hash = xxhash.xxh64("".join(sorted(hashes))).hexdigest()
    assert token_hash == raw["summary_tasks"][definition["task"]]["hashes"]["hash_input_tokens"]
    endings = None
    if not historical:
        requests = [r for p in (target / "evaluation/generation_observations").glob("batch_*.json") for r in read(p)]
        assert Counter(_sha256(r["prompt_token_ids"]) for r in requests) == Counter(
            r["input_tokens_sha256"] for r in reference.values())
        assert all(len(r["outputs"]) == definition["generations"] for r in requests)
        endings = Counter(o["finish_reason"] for r in requests for o in r["outputs"])
        assert set(endings) <= {"stop", "length"}
    routing = {"average_experts": 8., "measured": False}
    if arm != "top8":
        routing = aggregate_routing(target / "routing")
        assert routing["tp_ranks"] == 2 and len(routing["layer_counters"]["tokens"]) == 48
        assert routing["selected_count"] == routing["dispatched_count"]
    final = formal
    if responses:
        save(target / "aime_responses.json", {"responses": responses, "unresolved": unresolved})
        final = None if unresolved else {key: math.fsum(r["score"] for r in responses if r["id"] == key) / 5
                                        for key in reference}
    report = {"status": "awaiting_aime_review" if unresolved else "complete", "arm": arm, "task": name,
              "short": short, "count": count, "generations_per_question": definition["generations"],
              "official_score_pct": 100 * score, "official_per_question": formal,
              "score_pct": None if final is None else 100 * math.fsum(final.values()) / count,
              "per_question": final, "scoring": "final_submission_avg5" if responses else definition["metric"],
              "routing": routing, "actual_input_tokens_hash": token_hash, "protocol": protocol(config),
              "results_file": str(result_path), "details_file": str(details_path), "prediction_cache": str(cache),
              "finish_reasons": dict(endings) if endings else None, "reused": bool(historical)}
    save(target / "report.json", report)
    return report
