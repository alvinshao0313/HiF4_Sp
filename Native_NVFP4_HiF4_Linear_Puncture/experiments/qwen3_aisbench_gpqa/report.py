"""Retain strict JSON scores and apply the unchanged final-submission parser."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from .common import OLD_RUNS, read, save
from ..qwen3_mxfp4.review_answers import RULES, parse_final


def canonical(directory):
    ranks = {}
    for path in Path(directory).glob("worker-*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        experts = [row for row in rows if row["event"] == "loaded_experts"]
        if not experts:
            continue
        rank_set = {row["tp_rank"] for row in experts}
        if len(rank_set) != 1 or len(experts) != 48:
            raise ValueError("Missing rank/layer runtime evidence")
        rank = rank_set.pop()
        if rank in ranks:
            raise ValueError("Duplicated runtime rank")
        ranks[rank] = {
            "experts": {row["prefix"]: {key: row[key] for key in ("method", "variant", "tensors")} for row in experts},
            "non_experts": {row["prefix"]: row["method"] for row in rows if row["event"] == "non_expert_method"},
        }
    if set(ranks) != {0, 1, 2, 3}:
        raise ValueError("Missing TP4 runtime evidence")
    return ranks


def validate_runtime(root, variant, phase):
    target = Path(root) / variant / phase
    status = read(target / "status.json")
    result = read(status["results_file"])
    previous = read(OLD_RUNS[variant] / "formal/status.json")
    old_config = dict(read(previous["results_file"])["config_general"]["model_config"])
    config = dict(result["config_general"]["model_config"])
    for cfg in (config, old_config):
        cfg.pop("model_name", None)
    if phase == "probe":
        old_config["generation_parameters"]["max_new_tokens"] = 64
    if config != old_config or config["kv_cache_dtype"] != "bfloat16":
        raise ValueError("Actual saved inference configuration changed beyond the prompt/model view")
    report = {"status": "passed", "kv_cache_dtype": "bfloat16", "original_inference_config_equal": True}
    if variant != "native":
        old_audit = OLD_RUNS[variant] / "formal/audit"
        if canonical(target / "audit") != canonical(old_audit):
            raise ValueError("Expert bytes, TP4 slices or non-expert methods changed")
        report.update(reference=str(old_audit), ranks=4, layers_per_rank=48, packed_bytes_equal=True)
    save(target / "runtime_validation.json", report)


def final_submission_summary(root, variant):
    target = Path(root) / variant / "formal"
    original = read(target / "summary.json")
    protocol = {str(row["id"]): row for row in read(Path(root) / "protocol_inputs.json")}
    counts, totals, pending, rows = Counter(), [0] * 10, [], []
    for path in original["raw_caches"]:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=["sample_id", "sample.text"]):
            for item in batch.to_pylist():
                qid = str(item["sample_id"])
                for index, raw in enumerate(item["sample"]["text"]):
                    parsed = parse_final(raw)
                    identity = hashlib.sha256(f"{variant}:{qid}:{index}".encode()).hexdigest()[:16]
                    score = int(parsed["answer"] == protocol[qid]["gold"])
                    totals[index] += score
                    counts[parsed["status"]] += 1
                    rows.append({"id": identity, "question_id": qid, "sampling_index": index,
                                 "answer": parsed["answer"], "status": parsed["status"], "score": score,
                                 "raw_sha256": hashlib.sha256(raw.encode()).hexdigest()})
                    if parsed["status"].startswith("review_"):
                        pending.append({"id": identity, "final": parsed["final"], "candidates": parsed["candidates"],
                                        "status": parsed["status"]})
    if len(rows) != 1980:
        raise ValueError("Incomplete final-submission review coverage")
    with (target / "final_submission_scores.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(target / "blind_review_queue.json", sorted(pending, key=lambda x: x["id"]))
    save(target / "final_submission_summary.json", {
        "status": "awaiting_blind_review" if pending else "completed", "rules": RULES,
        "generations": 1980, "correct": sum(totals), "accuracy": sum(totals) / 1980,
        "correct_by_sampling_index": totals, "accuracy_by_sampling_index": [n / 198 for n in totals],
        "parse_status": dict(counts), "pending_review_count": len(pending),
        "provisional_if_pending": bool(pending), "strict_json_accuracy": original["accuracy"],
        "truncation_rate": original["truncation_rate"], "original_metrics_preserved": True,
    })
