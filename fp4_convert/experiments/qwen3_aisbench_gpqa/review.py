"""Audit persisted GPQA responses, apply blind decisions, and report sampling variance."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics

import pyarrow.parquet as pq

from ..qwen3_mxfp4.artifacts import file_hash, save
from ..qwen3_mxfp4 import review_answers
from ..qwen3_mxfp4.review_answers import RULES, digest, parse_final, plain


LABELS = {
    "native": "原生 NVFP4",
    "direct": "直接 MXFP4",
    "r64": "能量均衡 R64＋MXFP4",
    "fht32": "标准 Walsh–Hadamard H32＋MXFP4",
    "r16x2": "能量均衡 R16 扩展 R32＋MXFP4",
    "direct_uos_c7p25": "UOS c7.25",
    "dualscale_g512": "二级 scale G512",
}


def read(path):
    return json.loads(Path(path).read_text())


def extract(root, tp2_root, output):
    output.mkdir(exist_ok=False)
    protocol_path = root / "protocol_inputs.json"
    protocol = {str(row["id"]): row for row in read(protocol_path)}
    assert len(protocol) == 198
    assert read(protocol_path) == read(tp2_root / "protocol_inputs.json")
    hashes = {str(protocol_path): file_hash(protocol_path)}
    records, queue, evidence = [], [], {}
    for variant in LABELS:
        base = tp2_root if variant in ("direct_uos_c7p25", "dualscale_g512") else root
        formal = base / variant / "formal"
        state, summary = read(formal / "status.json"), read(formal / "summary.json")
        automatic = read(formal / "final_submission_summary.json")
        assert state["status"] == "completed" and state["returncode"] == 0
        assert summary["status"] == "verified" and summary["input_prompt_tokens_options_verified"]
        assert summary["questions"] == 198 and summary["generations"] == 1980
        assert read(formal / "runtime_validation.json")["status"] == "passed"
        config = read(state["results_file"])["config_general"]["model_config"]
        saved = {row["id"]: row for row in map(json.loads, (formal / "final_submission_scores.jsonl").read_text().splitlines())}
        old_queue = {row["id"]: row for row in read(formal / "blind_review_queue.json")}
        assert len(saved) == 1980
        sources = [formal / name for name in (
            "status.json", "summary.json", "runtime_validation.json",
            "final_submission_summary.json", "final_submission_scores.jsonl", "blind_review_queue.json")]
        sources += [Path(state["results_file"])]
        sources += list(map(Path, summary["raw_caches"]))
        hashes.update({str(path): file_hash(path) for path in sources})
        totals, seen, queued, truncated, statuses = [0] * 10, set(), set(), 0, Counter()
        for path in summary["raw_caches"]:
            columns = ["sample_id", "sample.input", "sample.input_tokens", "sample.text", "sample.output_tokens"]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1, columns=columns):
                for item in batch.to_pylist():
                    qid, sample = str(item["sample_id"]), item["sample"]
                    assert qid not in seen and qid in protocol
                    seen.add(qid)
                    ref = protocol[qid]
                    assert digest(sample["input"]) == ref["prompt_sha256"]
                    assert digest(json.dumps(sample["input_tokens"], separators=(",", ":"))) == ref["input_tokens_sha256"]
                    assert len(sample["text"]) == len(sample["output_tokens"]) == 10
                    for index, raw in enumerate(sample["text"]):
                        identity = digest(f"{variant}:{qid}:{index}")[:16]
                        parsed = parse_final(raw)
                        original = saved[identity]
                        assert original["raw_sha256"] == digest(raw)
                        assert original["answer"] == parsed["answer"] and original["status"] == parsed["status"]
                        score = int(parsed["answer"] == ref["gold"])
                        assert original["score"] == score
                        totals[index] += score
                        statuses[parsed["status"]] += 1
                        length = len(sample["output_tokens"][index])
                        is_truncated = length >= summary["max_new_tokens"]
                        truncated += is_truncated
                        reasons = []
                        if parsed["status"].startswith("review_"):
                            assert old_queue[identity]["final"] == parsed["final"]
                            assert old_queue[identity]["candidates"] == parsed["candidates"]
                            queued.add(identity)
                            reasons.append("original_review_queue")
                        # Independently inspect the requested terminal field, without changing the parser.
                        terminal = re.search(r"(?:^|\n)\s*(?:Final\s+)?Answer\s*:\s*([A-D])\s*[.。]?\s*$", plain(parsed["final"]), re.I)
                        if terminal and parsed["answer"] and terminal[1].upper() != parsed["answer"]:
                            reasons.append("terminal_field_disagreement")
                        if is_truncated and parsed["answer"]:
                            reasons.append("accepted_answer_at_length_limit")
                        if reasons:
                            queue.append({"id": identity, "reasons": reasons, "final": parsed["final"],
                                          "final_sha256": digest(parsed["final"]), "output_tokens": length,
                                          "candidates": parsed["candidates"]})
                        records.append({**original, "variant": variant, "gold": ref["gold"],
                                        "final_sha256": digest(parsed["final"]), "output_tokens": length,
                                        "truncated": is_truncated})
        assert seen == set(protocol) and queued == set(old_queue)
        assert totals == automatic["correct_by_sampling_index"]
        assert dict(statuses) == automatic["parse_status"]
        assert abs(truncated / 1980 - summary["truncation_rate"]) < 1e-12
        evidence[variant] = {"formal": str(formal), "questions": len(seen), "generations": 1980,
                             "automatic_correct": sum(totals), "strict_json_accuracy": summary["accuracy"],
                             "tp": config["tensor_parallel_size"], "kv_cache_memory_bytes": config["kv_cache_memory_bytes"],
                             "kv_cache_dtype": config["kv_cache_dtype"], "model_config": config,
                             "truncated_count": truncated, "original_pending": len(old_queue)}
        print(variant, "1980 raw responses verified", flush=True)
    with (output / "verified_scores.jsonl").open("x") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(output / "blind_review_queue.json", sorted(queue, key=lambda row: row["id"]))
    save(output / "manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(),
         "root": str(root), "tp2_root": str(tp2_root), "rules": RULES, "source_sha256": hashes,
         "parser_sha256": file_hash(review_answers.__file__), "script_sha256": file_hash(__file__),
         "verified_scores_sha256": file_hash(output / "verified_scores.jsonl"),
         "queue_sha256": file_hash(output / "blind_review_queue.json"), "review_count": len(queue), "evidence": evidence})
    print("review_count", len(queue), flush=True)


def score(output, decisions_path):
    if (output / "reviewed_summary.json").exists():
        raise FileExistsError("Preserve completed review; choose a new output for a new audit")
    manifest = read(output / "manifest.json")
    assert file_hash(__file__) == manifest["script_sha256"]
    assert file_hash(review_answers.__file__) == manifest["parser_sha256"]
    assert file_hash(output / "verified_scores.jsonl") == manifest["verified_scores_sha256"]
    assert file_hash(output / "blind_review_queue.json") == manifest["queue_sha256"]
    for path, expected in manifest["source_sha256"].items():
        assert file_hash(path) == expected, f"Source changed: {path}"
    decisions = read(decisions_path)
    queue = {row["id"]: row for row in read(output / "blind_review_queue.json")}
    assert set(decisions) == set(queue)
    rows = list(map(json.loads, (output / "verified_scores.jsonl").read_text().splitlines()))
    for row in rows:
        row["automatic_answer"], row["automatic_score"], row["automatic_status"] = row["answer"], row["score"], row["status"]
        if row["id"] in decisions:
            decision = decisions[row["id"]]
            assert decision["final_sha256"] == row["final_sha256"] == queue[row["id"]]["final_sha256"]
            assert decision["answer"] in (None, "A", "B", "C", "D") and decision["reason"]
            row.update(answer=decision["answer"], review_reason=decision["reason"],
                       status="reviewed_final" if decision["answer"] else "reviewed_no_unambiguous_submission")
        row["score"] = int(row["answer"] == row["gold"])
    results, sampling = {}, []
    for variant, label in LABELS.items():
        items = [row for row in rows if row["variant"] == variant]
        counts = [sum(row["score"] for row in items if row["sampling_index"] == index) for index in range(10)]
        assert all(sum(row["sampling_index"] == index for row in items) == 198 for index in range(10))
        fractions = [count / 198 for count in counts]
        percentages = [100 * value for value in fractions]
        correct = sum(counts)
        assert correct == sum(row["score"] for row in items)
        assert abs(statistics.mean(fractions) - correct / 1980) < 1e-12
        result = {"label": label, "correct": correct, "generations": 1980, "accuracy": correct / 1980,
                  "accuracy_percent": correct / 1980 * 100, "correct_by_sampling_index": counts,
                  "accuracy_percent_by_sampling_index": percentages,
                  "sample_variance_fraction_squared": statistics.variance(fractions),
                  "sample_variance_pp_squared": statistics.variance(percentages),
                  "sample_std_pp": statistics.stdev(percentages),
                  "population_variance_pp_squared": statistics.pvariance(percentages),
                  "population_std_pp": statistics.pstdev(percentages),
                  "review_count": sum(row["id"] in decisions for row in items),
                  "correct_change": sum(row["score"] - row["automatic_score"] for row in items),
                  "unresolved_count": sum(row["answer"] is None for row in items),
                  "parse_failure_rate": sum(row["answer"] is None for row in items) / 1980,
                  "truncated_count": sum(row["truncated"] for row in items),
                  "final_status": dict(Counter(row["status"] for row in items)),
                  "tp": manifest["evidence"][variant]["tp"],
                  "kv_gib_per_card": manifest["evidence"][variant]["kv_cache_memory_bytes"] / 2**30}
        results[variant] = result
        sampling.extend({"variant": variant, "sampling_index": i + 1, "correct": count,
                         "questions": 198, "accuracy_percent": percentages[i]} for i, count in enumerate(counts))
    for variant, result in results.items():
        result["delta_native_pp"] = result["accuracy_percent"] - results["native"]["accuracy_percent"]
        result["delta_direct_pp"] = result["accuracy_percent"] - results["direct"]["accuracy_percent"]
    with (output / "reviewed_scores.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(output / "reviewed_summary.json", {"status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
         "rules": RULES, "reviewer": "Codex; model identity and gold hidden during case decisions",
         "variance_definition": "Across ten sampling-index accuracies on the same 198 questions; primary ddof=1 (divide by 9); population ddof=0 also saved. Not ten independently seeded runs or question-level uncertainty.",
         "source_integrity_verified": True, "original_scores_preserved": True, "pending_review_count": 0,
         "decisions_sha256": file_hash(decisions_path), "reviewed_scores_sha256": file_hash(output / "reviewed_scores.jsonl"),
         "results": results})
    table_keys = ["variant", "label", "tp", "kv_gib_per_card", "correct", "generations", "accuracy_percent",
                  "sample_variance_pp_squared", "sample_std_pp", "population_variance_pp_squared", "population_std_pp",
                  "correct_change", "unresolved_count", "parse_failure_rate", "truncated_count", "delta_native_pp", "delta_direct_pp"]
    for name, data, keys in [("comparison.csv", [{"variant": v, **r} for v, r in results.items()], table_keys),
                              ("sampling_accuracy.csv", sampling, list(sampling[0]))]:
        with (output / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    extraction = commands.add_parser("extract")
    extraction.add_argument("--root", type=Path, required=True)
    extraction.add_argument("--tp2-root", type=Path, required=True)
    extraction.add_argument("--output", type=Path, required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--decisions", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "extract":
        extract(args.root.resolve(), args.tp2_root.resolve(), args.output.resolve())
    else:
        score(args.output.resolve(), args.decisions.resolve())
