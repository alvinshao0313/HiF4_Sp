"""Reuse five completed reviews; apply identical final-submission rules to UOS."""
import argparse
import csv
import json
from pathlib import Path

from .artifacts import file_hash, save
from .format import ALGORITHM, SCALE_RULE, VARIANT
from ..qwen3_mxfp4.review_answers import digest, extract, score
from ..qwen3_mxfp4.workflow import now


def prior_evidence(table_path):
    table_path = Path(table_path)
    table = json.loads(table_path.read_text())
    if table["status"] != "completed" or {r["variant"] for r in table["results"]} != {"native", "direct", "r64", "fht32", "r16x2"}:
        raise ValueError("Expected the completed five-model comparison")
    paths, directories = {table_path}, set()
    for row in table["results"]:
        summary_path = Path(row["reviewed_source"])
        if file_hash(summary_path) != row["reviewed_source_sha256"]:
            raise ValueError(f"Prior reviewed source changed: {summary_path}")
        summary = json.loads(summary_path.read_text())
        item = summary["results"][row["variant"]]
        if summary["status"] != "completed" or item["generations"] != 1980 or item["correct"] != row["correct"]:
            raise ValueError("Prior reviewed coverage/score differs")
        directory = summary_path.parent
        directories.add(str(directory))
        paths.add(directory.parent / "comparison.json")
        for name in ("reviewed_summary.json", "reviewed_scores.jsonl", "manifest.json", "manual_decisions.json",
                     "extractions.jsonl", "completion_validation.json"):
            paths.add(directory / name)
    return sorted(paths), sorted(directories)


def check_reused(manifest):
    for path, expected in manifest["reused_evidence_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Reused evidence changed: {path}")


def prepare_review(root, manifest):
    check_reused(manifest)
    table = json.loads(Path(manifest["prior_table"]).read_text())
    results = {}
    prior, decisions = {}, {}
    for directory in map(Path, manifest["prior_reviews"]):
        for line in (directory / "extractions.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record["id"] in prior and prior[record["id"]] != record:
                raise ValueError("Prior extractions disagree across reviews")
            prior[record["id"]] = record
        for key, value in json.loads((directory / "manual_decisions.json").read_text()).items():
            if key in decisions and decisions[key] != value:
                raise ValueError("Prior manual decisions disagree")
            if value.get("final_sha256") != digest(prior[key]["final"]):
                raise ValueError("Prior manual decision final-text hash differs")
            decisions[key] = value
    for row in table["results"]:
        source = Path(row["reviewed_source"]).parent.parent / "comparison.json"
        results[row["variant"]] = json.loads(source.read_text())["results"][row["variant"]]
    results[VARIANT] = json.loads((root / f"{VARIANT}_summary.json").read_text())
    save(root / "comparison.json", {"status": "completed", "results": results,
         "scoring_basis": "Original strict JSON only; final-submission review is separate"})
    review = root / "answer_review"
    extract(root, review)
    for line in (review / "extractions.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["model"] != VARIANT and row != prior[row["id"]]:
            raise ValueError("Reused final-answer extraction changed")
    queue = json.loads((review / "blind_review_queue.json").read_text())
    reused = {row["id"]: decisions[row["id"]] for row in queue if row["id"] in decisions}
    pending = [row for row in queue if row["id"] not in decisions]
    save(review / "reused_manual_decisions.json", reused)
    save(review / "pending_blind_review_queue.json", pending)
    save(review / "review_status.json", {"status": "awaiting_final_answer_review",
         "new_decisions_needed": len(pending), "prior_decisions_reused": len(reused),
         "original_scores_preserved": True})


def finish(root, decisions_path):
    """Called after blind review; never invent decisions or overwrite old results."""
    root = Path(root).absolute()
    manifest = json.loads((root / "workflow_manifest.json").read_text())
    check_reused(manifest)
    review = root / "answer_review"
    if (root / "six_model_summary.json").exists() or (review / "reviewed_summary.json").exists():
        raise FileExistsError("Review already scored")
    decisions = json.loads(Path(decisions_path).read_text())
    reused = json.loads((review / "reused_manual_decisions.json").read_text())
    if any(decisions.get(key) != value for key, value in reused.items()):
        raise ValueError("Previously completed decisions must be preserved")
    queue = json.loads((review / "blind_review_queue.json").read_text())
    if set(decisions) != {row["id"] for row in queue}:
        raise ValueError("Every queued submission requires an explicit decision")
    for row in queue:
        if decisions[row["id"]].get("final_sha256") != digest(row["final"]):
            raise ValueError("Manual decision must identify the exact final text")
    # Persist this run's complete decisions separately from all prior runs.
    destination = review / "manual_decisions.json"
    if destination.exists() and json.loads(destination.read_text()) != decisions:
        raise FileExistsError("Different manual decisions already exist")
    if not destination.exists():
        save(destination, decisions)
    score(review, destination)
    summary = json.loads((review / "reviewed_summary.json").read_text())
    previous = json.loads(Path(manifest["prior_table"]).read_text())
    for row in previous["results"]:
        current = summary["results"][row["variant"]]
        if current["correct"] != row["correct"] or current["correct_by_sampling_index"] != row["correct_by_sampling_index"]:
            raise ValueError("Prior reviewed scores failed to reproduce")
    raw = json.loads((root / f"{VARIANT}_summary.json").read_text())
    item = summary["results"][VARIANT]
    row = {"variant": VARIANT, "label": ALGORITHM, "correct": item["correct"], "generations": 1980,
           "accuracy_percent": 100 * item["accuracy"],
           "delta_native_pp": 100 * (item["accuracy"] - summary["results"]["native"]["accuracy"]),
           "delta_direct_pp": 100 * (item["accuracy"] - summary["results"]["direct"]["accuracy"]),
           "correct_by_sampling_index": item["correct_by_sampling_index"],
           "accuracy_percent_by_sampling_index": [100 * a for a in item["accuracy_by_sampling_index"]],
           "truncation_percent": 100 * raw["truncation_rate"],
           "unresolved_final_percent": 100 * item["unresolved_count"] / 1980,
           "final_status": item["final_status"],
           "original_strict_json_accuracy_percent": 100 * item["original_accuracy"],
           "original_strict_json_parse_failure_percent": 100 * raw["answer_parse_failure_rate"],
           "reviewed_source": str(review / "reviewed_summary.json"),
           "reviewed_source_sha256": file_hash(review / "reviewed_summary.json"),
           "cuda_visible_devices": "0,1,2,3", "scale_rule": SCALE_RULE}
    previous["results"].append(row)
    previous["scale_rule"] = "Per variant: five prior models unchanged; direct_uos_c7p25 uses ceil(log2(amax/7.25))"
    previous.update(created_at_utc=now(), prior_table=manifest["prior_table"],
                    operator_performance=str(root / f"{VARIANT}_gpu_validation.json"))
    save(root / "six_model_summary.json", previous)
    fields = ["variant", "label", "correct", "generations", "accuracy_percent", "delta_native_pp", "delta_direct_pp",
              "truncation_percent", "unresolved_final_percent", "original_strict_json_accuracy_percent",
              "original_strict_json_parse_failure_percent"] + [f"accuracy_sample_{i}_percent" for i in range(10)]
    with (root / "six_model_summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in previous["results"]:
            writer.writerow({**{key: entry[key] for key in fields[:11]},
                             **{f"accuracy_sample_{i}_percent": a for i, a in enumerate(entry["accuracy_percent_by_sampling_index"])}})
    save(review / "completion_validation.json", {"status": "completed", "prior_five_scores_reproduced": True,
         "manual_final_text_hashes_verified": True, "summary_sha256": file_hash(root / "six_model_summary.json"),
         "csv_sha256": file_hash(root / "six_model_summary.csv"), "completed_at": now()})
    save(review / "review_status.json", {"status": "completed", "original_scores_preserved": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    args = parser.parse_args()
    finish(args.root, args.decisions)
