"""Run a prepared phase and verify actual inputs, raw generations and scores."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from .download import save, sha256


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def summarize(root, phase, manifest):
    import pyarrow.parquet as pq
    from .task import parse_final, extract_final_answer

    directory = root / phase
    results = list((directory / "evaluation").rglob("results_*.json"))
    details = list((directory / "evaluation").rglob("details_*.json"))
    caches = list((directory / "model").rglob("GENERATIVE.parquet"))
    if len(results) != 1 or len(details) != 1 or not caches:
        raise ValueError("Missing or duplicate results/details/raw cache")
    records = json.loads(details[0].read_text())
    expected = manifest["phases"][phase]["questions"]
    n = manifest["sampling"]["n"]
    is_formal = manifest["phases"][phase].get("kind", phase) == "formal"
    protocol = {row["prompt_sha256"]: row for row in json.loads((root / "protocol_inputs.json").read_text())}
    if len(records) != expected:
        raise ValueError(f"Expected {expected} questions, got {len(records)}")
    by_prompt = {}
    for row in records:
        key = digest(row["model_response"]["input"])
        if key in by_prompt or key not in protocol or row["gold"] != [protocol[key]["gold"]]:
            raise ValueError("Duplicate, changed input or changed gold")
        by_prompt[key] = row["doc"]["specific"]["gpqa_scoring"]
    rows, review, seen = [], [], set()
    for cache in caches:
        for cached in pq.read_table(cache).to_pylist():
            sample = cached["sample"]
            key = digest(sample["input"])
            if key in seen or key not in by_prompt:
                raise ValueError("Duplicate or unexpected raw cache input")
            seen.add(key)
            reference = protocol[key]
            token_hash = digest(json.dumps(sample["input_tokens"], separators=(",", ":")))
            if token_hash != reference["input_tokens_sha256"]:
                raise ValueError("Actual input tokens differ from prepared protocol")
            if len(sample["text"]) != n or len(sample["output_tokens"]) != n:
                raise ValueError(f"Incomplete {n}-sample raw cache")
            evidence = by_prompt[key]
            for i, raw in enumerate(sample["text"]):
                parsed = parse_final(raw)
                original, _ = extract_final_answer(raw)
                score = int(parsed["answer"] == reference["gold"])
                if (parsed["answer"] != evidence["answers"][i]
                        or parsed["status"] != evidence["parse_status"][i]
                        or score != evidence["scores"][i]
                        or int(original == reference["gold"]) != evidence["strict_json_scores"][i]):
                    raise ValueError("Saved scoring disagrees with raw response")
                row = {"id": digest(f"{key}:{i}")[:20], "question_sha256": reference["question_sha256"],
                       "sampling_index": i, "gold": reference["gold"], "answer": parsed["answer"],
                       "score": score, "parse_status": parsed["status"], "raw_sha256": digest(raw),
                       "output_tokens": len(sample["output_tokens"][i]),
                       "strict_json_score": int(original == reference["gold"])}
                rows.append(row)
                if parsed["status"].startswith("review_"):
                    review.append({"id": row["id"], "final": parsed["final"],
                                   "final_sha256": digest(parsed["final"]), "prompt": sample["input"],
                                   "candidates": parsed["candidates"], "status": parsed["status"]})
    if seen != set(by_prompt) or len(rows) != expected * n:
        raise ValueError("Raw cache and result coverage differ")
    limit = manifest["phases"][phase]["max_new_tokens"]
    if not all(0 < row["output_tokens"] <= limit for row in rows):
        raise ValueError("Invalid output token coverage")
    if not is_formal and max(row["output_tokens"] for row in rows) != limit:
        raise ValueError("Probe did not exercise the planned decoding steps")
    correct = sum(row["score"] for row in rows)
    accuracy = correct / len(rows)
    payload = json.loads(results[0].read_text())
    metric_name = manifest.get("metric_name", "gpqa_accuracy_avg10")
    saved = [value[metric_name] for value in payload["results"].values() if metric_name in value]
    if not saved or not all(math.isclose(value, accuracy, abs_tol=1e-12) for value in saved):
        raise ValueError("Saved aggregate differs from the raw response scores")
    by_index = [sum(row["score"] for row in rows if row["sampling_index"] == i) for i in range(n)]
    with (directory / "generation_scores.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(directory / "blind_review_queue.json", review)
    summary = {"questions": expected, "generations": len(rows), "correct": correct,
               "accuracy": accuracy, "correct_by_sampling_index": by_index,
               "accuracy_by_sampling_index": [n / expected for n in by_index],
               "mean_definition": "one full-dataset round" if n == 1 else "mean of 10 sampled accuracies; not pass@10 or majority vote or independent seeds",
               "seed": manifest["phases"][phase].get("seed", manifest["seed"]),
               "parse_status": dict(Counter(row["parse_status"] for row in rows)),
               "needs_answer_review": bool(review), "review_count": len(review),
               "accuracy_is_final": is_formal and not review,
               "accuracy_is_benchmark": is_formal,
               "raw_input_tokens_and_scores_verified": True,
               "generated_tokens": sum(row["output_tokens"] for row in rows),
               "max_decode_steps": max(row["output_tokens"] for row in rows),
               "output_limit_count": sum(row["output_tokens"] == limit for row in rows),
               "raw_caches": [str(path) for path in caches],
               "results_file": str(results[0]), "details_file": str(details[0])}
    save(directory / "summary.json", summary)
    return summary


def wait_for_workers(session_id):
    """Wait only for this stage's session; never stop other GPU jobs."""
    import psutil

    deadline = time.monotonic() + 45
    while True:
        live = []
        for process in psutil.process_iter(["pid", "status"]):
            try:
                if os.getsid(process.pid) == session_id and process.info["status"] != psutil.STATUS_ZOMBIE:
                    live.append(process.pid)
            except (ProcessLookupError, psutil.NoSuchProcess):
                pass
        if not live:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Stage workers did not exit: {live}")
        time.sleep(1)


def run(root, phase):
    root = Path(root).resolve()
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "manifest.json").read_text())
    directory = root / phase
    if (directory / "status.json").exists():
        raise FileExistsError("Phase already exists; preserve its evidence")
    if json.loads((root / "download_status.json").read_text())["status"] != "completed":
        raise ValueError("Official model verification has not completed")
    if manifest["phases"][phase].get("kind", phase) == "formal":
        for prerequisite in manifest.get("qualification_phases", ["probe"]):
            previous = json.loads((root / prerequisite / "status.json").read_text())
            if previous["status"] != "completed":
                raise ValueError(f"Required {prerequisite} did not complete")
    for name, expected in manifest["source_sha256"].items():
        if sha256(repo / name) != expected:
            raise ValueError(f"Prepared source changed: {name}")
    if sha256(manifest["dataset"]["file"]) != manifest["dataset"]["sha256"]:
        raise ValueError("Prepared GPQA data changed")
    env = dict(os.environ)
    env.update(manifest["environment"])
    state = {"phase": phase, "status": "running", "controller_pid": os.getpid(),
             "started_at": datetime.now(timezone.utc).isoformat(),
             "command": manifest["commands"][phase], "log": str(directory / "main.log")}
    try:
        with (directory / "main.log").open("x") as log:
            process = subprocess.Popen(state["command"], cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["pid"] = process.pid
            save(directory / "status.json", state)
            save(root / "status.json", state)
            state["returncode"] = process.wait()
            wait_for_workers(process.pid)
        if state["returncode"]:
            raise RuntimeError(f"main.py exited with {state['returncode']}")
        state.update(summarize(root, phase, manifest), status="completed")
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(directory / "status.json", state)
        save(root / "status.json", state)
    return state


def run_sequence(root, qualification=False):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    phases = manifest["qualification_phases" if qualification else "formal_phases"]
    if not qualification:
        seeds = [manifest["phases"][name]["seed"] for name in phases]
        if len(phases) != 3 or len(set(seeds)) != 3 or manifest["sampling"]["n"] != 1:
            raise ValueError("Expected three sequential single-sample rounds with distinct seeds")
    completed = []
    for phase in phases:
        completed.append(run(root, phase))
        if qualification:
            continue
        if any(row["questions"] != 198 or row["generations"] != 198 for row in completed):
            raise ValueError("A round has incomplete GPQA coverage")
        done = len(completed) == len(phases)
        summary = {
            "status": "completed" if done else "running", "completed_rounds": len(completed),
            "requested_rounds": len(phases), "generations": sum(row["generations"] for row in completed),
            "seeds": [row["seed"] for row in completed],
            "accuracy_by_round": [row["accuracy"] for row in completed],
            "mean_accuracy": sum(row["accuracy"] for row in completed) / len(completed),
            "mean_definition": "arithmetic mean of full 198-question round accuracies, n=1 in each round",
            "accuracy_is_final": done and all(row["accuracy_is_final"] for row in completed),
            "needs_answer_review": any(row["needs_answer_review"] for row in completed),
            "review_count": sum(row["review_count"] for row in completed),
            "round_summaries": [str(root / row["phase"] / "summary.json") for row in completed],
        }
        save(root / "summary.json", summary)
    if not qualification:
        save(root / "status.json", {**summary, "controller_pid": os.getpid(),
                                    "finished_at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    if args.phase in ("qualification", "all_rounds"):
        run_sequence(args.run_dir, qualification=args.phase == "qualification")
    else:
        run(args.run_dir, args.phase)
