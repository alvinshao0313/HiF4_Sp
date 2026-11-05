"""Choose candidates only on the unchanged 64-math/64-code development set."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from ..reasoning_trace_calibration.generate import close_engine, engine, prompt
from ..reasoning_trace_calibration.scoring import code_score, math_score
from ..reasoning_trace_calibration.sources import reference
from .artifacts import digest, read_json, sha256, validate_root, verify_sources, write_json
from .config import CANDIDATES, ENGINE, GENERATION, NATIVE, SOURCE_RUN


def bind_inputs(root):
    """Bind question and reference manifests without opening reference answers."""
    root = validate_root(root)
    prepared = read_json(root / "prepare.inputs.json")
    if (not prepared["complete"]
            or digest({k: v for k, v in prepared.items() if k != "content_sha256"}) != prepared["content_sha256"]
            or Path(prepared["source_run"]).resolve() != SOURCE_RUN.resolve()):
        raise ValueError("development requires the verified source preparation")
    relative = "calibration/development.json"
    if sha256(root / relative) != prepared["files"][relative]:
        raise ValueError("fixed development questions changed")
    questions = read_json(root / relative)
    if ([r["id"] for r in questions] != prepared["development_ids"]
            or len({r["id"] for r in questions}) != 128
            or any(sum(r["domain"] == d for r in questions) != 64 for d in ("math", "code"))):
        raise ValueError("development must retain exactly 64 math and 64 code questions")
    pools_path = SOURCE_RUN / "isolation/pools.json"
    if sha256(pools_path) != prepared["source_files"][str(pools_path)]:
        raise ValueError("source development pools changed after preparation")
    pools = read_json(pools_path)
    if questions != [row for domain in ("math", "code") for row in pools[domain]["dev"]]:
        raise ValueError("development questions differ from the fixed source pool")
    references = read_json(SOURCE_RUN / "judging/reference_manifest.json")
    if not references["complete"]:
        raise ValueError("development code references were not prepared")
    for row in questions:
        if row["domain"] == "code":
            path = "judging/source_tests/" + Path(row["locator"]["file"]).name + ".json"
            if path not in references["files"] or not (SOURCE_RUN / path).is_file():
                raise ValueError(f"missing prepared development reference: {path}")
    binding = {
        "source_run": str(SOURCE_RUN), "count": 128, "domain_counts": {"math": 64, "code": 64},
        "development_sha256": sha256(root / relative), "prepare_inputs_sha256": sha256(root / "prepare.inputs.json"),
        "source_files": {str(SOURCE_RUN / name): sha256(SOURCE_RUN / name)
                         for name in ("isolation/pools.json", "data_sources/manifest.json", "judging/reference_manifest.json")},
        "sample_ids": [r["id"] for r in questions], "engine": ENGINE, "generation": GENERATION,
    }
    destination = root / "development/inputs.json"
    if destination.exists():
        if read_json(destination) != binding:
            raise ValueError("bound development questions, references, or generation protocol changed")
    else:
        write_json(binding, destination)
    return questions, binding


def score_summary(rows, questions):
    expected = {row["id"]: row["domain"] for row in questions}
    if len(rows) != len(expected) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("incomplete or duplicate development responses")
    for row in rows:
        if expected.get(row["id"]) != row["domain"] or row["score"] not in (0, 1):
            raise ValueError("development membership or binary scoring changed")
    scores = {domain: sum(row["score"] for row in rows if row["domain"] == domain)
              / sum(value == domain for value in expected.values()) for domain in ("math", "code")}
    return scores, sum(scores.values()) / 2


def evaluate(root, candidate, model_dir=None, *, smoke=False):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    root = validate_root(root)
    verify_sources(root)
    if candidate not in CANDIDATES:
        raise ValueError(candidate)
    questions, binding = bind_inputs(root)
    if smoke:
        questions = [next(row for row in questions if row["domain"] == domain) for domain in ("math", "code")]
    model = Path(model_dir).resolve() if model_dir is not None else root / ("qualification/models" if smoke else "models") / candidate
    exported = read_json(model / "export.json")
    if (not exported["complete"] or exported.get("candidate") != candidate
            or bool(exported.get("qualification")) != smoke
            or Path(exported["native_snapshot"]).resolve() != NATIVE.resolve()):
        raise ValueError("development checkpoint does not match the candidate, native source, or run mode")
    export_hash = sha256(model / "export.json")
    output = root / ("qualification/development" if smoke else "development") / candidate
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
    requests = []
    for row in questions:
        ids = prompt(tokenizer, row)
        if len(ids) + GENERATION["max_tokens"] > ENGINE["max_model_len"]:
            raise ValueError(f"development question exceeds the fixed prompt budget: {row['id']}")
        requests.append((row, ids))
    write_json({"complete": False, "candidate": candidate, "smoke": smoke,
                "selection_eligible": False, "question_count": len(questions),
                "inputs_sha256": sha256(root / "development/inputs.json"),
                "export_sha256": export_hash}, output / "manifest.json")
    llm = engine(model)
    rows = []
    try:
        results = llm.generate([{"prompt_token_ids": ids} for _, ids in requests],
                               SamplingParams(**GENERATION), use_tqdm=False)
        if len(results) != len(requests):
            raise ValueError("incomplete fixed development generation")
        for (source, ids), generation in zip(requests, results):
            if len(generation.outputs) != 1:
                raise ValueError("development requires exactly one response per question")
            result = generation.outputs[0]
            saved = {"id": source["id"], "domain": source["domain"], "prompt_token_ids": ids,
                     "generated_token_ids": list(result.token_ids), "text": result.text,
                     "finish_reason": result.finish_reason}
            write_json(saved, output / f"responses/{len(rows):03d}.json")
            rows.append(saved)
    finally:
        close_engine(llm)
    # Preserve the original separation: references are read after vLLM has closed.
    sources = {row["id"]: row for row in questions}
    for row in rows:
        answer = reference(SOURCE_RUN, sources[row["id"]])
        row["score"] = (math_score(row["text"], answer["solution"]) if row["domain"] == "math"
                        else code_score(row["text"], answer))
        print(json.dumps({"event": "development_score", "id": row["id"], "score": row["score"]}), flush=True)
    means, mean = score_summary(rows, questions)
    verify_sources(root)
    if bind_inputs(root)[1] != binding or sha256(model / "export.json") != export_hash:
        raise ValueError("development inputs or checkpoint changed during evaluation")
    metrics = {"candidate": candidate, "complete": True, "smoke": smoke, "selection_eligible": not smoke,
               "model_dir": str(model), "per_question": rows, "domain_scores": means, "equal_mean": mean,
               "export_sha256": export_hash, "inputs_sha256": sha256(root / "development/inputs.json"),
               "seconds": time.monotonic() - started}
    write_json(metrics, output / "metrics.json")
    write_json({key: value for key, value in metrics.items() if key != "per_question"}, output / "manifest.json")
    return metrics


def choose_candidate(scores, seconds):
    if set(scores) != set(CANDIDATES) or set(seconds) != set(CANDIDATES):
        raise ValueError("all predefined candidates need complete development scores and training times")
    if (any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values())
            or any(not math.isfinite(value) or value <= 0 for value in seconds.values())):
        raise ValueError("invalid development score or measured optimization time")
    order = {name: index for index, name in enumerate(CANDIDATES)}
    return min(CANDIDATES, key=lambda name: (-scores[name], seconds[name], order[name]))


def select(root):
    from .training import training_dir

    root = validate_root(root)
    verify_sources(root)
    questions, _ = bind_inputs(root)
    target = root / "development/selection.json"
    if target.exists():
        raise FileExistsError("the development winner was already selected")
    inputs_hash = sha256(root / "development/inputs.json")
    scores, seconds, evidence = {}, {}, {}
    for name in CANDIDATES:
        path = root / "development" / name / "metrics.json"
        result = read_json(path)
        if (not result["complete"] or result.get("smoke") or not result.get("selection_eligible")
                or result.get("candidate") != name or result["inputs_sha256"] != inputs_hash):
            raise ValueError("all candidates require complete formal development results on the bound set")
        means, score = score_summary(result["per_question"], questions)
        if result["domain_scores"] != means or result["equal_mean"] != score:
            raise ValueError("development aggregate differs from the saved per-question scores")
        model = Path(result["model_dir"])
        if sha256(model / "export.json") != result["export_sha256"]:
            raise ValueError("development checkpoint changed")
        training_manifest = training_dir(root, name) / "manifest.json"
        seconds[name] = read_json(training_manifest)["optimization_seconds"]
        scores[name] = score
        evidence[name] = {"metrics_sha256": sha256(path), "training_manifest_sha256": sha256(training_manifest),
                          "export_sha256": result["export_sha256"]}
    winner = choose_candidate(scores, seconds)
    write_json({"winner": winner, "complete": True, "scores": scores, "optimization_seconds": seconds,
                "inputs_sha256": inputs_hash, "evidence": evidence,
                "rule": "maximum equal mean of math accuracy and code pass@1; ties choose lower measured optimization_seconds, then CANDIDATES order"}, target)
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    if args.select:
        select(args.run)
    else:
        if args.candidate is None:
            parser.error("--candidate is required for development evaluation")
        evaluate(args.run, args.candidate, args.model_dir, smoke=args.smoke)
