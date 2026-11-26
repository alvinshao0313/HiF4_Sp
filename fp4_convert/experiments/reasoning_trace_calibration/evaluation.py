"""Full paired downstream protocol with separately corrected AIME submissions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from ..downstream_accuracy_recovery import evaluate as shared
from ..downstream_accuracy_recovery.protocol import protocol_for, reasoning_details
from ..downstream_accuracy_recovery.reporting import audit_prompts, validate_metrics
from ..e2e_diag_reconstruction.evaluation.vllm_runner import parse_main_py_results
from ..module_design_ablation.aime import aime_protocol, audit_aime, load_details, validate_result
from ..module_design_ablation.evaluate_aime import model_view
from .artifacts import digest, read_json, sha256, validate_root, verify_sources, write_json
from .config import MODELS, MODULE, NATIVE, REPO
from .pinned import install
from .scoring import aime_submission


class AimeReviewRequired(Exception):
    """Raw answers are complete; final scores await review of ambiguous submissions."""


def run_reasoning(model, output, protocol, model_kind, spec):
    output.mkdir(parents=True, exist_ok=False)
    command = shared.reasoning_command(model, output, protocol, model_kind, spec)
    command[1:2] = ["-m", MODULE + ".reasoning_main"]
    write_json({"command": command, "protocol": protocol}, output / "command.json")
    logfile = output / "main_py.log"
    with logfile.open("w", encoding="utf-8") as log:
        child = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False)
    write_json({"command": command, "exit_code": child.returncode}, output / "exit.json")
    if child.returncode:
        raise RuntimeError(f"reasoning main.py failed: {logfile}")
    actual = re.findall(r"\bquantization=([^,\s]+)", logfile.read_text())
    expected = "modelopt_fp4" if model_kind == "native_nvfp4" else "None"
    if actual != [expected]:
        raise ValueError(f"unexpected engine quantization: {actual}")
    return {**parse_main_py_results(output), "actual_quantization": expected, "exit_code": 0,
            "log_file": str(logfile), "command": command}


def correct_aime(output, result, *, smoke=False):
    import pyarrow.parquet as pq
    _, details = load_details(result)
    paths = list((output / "model").rglob("GENERATIVE.parquet"))
    if len(paths) != 1:
        raise ValueError("expected exactly one raw AIME response cache")
    cached = pq.read_table(paths[0]).to_pylist()
    cache = {str(row["sample_id"]): row["sample"] for row in cached}
    if (len(cached) != len(cache) or len(details) != (1 if smoke else 30)
            or set(cache) != {str(row["doc"]["id"]) for row in details}):
        raise ValueError("duplicate/incomplete AIME coverage")
    overrides_path = output / "reviews.json"
    overrides = read_json(overrides_path) if overrides_path.exists() else {}
    responses, unresolved = [], []
    for row in details:
        qid = str(row["doc"]["id"])
        raw = cache[qid]
        if raw["input"] != row["model_response"]["input"] or len(raw["text"]) != 5 or len(raw["output_tokens"]) != 5:
            raise ValueError("AIME raw prompt/generation coverage differs")
        for i, text in enumerate(raw["text"]):
            submission = aime_submission(text, row["gold"][0])
            record = {"id": qid, "response_index": i + 1, "raw_sha256": digest(text),
                      "gold": row["gold"][0], "tokens": len(raw["output_tokens"][i]), **submission}
            key = f"{qid}:{i + 1}"
            if submission["review"]:
                review = overrides.get(key)
                if review is None:
                    unresolved.append({**record, "final_tail": text[-6000:]})
                elif (review.get("raw_sha256") != record["raw_sha256"] or review.get("score") not in (0, 1)
                      or not review.get("reason")):
                    raise ValueError(f"invalid AIME manual review: {key}")
                else:
                    record.update(score=review["score"], review=False, decision="manually_reviewed: " + review["reason"])
            responses.append(record)
    write_json({"responses": responses, "unresolved": unresolved, "raw_cache_sha256": sha256(paths[0])},
               output / "corrected_responses.json")
    if unresolved:
        raise AimeReviewRequired(f"{len(unresolved)} ambiguous AIME submissions require review: {output / 'corrected_responses.json'}")
    per_question = {qid: sum(r["score"] for r in responses if r["id"] == qid) / 5 for qid in cache}
    if len(responses) != len(per_question) * 5:
        raise ValueError("AIME avg5 coverage is incomplete")
    write_json({"complete": True, "score": sum(per_question.values()) / len(per_question),
                "per_question": per_question, "responses": len(responses),
                "corrected_responses_sha256": sha256(output / "corrected_responses.json")}, output / "corrected_metrics.json")


def evaluate(root, candidate, task, *, qualification=False):
    root = validate_root(root)
    verify_sources(root)
    if candidate not in MODELS or task not in ("arc", "mmlu_pro", "lcb", "gsm8k", "aime25"):
        raise ValueError("unknown model/task")
    if not qualification:
        selection = read_json(root / "development/selection.json")
        for name in MODELS[:2]:
            if sha256(root / "models" / name / "export.json") != selection["export_sha256"][name]:
                raise ValueError("checkpoint changed after development selection")
    source = root / ("qualification/models" if qualification else "models") / candidate
    export = read_json(source / "export.json")
    if export["qualification"] != qualification or not export["complete"]:
        raise ValueError("wrong model export status")
    target = root / ("qualification/evaluation" if qualification else "evaluation") / candidate / task
    install()
    if task != "aime25":
        model = model_view(source, target / "model")
        audit = audit_prompts({"native_nvfp4": NATIVE, candidate: model}, target / "prompt_audit.json")
        original = shared.run_reasoning
        shared.run_reasoning = run_reasoning
        try:
            result = shared.evaluate(model, target, task, model_kind="hif4", smoke=qualification, tensor_parallel_size=2)
        finally:
            shared.run_reasoning = original
        reference = None if task == "arc" else audit["models"][candidate]["tasks"][task]
        scores = validate_metrics(result, task, smoke=qualification, prompt_reference=reference, model_kind="hif4")
        if task != "arc":
            _, details = reasoning_details(result, task)
            definition = protocol_for(task, smoke=qualification)
            rows = {str(r["doc"]["id"]): r["metric"][definition["metric"]] for r in details}
            write_json(rows, target / "paired_scores.json")
        write_json(scores, target / "verified_scores.json")
        return scores
    model = model_view(source, target / "model")
    audit = audit_aime(model)
    write_json(audit, target / "prompt_audit.json")
    protocol = aime_protocol(smoke=qualification)
    spec, metadata = shared.check_model(model, "hif4")
    result = run_reasoning(model, target / "vllm_run", protocol, "hif4", spec)
    result.update(protocol=protocol, model_kind="hif4", model=metadata)
    validate_result(result, audit=audit, model_kind="hif4", smoke=qualification)
    write_json(result, target / "metrics.json")
    correct_aime(target, result, smoke=qualification)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", choices=MODELS, required=True)
    parser.add_argument("--task", choices=("arc", "mmlu_pro", "lcb", "gsm8k", "aime25"), required=True)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--rescore-aime", action="store_true")
    args = parser.parse_args()
    try:
        if args.rescore_aime:
            target = args.run / ("qualification/evaluation" if args.qualification else "evaluation") / args.candidate / "aime25"
            correct_aime(target, read_json(target / "metrics.json"), smoke=args.qualification)
        else:
            evaluate(args.run, args.candidate, args.task, qualification=args.qualification)
    except AimeReviewRequired as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(21) from error
