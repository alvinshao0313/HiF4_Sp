"""TP8 four-task paired evaluation, retaining raw and reviewed AIME scores."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from ..downstream_accuracy_recovery.evaluate import check_model, reasoning_command
from ..downstream_accuracy_recovery.reporting import _check_reasoning_config
from ..e2e_diag_reconstruction.evaluation.common import require_visible_cuda_count
from ..e2e_diag_reconstruction.evaluation.vllm_runner import parse_main_py_results
from ..module_design_ablation.aime import load_details, validate_details
from ..module_design_ablation.evaluate_aime import model_view
from ..reasoning_trace_calibration.evaluation import AimeReviewRequired, correct_aime
from .artifacts import digest, read_json, sha256, verify_inputs, verify_sources, write_json
from .config import MODELS, MODULE, NATIVE, REPO, TASKS, task_protocol
from .prepare import audit_model


def result_details(result, task):
    if task == "aime25":
        return load_details(result)
    from ..downstream_accuracy_recovery.protocol import reasoning_details
    return reasoning_details(result, task)


def validate_result(result, task, audit, *, model_kind, smoke=False):
    expected = task_protocol(task, smoke=smoke)
    if (result.get("protocol") != expected or result.get("model_kind") != model_kind
            or result.get("exit_code") != 0):
        raise ValueError("evaluation protocol, model kind or exit status differs")
    actual = {"tensor_parallel_size": 8, "quantization": "modelopt_fp4" if model_kind == "native_nvfp4" else "None"}
    if result.get("actual_engine") != actual:
        raise ValueError("actual TP8 engine does not match the declared model kind")
    raw = _check_reasoning_config(result, expected, model_kind)
    definition = raw["config_tasks"][expected["task_name"]]
    if definition["hf_repo"] != audit["hf_repo"] or definition["hf_revision"] != audit["hf_revision"]:
        raise ValueError("task source/revision differs from the pinned input audit")
    score = result["results"][expected["task_name"]][expected["metric"]]
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("invalid benchmark score")
    if raw["results"][expected["task_name"]][expected["metric"]] != score:
        raise ValueError("score differs from the saved raw result")
    _, details = result_details(result, task)
    if len(details) != expected["count"] or len({str(row["doc"]["id"]) for row in details}) != expected["count"]:
        raise ValueError("missing or duplicate evaluation questions")
    ids, paired = set(), {}
    for row in details:
        key = str(row["doc"]["id"])
        reference = audit["documents"][key]
        response = row["model_response"]
        if (len(response["text_post_processed"]) != expected["generations_per_doc"]
                or digest(response["input"]) != reference["prompt_sha256"]
                or digest(row["gold"]) != reference["gold_sha256"]):
            raise ValueError("actual prompt, gold or response count differs")
        paired[key] = row["metric"][expected["metric"]]
        ids.add(key)
    if not smoke and ids != set(audit["documents"]):
        raise ValueError("formal task does not cover all audited question IDs")
    if not math.isclose(math.fsum(paired.values()) / len(paired), score, rel_tol=0, abs_tol=1e-12):
        raise ValueError("aggregate score differs from per-question scores")
    if task == "aime25":
        validate_details(details, score, audit=audit, smoke=smoke)
    return {"score": score, "paired_scores": paired, "questions": len(details),
            "responses": len(details) * expected["generations_per_doc"]}


def validate_raw_cache(target, task, audit, *, smoke=False):
    """Verify actual token IDs and multiplicity, not just the metric's name."""
    import pyarrow.parquet as pq
    expected = task_protocol(task, smoke=smoke)
    paths = list((target / "model").rglob("GENERATIVE.parquet"))
    if len(paths) != 1:
        raise ValueError("expected one isolated raw generation cache")
    observed = {}
    for batch in pq.ParquetFile(paths[0]).iter_batches(batch_size=8):
        for row in batch.to_pylist():
            key, sample = str(row["sample_id"]), row["sample"]
            reference = audit["documents"][key]
            if (key in observed or digest(sample["input"]) != reference["prompt_sha256"]
                    or digest(sample["input_tokens"]) != reference["token_sha256"]
                    or sample["truncated_tokens_count"] != 0
                    or len(sample["text"]) != expected["generations_per_doc"]
                    or len(sample["output_tokens"]) != expected["generations_per_doc"]
                    or any(not ids for ids in sample["output_tokens"])):
                raise ValueError("raw generation tokens, coverage or multiplicity differs")
            observed[key] = [len(ids) for ids in sample["output_tokens"]]
    if len(observed) != expected["count"] or (not smoke and set(observed) != set(audit["documents"])):
        raise ValueError("raw generation cache has incomplete question coverage")
    checked = {"complete": True, "path": str(paths[0]), "sha256": sha256(paths[0]),
               "questions": len(observed), "responses": sum(map(len, observed.values())),
               "output_token_counts": observed, "actual_input_tokens_match_audit": True}
    write_json(checked, target / "raw_cache_validation.json")
    return checked


def evaluate(root, candidate, task, *, smoke=False):
    root = verify_inputs(root)
    verify_sources(root)
    require_visible_cuda_count(8)
    if candidate not in MODELS or task not in TASKS:
        raise ValueError("unknown model/task")
    kind = "native_nvfp4" if candidate == MODELS[0] else "hif4"
    source = NATIVE if kind == "native_nvfp4" else root / "models/matrix_norm"
    if kind == "hif4":
        export = read_json(source / "export.json")
        if not export["complete"] or export["qualification"] or export["layers"] != 94:
            raise ValueError("full 235B export is incomplete or belongs to the wrong phase")
    target = root / ("qualification/evaluation" if smoke else "evaluation") / candidate / task
    model = model_view(source, target / "model")
    expected_audit = read_json(root / "prompt_audit.json")
    actual_audit = audit_model(root, model, task_keys=(task,))
    if (actual_audit["tokenizer_files"] != expected_audit["tokenizer_files"]
            or actual_audit["tasks"][task] != expected_audit["tasks"][task]):
        raise ValueError("model tokenizer or task contents differ from the central native audit")
    write_json(actual_audit, target / "prompt_audit.json")
    protocol = task_protocol(task, smoke=smoke)
    spec, metadata = check_model(model, kind)
    output = target / "vllm_run"
    output.mkdir()
    command = reasoning_command(model, output, protocol, kind, spec)
    command[1:2] = ["-m", MODULE + ".reasoning_main"]
    env = {**os.environ, "HIF4_235B_RUN": str(root)}
    env.pop("HIF4_RUNTIME_SPEC_PATH", None)
    if spec is not None:
        env["HIF4_RUNTIME_SPEC_PATH"] = str(spec)
    write_json({"command": command, "protocol": protocol, "model": metadata}, output / "command.json")
    logfile = output / "main_py.log"
    with logfile.open("w", encoding="utf-8") as log:
        child = subprocess.run(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, check=False)
    write_json({"command": command, "exit_code": child.returncode}, output / "exit.json")
    if child.returncode:
        raise RuntimeError(f"main.py failed with exit {child.returncode}: {logfile}")
    engines = [json.loads(line.split("MATRIX235_ENGINE_CONFIG ", 1)[1])
               for line in logfile.read_text().splitlines() if line.startswith("MATRIX235_ENGINE_CONFIG ")]
    if len(engines) != 1:
        raise ValueError("expected exactly one actual engine configuration")
    result = {**parse_main_py_results(output), "protocol": protocol, "model_kind": kind, "model": metadata,
              "actual_engine": engines[0], "actual_quantization": engines[0]["quantization"],
              "exit_code": child.returncode, "log_file": str(logfile), "command": command}
    checked = validate_result(result, task, actual_audit["tasks"][task], model_kind=kind, smoke=smoke)
    validate_raw_cache(target, task, actual_audit["tasks"][task], smoke=smoke)
    # Preserve raw metrics before any separate final-submission review.
    write_json(result, target / "metrics.json")
    write_json(checked, target / "verified_scores.json")
    verify_sources(root)
    if task == "aime25":
        correct_aime(target, result, smoke=smoke)
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--candidate", choices=MODELS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rescore-aime", action="store_true")
    args = parser.parse_args()
    try:
        if args.rescore_aime:
            if args.task != "aime25":
                parser.error("--rescore-aime requires --task aime25")
            root = verify_inputs(args.run)
            target = root / ("qualification/evaluation" if args.smoke else "evaluation") / args.candidate / "aime25"
            correct_aime(target, read_json(target / "metrics.json"), smoke=args.smoke)
        else:
            evaluate(args.run, args.candidate, args.task, smoke=args.smoke)
    except AimeReviewRequired as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(21) from error
