"""AIME25 avg5 protocol and checks against actual prompts and question scores."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..downstream_accuracy_recovery.protocol import protocol_for
from ..downstream_accuracy_recovery.reporting import tokenizer_asset_hashes

REPO = Path(__file__).resolve().parents[3]
TASK_NAME = "aime25_avg5|0"
METRIC = "avg@n:n=5"
COUNT = 30
GENERATIONS = 5


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def aime_protocol(*, smoke=False):
    # Reuse the reasoning engine settings, without changing the five-task protocol.
    return {**protocol_for("lcb", smoke=smoke), "task_name": TASK_NAME,
            "metric": METRIC, "count": 1 if smoke else COUNT,
            "generations_per_doc": GENERATIONS}


def audit_aime(model_dir):
    """Read the actual task and tokenizer; no model inference or test-set tuning."""
    from transformers import AutoTokenizer
    from lighteval.tasks.registry import Registry
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.prompt_manager import PromptManager

    model = Path(model_dir)
    tasks = Registry(tasks=TASK_NAME, custom_tasks=str(REPO / "tasks/aime.py")).load_tasks()
    LightevalTask.load_datasets(tasks)
    task = tasks[TASK_NAME]
    docs = task.get_docs(max_samples=None)
    if len(docs) != COUNT or len({str(doc.id) for doc in docs}) != COUNT:
        raise ValueError("AIME25 requires all 30 unique questions")
    if any(doc.num_samples != GENERATIONS for doc in docs):
        raise ValueError("AIME25 task must request five generations per question")
    if [metric.metric_name for metric in task.metrics] != [METRIC]:
        raise ValueError("AIME25 must use the avg5 math metric")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError("AIME25 reasoning requires the model chat template")
    default = PromptManager(True, tokenizer, enable_thinking=None)
    thinking = PromptManager(True, tokenizer, enable_thinking=True)
    no_thinking = PromptManager(True, tokenizer, enable_thinking=False)
    documents = {}
    lengths = []
    for doc in docs:
        prompt = default.prepare_prompt(doc)
        if prompt != thinking.prepare_prompt(doc) or prompt == no_thinking.prepare_prompt(doc):
            raise ValueError("default chat template must enable thinking")
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        lengths.append(len(tokens))
        documents[str(doc.id)] = {"prompt_sha256": digest(prompt),
                                  "token_sha256": digest(tokens), "gold_sha256": digest(doc.get_golds())}
    protocol = aime_protocol()
    if max(lengths) + protocol["max_new_tokens"] > protocol["max_model_length"]:
        raise ValueError("AIME25 prompt plus generation exceeds the context budget")
    return {"task_name": TASK_NAME, "metric": METRIC, "count": COUNT,
            "generations_per_doc": GENERATIONS, "max_prompt_tokens": max(lengths),
            "hf_repo": task.dataset_path, "hf_revision": task.dataset_revision,
            "dataset_fingerprint": task.dataset["train"]._fingerprint,
            "documents": documents, "documents_sha256": digest(documents),
            "tokenizer_files": tokenizer_asset_hashes(model, tokenizer)}


def _fraction(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    return float(value)


def validate_details(details, score, *, audit, smoke=False):
    """Check coverage and aggregation; saved avg5 scores are not pass@5 scores."""
    count = 1 if smoke else COUNT
    score = _fraction(score, "aggregate score")
    if (audit.get("task_name") != TASK_NAME or audit.get("metric") != METRIC
            or audit.get("count") != COUNT or audit.get("generations_per_doc") != GENERATIONS
            or len(audit.get("documents", {})) != COUNT
            or audit.get("documents_sha256") != digest(audit["documents"])):
        raise ValueError("a complete AIME25 avg5 input audit is required")
    if len(details) != count:
        raise ValueError(f"expected {count} AIME25 questions, got {len(details)}")
    ids = [str(row["doc"]["id"]) for row in details]
    if len(set(ids)) != count:
        raise ValueError("duplicate AIME25 question IDs")
    if not set(ids).issubset(audit["documents"]) or (not smoke and set(ids) != set(audit["documents"])):
        raise ValueError("AIME25 question IDs differ from the input audit")
    per_question = []
    for row, sample_id in zip(details, ids):
        responses = row["model_response"]["text_post_processed"]
        if not isinstance(responses, list) or len(responses) != GENERATIONS or not all(
                isinstance(response, str) for response in responses):
            raise ValueError(f"AIME25 question {sample_id} must have exactly five text responses")
        reference = audit["documents"][sample_id]
        if digest(row["model_response"]["input"]) != reference["prompt_sha256"]:
            raise ValueError(f"AIME25 prompt differs from audit: {sample_id}")
        if digest(row["gold"]) != reference["gold_sha256"]:
            raise ValueError(f"AIME25 gold differs from audit: {sample_id}")
        if set(row["metric"]) != {METRIC}:
            raise ValueError("AIME25 requires avg@n:n=5, not pass@5 or majority voting")
        value = _fraction(row["metric"][METRIC], f"question {sample_id} score")
        # Each of the five official math matches is binary.
        if not math.isclose(value * GENERATIONS, round(value * GENERATIONS), rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"AIME25 question {sample_id} score is not an average of five binary matches")
        per_question.append({"id": sample_id, "avg5": value})
    mean = math.fsum(row["avg5"] for row in per_question) / count
    if not math.isclose(mean, score, rel_tol=0, abs_tol=1e-12):
        raise ValueError("AIME25 aggregate differs from mean per-question avg5")
    return {"score": score, "question_count": count, "generation_count": count * GENERATIONS,
            "per_question": per_question, "smoke_only": smoke,
            "evaluated_documents_sha256": digest({sample_id: audit["documents"][sample_id] for sample_id in ids})}


def load_details(result):
    root = Path(result["results_file"]).parent.parent / "details"
    files = list(root.rglob(f"details_{TASK_NAME}_*.json"))
    if len(files) != 1:
        raise ValueError(f"expected one AIME25 details file under {root}, got {len(files)}")
    return files[0], json.loads(files[0].read_text(encoding="utf-8"))


def validate_sample_limit(limit, *, smoke):
    if smoke:
        valid = type(limit) in (int, float) and limit == 1
    else:
        valid = limit is None or (type(limit) in (int, float) and math.isfinite(limit) and limit >= COUNT)
    if not valid:
        raise ValueError("AIME25 sample limit does not cover the requested questions")


def validate_result(result, *, audit, model_kind, smoke=False):
    from ..downstream_accuracy_recovery.reporting import _check_reasoning_config

    expected = aime_protocol(smoke=smoke)
    if result.get("protocol") != expected:
        raise ValueError("AIME25 result does not match the requested protocol")
    raw = json.loads(Path(result["results_file"]).read_text(encoding="utf-8"))
    limit = raw["config_general"]["max_samples"]
    validate_sample_limit(limit, smoke=smoke)
    # A cap >= 30 and no cap both cover the full task. Actual IDs/prompts/golds
    # below establish coverage; unrelated tasks' larger caps are not a mismatch.
    raw = _check_reasoning_config(result, {**expected, "max_samples": limit}, model_kind)
    if (result.get("model_kind") != model_kind or result.get("exit_code") != 0
            or result.get("actual_quantization") != ("modelopt_fp4" if model_kind == "native_nvfp4" else "None")):
        raise ValueError("AIME25 model kind, engine quantization or exit status is invalid")
    score = result["results"][TASK_NAME][METRIC]
    if raw["results"][TASK_NAME][METRIC] != score:
        raise ValueError("AIME25 score differs from the raw result")
    config = raw["config_tasks"][TASK_NAME]
    if (config["hf_repo"] != audit["hf_repo"] or config["hf_revision"] != audit["hf_revision"]
            or [metric["metric_name"] for metric in config["metrics"]] != [METRIC]):
        raise ValueError("AIME25 task definition differs from the audited task")
    details_path, details = load_details(result)
    checked = validate_details(details, score, audit=audit, smoke=smoke)
    return {**checked, "details_file": str(details_path)}
