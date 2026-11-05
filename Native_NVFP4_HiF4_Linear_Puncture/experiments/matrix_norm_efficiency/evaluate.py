"""Five full reasoning tasks with isolated caches and reviewed AIME avg5."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..downstream_accuracy_recovery.evaluate import check_model, reasoning_command
from ..downstream_accuracy_recovery.protocol import protocol_for
from ..downstream_accuracy_recovery.reporting import _check_reasoning_config, tokenizer_asset_hashes
from ..e2e_diag_reconstruction.evaluation.vllm_runner import parse_main_py_results
from ..module_design_ablation.evaluate_aime import model_view
from ..reasoning_trace_calibration.evaluation import AimeReviewRequired
from ..reasoning_trace_calibration.scoring import aime_submission
from .artifacts import digest, read_json, sha256, validate_root, verify_sources, write_json
from .config import ENGINE, MMLU_DATA, NATIVE, TASKS
from .pinned import install, manifest as data_manifest

REPO = Path(__file__).resolve().parents[3]
MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.matrix_norm_efficiency"
AIME24_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
AIME24_FILE = "data/train-00000-of-00001.parquet"
AIME24_SHA256 = "26139847601a5037c237d5928b195e7260ca8074cf4f264b794af42847f79ccf"


def task_protocol(task, *, smoke=False):
    if task not in TASKS:
        raise ValueError(f"unknown reasoning task: {task}")
    base = protocol_for("lcb" if task.startswith("aime") else task, smoke=smoke,
                        tensor_parallel_size=ENGINE["tensor_parallel_size"])
    if task.startswith("aime"):
        base.update(task_name=f"{task}_avg5|0", metric="avg@n:n=5",
                    count=1 if smoke else 30, generations_per_doc=5)
    return base


def task_registry():
    from lighteval.tasks.registry import Registry
    return Registry(tasks=",".join(task_protocol(t)["task_name"] for t in TASKS),
                    custom_tasks=str(REPO / "tasks/custom_tasks.py")).load_tasks()


def prepare_data(root):
    """Copy existing pinned contents; fetch only the missing official AIME24 source."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    from ..reasoning_trace_calibration.sources import evaluation_questions

    root = Path(root).resolve()
    destination = root / "evaluation/data"
    if (destination / "manifest.json").exists():
        value = data_manifest(root)
        if set(value["tasks"]) != set(TASKS):
            raise ValueError("prepared evaluation task set differs")
        return value
    destination.mkdir(parents=True, exist_ok=True)
    _, prior = evaluation_questions()
    registry, entries = task_registry(), {}
    for key in TASKS:
        task = registry[task_protocol(key)["task_name"]]
        split = "train" if key.startswith("aime") else "test"
        target = destination / key / f"{split}.arrow"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"partial preparation must be inspected before reuse: {target}")
        if key == "aime24":
            source = Path(hf_hub_download("HuggingFaceH4/aime_2024", AIME24_FILE,
                          repo_type="dataset", revision=AIME24_REVISION, endpoint="https://huggingface.co"))
            if sha256(source) != AIME24_SHA256:
                raise ValueError("official pinned AIME24 parquet content differs")
            table = pq.read_table(source)
            if table.num_rows != 30 or not {"problem", "answer"}.issubset(table.column_names):
                raise ValueError("official AIME24 shape or fields differ")
            with pa.OSFile(str(target), "wb") as stream:
                with pa.ipc.new_stream(stream, table.schema) as writer:
                    writer.write_table(table)
            provenance = {"source_revision": AIME24_REVISION, "source_file": AIME24_FILE,
                          "source_file_sha256": AIME24_SHA256}
        else:
            source = Path(prior[key]["path"])
            if sha256(source) != prior[key]["sha256"]:
                raise ValueError(f"existing pinned source changed: {source}")
            shutil.copyfile(source, target)
            provenance = {"source_revision": source.parent.name, "source_path": str(source),
                          "source_file_sha256": prior[key]["sha256"]}
        expected = task_protocol(key)["count"]
        if len(Dataset.from_file(str(target))) != expected:
            raise ValueError(f"unexpected full dataset count: {key}")
        splits = {split: {"path": str(target.relative_to(root)), "sha256": sha256(target)}}
        if key == "mmlu_pro":
            canonical = read_json(MMLU_DATA / "dataset_manifest.json")
            if task.dataset_revision != canonical["revision"]:
                raise ValueError("MMLU-Pro registry revision differs from canonical content")
            for name in ("test", "validation"):
                source = MMLU_DATA / f"data/{name}.arrow"
                if sha256(source) != canonical["splits"][name]["sha256"]:
                    raise ValueError(f"canonical MMLU-Pro {name} changed")
                target = destination / key / f"{name}.arrow"
                if name != "test":
                    shutil.copyfile(source, target)
                splits[name] = {"path": str(target.relative_to(root)), "sha256": sha256(target)}
            provenance["source_revision"] = canonical["revision"]
        entries[key] = {"repo": task.dataset_path, "registry_revision": task.dataset_revision,
                        "task_revision": provenance["source_revision"],
                        "subset": task.dataset_config_name, "count": expected,
                        "splits": splits, **provenance}
    value = {"complete": True, "tasks": entries}
    write_json(value, destination / "manifest.json")
    return data_manifest(root)


def audit_model(root, model, *, task_keys=None):
    from transformers import AutoTokenizer
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.prompt_manager import PromptManager

    install(root)
    tasks = task_registry()
    LightevalTask.load_datasets(tasks)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    managers = [PromptManager(True, tokenizer, enable_thinking=x) for x in (None, True, False)]
    result = {"tokenizer_files": tokenizer_asset_hashes(model, tokenizer), "tasks": {}}
    for key in TASKS if task_keys is None else task_keys:
        protocol = task_protocol(key)
        task = tasks[protocol["task_name"]]
        docs = task.get_docs(max_samples=None)
        if (len(docs) != protocol["count"] or len({str(d.id) for d in docs}) != len(docs)
                or any(d.num_samples != protocol["generations_per_doc"] for d in docs)):
            raise ValueError(f"actual task count or generation multiplicity differs: {key}")
        documents, maximum = {}, 0
        for doc in docs:
            default, thinking, direct = [manager.prepare_prompt(doc) for manager in managers]
            if default != thinking or default == direct:
                raise ValueError("default reasoning template must enable thinking")
            tokens = tokenizer.encode(default, add_special_tokens=False)
            maximum = max(maximum, len(tokens))
            documents[str(doc.id)] = {"prompt_sha256": digest(default), "token_sha256": digest(tokens),
                                      "gold_sha256": digest(doc.get_golds())}
        if maximum + protocol["max_new_tokens"] > protocol["max_model_length"]:
            raise ValueError(f"full task exceeds model context: {key}")
        result["tasks"][key] = {"task_name": protocol["task_name"], "metric": protocol["metric"],
            "hf_repo": task.dataset_path, "hf_revision": task.dataset_revision,
            "count": len(docs), "generations_per_doc": protocol["generations_per_doc"],
            "max_prompt_tokens": maximum, "documents": documents, "documents_sha256": digest(documents)}
        print(f"audited {key}: {len(docs)} questions x {protocol['generations_per_doc']}", flush=True)
    return result


def prepare(root):
    root = Path(root).resolve()
    prepare_data(root)
    path = root / "evaluation/prompt_audit.json"
    observed = audit_model(root, NATIVE)
    if path.exists() and read_json(path) != observed:
        raise ValueError("central native input audit changed")
    if not path.exists():
        write_json(observed, path)
    return observed


def result_details(result, task):
    folder = Path(result["results_file"]).parent.parent / "details"
    files = list(folder.rglob(f"details_{task_protocol(task)['task_name']}_*.json"))
    if len(files) != 1:
        raise ValueError(f"expected exactly one saved details file for {task}")
    return read_json(files[0])


def validate_result(result, task, audit, *, model_kind, smoke=False):
    expected = task_protocol(task, smoke=smoke)
    engine = {"tensor_parallel_size": expected["tensor_parallel_size"],
              "quantization": "modelopt_fp4" if model_kind == "native_nvfp4" else "None"}
    if (result.get("protocol") != expected or result.get("model_kind") != model_kind
            or result.get("exit_code") != 0 or result.get("actual_engine") != engine):
        raise ValueError("actual evaluation protocol, engine or exit status differs")
    raw = _check_reasoning_config(result, expected, model_kind)
    definition = raw["config_tasks"][expected["task_name"]]
    if (definition["hf_repo"] != audit["hf_repo"] or definition["hf_revision"] != audit["hf_revision"]
            or [m["metric_name"] for m in definition["metrics"]] != [expected["metric"]]):
        raise ValueError("saved task definition differs from audited task")
    score = result["results"][expected["task_name"]][expected["metric"]]
    if (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
            or not 0 <= score <= 1 or raw["results"][expected["task_name"]][expected["metric"]] != score):
        raise ValueError("invalid or inconsistent aggregate task score")
    details, paired = result_details(result, task), {}
    for row in details:
        key = str(row["doc"]["id"])
        reference = audit["documents"][key]
        response = row["model_response"]
        if (key in paired or len(response["text_post_processed"]) != expected["generations_per_doc"]
                or digest(response["input"]) != reference["prompt_sha256"]
                or digest(row["gold"]) != reference["gold_sha256"]
                or set(row["metric"]) != {expected["metric"]}):
            raise ValueError("actual prompt, gold, metric or generation count differs")
        value = row["metric"][expected["metric"]]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not 0 <= value <= 1
                or not math.isclose(value * expected["generations_per_doc"],
                                    round(value * expected["generations_per_doc"]), abs_tol=1e-10)):
            raise ValueError("question score is not the requested mean of binary matches")
        paired[key] = value
    if (len(paired) != expected["count"] or (not smoke and set(paired) != set(audit["documents"]))
            or not math.isclose(math.fsum(paired.values()) / len(paired), score, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("task coverage or per-question aggregation differs")
    return {"score": score, "paired_scores": paired, "questions": len(paired),
            "responses": len(paired) * expected["generations_per_doc"]}


def raw_cache_path(target):
    paths = list((Path(target) / "model").rglob("GENERATIVE.parquet"))
    if len(paths) != 1:
        raise ValueError("expected one isolated raw generation cache")
    return paths[0]


def validate_raw_cache(target, task, audit, *, smoke=False):
    import pyarrow.parquet as pq
    expected, observed = task_protocol(task, smoke=smoke), {}
    path = raw_cache_path(target)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
        for row in batch.to_pylist():
            key, sample = str(row["sample_id"]), row["sample"]
            reference = audit["documents"][key]
            if (key in observed or digest(sample["input"]) != reference["prompt_sha256"]
                    or digest(sample["input_tokens"]) != reference["token_sha256"]
                    or sample["truncated_tokens_count"] != 0
                    or len(sample["text"]) != expected["generations_per_doc"]
                    or len(sample["output_tokens"]) != expected["generations_per_doc"]
                    or any(not ids for ids in sample["output_tokens"])):
                raise ValueError("raw generation token IDs, truncation or multiplicity differs")
            observed[key] = [len(ids) for ids in sample["output_tokens"]]
    if len(observed) != expected["count"] or (not smoke and set(observed) != set(audit["documents"])):
        raise ValueError("raw generation cache has incomplete task coverage")
    checked = {"complete": True, "path": str(path), "sha256": sha256(path),
               "questions": len(observed), "responses": sum(map(len, observed.values())),
               "output_token_counts": observed, "actual_input_tokens_match_audit": True}
    write_json(checked, Path(target) / "raw_cache_validation.json")
    return checked


def correct_aime(target, result, task, *, smoke=False):
    import pyarrow.parquet as pq
    if task not in ("aime24", "aime25"):
        raise ValueError("final-submission correction is defined only for AIME")
    target = Path(target)
    details = result_details(result, task)
    path = raw_cache_path(target)
    cached = pq.read_table(path).to_pylist()
    cache = {str(row["sample_id"]): row["sample"] for row in cached}
    if (len(cache) != len(cached) or len(details) != task_protocol(task, smoke=smoke)["count"]
            or set(cache) != {str(row["doc"]["id"]) for row in details}):
        raise ValueError("duplicate or incomplete raw AIME coverage")
    reviews = read_json(target / "reviews.json") if (target / "reviews.json").exists() else {}
    responses, unresolved = [], []
    for row in details:
        key, gold = str(row["doc"]["id"]), row["gold"][0]
        sample = cache[key]
        if (sample["input"] != row["model_response"]["input"] or len(sample["text"]) != 5
                or len(sample["output_tokens"]) != 5):
            raise ValueError("AIME raw prompt or actual five responses differ")
        for i, text in enumerate(sample["text"]):
            answer = aime_submission(text, gold)
            record = {"id": key, "response_index": i + 1, "raw_sha256": digest(text), "gold": gold,
                      "tokens": len(sample["output_tokens"][i]), **answer}
            if answer["review"]:
                review = reviews.get(f"{key}:{i + 1}")
                if review is None:
                    unresolved.append({**record, "final_tail": text[-6000:]})
                elif (review.get("raw_sha256") != record["raw_sha256"]
                      or type(review.get("score")) is not int or review["score"] not in (0, 1)
                      or not isinstance(review.get("reason"), str) or not review["reason"].strip()):
                    raise ValueError("invalid manual AIME review or mismatched response hash")
                else:
                    record.update(score=review["score"], review=False,
                                  decision="manually_reviewed: " + review["reason"])
            responses.append(record)
    write_json({"task": task, "responses": responses, "unresolved": unresolved,
                "raw_cache_sha256": sha256(path)}, target / "corrected_responses.json")
    if unresolved:
        raise AimeReviewRequired(f"{len(unresolved)} {task} submissions await review: {target}")
    per_question = {key: math.fsum(r["score"] for r in responses if r["id"] == key) / 5 for key in cache}
    corrected = {"complete": True, "task": task, "scoring": "final_submission_avg5",
        "score": math.fsum(per_question.values()) / len(per_question), "per_question": per_question,
        "responses": len(responses), "corrected_responses_sha256": sha256(target / "corrected_responses.json")}
    write_json(corrected, target / "corrected_metrics.json")
    return corrected


def evaluate(root, candidate, task, *, model_dir=None, smoke=False):
    from ..ban_trace_calibration.processes import join_session
    from ..e2e_diag_reconstruction.evaluation.common import require_visible_cuda_count

    root = Path(validate_root(root))
    verify_sources(root)
    protocol = task_protocol(task, smoke=smoke)
    require_visible_cuda_count(protocol["tensor_parallel_size"])
    if not candidate or Path(candidate).name != candidate or candidate in (".", ".."):
        raise ValueError("candidate must be a single model name")
    kind = "native_nvfp4" if candidate == "native_nvfp4" else "hif4"
    source = (Path(model_dir) if model_dir is not None else NATIVE if kind == "native_nvfp4"
              else root / ("qualification/models" if smoke else "models") / candidate)
    if kind == "native_nvfp4":
        if source.resolve() != NATIVE.resolve():
            raise ValueError("native evaluation must use the pinned native snapshot")
    else:
        export = read_json(source / "export.json")
        if (export.get("complete") is not True or export.get("candidate") != candidate
                or export.get("qualification") is not smoke
                or Path(export.get("native_snapshot", "")).resolve() != NATIVE.resolve()):
            raise ValueError("candidate export identity, native source or qualification differs")
    target = root / ("qualification/evaluation" if smoke else "evaluation") / candidate / task
    model = model_view(source, target / "model")
    baseline = read_json(root / "evaluation/prompt_audit.json")
    audit = audit_model(root, model, task_keys=(task,))
    if (audit["tokenizer_files"] != baseline["tokenizer_files"]
            or audit["tasks"][task] != baseline["tasks"][task]):
        raise ValueError("candidate inputs differ from the central native audit")
    write_json(audit, target / "prompt_audit.json")
    spec, metadata = check_model(model, kind)
    output = target / "vllm_run"
    output.mkdir()
    command = reasoning_command(model, output, protocol, kind, spec)
    command[1:2] = ["-m", MODULE + ".reasoning_main"]
    environment = {**os.environ, "HIF4_EFFICIENCY_RUN": str(root)}
    environment.pop("HIF4_RUNTIME_SPEC_PATH", None)
    if spec is not None:
        environment["HIF4_RUNTIME_SPEC_PATH"] = str(spec)
    write_json({"command": command, "protocol": protocol, "model": metadata}, output / "command.json")
    logfile = output / "main_py.log"
    started = time.monotonic()
    with logfile.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json({"pid": child.pid, "session_id": child.pid, "command": command}, output / "process.json")
        status = child.wait()
    write_json({"command": command, "exit_code": status, "wall_seconds": time.monotonic() - started}, output / "exit.json")
    write_json(join_session(child.pid), output / "children_exit.json")
    if status:
        raise RuntimeError(f"reasoning main.py failed with exit {status}: {logfile}")
    engines = [json.loads(line.split("MATRIX_EFFICIENCY_ENGINE ", 1)[1])
               for line in logfile.read_text().splitlines() if line.startswith("MATRIX_EFFICIENCY_ENGINE ")]
    if len(engines) != 1:
        raise ValueError("expected exactly one actual engine configuration")
    result = {**parse_main_py_results(output), "protocol": protocol, "model_kind": kind,
              "model": metadata, "actual_engine": engines[0], "exit_code": status,
              "actual_quantization": engines[0]["quantization"], "log_file": str(logfile), "command": command}
    checked = validate_result(result, task, audit["tasks"][task], model_kind=kind, smoke=smoke)
    validate_raw_cache(target, task, audit["tasks"][task], smoke=smoke)
    write_json(result, target / "metrics.json")
    write_json(checked, target / "verified_scores.json")
    write_json(checked["paired_scores"], target / "paired_scores.json")
    verify_sources(root)
    if task.startswith("aime"):
        return correct_aime(target, result, task, smoke=smoke)
    return checked


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", help="CPU-only pinned-data and full-prompt preparation")
    prep.add_argument("--run", type=Path, required=True)
    for action in ("evaluate", "rescore-aime"):
        command = sub.add_parser(action)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--candidate", required=True)
        command.add_argument("--task", choices=TASKS, required=True)
        command.add_argument("--smoke", action="store_true")
        if action == "evaluate":
            command.add_argument("--model-dir", type=Path)
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "prepare":
        prepare(args["run"])
    elif action == "evaluate":
        args["root"] = args.pop("run")
        evaluate(**args)
    else:
        root = Path(validate_root(args["run"]))
        verify_sources(root)
        target = root / ("qualification/evaluation" if args["smoke"] else "evaluation") / args["candidate"] / args["task"]
        correct_aime(target, read_json(target / "metrics.json"), args["task"], smoke=args["smoke"])


if __name__ == "__main__":
    try:
        main()
    except AimeReviewRequired as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(21) from error
