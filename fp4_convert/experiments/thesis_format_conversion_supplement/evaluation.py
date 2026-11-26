"""Independent five-task entry, with pinned reasoning data and isolated caches."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..downstream_accuracy_recovery.evaluate import check_model, reasoning_command
from ..downstream_accuracy_recovery.protocol import protocol_for
from ..downstream_accuracy_recovery.reporting import tokenizer_asset_hashes
from ..e2e_diag_reconstruction.evaluation.vllm_runner import parse_main_py_results
from ..module_design_ablation.evaluate_aime import model_view
from ..matrix_norm_efficiency.pinned import install
from ..matrix_norm_efficiency.evaluate import validate_result, validate_raw_cache
from ..ban_trace_calibration.processes import require_free, finish_stage
from .artifacts import read_json, write_json, digest, verify_sources, run_root
from .config import *


def audit_model(root, model, tasks=REASONING):
    from transformers import AutoTokenizer
    from lighteval.tasks.registry import Registry
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.prompt_manager import PromptManager
    install(root)
    registry = Registry(tasks=",".join(protocol_for(t)["task_name"] for t in tasks)).load_tasks()
    LightevalTask.load_datasets(registry)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    managers = [PromptManager(True, tokenizer, enable_thinking=x) for x in (None, True, False)]
    result = {"tokenizer_files": tokenizer_asset_hashes(model, tokenizer), "tasks": {}}
    for key in tasks:
        protocol = protocol_for(key)
        task = registry[protocol["task_name"]]
        documents, maximum = {}, 0
        for doc in task.get_docs(max_samples=None):
            default, thinking, direct = [m.prepare_prompt(doc) for m in managers]
            if default != thinking or default == direct or doc.num_samples != 1 or str(doc.id) in documents:
                raise ValueError("prompt thinking, multiplicity or ID contract differs")
            tokens = tokenizer.encode(default, add_special_tokens=False)
            maximum = max(maximum, len(tokens))
            documents[str(doc.id)] = {"prompt_sha256": digest(default), "token_sha256": digest(tokens),
                                      "gold_sha256": digest(doc.get_golds())}
        if len(documents) != COUNTS[key] or maximum + protocol["max_new_tokens"] > 40960:
            raise ValueError(f"task coverage/context differs: {key}")
        result["tasks"][key] = {"task_name": protocol["task_name"], "metric": protocol["metric"],
            "hf_repo": task.dataset_path, "hf_revision": task.dataset_revision, "count": len(documents),
            "generations_per_doc": 1, "max_prompt_tokens": maximum, "documents": documents,
            "documents_sha256": digest(documents)}
        print(f"audited {key}: {len(documents)}", flush=True)
    return result


def model_for(root, candidate):
    fresh = root / "models" / candidate
    if (fresh / "ablation_export.json").is_file() and candidate in ("direct", "r64"):
        marker = read_json(fresh / "ablation_export.json")
        expected = {"direct": "direct_hif4", "r64": "fixed_r64"}[candidate]
        if (not marker["complete"] or marker["smoke_only"] or marker["variant"] != expected
                or Path(marker["source_snapshot"]).resolve() != NATIVE.resolve()):
            raise ValueError("re-exported reference identity differs")
        return fresh
    inventory = read_json(root / "inventory.json")
    if candidate in inventory:
        if not inventory[candidate]["model_present"]:
            raise FileNotFoundError(f"historical model absent: {candidate}; export is required")
        return Path(inventory[candidate]["model_path"])
    model = root / "models" / candidate
    export = read_json(model / "export.json")
    if not export["complete"] or export.get("qualification") or export["candidate"] != candidate:
        raise ValueError("incomplete or diagnostic export")
    return model


def evaluate(root, candidate, task, *, smoke=False, repetition=1):
    root = run_root(root)
    verify_sources(root)
    pair = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if pair not in ("0,1", "6,7"):
        raise ValueError("evaluation requires an allocated TP2 pair (0,1 or 6,7)")
    require_free(pair)
    if repetition not in range(1, 6):
        raise ValueError("repetition must be 1..5")
    target = root / ("qualification/evaluation" if smoke else "evaluation") / candidate / f"round_{repetition:02d}" / task
    # Never overwrite an unfinished stage or attach its cache to a new run.
    model = model_view(model_for(root, candidate), target / "model")
    kind = "native_nvfp4" if candidate == "native" else "hif4"
    spec, metadata = check_model(model, kind)
    if task == "arc":
        command = [sys.executable, "-m", MODULE + ".arc_main",
                   "--model_dir", str(model), "--output_dir", str(target), "--task", "arc", "--model_kind", kind]
        if smoke:
            command.append("--smoke")
        protocol = protocol_for("arc", smoke=smoke)
    else:
        protocol = protocol_for(task, smoke=smoke)
        audit = audit_model(root, model, (task,))
        baseline = read_json(root / "evaluation/prompt_audit.json")
        if audit["tokenizer_files"] != baseline["tokenizer_files"] or audit["tasks"][task] != baseline["tasks"][task]:
            raise ValueError("actual task inputs differ from native")
        write_json(audit, target / "prompt_audit.json")
        command = reasoning_command(model, target / "vllm_run", protocol, kind, spec)
        command[1:2] = ["-m", MODULE + ".reasoning_main"]
    env = {**os.environ, "HIF4_SUPPLEMENT_RUN": str(root)}
    env.pop("HIF4_RUNTIME_SPEC_PATH", None)
    if spec is not None:
        env["HIF4_RUNTIME_SPEC_PATH"] = str(spec)
    output = target / "vllm_run"
    output.mkdir()
    logfile = output / "main_py.log"
    write_json({"command": command, "protocol": protocol, "model": metadata}, output / "command.json")
    started = time.monotonic()
    with logfile.open("w") as log:
        child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json({"pid": child.pid, "session_id": child.pid, "command": command,
                    "gpus": pair, "log": str(logfile)}, output / "process.json")
        status = child.wait()
    write_json({"exit_code": status, "seconds": time.monotonic() - started}, output / "exit.json")
    write_json(finish_stage(child.pid, pair), output / "children_exit.json")
    if status:
        raise RuntimeError(f"evaluation exit={status}: {logfile}")
    if task != "arc":
        engines = [json.loads(line.split("MATRIX_EFFICIENCY_ENGINE ", 1)[1]) for line in logfile.read_text().splitlines()
                   if line.startswith("MATRIX_EFFICIENCY_ENGINE ")]
        if len(engines) != 1:
            raise ValueError("expected one engine record")
        result = {**parse_main_py_results(output), "protocol": protocol, "model_kind": kind,
                  "model": metadata, "actual_engine": engines[0], "exit_code": status,
                  "actual_quantization": engines[0]["quantization"], "log_file": str(logfile), "command": command}
        checked = validate_result(result, task, audit["tasks"][task], model_kind=kind, smoke=smoke)
        validate_raw_cache(target, task, audit["tasks"][task], smoke=smoke)
        write_json(result, target / "metrics.json")
        write_json(checked, target / "verified_scores.json")
    verify_sources(root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate", required=True, choices=MAIN + EFFICIENCY_ROLES + ("orthogonal_a20",))
    p.add_argument("--task", choices=REASONING + ("arc",), required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--repetition", type=int, default=1)
    a = vars(p.parse_args()); a["root"] = a.pop("run")
    evaluate(**a)
