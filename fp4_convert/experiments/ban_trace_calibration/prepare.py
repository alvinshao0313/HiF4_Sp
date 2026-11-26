"""Freeze original inputs and complete task audits before GPU execution."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..ban.additional_main import install_datasets
from ..ban.artifact import load_artifact, model_identity
from ..ban.evaluate import file_sha256
from ..ban.evaluation_report import _sha256
from .config import C4, DATASETS, INFERENCE, MODEL, TASKS, TRACES, now, read, save, source_identity
from .data import budget_inputs, load_traces, sample_windows


def task_audits(root, model):
    from transformers import AutoTokenizer
    from lighteval.tasks.prompt_manager import PromptManager
    from lighteval.tasks.registry import Registry
    from ..ban.evaluate import REPO

    install_datasets(root / "datasets.json")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    manager = PromptManager(use_chat_template=True, tokenizer=tokenizer, enable_thinking=None)
    tasks = Registry(tasks=",".join(d["task"] for d in TASKS.values()),
                     custom_tasks=str(REPO / "tasks/custom_tasks.py")).load_tasks()
    for name, definition in TASKS.items():
        task = tasks[definition["task"]]
        docs = task.get_docs(max_samples=None)
        if len(docs) != definition["count"] or any(doc.num_samples != 1 for doc in docs):
            raise ValueError(f"full task coverage/generation count differs: {name}")
        documents = []
        for doc in docs:
            prompt = manager.prepare_prompt(doc)
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            limit = 40960 - definition["max_new_tokens"]
            documents.append({"id": str(doc.id), "prompt_sha256": _sha256(prompt),
                              "gold_sha256": _sha256(doc.get_golds()),
                              "input_tokens_sha256": _sha256(ids[-limit:]),
                              "full_input_tokens": len(ids)})
        if len({row["id"] for row in documents}) != len(documents):
            raise ValueError("duplicate formal task IDs")
        save(root / f"{name}_input_audit.json", {**definition, "documents": documents,
             "dataset_path": task.dataset_path, "revision": task.dataset_revision})


def prepare(output, target_experts=6.07):
    from .config import budget_config
    budget = budget_config(target_experts)
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(f"output must be a new directory: {root}")
    spec = MODEL / "hif4_runtime_spec.pt"
    identity = model_identity(MODEL, spec)
    c4 = load_artifact(C4, identity)
    if (c4.get("phase"), c4.get("num_samples"), c4.get("k_pruned"), c4.get("topn"), c4.get("kl_positions")) != (
            "formal", 512, 3, 1000, "last"):
        raise ValueError("C4 reference must be the qualified 512-window formal calibration")
    c4_manifest = read(C4.parent / "manifest.json")
    if c4_manifest["status"] != "complete" or c4_manifest["artifact_sha256"] != file_sha256(C4):
        raise ValueError("C4 reference completion/hash differs")
    # The existing original/native-router, source parity and compact-kernel
    # evidence applies only while its actual dependency sources are unchanged.
    qualification = read(DATASETS.parent / "manifest.json")["code"]
    from ..ban.evaluate import REPO
    for name, digest in qualification["source_sha256"].items():
        if file_sha256(REPO / name) != digest:
            raise ValueError(f"old Ban qualification source changed: {name}")
    records, inputs = load_traces(TRACES)
    windows, window_data = sample_windows(records["train"])
    diagnostic = torch.stack([windows[0], windows[256]])
    held_out = budget_inputs(records["val"])
    if {r["id"] for r in held_out} & {r["id"] for r in window_data["windows"]}:
        raise ValueError("budget prompts overlap calibration questions")
    routes = {name: read(DATASETS)[name] for name in ("TIGER-Lab/MMLU-Pro", "lighteval/code_generation_lite")}
    if routes["TIGER-Lab/MMLU-Pro"]["revision"] != "3373e0b32277875b8db2aa555a333b78a08477ea":
        raise ValueError("unexpected MMLU-Pro revision")
    for route in routes.values():
        for split in route["splits"].values():
            if file_sha256(split["path"]) != split["sha256"]:
                raise ValueError("pinned formal Arrow changed")
            inputs[split["path"]] = split["sha256"]
    for path in (C4, C4.parent / "manifest.json", DATASETS):
        inputs[str(path)] = file_sha256(path)
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    torch.save(windows, root / "calibration_inputs.pt")
    torch.save(diagnostic, root / "diagnostic_inputs.pt")
    save(root / "windows.json", window_data)
    save(root / "budget_inputs.json", held_out)
    save(root / "datasets.json", routes)
    task_audits(root, MODEL)
    for name in ("calibration_inputs.pt", "diagnostic_inputs.pt", "windows.json", "budget_inputs.json",
                 "datasets.json", "lcb_input_audit.json", "mmlu_pro_input_audit.json"):
        inputs[str(root / name)] = file_sha256(root / name)
    save(root / "manifest.json", {"kind": "ban_trace_calibration", "created_at": now(),
         "model": str(MODEL), "spec": str(spec), "model_identity": identity,
         "c4_artifact": str(C4), "trace_source": str(TRACES), "input_sha256": inputs,
         "inference": INFERENCE, "tasks": TASKS, "seed": 1234, "calibration_seed": 0,
         "budget": budget,
         "code": source_identity(), "reuse_evidence": qualification,
         "storage": "Existing BF16 QDQ export; weights and online HiF4/R64 unchanged."})
    save(root / "status.json", {"status": "prepared", "created_at": now(), "phases": {}, "errors": []})
    print(f"BAN_TRACE_PREPARED {root}", flush=True)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_experts", type=float, default=6.07)
    args = parser.parse_args()
    prepare(args.output, args.target_experts)
