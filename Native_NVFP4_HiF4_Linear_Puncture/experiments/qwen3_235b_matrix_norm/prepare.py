"""Reuse verified source projections; cluster and audit before 235B generation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil

from ..e2e_diag_reconstruction.core.moe_model_spec import Qwen3MoeModelSpec
from ..reasoning_trace_calibration.config import MMLU_DATA, SOURCES
from ..reasoning_trace_calibration.generate import prompt
from ..reasoning_trace_calibration.isolation import cluster_questions, normalize, split_pools
from ..reasoning_trace_calibration.sources import evaluation_questions, fetch_sources, source_identity
from .artifacts import digest, now, read_json, sha256, source_hashes, verify_inputs, write_json
from .config import ENGINE, GENERATION, NATIVE, PROJECTION_SOURCE, PROTOCOL, REPO, TASKS, task_protocol


def reuse_projections(root):
    destination = root / "data_sources"
    if not (PROJECTION_SOURCE / "manifest.json").is_file():
        return fetch_sources(root)
    evidence = read_json(PROJECTION_SOURCE / "manifest.json")
    records = []
    for relative, entry in evidence.items():
        domain = relative.split("/")[1]
        source = PROJECTION_SOURCE.parent / relative
        if (entry["revision"] != SOURCES[domain]["revision"] or entry["repo"] != SOURCES[domain]["repo"]
                or not entry.get("source_file_sha256") or sha256(source) != entry["sha256"]):
            raise ValueError(f"source projection is not the fixed version: {source}")
        rows = read_json(source)
        if len(rows) != entry["count"]:
            raise ValueError(f"incomplete source projection: {source}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        filename = entry["file"]
        split = Path(filename).name.split("-")[0]
        if split not in {"train", "test", "valid", "validation"}:
            raise ValueError(f"unknown source split: {filename}")
        for index, row in enumerate(rows):
            question = row["problem" if domain == "math" else "description"]
            if not isinstance(question, str) or not question.strip():
                raise ValueError("empty source question")
            records.append({"id": f"{domain}:{split}:{Path(filename).stem}:{index}",
                            "domain": domain, "role": "candidate" if split == "train" else "test",
                            "question": question, "identities": source_identity(domain, row),
                            "locator": {"domain": domain, "file": filename, "row": index}})
    write_json(evidence, destination / "manifest.json")
    write_json(records, destination / "questions.json")
    return records


def task_registry():
    from lighteval.tasks.registry import Registry
    names = ",".join(task_protocol(task)["task_name"] for task in TASKS)
    return Registry(tasks=names, custom_tasks=str(REPO / "tasks/custom_tasks.py")).load_tasks()


def prepare_datasets(root):
    tests, evidence = evaluation_questions()
    tests = [row for row in tests if row["id"].split(":")[1] in TASKS]
    tasks = task_registry()
    manifest = {}
    for name in TASKS:
        task = tasks[task_protocol(name)["task_name"]]
        split = "train" if name == "aime25" else "test"
        sources = {split: Path(evidence[name]["path"])}
        if name == "mmlu_pro":
            sources["validation"] = MMLU_DATA / "data/validation.arrow"
            pinned = read_json(MMLU_DATA / "dataset_manifest.json")
            if task.dataset_revision != pinned["revision"]:
                raise ValueError("MMLU-Pro task no longer selects the approved revision")
            for key, path in sources.items():
                if sha256(path) != pinned["splits"][key]["sha256"]:
                    raise ValueError(f"canonical MMLU-Pro {key} changed")
        entry = {"repo": task.dataset_path, "task_revision": task.dataset_revision, "splits": {}}
        for key, source in sources.items():
            target = root / f"datasets/{name}/{key}.arrow"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entry["splits"][key] = {"path": str(target.relative_to(root)), "sha256": sha256(target)}
        manifest[name] = entry
    write_json(manifest, root / "datasets/manifest.json")
    return tests


def audit_model(root, model, *, task_keys=TASKS):
    from transformers import AutoTokenizer
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.prompt_manager import PromptManager
    from ..downstream_accuracy_recovery.reporting import _digest, tokenizer_asset_hashes
    from .pinned import install
    install(root)
    tasks = task_registry()
    LightevalTask.load_datasets(tasks)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    default = PromptManager(True, tokenizer, enable_thinking=None)
    thinking = PromptManager(True, tokenizer, enable_thinking=True)
    direct = PromptManager(True, tokenizer, enable_thinking=False)
    result = {"tokenizer_files": tokenizer_asset_hashes(model, tokenizer), "tasks": {}}
    for key in task_keys:
        definition = task_protocol(key)
        task = tasks[definition["task_name"]]
        docs = task.get_docs(max_samples=None)
        if len(docs) != definition["count"] or len({str(d.id) for d in docs}) != len(docs):
            raise ValueError(f"unexpected full task coverage: {key}")
        if any(d.num_samples != definition["generations_per_doc"] for d in docs):
            raise ValueError(f"actual generation multiplicity differs for {key}")
        documents, prompts = {}, []
        maximum = 0
        for doc in docs:
            text = default.prepare_prompt(doc)
            if text != thinking.prepare_prompt(doc) or text == direct.prepare_prompt(doc):
                raise ValueError("default chat template must enable thinking")
            ids = tokenizer.encode(text, add_special_tokens=False)
            maximum = max(maximum, len(ids))
            documents[str(doc.id)] = {"prompt_sha256": digest(text), "token_sha256": digest(ids),
                                      "gold_sha256": digest(doc.get_golds())}
            prompts.append(text)
        if maximum + definition["max_new_tokens"] > definition["max_model_length"]:
            raise ValueError(f"task exceeds full context budget: {key}")
        result["tasks"][key] = {"count": len(docs), "generations_per_doc": definition["generations_per_doc"],
            "max_prompt_tokens": maximum, "documents": documents, "documents_sha256": digest(documents),
            "prompt_set_sha256": _digest(sorted(prompts)), "hf_repo": task.dataset_path,
            "hf_revision": task.dataset_revision, "task_name": definition["task_name"],
            "metric": definition["metric"]}
        print(f"audited {key}: {len(docs)} questions x {definition['generations_per_doc']}", flush=True)
    return result


def prepare(root):
    root = Path(root).resolve()
    if root.exists():
        verify_inputs(root)
        return root
    spec = Qwen3MoeModelSpec.from_snapshot(NATIVE)
    if (spec.num_layers, spec.hidden_size, spec.num_attention_heads) != (94, 4096, 64):
        raise ValueError("expected the complete native 235B model")
    index = read_json(NATIVE / "model.safetensors.index.json")
    shards = {}
    for name in sorted(set(index["weight_map"].values())):
        path = NATIVE / name
        stat = path.stat()
        shards[name] = {"resolved": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    root.mkdir(parents=True, exist_ok=False)
    write_json({"kind": "qwen3_235b_matrix_norm", "created_at": now(), "protocol": PROTOCOL,
                "protocol_sha256": digest(PROTOCOL), "geometry": asdict(spec), "status": "preparing"},
               root / "manifest.json")
    candidates = reuse_projections(root)
    tests = prepare_datasets(root)
    rows = candidates + tests
    print(f"clustering {len(rows)} real source/test records", flush=True)
    groups, edges = cluster_questions(rows)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
    eligible, budget_exclusions = [], []
    for group in groups:
        if any(row["role"] == "test" for row in group):
            eligible.append(group)
            continue
        chosen = min(group, key=lambda row: row["id"])
        if len(prompt(tokenizer, chosen)) + GENERATION["max_tokens"] > ENGINE["max_model_len"]:
            budget_exclusions.append([row["id"] for row in group])
        else:
            eligible.append(group)
    pools, exclusions, _ = split_pools(eligible)
    for domain in pools:
        for split, records in pools[domain].items():
            for row in records:
                row["question_sha256"] = digest(normalize(row["question"]))
    write_json(rows, root / "isolation/questions.json")
    write_json(edges, root / "isolation/edges.json")
    write_json(pools, root / "isolation/pools.json")
    write_json(exclusions, root / "isolation/exclusions.json")
    write_json(budget_exclusions, root / "isolation/prompt_budget_exclusions.json")
    write_json({"complete": True, "source_versions": SOURCES, "source_records": len(candidates),
                "formal_test_records": len(tests), "includes_official_source_tests": True,
                "cluster_count": len(groups), "excluded_clusters": len(exclusions),
                "prompt_budget_excluded_clusters": len(budget_exclusions),
                "pool_counts": {d: {s: len(v) for s, v in p.items()} for d, p in pools.items()},
                "rule": "source identity or normalized/numeric-masked 5-gram Jaccard >= .8; cluster before split"},
               root / "isolation/audit.json")
    write_json(audit_model(root, NATIVE), root / "prompt_audit.json")
    files = [p for directory in ("data_sources", "datasets", "isolation") for p in (root / directory).rglob("*") if p.is_file()]
    files.append(root / "prompt_audit.json")
    assets = ("config.json", "hf_quant_config.json", "model.safetensors.index.json", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json", "generation_config.json")
    write_json({"complete": True, "files": {str(p.relative_to(root)): sha256(p) for p in files},
                "native_assets": {name: sha256(NATIVE / name) for name in assets if (NATIVE / name).is_file()},
                "native_shards": shards}, root / "inputs.json")
    write_json(source_hashes(), root / "source_hashes.json")
    manifest = read_json(root / "manifest.json")
    write_json({**manifest, "status": "prepared"}, root / "manifest.json")
    verify_inputs(root)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    prepare(parser.parse_args().run)
