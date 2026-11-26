"""CPU preparation only: checkpoint provenance, pinned data and paired inputs."""
import argparse
import importlib.util
from pathlib import Path
import random

from .common import HERE, MODELS, REPO, RESULTS, RUNS, TASKS, digest, file_hash, install_datasets, now, read, save


def sources():
    from ..qwen3_mxfp4_r16x2.artifacts import code_hashes
    paths = list(HERE.glob("*.py")) + [REPO / "main.py"]
    light = REPO / "3rdparty/lighteval/src/lighteval"
    paths += [light / name for name in (
        "models/vllm/vllm_model.py", "models/model_output.py", "pipeline.py",
        "tasks/lighteval_task.py", "tasks/prompt_manager.py", "tasks/tasks/mmlu_pro.py",
        "tasks/tasks/lcb/main.py", "tasks/tasks/lcb/codegen_metrics.py",
        "metrics/metrics.py", "metrics/metrics_sample.py", "metrics/dynamic_metrics.py",
        "utils/cache_management.py")]
    lm_root = Path(importlib.util.find_spec("lm_eval").origin).parent
    paths += [lm_root / name for name in ("models/vllm_causallms.py", "api/task.py", "evaluator.py",
                                          "tasks/arc/arc_easy.yaml", "tasks/arc/arc_challenge.yaml")]
    result = {str((REPO / path).resolve()): value for path, value in code_hashes().items()}
    result.update({str(path): file_hash(path) for path in paths})
    return result


def inputs(root):
    from datasets import Dataset
    source = RESULTS / "ban/supplement_20261025T150926Z/datasets.json"
    previous = read(source)
    routes = {name: previous[name] for name in ("TIGER-Lab/MMLU-Pro", "lighteval/code_generation_lite")}
    routes["TIGER-Lab/MMLU-Pro"]["subset"] = "default"
    routes["lighteval/code_generation_lite"]["subset"] = "v6"
    for route in routes.values():
        for item in route["splits"].values():
            if file_hash(item["path"]) != item["sha256"] or len(Dataset.from_file(item["path"])) != item["count"]:
                raise ValueError("Previously pinned reasoning dataset changed")
    save(root / "datasets.json", routes)
    arc = {}
    for subset, count in (("ARC-Easy", 2376), ("ARC-Challenge", 1172)):
        directory = Path.home() / ".cache/huggingface/datasets/allenai___ai2_arc" / subset / "0.0.0/210d026faf9955653af8916fad021475a3f00453"
        splits = {}
        for split in ("train", "validation", "test"):
            path = directory / f"ai2_arc-{split}.arrow"
            splits[split] = {"path": str(path), "sha256": file_hash(path), "count": len(Dataset.from_file(str(path)))}
        if splits["test"]["count"] != count:
            raise ValueError("Unexpected ARC test coverage")
        arc[subset] = {"revision": "210d026faf9955653af8916fad021475a3f00453", "splits": splits}
    save(root / "arc_datasets.json", arc)
    return {"source_manifest": str(source), "source_sha256": file_hash(source)}


def audits(root):
    from transformers import AutoTokenizer
    from lighteval.tasks.prompt_manager import PromptManager
    from lighteval.tasks.registry import Registry

    install_datasets(root / "datasets.json")
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS["fht32"]), local_files_only=True)
    manager = PromptManager(use_chat_template=True, tokenizer=tokenizer, enable_thinking=None)
    for name, definition in TASKS.items():
        random.seed(1234)
        task = Registry(tasks=definition["tag"], custom_tasks=str(HERE / "tasks.py")).load_tasks()[definition["tag"]]
        docs = task.get_docs(max_samples=None)
        expected = {"mmlu_pro": 12032, "lcb": 175}[name]
        if len(docs) != expected or any(doc.num_samples != 10 for doc in docs):
            raise ValueError("Reasoning dataset coverage or actual sample count differs")
        rows = []
        for doc in docs:
            prompt = manager.prepare_prompt(doc)
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            rows.append({"id": str(doc.id), "prompt_sha256": digest(prompt),
                         "gold_sha256": digest(doc.get_golds()), "full_input_tokens": len(tokens),
                         "input_tokens_sha256": digest(tokens[-(40960-definition["max_new_tokens"]):]),
                         "probe_input_tokens_sha256": digest(tokens[-(40960-64):])})
        if len({row["id"] for row in rows}) != expected:
            raise ValueError("Duplicate reasoning question IDs")
        save(root / f"{name}_inputs.json", {**definition, "count": expected, "generations": 10, "documents": rows,
             "prompt_policy": "Existing task formatter/chat template; actual input truncation follows unchanged main.py"})
        print(f"Pinned {name}: {expected} questions, 10 responses each; longest prompt {max(row['full_input_tokens'] for row in rows)} tokens", flush=True)


def prepare(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    models = {}
    tokenizer_identity = None
    for variant, model in MODELS.items():
        conversion = read(model / "conversion.json")
        if (conversion["status"] != "completed" or conversion["variant"] != variant
                or conversion["converted_matrices"] != 18432 or not conversion["non_expert_byte_exact"]
                or not conversion["saved_reload_byte_exact"]):
            raise ValueError(f"Incomplete converted checkpoint: {variant}")
        artifacts = {str(model / "conversion.json"): file_hash(model / "conversion.json"),
                     str(model / "model.safetensors.index.json"): file_hash(model / "model.safetensors.index.json")}
        for filename, record in conversion["output_shards"].items():
            path = model / filename
            if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
                raise ValueError(f"Converted checkpoint bytes changed: {path}")
            artifacts[str(path)] = record["sha256"]
        small = {name: file_hash(model / name) for name in conversion["copied_files"]}
        if small != conversion["copied_files"]:
            raise ValueError("Tokenizer or source auxiliary file changed")
        if tokenizer_identity is not None and small != tokenizer_identity:
            raise ValueError("The two models have different tokenizer/generation files")
        tokenizer_identity = small
        numerical_path = RUNS[variant] / f"{variant}_gpu_validation.json"
        numerical = read(numerical_path)
        if numerical["status"] != "passed":
            raise ValueError("Existing numerical qualification is missing")
        # Only the inference dependencies determine whether old operator evidence
        # applies; report/controller edits are not numerical changes.
        names = {"runtime.py", "kernels.py", "format.py", "transform.py", "fht32.py",
                 "integration.py", "nvfp4_emulation_moe.py", "modelopt.py", "qwen3_moe.py", "layer.py"}
        for name, expected in numerical["source_sha256"].items():
            if Path(name).name in names and file_hash(REPO / name) != expected:
                raise ValueError(f"Qualified inference source changed: {name}")
        models[variant] = {"checkpoint": str(model), "artifacts_sha256": artifacts,
                           "reused_gpu_validation": str(numerical_path),
                           "reused_gpu_validation_sha256": file_hash(numerical_path),
                           "source_revision": conversion["source_revision"],
                           "nominal_weight_bits_including_scales": 4.25}
        print(f"Verified {variant}: 18,432 matrices; checkpoint byte hashes and existing operator evidence", flush=True)
    data = inputs(root)
    audits(root)
    save(root / "manifest.json", {"status": "prepared", "created_at": now(), "models": models,
         "data": data, "source_sha256": sources(), "reasoning_generations": 10, "arc_rounds": 5,
         "pending": ["GPU allocation", "MMLU-Pro 300 or 12032 questions"],
         "notes": "CPU preparation only; no GPU inference started. Existing MXFP4 floor quantizers/checkpoints unchanged."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args().output)
