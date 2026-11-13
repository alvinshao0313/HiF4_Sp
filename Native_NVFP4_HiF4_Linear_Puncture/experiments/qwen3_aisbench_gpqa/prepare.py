"""Freeze the official prompt and prove that every other input/config stays fixed."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import urllib.request

from .common import HERE, LANES, NATIVE, OLD_RUNS, REPO, command_protocol, file_hash, now, read, save


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prepare(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    native = read(NATIVE / "manifest.json")
    os.environ.update(native["environment"])
    from .task import INSTRUCTION, TEMPLATE, UPSTREAM_PATH, UPSTREAM_REVISION
    from ..nvfp4_gpqa.task import INSTRUCTION as original_instruction
    from transformers import AutoTokenizer
    from lighteval.tasks.prompt_manager import PromptManager
    from lighteval.tasks.registry import Registry

    url = f"https://raw.githubusercontent.com/AISBench/benchmark/{UPSTREAM_REVISION}/{UPSTREAM_PATH}"
    upstream = urllib.request.urlopen(url, timeout=30).read()
    (root / "aisbench_gpqa_config.py").write_bytes(upstream)
    node = next(node.value for node in ast.parse(upstream).body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "align_prompt" for t in node.targets))
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "strip" and ast.literal_eval(node.func.value).strip() == TEMPLATE):
        raise ValueError("Local wording differs from the pinned AISBench template")
    dataset = Path(native["dataset"]["local_path"]) / "train.jsonl"
    if file_hash(dataset) != native["dataset"]["sha256"]:
        raise ValueError("Original GPQA dataset changed")
    tokenizer = AutoTokenizer.from_pretrained(native["source_snapshot"], local_files_only=True)
    manager = PromptManager(True, tokenizer)
    explicit = PromptManager(True, tokenizer, enable_thinking=True)
    protocol = {str(row["id"]): row for row in read(NATIVE / "protocol_inputs.json")}
    docs = {}
    for name, path in (("gpqa_diamond_qwen_avg10", HERE.parent / "nvfp4_gpqa/task.py"),
                       ("gpqa_diamond_aisbench_avg10", HERE / "task.py")):
        random.seed(1234)
        task = Registry(tasks=f"{name}|0", custom_tasks=str(path)).load_tasks()[f"{name}|0"]
        docs[name] = task.get_docs()
    old, new = docs.values()
    if len(old) != 198 or len(new) != 198:
        raise ValueError("Expected all 198 GPQA Diamond questions")
    inputs = []
    for before, after in zip(old, new):
        prompt_old, prompt_new = manager.prepare_prompt(before), manager.prepare_prompt(after)
        ref = protocol[str(before.id)]
        if (before.id != after.id or before.get_golds() != after.get_golds()
                or before.specific != after.specific or after.num_samples != 10
                or text_hash(prompt_old) != ref["prompt_sha256"]
                or after.get_golds()[0] != ref["gold"]
                or prompt_old.replace(original_instruction, INSTRUCTION, 1) != prompt_new
                or prompt_new != explicit.prepare_prompt(after)):
            raise ValueError("Change beyond the instruction text: question/options/thinking/count")
        tokens = tokenizer.encode(prompt_new, add_special_tokens=True)
        if len(tokens) + 32768 > 40960:
            raise ValueError("New prompt would be truncated")
        inputs.append({**ref, "prompt_sha256": text_hash(prompt_new),
                       "input_tokens_sha256": text_hash(json.dumps(tokens, separators=(",", ":"))),
                       "input_tokens": len(tokens), "prompt": prompt_new})
    save(root / "protocol_inputs.json", inputs)
    sources = {str(REPO / name): digest for name, digest in native["source_sha256"].items()}
    models = {}
    for variant, old_root in OLD_RUNS.items():
        previous = read(old_root / "manifest.json")
        status = read(old_root / "formal/status.json")
        if status["status"] != "completed" or status["returncode"] != 0:
            raise ValueError(f"Incomplete prior formal evaluation: {variant}")
        checkpoint = Path(previous["source_snapshot"]).resolve()
        # Current shared base includes H32 support added after the direct/R64 run.
        # Its inactive-branch equivalence is checked separately before GPU probes.
        reference = read(OLD_RUNS["fht32"] / "manifest.json") if variant in ("direct", "r64") else previous
        for name, digest in reference["source_sha256"].items():
            if file_hash(REPO / name) != digest:
                raise ValueError(f"Previously qualified dependency changed: {name}")
            sources[str(REPO / name)] = digest
        copied = ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja")
        for filename in copied:
            if file_hash(checkpoint / filename) != file_hash(Path(native["source_snapshot"]) / filename):
                raise ValueError(f"Tokenizer/template differs: {variant}/{filename}")
        artifacts = {}
        if variant != "native":
            conversion = read(checkpoint / "conversion.json")
            if (conversion["status"] != "completed" or conversion["converted_matrices"] != 18432
                    or not conversion["non_expert_byte_exact"] or not conversion["saved_reload_byte_exact"]):
                raise ValueError(f"Incomplete checkpoint conversion: {variant}")
            for filename, evidence in conversion["output_shards"].items():
                path = checkpoint / filename
                if path.stat().st_size != evidence["bytes"] or file_hash(path) != evidence["sha256"]:
                    raise ValueError(f"Checkpoint shard changed: {path}")
                artifacts[str(path)] = evidence["sha256"]
            artifacts[str(checkpoint / "conversion.json")] = file_hash(checkpoint / "conversion.json")
        config = deepcopy(previous)
        config.update(created_at=now(), status="prepared", variant=variant, validation_phase="probe",
                      baseline=str(NATIVE), source_snapshot=str(checkpoint),
                      authorization="User approved eight GPQA prompt-only reruns; BF16 KV and original hyperparameters.")
        config["protocol"]["prompt"] = f"AISBench {UPSTREAM_PATH} at {UPSTREAM_REVISION}; instruction text only"
        devices = next(gpus for gpus, variants in LANES.items() if variant in variants)
        config["environment"]["CUDA_VISIBLE_DEVICES"] = devices
        config["protocol"]["cuda_visible_devices"] = devices
        for phase in ("probe", "formal"):
            directory = root / variant / phase
            model = directory / "model"
            model.mkdir(parents=True)
            for path in checkpoint.iterdir():
                if path.is_file() and path.name != "conversion.json":
                    (model / path.name).symlink_to(path.resolve())
            cmd = config["commands"][phase]
            cmd[2] = str(HERE / "entry.py")
            for flag, value in (("--model_path", model), ("--output_dir", directory / "evaluation"),
                                ("--datasets", "gpqa_diamond_aisbench_avg10|0"), ("--custom_tasks", HERE / "task.py")):
                cmd[cmd.index(flag) + 1] = str(value)
            if command_protocol(cmd) != command_protocol(previous["commands"][phase]):
                raise ValueError(f"Original hyperparameters changed: {variant}/{phase}")
            if cmd[cmd.index("--kv_cache_dtype") + 1] != "bfloat16":
                raise ValueError("Only unquantized BF16 KV is authorized")
        models[variant] = {"checkpoint": str(checkpoint), "old_run": str(old_root),
                           "old_manifest_sha256": file_hash(old_root / "manifest.json"),
                           "checkpoint_sha256": artifacts, "devices": devices}
        save(root / variant / "manifest.json", config)
        print(f"Prepared {variant}: original checkpoint, parameters, BF16 KV; independent caches", flush=True)
    paths = list(HERE.glob("*.py")) + [HERE.parent / "qwen3_mxfp4/review_answers.py"]
    sources.update({str(path): file_hash(path) for path in paths})
    sources[str(root / "protocol_inputs.json")] = file_hash(root / "protocol_inputs.json")
    sources[str(root / "aisbench_gpqa_config.py")] = file_hash(root / "aisbench_gpqa_config.py")
    for variant in OLD_RUNS:
        path = root / variant / "manifest.json"
        config = read(path)
        config["source_sha256"] = sources
        save(path, config)
    for path, digest in sources.items():
        source = Path(path)
        if source.is_relative_to(REPO) and source.suffix == ".py":
            target = root / "source_frozen" / source.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    save(root / "workflow_manifest.json", {"status": "prepared", "created_at": now(), "models": models,
         "lanes": LANES, "source_sha256": sources, "upstream_url": url, "upstream_revision": UPSTREAM_REVISION,
         "prompt_only": True, "kv_cache_dtype": "bfloat16", "old_protocol": native["protocol"],
         "questions": 198, "generations_per_question": 10,
         "max_prompt_tokens": max(row["input_tokens"] for row in inputs),
         "failure_policy": "Exit on first failure, no retries or protocol changes"})
    print(root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args().output)
