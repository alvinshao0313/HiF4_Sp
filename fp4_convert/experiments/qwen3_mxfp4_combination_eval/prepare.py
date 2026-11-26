"""Reuse qualified checkpoints and freeze data plus independently checked AISBench prompts."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import random
import urllib.request

from .common import (DATA_REFERENCE, DEVICES, HERE, REFERENCE, REPO, TASKS, UPSTREAM_FILES,
                     UPSTREAM_REVISION, VARIANTS, digest, file_hash, install_datasets, now, read, save)


def upstream(root):
    destination = root / "aisbench_sources"
    destination.mkdir()
    records = {}
    for name, path in UPSTREAM_FILES.items():
        url = f"https://raw.githubusercontent.com/AISBench/benchmark/{UPSTREAM_REVISION}/{path}"
        with urllib.request.urlopen(url, timeout=30) as response:
            (destination / name).write_bytes(response.read())
        records[name] = {"url": url, "sha256": file_hash(destination / name)}
    return records


def assignment(path, name):
    tree = ast.parse(path.read_text())
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous upstream constant {name}")
    # Evaluate only the selected constant expression, without importing AISBench.
    return eval(compile(ast.Expression(matches[0]), str(path), "eval"), {"__builtins__": {}})


def inputs(root):
    from datasets import Dataset
    from transformers import AutoTokenizer
    from lighteval.tasks.prompt_manager import PromptManager
    from lighteval.tasks.registry import Registry
    from .tasks import LCB_TEMPLATE, LCB_WITH_STARTER, LCB_WITHOUT_STARTER, MMLU_TEMPLATE

    routes = read(DATA_REFERENCE)
    for route in routes.values():
        for split in route["splits"].values():
            if file_hash(split["path"]) != split["sha256"] or len(Dataset.from_file(split["path"])) != split["count"]:
                raise ValueError("Pinned dataset bytes or coverage changed")
    save(root / "datasets.json", routes)
    install_datasets(root / "datasets.json")
    upstream_root = root / "aisbench_sources"
    mmlu_template = assignment(upstream_root / "mmlu_config.py", "QUERY_TEMPLATE")
    lcb_template = assignment(upstream_root / "lcb_config.py", "prompt_template")
    constants = next(node for node in ast.parse((upstream_root / "lcb_prompts.py").read_text()).body
                     if isinstance(node, ast.ClassDef) and node.name == "CodeGenerationPromptConstants")
    formatting = {node.targets[0].id: ast.literal_eval(node.value) for node in constants.body if isinstance(node, ast.Assign)}
    if (mmlu_template != MMLU_TEMPLATE or lcb_template != LCB_TEMPLATE
            or formatting["FORMATTING_MESSAGE_WITH_STARTER_CODE"] != LCB_WITH_STARTER
            or formatting["FORMATTING_WITHOUT_STARTER_CODE"] != LCB_WITHOUT_STARTER):
        raise ValueError("Local prompt literals differ from the pinned AISBench revision")
    parser_tree = ast.parse((upstream_root / "mmlu_dataset.py").read_text())
    parse_function = next(node for node in parser_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_parse")
    namespace = {"CHOICES": list("ABCDEFGHIJKLMNOP")}
    exec(compile(ast.Module(body=[parse_function], type_ignores=[]), "aisbench_mmlu_parse", "exec"), namespace)
    tokenizer = AutoTokenizer.from_pretrained(REFERENCE / "weights" / VARIANTS[0], local_files_only=True)
    manager = PromptManager(use_chat_template=True, tokenizer=tokenizer, enable_thinking=None)
    reports = {}
    for task_name, definition in TASKS.items():
        random.seed(1234)
        task = Registry(tasks=definition["tag"], custom_tasks=str(HERE / "tasks.py")).load_tasks()[definition["tag"]]
        docs = task.get_docs(max_samples=None)
        if len(docs) != definition["questions"] or any(doc.num_samples != 5 for doc in docs):
            raise ValueError("Wrong question coverage or actual generation count")
        data_name = "TIGER-Lab/MMLU-Pro" if task_name == "mmlu_pro" else "lighteval/code_generation_lite"
        original = Dataset.from_file(routes[data_name]["splits"]["test"]["path"])
        # get_docs shuffles deterministically; compare query multisets, independently of IDs.
        reference_queries = []
        for row in original:
            if task_name == "mmlu_pro":
                parsed = namespace["_parse"](dict(row))
                reference_queries.append(mmlu_template.format(**parsed))
            else:
                starter = row.get("starter_code")
                instruction = formatting["FORMATTING_MESSAGE_WITH_STARTER_CODE" if starter else "FORMATTING_WITHOUT_STARTER_CODE"]
                formatted = f"### Format: {instruction}\n```python\n{starter or '# YOUR CODE HERE'}\n```\n\n"
                reference_queries.append(lcb_template.format(question_content=row["question_content"], format_prompt=formatted))
        if sorted(reference_queries) != sorted(doc.query for doc in docs):
            raise ValueError("Actual task prompts differ from upstream formatting of the pinned dataset")
        rows = []
        for doc in docs:
            prompt = manager.prepare_prompt(doc)
            explicit = tokenizer.apply_chat_template([{"role": "user", "content": doc.query}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            if prompt != explicit:
                raise ValueError("Default chat template differs from explicit thinking=True")
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            rows.append({"id": str(doc.id), "prompt_sha256": digest(prompt),
                         "gold_sha256": digest(doc.get_golds()), "full_input_tokens": len(tokens),
                         "input_tokens_sha256": digest(tokens[-(40960-definition["max_new_tokens"]):]),
                         "probe_input_tokens_sha256": digest(tokens[-(40960-64):])})
        if len({row["id"] for row in rows}) != len(docs):
            raise ValueError("Duplicate question IDs")
        report = {**definition, "generations_per_question": 5, "documents": rows,
                  "upstream_prompt_exact": True, "thinking_default_equals_true": True,
                  "input_truncated_questions": sum(row["full_input_tokens"] > 40960-definition["max_new_tokens"] for row in rows)}
        save(root / f"{task_name}_inputs.json", report)
        save(root / f"{task_name}_prompt_example.json", {"query": docs[0].query, "chat_prompt": manager.prepare_prompt(docs[0])})
        reports[task_name] = {key: value for key, value in report.items() if key != "documents"}
        print(f"{task_name}: {len(docs)} questions x 5; all AISBench queries and thinking template verified", flush=True)
    return reports


def prepare(root):
    from pathlib import Path
    from ..qwen3_mxfp4_combinations.artifacts import code_hashes

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    save(root / "authorization.json", {"at": now(), "request": "Four existing combinations, LCB and MMLU-Pro, five-response mean",
        "gpu_authorization": "User: 所有卡都可以用", "devices": dict(zip(VARIANTS, DEVICES)),
        "scope": "Full MMLU-Pro 12032 and existing LCB v6 175; AISBench prompts; TP2 BF16 KV56GiB",
        "no_new_quantization": True})
    upstream_records = upstream(root)
    models, shards, source_checks = {}, {}, {}
    inference_names = {"kernels.py", "runtime.py", "format.py", "integration.py", "model_io.py", "transform.py",
                       "fht32.py", "r64_transform.py", "modelopt.py", "qwen3_moe.py", "layer.py", "nvfp4_emulation_moe.py"}
    tokenizer_identity = None
    for variant in VARIANTS:
        model = REFERENCE / "weights" / variant
        conversion = read(model / "conversion.json")
        operators = read(REFERENCE / variant / "operators.json")
        if (conversion["status"] != "completed" or conversion["converted_matrices"] != 18432
                or not conversion["non_expert_byte_exact"] or not conversion["saved_reload_byte_exact"]
                or operators["status"] != "passed"
                or read(REFERENCE / variant / "formal/runtime_validation.json")["status"] != "passed"):
            raise ValueError(f"Missing applicable checkpoint/operator/runtime evidence: {variant}")
        source_checks.update({str(REPO / path): value for path, value in operators["source_sha256"].items()
                              if Path(path).name in inference_names})
        for name, record in conversion["output_shards"].items():
            path = model / name
            if path.stat().st_size != record["bytes"]:
                raise ValueError(f"Checkpoint size changed: {path}")
            shards[str(path)] = record["sha256"]
        identity = {name: file_hash(model / name) for name in conversion["copied_files"]}
        if identity != conversion["copied_files"] or (tokenizer_identity is not None and tokenizer_identity != identity):
            raise ValueError("Tokenizer/chat/generation files differ")
        tokenizer_identity = identity
        models[variant] = {"checkpoint": str(model), "conversion_sha256": file_hash(model / "conversion.json"),
            "operators": str(REFERENCE / variant / "operators.json"), "operators_sha256": file_hash(REFERENCE / variant / "operators.json"),
            "runtime_reference": str(REFERENCE / variant / "formal"), "converted_matrices": 18432}
    for path, expected in source_checks.items():
        if file_hash(path) != expected:
            raise ValueError(f"Qualified inference source changed: {path}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path, actual in zip(shards, pool.map(file_hash, shards)):
            if actual != shards[path]:
                raise ValueError(f"Checkpoint digest changed: {path}")
    print("All four existing checkpoints and inference dependencies verified", flush=True)
    input_reports = inputs(root)
    sources = {str(REPO / path): value for path, value in code_hashes().items()}
    reference_sources = read(REFERENCE / "preparation.json")["source_sha256"]
    for path, expected in reference_sources.items():
        absolute = REPO / path
        if "3rdparty/lighteval/" in str(path) or absolute == REPO / "main.py":
            if file_hash(absolute) != expected:
                raise ValueError(f"Reference inference framework changed: {path}")
            sources[str(absolute)] = expected
    for path in list(HERE.glob("*.py")) + list((HERE.parent / "qwen3_mxfp4_rotation_eval").glob("*.py")):
        sources[str(path)] = file_hash(path)
    for path in list((root / "aisbench_sources").glob("*.py")) + list(root.glob("*_inputs.json")) + [root / "datasets.json"]:
        sources[str(path)] = file_hash(path)
    save(root / "expected_runtime.json", read(REFERENCE / "expected_runtime.json"))
    sources[str(root / "expected_runtime.json")] = file_hash(root / "expected_runtime.json")
    save(root / "manifest.json", {"status": "prepared", "created_at": now(), "reference": str(REFERENCE),
        "models": models, "checkpoint_sha256": shards, "source_sha256": sources,
        "datasets_sha256": file_hash(root / "datasets.json"), "inputs": input_reports,
        "aisbench_revision": UPSTREAM_REVISION, "aisbench_sources": upstream_records,
        "devices": dict(zip(VARIANTS, DEVICES)), "generations_per_question": 5,
        "scoring": "Existing lighteval MMLU extractive_match and LCB code tests; preserve individual scores and raw answers",
        "metric": "mean single-response accuracy, not pass@5 or voting", "kv_dtype": "bfloat16",
        "kv_gib_per_card": 56, "tp": 2, "qualification": "not_run"})
    print(f"Prepared {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args().output)
