"""Five-model evaluation using the confirmed artifacts and task protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from .artifact import sha256, write_json

MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.residual_lora_compensation"
TASK_DIRS = {"arc": "arc", "mmlu_pro": "mmlu_pro", "lcb": "livecodebench"}


def models_for(root, baseline_run, e4_run):
    return {"moe": root / "moe/model", "attention": root / "attention/model", "both": root / "both/model",
            "group_top_mass": Path(baseline_run).resolve() / "model", "E4": Path(e4_run).resolve() / "model"}


def prompt_audit(models, output):
    from transformers import AutoTokenizer
    from lighteval.models.utils import uses_chat_template
    from lighteval.tasks.registry import Registry
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.prompt_manager import PromptManager

    tasks = Registry(tasks="mmlu_pro|0,lcb:codegeneration_v6|0").load_tasks()
    LightevalTask.load_datasets(tasks)
    docs = {name: task.get_docs(max_samples=300 if name.startswith("mmlu_pro") else None)
            for name, task in tasks.items()}
    rows = {}
    for name, model in models.items():
        tokenizer = AutoTokenizer.from_pretrained(model)
        if not uses_chat_template(tokenizer=tokenizer):
            raise RuntimeError(f"missing reasoning chat template for {name}")
        manager = PromptManager(True, tokenizer, enable_thinking=None)
        thinking_manager = PromptManager(True, tokenizer, enable_thinking=True)
        direct_manager = PromptManager(True, tokenizer, enable_thinking=False)
        digests = {}
        for task, samples in docs.items():
            prompts = [manager.prepare_prompt(doc) for doc in samples]
            if any(p != thinking_manager.prepare_prompt(doc) or p == direct_manager.prepare_prompt(doc)
                   for p, doc in zip(prompts, samples)):
                raise RuntimeError("template defaults do not match explicitly enabled thinking")
            token_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
            digests[task] = {"count": len(prompts), "generations_per_doc": samples[0].num_samples,
                             "max_prompt_tokens": max(map(len, token_ids)),
                             "token_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
                             "prompt_set_sha256": hashlib.sha256(json.dumps(sorted(prompts)).encode()).hexdigest(),
                             "prompt_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest()}
        tokenizer_files = {f: sha256(model / f) for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json")}
        rows[name] = {"tasks": digests, "tokenizer_files": tokenizer_files}
    if any(row != rows["E4"] for row in rows.values()):
        raise RuntimeError("five-model prompt or tokenizer protocol differs")
    write_json({"complete": True, "models": rows}, output)
    return rows


def validate_metrics(result, task, *, smoke, prompt_reference=None):
    if result.get("smoke_only") is not smoke or result.get("tensor_parallel_size") != 2:
        raise ValueError("wrong evaluation scope or TP size")
    if task == "arc":
        scores = result["scores"]
        if set(scores) != {"arc_easy", "arc_challenge"}:
            raise ValueError("missing ARC scores")
        for name, n in result["n_samples"].items():
            if n["effective"] != (min(2, n["original"]) if smoke else n["original"]):
                raise ValueError(f"incomplete ARC samples for {name}")
    else:
        key = "mmlu_pro|0" if task == "mmlu_pro" else "lcb:codegeneration_v6|0"
        metric = "extractive_match" if task == "mmlu_pro" else "codegen_pass@1:16"
        scores = {key: result["results"][key][metric]}
        expected_samples = 1 if smoke else (300 if task == "mmlu_pro" else None)
        if result.get("max_samples") != expected_samples:
            raise ValueError("wrong reasoning sample limit")
        detail_files = list((Path(result["results_file"]).parent.parent / "details").rglob(f"details_{key}_*.json"))
        if len(detail_files) != 1:
            raise ValueError("missing or ambiguous reasoning sample details")
        details = json.loads(detail_files[0].read_text())
        expected_count = 1 if smoke else prompt_reference["count"]
        if len(details) != expected_count or len({row["doc"]["id"] for row in details}) != expected_count:
            raise ValueError("incomplete or duplicate reasoning samples")
        if prompt_reference is not None and not smoke:
            actual_prompts = sorted(row["model_response"]["input"] for row in details)
            if hashlib.sha256(json.dumps(actual_prompts).encode()).hexdigest() != prompt_reference["prompt_set_sha256"]:
                raise ValueError("actual evaluated prompts differ from the audited protocol")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in scores.values()):
        raise ValueError("invalid benchmark score")
    return scores


def summarize_completed(out, models):
    rows = []
    for name in ("E4", "group_top_mass", "attention", "moe", "both"):
        row = {"name": name, "model_dir": str(models[name])}
        row.update({task: json.loads((out / name / f"eval/{folder}/metrics.json").read_text())
                    for task, folder in TASK_DIRS.items()})
        row["scores"] = {"ARC-C": row["arc"]["scores"]["arc_challenge"],
                         "ARC-E": row["arc"]["scores"]["arc_easy"],
                         "MMLU-Pro": row["mmlu_pro"]["results"]["mmlu_pro|0"]["extractive_match"],
                         "LCB": row["lcb"]["results"]["lcb:codegeneration_v6|0"]["codegen_pass@1:16"]}
        rows.append(row)
    write_json({"complete": True, "models": rows}, out / "comparison.json")
    lines = ["# 残差 LoRA 五组下游评测", "", "五组完成同协议评测；ARC 为 accuracy，MMLU-Pro 为 extractive match，LCB 为当前官方 checker 的 pass@1。分数以百分比展示。",
             "", "| 模型 | ARC-C | ARC-E | MMLU-Pro（300） | LCB（全量） |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append("| " + row["name"] + " | " + " | ".join(f"{100*v:.2f}" for v in row["scores"].values()) + " |")
    lines += ["", "相对复用 group_top_mass 模型的变化（百分点）：", "",
              "| 模型 | ARC-C | ARC-E | MMLU-Pro | LCB |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows[2:]:
        lines.append("| " + row["name"] + " | " + " | ".join(f"{100*(v-rows[1]['scores'][k]):+.2f}" for k,v in row["scores"].items()) + " |")
    lines += ["", "三种 LoRA 均从 E4 初始化联合训练，rank=4、alpha=8。对照复用已有权重，本次分数重新评测。局部 NMSE 与下游分数分别解释；单次采样的分数差不等于统计显著结论，也不独立证明累计误差抵消机制。", "",
              "- [原始指标与完整配置](comparison.json)", "- [任务命令和完成状态](status.json)",
              "- [逐层训练结果](../TRAINING_REPORT.md)", "- [五组实际 prompt 核对](../downstream_validation/prompt_audit.json)", ""]
    (out / "COMPARISON.md").write_text("\n".join(lines))


def run(run_root, baseline_run, e4_run, *, audit_only=False):
    root = Path(run_root).resolve()
    models = models_for(root, baseline_run, e4_run)
    validation = root / "downstream_validation"
    for name, model in models.items():
        marker = model / ("residual_lora_export.json" if name in ("moe", "attention", "both") else "non_equivalent_export.json")
        metadata = json.loads(marker.read_text())
        if not metadata["complete"] or metadata.get("smoke_only"):
            raise ValueError(f"not a formal complete model: {model}")
        index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        if any(not (model / shard).is_file() for shard in set(index.values())):
            raise FileNotFoundError(f"missing model shard: {model}")
    if audit_only:
        prompt_audit(models, validation / "prompt_audit.json")
        return
    audit = json.loads((validation / "prompt_audit.json").read_text())
    if not audit["complete"] or set(audit["models"]) != set(models):
        raise ValueError("prompt audit incomplete")
    for name, model in models.items():
        if any(sha256(model / f) != digest for f, digest in audit["models"][name]["tokenizer_files"].items()):
            raise ValueError("tokenizer changed after prompt audit")
    for name in ("moe", "attention", "both"):
        result = json.loads((validation / name / "verification.json").read_text())
        if not result["complete"] or result["runtime_spec_sha256"] != sha256(models[name] / "hif4_runtime_spec.pt"):
            raise ValueError(f"missing/current TP2 verification for {name}")
    for task, folder in TASK_DIRS.items():
        validate_metrics(json.loads((validation / f"task_smoke/eval/{folder}/metrics.json").read_text()), task, smoke=True)
    out = root / "downstream"
    out.mkdir(exist_ok=False)
    # lighteval keys its prediction cache by model path. Give this formal run
    # fresh paths so it cannot consume earlier baseline or short-test outputs.
    sources = models
    models = {}
    for name, source in sources.items():
        view = out / "model_views" / name
        view.mkdir(parents=True)
        for file in source.iterdir():
            if file.is_file():
                (view / file.name).symlink_to(file)
        models[name] = view
    state = {"status": "running", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "source_models": {k: str(v) for k, v in sources.items()},
             "models": {k: str(v) for k, v in models.items()}, "completed": [], "commands": []}
    write_json(state, out / "status.json")
    try:
        for task in TASK_DIRS:
            for name, model in models.items():
                target = out / name
                target.mkdir(exist_ok=True)
                command = [sys.executable, "-m", f"{MODULE}.evaluate", "--model_dir", str(model),
                           "--output_dir", str(target), "--task", task]
                state["current"] = {"model": name, "task": task}
                state["commands"].append(command)
                write_json(state, out / "status.json")
                with (target / f"{task}.log").open("w") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                result = json.loads((target / f"eval/{TASK_DIRS[task]}/metrics.json").read_text())
                task_key = "mmlu_pro|0" if task == "mmlu_pro" else "lcb:codegeneration_v6|0"
                reference = None if task == "arc" else audit["models"][name]["tasks"][task_key]
                scores = validate_metrics(result, task, smoke=False, prompt_reference=reference)
                state["completed"].append({"model": name, "task": task, "scores": scores})
                write_json(state, out / "status.json")
        if len(state["completed"]) != 15:
            raise ValueError("incomplete five-model comparison")
        summarize_completed(out, models)
        state["status"] = "complete"
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        raise
    finally:
        write_json(state, out / "status.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--baseline_run", required=True)
    parser.add_argument("--e4_run", required=True)
    parser.add_argument("--audit_only", action="store_true")
    run(**vars(parser.parse_args()))
