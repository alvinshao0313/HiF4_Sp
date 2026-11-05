"""Audit the five-task comparison and report its positive-part accuracy loss."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from Native_NVFP4_HiF4_Linear_Puncture.experiments.residual_lora_compensation.artifact import sha256, write_json
from .protocol import ARC_COUNTS, REASONING_TASKS, TASK_DIRS, protocol_for, reasoning_details

METRICS = ("MMLU-Pro", "LCB", "GSM8K", "ARC-E", "ARC-C")
REASONING_METRICS = {"mmlu_pro": "MMLU-Pro", "lcb": "LCB", "gsm8k": "GSM8K"}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json")


def _digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def tokenizer_asset_hashes(model, tokenizer):
    """Compare the effective template even when the source embeds it in JSON."""
    hashes = {}
    for file in TOKENIZER_FILES:
        path = Path(model) / file
        if file == "chat_template.jinja" and not path.is_file():
            if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
                raise ValueError("source tokenizer has no effective chat template")
            hashes[file] = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
        else:
            hashes[file] = sha256(path)
    return hashes


def audit_prompts(models: dict[str, Path], output: Path) -> dict:
    from transformers import AutoTokenizer
    from lighteval.models.utils import uses_chat_template
    from lighteval.tasks.registry import Registry
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.prompt_manager import PromptManager

    if "native_nvfp4" not in models or len(models) < 2:
        raise ValueError("prompt audit requires native_nvfp4 and at least one candidate")
    tasks = Registry(tasks=",".join(d["task_name"] for d in REASONING_TASKS.values())).load_tasks()
    LightevalTask.load_datasets(tasks)
    docs = {key: tasks[definition["task_name"]].get_docs(max_samples=None)
            for key, definition in REASONING_TASKS.items()}
    for key, samples in docs.items():
        if len(samples) != REASONING_TASKS[key]["count"]:
            raise ValueError(f"unexpected full dataset count for {key}: {len(samples)}")
        if len({str(doc.id) for doc in samples}) != len(samples):
            raise ValueError(f"duplicate document IDs in {key}")
        if any(doc.num_samples != 1 for doc in samples):
            raise ValueError(f"{key} requires exactly one generation per document")
    rows = {}
    for name, model_dir in models.items():
        model = Path(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model)
        if not uses_chat_template(tokenizer=tokenizer):
            raise ValueError(f"missing reasoning chat template: {name}")
        default = PromptManager(True, tokenizer, enable_thinking=None)
        thinking = PromptManager(True, tokenizer, enable_thinking=True)
        direct = PromptManager(True, tokenizer, enable_thinking=False)
        task_rows = {}
        for key, samples in docs.items():
            prompts = [default.prepare_prompt(doc) for doc in samples]
            if any(prompt != thinking.prepare_prompt(doc) or prompt == direct.prepare_prompt(doc)
                   for prompt, doc in zip(prompts, samples)):
                raise ValueError(f"default template does not enable thinking: {name}/{key}")
            tokens = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
            longest = max(map(len, tokens))
            protocol = protocol_for(key)
            if longest + protocol["max_new_tokens"] > protocol["max_model_length"]:
                raise ValueError(f"prompt plus generation exceeds context: {name}/{key}: {longest}")
            task_rows[key] = {"count": len(samples), "generations_per_doc": 1,
                              "max_prompt_tokens": longest,
                              "document_ids_sha256": _digest([str(doc.id) for doc in samples]),
                              "prompt_set_sha256": _digest(sorted(prompts)),
                              "prompt_sha256": _digest(prompts), "token_sha256": _digest(tokens)}
        rows[name] = {"tasks": task_rows,
                      "tokenizer_files": tokenizer_asset_hashes(model, tokenizer)}
    reference = rows["native_nvfp4"]
    if any(row != reference for row in rows.values()):
        raise ValueError("candidate tokenizer, document IDs, prompts or input tokens differ from native")
    result = {"complete": True, "models": rows}
    write_json(result, output)
    return result


def _check_scores(scores):
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) and 0 <= value <= 1 for value in scores.values()):
        raise ValueError("benchmark scores must be finite fractions in [0, 1]")


def _check_reasoning_config(result, expected, model_kind):
    raw = json.loads(Path(result["results_file"]).read_text(encoding="utf-8"))
    general = raw["config_general"]
    config = general["model_config"]
    if general["max_samples"] != expected["max_samples"]:
        raise ValueError("actual reasoning result used the wrong sample limit")
    keys = ("tensor_parallel_size", "kv_cache_dtype", "enforce_eager", "max_num_batched_tokens",
            "seed", "max_model_length", "gpu_memory_utilization")
    for key in keys:
        if config.get(key) != expected[key]:
            raise ValueError(f"actual reasoning model config differs for {key}")
    if config.get("max_num_seqs") != expected["batch_size_max_num_seqs"]:
        raise ValueError("actual reasoning batch capacity differs")
    generation = config["generation_parameters"]
    for key in ("max_new_tokens", "temperature", "top_p", "top_k", "min_p"):
        if generation.get(key) != expected[key]:
            raise ValueError(f"actual generation config differs for {key}")
    if config.get("enable_thinking") is False:
        raise ValueError("actual reasoning config disables thinking")
    if model_kind is not None:
        backends = ("emulation", "emulation") if model_kind == "native_nvfp4" else ("auto", "triton")
        if (config.get("linear_backend"), config.get("moe_backend")) != backends:
            raise ValueError("actual model execution backends differ from the declared model kind")
    return raw


def validate_metrics(result, task, *, smoke, prompt_reference=None, model_kind=None,
                     tensor_parallel_size=2):
    expected = protocol_for(task, smoke=smoke, tensor_parallel_size=tensor_parallel_size)
    if result.get("protocol") != expected:
        raise ValueError(f"wrong evaluation protocol for {task}; full task coverage is required")
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"metric metadata differs from protocol: {key}")
    if result.get("task") != task:
        raise ValueError("metric task name differs")
    if model_kind is not None:
        if model_kind not in ("native_nvfp4", "hif4") or result.get("model_kind") != model_kind:
            raise ValueError("metric model kind differs")
        if result.get("native_nvfp4") is not (model_kind == "native_nvfp4"):
            raise ValueError("native model flag differs")
    if task == "arc":
        if set(result["scores"]) != set(ARC_COUNTS) or set(result["n_samples"]) != set(ARC_COUNTS):
            raise ValueError("both ARC-C and ARC-E results are required")
        for name, count in ARC_COUNTS.items():
            observed = result["n_samples"][name]
            if observed["original"] != count or observed["effective"] != expected["sample_counts"][name]:
                raise ValueError(f"incomplete ARC sample count: {name}")
            if result["raw_results"][name]["acc,none"] != result["scores"][name]:
                raise ValueError(f"ARC result does not use raw accuracy: {name}")
        scores = {"ARC-C": result["scores"]["arc_challenge"], "ARC-E": result["scores"]["arc_easy"]}
    else:
        raw = _check_reasoning_config(result, expected, model_kind)
        key, metric = expected["task_name"], expected["metric"]
        score = result["results"][key][metric]
        if raw["results"][key][metric] != score:
            raise ValueError("reasoning aggregate differs from its raw result")
        _, details = reasoning_details(result, task)
        count = expected["count"]
        if len(details) != count or len({str(row["doc"]["id"]) for row in details}) != count:
            raise ValueError(f"incomplete or duplicate {task} samples; expected {count}")
        if result.get("n_samples") != count or result.get("actual_generations_per_doc") != 1:
            raise ValueError("reasoning sample metadata differs from actual full coverage")
        if any(len(row["model_response"]["text_post_processed"]) != 1 for row in details):
            raise ValueError("exactly one generation per document is required")
        prompt_hash = _digest(sorted(row["model_response"]["input"] for row in details))
        if result.get("actual_prompt_set_sha256") != prompt_hash:
            raise ValueError("actual prompt hash differs from saved metric metadata")
        if not smoke:
            if prompt_reference is None or prompt_reference["count"] != count:
                raise ValueError("a full-count prompt audit reference is required")
            if prompt_hash != prompt_reference["prompt_set_sha256"]:
                raise ValueError("evaluated prompts differ from the audited full task")
        sample_scores = [row["metric"][metric] for row in details]
        _check_scores(dict(enumerate(sample_scores)))
        if not math.isclose(math.fsum(sample_scores) / count, score, rel_tol=0, abs_tol=1e-12):
            raise ValueError("reasoning score does not match the per-document results")
        scores = {REASONING_METRICS[task]: score}
    _check_scores(scores)
    return scores


def compare_scores(native, candidate):
    if set(native) != set(METRICS) or set(candidate) != set(METRICS):
        raise ValueError("target calculation requires all five benchmark scores")
    _check_scores(native)
    _check_scores(candidate)
    losses = {name: max(0.0, native[name] - candidate[name]) * 100 for name in METRICS}
    mean_loss = math.fsum(losses.values()) / len(METRICS)
    return {"positive_loss_pp": losses,
            "delta_vs_native_pp": {name: (candidate[name] - native[name]) * 100 for name in METRICS},
            "mean_positive_loss_pp": mean_loss, "threshold_pp": 0.5,
            "observed_target_met": mean_loss <= 0.5 + 1e-12}


def summarize(run_root):
    root = Path(run_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    tensor_parallel_size = manifest.get("tensor_parallel_size", 2)
    models = manifest["models"]
    if "native_nvfp4" not in models or len(models) < 2:
        raise ValueError("comparison requires native_nvfp4 and at least one candidate")
    audit = json.loads((root / "prompt_audit.json").read_text(encoding="utf-8"))
    if audit.get("complete") is not True or set(audit["models"]) != set(models):
        raise ValueError("complete prompt audit for every model is required")
    if any(row != audit["models"]["native_nvfp4"] for row in audit["models"].values()):
        raise ValueError("model prompt audits do not match the native reference")
    rows = []
    for name, model in models.items():
        expected_kind = "native_nvfp4" if name == "native_nvfp4" else "hif4"
        if model["kind"] != expected_kind:
            raise ValueError(f"unexpected model kind for {name}")
        row = {"name": name, "model": model, "scores": {}, "metrics_files": {}}
        for task, folder in TASK_DIRS.items():
            path = root / "formal" / name / "eval" / folder / "metrics.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            reference = None if task == "arc" else audit["models"][name]["tasks"][task]
            row["scores"].update(validate_metrics(result, task, smoke=False,
                                                  prompt_reference=reference, model_kind=model["kind"],
                                                  tensor_parallel_size=tensor_parallel_size))
            row["metrics_files"][task] = str(path.relative_to(root))
        rows.append(row)
    native = next(row["scores"] for row in rows if row["name"] == "native_nvfp4")
    for row in rows:
        row.update(compare_scores(native, row["scores"]))
    comparison = {"complete": True, "definition": "mean_i(max(0, E0_i - candidate_i) * 100)",
                  "metric_order": list(METRICS), "threshold_pp": 0.5,
                  "task_counts": {**{REASONING_METRICS[k]: v["count"] for k, v in REASONING_TASKS.items()},
                                  "ARC-E": ARC_COUNTS["arc_easy"], "ARC-C": ARC_COUNTS["arc_challenge"]},
                  "models": rows,
                  "observed_target_met": any(row["observed_target_met"] for row in rows if row["name"] != "native_nvfp4")}
    write_json(comparison, root / "comparison.json")
    lines = ["# NVFP4 → HiF4 五项下游精度验收", "",
             "所有任务使用已确认的全量覆盖：MMLU-Pro 12,032 题、LiveCodeBench 175 题、GSM8K 1,319 题、ARC-E 2,376 题、ARC-C 1,172 题。分数为百分比。", "",
             "| 模型 | " + " | ".join(METRICS) + " | 平均正向下降（百分点） | 本轮观察达标 |",
             "| --- | " + " | ".join(["---:"] * 6) + " | --- |"]
    for row in rows:
        passed = "基准" if row["name"] == "native_nvfp4" else ("是" if row["observed_target_met"] else "否")
        lines.append("| " + row["name"] + " | " + " | ".join(f"{row['scores'][key] * 100:.4f}" for key in METRICS)
                     + f" | {row['mean_positive_loss_pp']:.4f} | {passed} |")
    lines += ["", "每项损失为 `max(0, E0 − candidate) × 100`，五项等权平均，阈值为 `≤ 0.5` 个百分点。某项提高不抵消其他任务的下降。", "",
              "| 候选模型 | " + " | ".join(METRICS) + " |", "| --- | " + " | ".join(["---:"] * 5) + " |"]
    for row in rows:
        if row["name"] != "native_nvfp4":
            lines.append("| " + row["name"] + " | " + " | ".join(f"{row['positive_loss_pp'][key]:.4f}" for key in METRICS) + " |")
    lines += ["", "上表为各任务正向下降（百分点）。MMLU-Pro/GSM8K 使用 extractive match，LCB 使用官方 checker 的单次生成 pass@1，ARC 使用 raw accuracy。", "",
              "本报告只说明当前模型和固定全量评测协议下的观察值；单次采样不是统计显著性或所有采样种子的保证。原生模型加载 packed NVFP4，HiF4 模型使用 HiF4 量化后反量化的 BF16 权重与运行时激活量化；本结果不构成低比特存储或推理加速测量。", "",
              "- [模型来源和运行配置](manifest.json)", "- [完整分数、逐项损失和原始指标路径](comparison.json)",
              "- [全量实际 prompt 与 tokenizer 核对](prompt_audit.json)", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return comparison
