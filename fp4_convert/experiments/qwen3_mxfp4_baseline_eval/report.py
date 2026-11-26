"""Native/direct runtime checks; reuse the unchanged five-response scorer audit."""
from copy import deepcopy
import csv
import json
from pathlib import Path

from ..qwen3_mxfp4_combination_eval.report import summarize
from ..qwen3_mxfp4_rotation_eval.report import one
from .common import TASKS, VARIANTS, read, save


def runtime_check(root, variant, target, task, short):
    expected = read(root / "expected_runtime.json")
    ranks = {}
    event = "loaded_experts" if variant == "direct" else "loaded_native_experts"
    prefixes = {f"model.layers.{i}.mlp.experts" for i in range(48)}
    for path in (target / "audit").glob("worker-*.jsonl"):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        experts = [row for row in events if row["event"] == event]
        if not experts:
            continue
        rank_set = {str(row["tp_rank"]) for row in experts}
        if len(rank_set) != 1 or len(experts) != 48 or {row["prefix"] for row in experts} != prefixes:
            raise ValueError("Missing TP rank or expert layer evidence")
        rank = rank_set.pop()
        if rank in ranks or rank not in {"0", "1"}:
            raise ValueError("Duplicate or unexpected TP rank")
        if variant == "direct":
            if ({row["prefix"]: row["tensors"] for row in experts} != expected["direct_ranks"][rank]
                    or any(row["variant"] != "direct" or row["method"] != "Qwen3ModelOptMXFP4FusedMoE" for row in experts)):
                raise ValueError("Direct expert methods or TP2 packed bytes differ")
        else:
            shapes = {"w13_weight": [128, 768, 1024], "w2_weight": [128, 2048, 192],
                      "w13_weight_scale": [128, 768, 128], "w2_weight_scale": [128, 2048, 24]}
            for row in experts:
                if (row["tp_size"] != 2 or row["method"] != "ModelOptNvFp4FusedMoE"
                        or row["experts_class"] != "Nvfp4QuantizationEmulationTritonExperts"
                        or {k: v["shape"] for k, v in row["tensors"].items()} != shapes
                        or any(row["tensors"][k]["dtype"] != "torch.uint8" for k in ("w13_weight", "w2_weight"))):
                    raise ValueError(f"Unexpected native NVFP4 method/backend/TP2 layout: {row}")
        methods = {row["prefix"]: row["method"] for row in events if row["event"] == "non_expert_method"}
        if methods != expected["non_expert_methods"]:
            raise ValueError("Non-expert method changed")
        ranks[rank] = 48
    if set(ranks) != {"0", "1"}:
        raise ValueError("Both TP2 ranks must be present")
    expected_config = deepcopy(read(root / "manifest.json")["protocol"])
    expected_config["generation_parameters"]["max_new_tokens"] = 64 if short else TASKS[task]["max_new_tokens"]
    actual = read(one((target / "evaluation").rglob("results_*.json")))["config_general"]["model_config"]
    actual.pop("model_name")
    if actual != expected_config or actual["kv_cache_dtype"] != "bfloat16":
        changed = {key: (expected_config.get(key), actual.get(key)) for key in set(expected_config) | set(actual)
                   if expected_config.get(key) != actual.get(key)}
        raise ValueError(f"Inference configuration changed: {changed}")
    result = {"status": "passed", "variant": variant, "ranks": ranks, "protocol": actual,
              "direct_packed_slices_exact": variant == "direct", "native_backend_verified": variant == "native",
              "non_expert_methods_equal": True}
    save(target / "runtime_validation.json", result)
    return result


def compare(root):
    output = {"status": "completed", "metric": "mean single-response accuracy", "tasks": {}}
    for task in TASKS:
        arms = {v: read(root / v / "formal" / task / "summary.json") for v in VARIANTS}
        if (arms["native"]["input_sha256"] != arms["direct"]["input_sha256"]
                or arms["native"]["protocol"] != arms["direct"]["protocol"]):
            raise ValueError("Comparison inputs or inference protocols differ")
        output["tasks"][task] = arms
    save(root / "comparison.json", output)
    with (root / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "variant", "mean_accuracy_percent", "sample_variance_pp_squared", *[f"sampling_{i}" for i in range(1, 6)]])
        for task, arms in output["tasks"].items():
            for variant, row in arms.items():
                writer.writerow([task, variant, row["accuracy_percent"], row["sample_variance_pp_squared"], *row["accuracy_percent_by_sampling_index"]])
    return output
