"""Verify the new expert bytes and unchanged TP2 evaluation configuration."""
from copy import deepcopy
import json
from pathlib import Path

from ..qwen3_aisbench_gpqa.common import read, save


def validate_runtime(root, variant, phase):
    root = Path(root)
    target = root / variant / phase
    original = Path(read(root / "preparation.json")["reference_candidate"]) / "probe"
    old = read(read(original / "status.json")["results_file"])
    expected_config = deepcopy(old["config_general"]["model_config"])
    expected_config.pop("model_name")
    expected_config["tensor_parallel_size"] = 2
    expected_config["kv_cache_memory_bytes"] = read(root / variant / "manifest.json")["protocol"]["kv_cache_memory_bytes"]
    expected_config["generation_parameters"]["max_new_tokens"] = 64 if phase == "probe" else 32768
    actual = read(read(target / "status.json")["results_file"])["config_general"]["model_config"]
    actual.pop("model_name")
    if actual != expected_config or actual["kv_cache_dtype"] != "bfloat16":
        changed = {key: (expected_config.get(key), actual.get(key)) for key in set(expected_config) | set(actual)
                   if expected_config.get(key) != actual.get(key)}
        raise ValueError(f"Unexpected model configuration differences: {changed}")
    expected = read(root / "expected_runtime.json")[variant]
    ranks = {}
    for path in (target / "audit").glob("worker-*.jsonl"):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        experts = [row for row in events if row["event"] == "loaded_experts"]
        if not experts:
            continue
        rank_set = {str(row["tp_rank"]) for row in experts}
        if len(rank_set) != 1 or len(experts) != 48:
            raise ValueError("Incomplete expert coverage")
        rank = rank_set.pop()
        if rank in ranks or rank not in expected["ranks"]:
            raise ValueError("Unexpected or duplicated rank")
        if ({row["prefix"]: row["tensors"] for row in experts} != expected["ranks"][rank]
                or any(row["variant"] != variant or row["method"] != "Qwen3ModelOptMXFP4FusedMoE" for row in experts)):
            raise ValueError("Expert TP2 weights/scales/method changed")
        methods = {row["prefix"]: row["method"] for row in events if row["event"] == "non_expert_method"}
        if methods != expected["non_expert_methods"]:
            raise ValueError("Non-expert implementation changed")
        ranks[rank] = 48
    if set(ranks) != {"0", "1"}:
        raise ValueError("Missing TP2 rank")
    save(target / "runtime_validation.json", {"status": "passed", "tensor_parallel_size": 2,
        "kv_cache_memory_bytes": actual["kv_cache_memory_bytes"], "kv_cache_dtype": "bfloat16",
        "ranks": ranks, "expert_slices_exact": True, "non_expert_methods_equal": True,
        "reference_tp2_configuration_equal": True})
