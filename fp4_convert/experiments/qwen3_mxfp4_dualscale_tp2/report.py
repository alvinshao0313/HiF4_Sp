"""Check TP2 model selection, checkpoint slices and unchanged sampling."""
from copy import deepcopy
import json
from pathlib import Path

from ..qwen3_aisbench_gpqa.common import read, save


def validate_runtime(root, phase):
    root = Path(root)
    target = root / "dualscale_g512" / phase
    preparation = read(root / "preparation.json")
    original = Path(preparation["parent_run"]) / "dualscale_g512"
    old_status = read(original / "probe/status.json")
    expected_config = deepcopy(read(old_status["results_file"])["config_general"]["model_config"])
    expected_config.pop("model_name")
    expected_config["tensor_parallel_size"] = 2
    expected_config["generation_parameters"]["max_new_tokens"] = 64 if phase == "probe" else 32768
    actual_status = read(target / "status.json")
    actual_config = dict(read(actual_status["results_file"])["config_general"]["model_config"])
    actual_config.pop("model_name")
    if actual_config != expected_config or actual_config["kv_cache_dtype"] != "bfloat16":
        raise ValueError("Saved model settings changed beyond the authorized TP4-to-TP2 change")
    expected = read(root / "expected_runtime.json")
    ranks = {}
    for path in (target / "audit").glob("worker-*.jsonl"):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        experts = [row for row in events if row["event"] == "loaded_experts"]
        if not experts:
            continue
        rank_set = {row["tp_rank"] for row in experts}
        if len(rank_set) != 1 or len(experts) != 48:
            raise ValueError("Incomplete rank/layer coverage")
        rank = str(rank_set.pop())
        if rank in ranks or rank not in expected["ranks"]:
            raise ValueError("Unexpected or duplicated TP2 worker")
        tensors = {row["prefix"]: row["tensors"] for row in experts}
        if (tensors != expected["ranks"][rank]
                or any(row["variant"] != "dualscale_g512"
                       or row["method"] != "Qwen3ModelOptMXFP4FusedMoE" for row in experts)):
            raise ValueError("TP2 packed weights/scales or expert method differs from the checkpoint")
        methods = {row["prefix"]: row["method"] for row in events if row["event"] == "non_expert_method"}
        if methods != expected["non_expert_methods"]:
            raise ValueError("Non-expert method selection changed")
        ranks[rank] = {"layers": len(experts), "packed_bytes_and_outer_scales_equal": True}
    if set(ranks) != {"0", "1"}:
        raise ValueError("Missing one of the two TP ranks")
    save(target / "runtime_validation.json", {
        "status": "passed", "tensor_parallel_size": 2, "ranks": ranks,
        "kv_cache_dtype": "bfloat16", "only_tp_changed_from_previous_config": True,
        "checkpoint_slices_exact": True, "non_expert_methods_equal": True,
    })
