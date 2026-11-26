"""Real TP2 engine teardown probe; its generations are never formal scores."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..ban.engine import create_engine
from .config import read, save, verify
from .stages import budget_probe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = verify(args.run)
    budget = read(args.run / "budget/trace/budget.json")
    engine = create_engine(manifest["model"], manifest["spec"], artifact_path=budget["artifact"],
                           k_min=5, max_model_len=40960, max_num_seqs=128, max_num_batched_tokens=4096,
                           gpu_memory_utilization=0.9, seed=1234, kv_cache_dtype="bfloat16", enable_chunked_prefill=None)
    result = budget_probe(engine, read(args.run / "budget_inputs.json"), budget["chosen_lambda"], 64)
    save(args.output, {"status": "complete", "diagnostic": True, "max_new_tokens": 64, "result": result})
    print("RELEASE_PROBE_COMPLETE", flush=True)
