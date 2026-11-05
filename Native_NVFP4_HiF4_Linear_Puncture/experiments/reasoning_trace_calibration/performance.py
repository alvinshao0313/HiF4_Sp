"""Real TP2 diagnostics for continuous requests and native-NVFP4 CUDA Graph."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import time

from .artifacts import write_json
from .config import GENERATION, NATIVE
from .generate import close_engine, engine, pooled_requests, prompt


def qualify(root, mode):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    output = Path(root) / "qualification/performance" / mode
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
    llm = engine(enforce_eager=mode == "eager")
    try:
        rows = [{"domain": "math", "question": f"Explain how to compute {i + 2} plus {i + 3}."} for i in range(20)]
        ids = [prompt(tokenizer, row) for row in rows]
        # Fixed 64-step greedy decode measures runtime separately from sampling
        # and task accuracy. It never contributes a calibration/development item.
        params = SamplingParams(**{**GENERATION, "temperature": 0., "max_tokens": 64, "ignore_eos": True})
        llm.generate([{"prompt_token_ids": ids[0]}], SamplingParams(max_tokens=2, temperature=0.))
        started = time.monotonic()
        values = llm.generate([{"prompt_token_ids": v} for v in ids[:16]], params, use_tqdm=False)
        seconds = time.monotonic() - started
        if len(values) != 16 or any(len(v.outputs[0].token_ids) != 64 for v in values):
            raise RuntimeError("TP2 decode coverage is incomplete")
        benchmark = {"seconds": seconds, "output_tokens": 1024, "tokens_per_second": 1024 / seconds,
                     "responses": [list(v.outputs[0].token_ids) for v in values], "prompt_token_ids": ids[:16]}
        write_json(benchmark, output / "decode.json")
        # A long first request and short later requests exercise refill while an
        # earlier request is still running. Only iteration counts are shortened.
        sampling = [SamplingParams(**{**GENERATION, "max_tokens": 256 if i == 0 else 16,
                                      "ignore_eos": True}) for i in range(20)]
        accepted, results = 0, []
        stream = StringIO()
        with redirect_stdout(stream):
            for index, row, tokens, result in pooled_requests(llm, tokenizer, rows, 20, lambda: accepted, sampling=sampling):
                if len(result.outputs[0].token_ids) != sampling[index].max_tokens or tokens != ids[index]:
                    raise RuntimeError("continuous request changed prompt or output coverage")
                results.append(index)
                accepted += 1
        events = [json.loads(line) for line in stream.getvalue().splitlines() if line.startswith('{"event": "enqueue_trace"')]
        if results != list(range(20)) or len(events) != 20:
            raise RuntimeError("continuous requests repeated, skipped, or reordered a question")
        if not any(e["index"] >= 16 and e["pending"] > 1 and e["completed_waiting_for_order"] > 0 for e in events):
            raise RuntimeError("continuous refill while earlier requests run was not exercised")
        write_json({"passed": True, "mode": mode, "native_snapshot": str(NATIVE), "tensor_parallel_size": 2,
                    "decode": benchmark, "continuous_events": events, "ordered_results": results,
                    "pending_after_completion": llm.llm_engine.get_num_unfinished_requests(),
                    "scope": "16-request 64-step greedy runtime; 20-request unequal-length refill; no accuracy claim"}, output / "metrics.json")
    finally:
        close_engine(llm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    args = parser.parse_args()
    qualify(args.run, args.mode)
