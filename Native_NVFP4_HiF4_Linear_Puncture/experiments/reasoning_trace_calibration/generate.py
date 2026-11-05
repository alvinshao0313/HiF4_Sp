"""One native packed-NVFP4 TP2 trace per isolated question, without answer access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .artifacts import atomic_save, digest, read_json, sha256, validate_root, write_json
from .config import COUNTS, ENGINE, GENERATION, NATIVE
from .prepare import require_isolation


def prompt(tokenizer, row):
    instruction = ("Solve the following mathematics problem. Put your final answer in \\boxed{}.\n\n" if row["domain"] == "math"
                   else "Solve the following programming problem in Python 3. Read standard input and write standard output. "
                        "Provide the complete solution in a Python code block.\n\n")
    # Deliberate whitelist: no source solution, hidden test, or gold can enter.
    return tokenizer.apply_chat_template([{"role": "user", "content": instruction + row["question"]}],
                                         tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)


def completed(text, finish_reason):
    if finish_reason != "stop" or "</think>" not in text:
        return False
    final = text.rsplit("</think>", 1)[1].strip()
    return bool(final)


def engine(model=NATIVE, *, enforce_eager=None):
    import os
    from vllm import LLM
    model = Path(model).resolve()
    native = model == NATIVE.resolve()
    os.environ.pop("HIF4_RUNTIME_SPEC_PATH", None)
    extra = {}
    if not native:
        spec = model / "hif4_runtime_spec.pt"
        if not spec.is_file():
            raise FileNotFoundError(spec)
        os.environ["HIF4_RUNTIME_SPEC_PATH"] = str(spec)
        extra["additional_config"] = {"hif4_runtime_spec_path": str(spec)}
    settings = dict(ENGINE)
    if enforce_eager is not None:
        settings["enforce_eager"] = enforce_eager
        if not enforce_eager:
            # Bounded diagnostic capture; formal engine settings stay in ENGINE.
            extra["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                           "cudagraph_capture_sizes": [1, 2, 4, 8, 16]}
    llm = LLM(model=str(model), trust_remote_code=True, dtype="auto", seed=1234,
              linear_backend="emulation" if native else "auto", moe_backend="emulation" if native else "triton",
              enable_prefix_caching=False, disable_log_stats=False, **settings, **extra)
    actual = llm.llm_engine.model_config.quantization
    if actual != ("modelopt_fp4" if native else None):
        raise ValueError(f"incorrect quantization backend: {actual}")
    return llm


def close_engine(llm):
    core = llm.llm_engine.engine_core
    thread = core.output_queue_thread
    core.shutdown(timeout=60)
    thread.join(timeout=60)
    if thread.is_alive():
        raise RuntimeError("vLLM shutdown did not complete")


def pooled_requests(llm, tokenizer, pool, quota, accepted, *, sampling=None):
    """Refill finished slots, retaining fixed pool order and exact acceptance quota.

    Completed later indices wait in ``ready`` for earlier indices. Pending and
    ready requests reserve acceptance slots, so no speculative extra question is
    generated or aborted when the quota is reached.
    """
    from vllm import SamplingParams
    pending, ready = {}, {}
    cursor = next_result = 0
    while accepted() < quota:
        while (cursor < len(pool) and len(pending) < 16
               and accepted() + len(pending) + len(ready) < quota):
            row = pool[cursor]
            ids = prompt(tokenizer, row)
            params = SamplingParams(**GENERATION) if sampling is None else sampling[cursor]
            if len(ids) + params.max_tokens > ENGINE["max_model_len"]:
                raise ValueError("question pool contains a prompt outside the audited token budget")
            requests = llm.enqueue({"prompt_token_ids": ids}, params, use_tqdm=False)
            if len(requests) != 1:
                raise ValueError("exactly one completion per question is required")
            # enqueue returns the randomized internal ID; RequestOutput carries
            # the external ID. Use vLLM's actual mapping rather than parsing IDs.
            external_id = llm.llm_engine.output_processor.request_states[requests[0]].external_req_id
            pending[external_id] = (cursor, row, ids)
            print(json.dumps({"event": "enqueue_trace", "index": cursor, "pending": len(pending),
                              "completed_waiting_for_order": len(ready)}), flush=True)
            cursor += 1
        if not pending:
            break
        for result in llm.llm_engine.step():
            if not result.finished:
                continue
            index, row, ids = pending.pop(result.request_id)
            ready[index] = (index, row, ids, result)
        while next_result in ready:
            yield ready.pop(next_result)
            next_result += 1
    if pending or ready or llm.llm_engine.has_unfinished_requests():
        raise RuntimeError("generation ended with unconsumed requests")


def generate(root):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from ..e2e_diag_reconstruction.data.calibration import CalibrationSample
    root = validate_root(root)
    require_isolation(root)
    output = root / "generation"
    output.mkdir(exist_ok=False)
    pools = read_json(root / "isolation/pools.json")
    tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
    llm = engine()
    selected, attempts = {"train": [], "val": []}, []
    started = time.monotonic()
    try:
        for split in ("train", "val"):
            for domain in ("math", "code"):
                accepted = 0
                for index, row, ids, result in pooled_requests(llm, tokenizer, pools[domain][split], COUNTS[split], lambda: accepted):
                    response = result.outputs[0]
                    tokens = list(response.token_ids)
                    record = {"id": row["id"], "split": split, "domain": domain, "locator": row["locator"],
                              "question_sha256": row["question_sha256"], "prompt_token_ids": ids,
                              "generated_token_ids": tokens, "text": response.text, "seed": GENERATION["seed"],
                              "finish_reason": response.finish_reason, "stop_reason": response.stop_reason,
                              "accepted": completed(response.text, response.finish_reason),
                              "generation_seconds": time.monotonic() - started}
                    filename = f"{split}_{domain}_{index:05d}.json"
                    write_json(record, output / filename)
                    attempts.append({"id": row["id"], "split": split, "domain": domain,
                                     "path": filename, "sha256": sha256(output / filename), "accepted": record["accepted"]})
                    if record["accepted"]:
                        full = torch.tensor(ids + tokens, dtype=torch.long)
                        if not tokens or full.numel() > ENGINE["max_model_len"]:
                            raise ValueError("invalid full trace token coverage")
                        sample = CalibrationSample(row["id"], row["locator"]["row"], full, torch.ones_like(full),
                                                   {"source": "native_traces", "domain": domain,
                                                    "question": row["question"], "trace_sha256": digest(record)})
                        selected[split].append(sample)
                        accepted += 1
                    write_json({"attempts": attempts, "complete": False}, output / "manifest.json")
                    print(json.dumps({"event": "trace", "split": split, "domain": domain, "accepted": accepted,
                                      "tokens": len(tokens), "finish_reason": response.finish_reason}), flush=True)
                if accepted != COUNTS[split]:
                    raise ValueError(f"isolated candidate pool exhausted: {split}/{domain}")
        lengths = {}
        for split, samples in selected.items():
            if len(samples) != {"train": 128, "val": 32}[split]:
                raise ValueError("incomplete calibration split")
            atomic_save(samples, root / "calibration/native_traces" / f"{split}.pt")
            sizes = sorted(s.input_ids.numel() for s in samples)
            lengths[split] = {"count": len(sizes), "total_tokens": sum(sizes), "min": sizes[0], "max": sizes[-1],
                              "median": sizes[len(sizes) // 2], "lengths": sizes}
        write_json({"attempts": attempts, "complete": True, "lengths": lengths,
                    "seconds": time.monotonic() - started, "sampling": GENERATION, "engine": ENGINE,
                    "source_snapshot": str(NATIVE), "weight_index_sha256": sha256(NATIVE / "model.safetensors.index.json"),
                    "calibration_sha256": {s: sha256(root / "calibration/native_traces" / f"{s}.pt") for s in selected}},
                   output / "manifest.json")
    finally:
        close_engine(llm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    generate(parser.parse_args().run)
