"""Native 235B TP8 full traces, with a separate time-compressed engine short test."""
from __future__ import annotations

import argparse
import os
import time

import torch

from ..e2e_diag_reconstruction.data.calibration import CalibrationSample
from ..reasoning_trace_calibration.generate import close_engine, completed, pooled_requests, prompt
from .artifacts import atomic_save, digest, read_json, sha256, verify_inputs, verify_sources, write_json
from .config import COUNTS, ENGINE, GENERATION, NATIVE


def engine():
    from vllm import LLM
    if torch.cuda.device_count() != ENGINE["tensor_parallel_size"]:
        raise RuntimeError("native 235B generation requires exactly eight visible GPUs")
    os.environ.pop("HIF4_RUNTIME_SPEC_PATH", None)
    llm = LLM(model=str(NATIVE), trust_remote_code=True, dtype="auto", seed=1234,
              linear_backend="emulation", moe_backend="emulation", enable_prefix_caching=False,
              disable_log_stats=False, **ENGINE)
    config = llm.llm_engine.vllm_config
    if config.model_config.quantization != "modelopt_fp4" or config.parallel_config.tensor_parallel_size != 8:
        raise ValueError("actual native generation engine differs from the TP8 NVFP4 protocol")
    return llm


def generate(root, *, smoke=False):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    root = verify_inputs(root)
    verify_sources(root)
    output = root / ("qualification/generation" if smoke else "generation")
    output.mkdir(parents=True, exist_ok=False)
    if not smoke and not read_json(root / "qualification/generation/manifest.json")["complete"]:
        raise ValueError("TP8 generation qualification is required")
    tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
    llm = engine()
    attempts, selected = [], {"train": [], "val": []}
    started = time.monotonic()
    try:
        if smoke:
            # Twenty requests exercise refilling beyond the sixteen live slots;
            # only decode time is reduced, never context, TP or engine capacity.
            rows = [{"domain": "math" if i % 2 == 0 else "code", "question": f"Compute {i} + 2."} for i in range(20)]
            params = [SamplingParams(**{**GENERATION, "max_tokens": 64}) for _ in rows]
            count = 0
            for index, _, ids, result in pooled_requests(llm, tokenizer, rows, 20, lambda: count, sampling=params):
                if len(result.outputs) != 1 or not result.outputs[0].token_ids:
                    raise ValueError("generation qualification produced an empty or multiple response")
                response = result.outputs[0]
                write_json({"prompt_token_ids": ids, "generated_token_ids": list(response.token_ids),
                            "text": response.text, "finish_reason": response.finish_reason}, output / f"{index:03d}.json")
                count += 1
            if count != 20:
                raise ValueError("incomplete generation qualification")
            write_json({"complete": True, "smoke_only": True, "requests": count, "max_tokens": 64,
                        "engine": ENGINE, "source_snapshot": str(NATIVE), "seconds": time.monotonic() - started},
                       output / "manifest.json")
            return
        pools = read_json(root / "isolation/pools.json")
        for split in ("train", "val"):
            for domain in ("math", "code"):
                accepted = 0
                for index, row, ids, result in pooled_requests(llm, tokenizer, pools[domain][split], COUNTS[split], lambda: accepted):
                    if len(result.outputs) != 1:
                        raise ValueError("one native completion per calibration question is required")
                    response = result.outputs[0]
                    tokens = list(response.token_ids)
                    record = {"id": row["id"], "split": split, "domain": domain, "locator": row["locator"],
                              "question_sha256": row["question_sha256"], "prompt_token_ids": ids,
                              "generated_token_ids": tokens, "text": response.text, "seed": GENERATION["seed"],
                              "finish_reason": response.finish_reason, "stop_reason": response.stop_reason,
                              "accepted": completed(response.text, response.finish_reason),
                              "source_snapshot": str(NATIVE)}
                    filename = f"{split}_{domain}_{index:05d}.json"
                    write_json(record, output / filename)
                    attempts.append({"id": row["id"], "split": split, "domain": domain,
                                     "path": filename, "sha256": sha256(output / filename), "accepted": record["accepted"]})
                    if record["accepted"]:
                        full = torch.tensor(ids + tokens, dtype=torch.long)
                        if not tokens or full.numel() > ENGINE["max_model_len"]:
                            raise ValueError("invalid complete calibration token sequence")
                        selected[split].append(CalibrationSample(row["id"], row["locator"]["row"], full,
                            torch.ones_like(full), {"source": "native_235b_traces", "domain": domain,
                            "question": row["question"], "trace_sha256": digest(record)}))
                        accepted += 1
                    write_json({"complete": False, "attempts": attempts}, output / "manifest.json")
                    print({"event": "trace", "split": split, "domain": domain, "accepted": accepted,
                           "tokens": len(tokens), "finish_reason": response.finish_reason}, flush=True)
                if accepted != COUNTS[split]:
                    raise ValueError(f"isolated candidate pool exhausted: {split}/{domain}")
        lengths = {}
        for split, samples in selected.items():
            atomic_save(samples, root / f"calibration/{split}.pt")
            sizes = sorted(s.input_ids.numel() for s in samples)
            lengths[split] = {"count": len(sizes), "total_tokens": sum(sizes), "lengths": sizes}
        write_json({"complete": True, "smoke_only": False, "attempts": attempts, "lengths": lengths,
                    "source_snapshot": str(NATIVE), "sampling": GENERATION, "engine": ENGINE,
                    "seconds": time.monotonic() - started,
                    "calibration_sha256": {s: sha256(root / f"calibration/{s}.pt") for s in selected}},
                   output / "manifest.json")
    finally:
        close_engine(llm)


def calibration(root):
    from .artifacts import load
    root = verify_inputs(root)
    manifest = read_json(root / "generation/manifest.json")
    if (not manifest["complete"] or manifest["smoke_only"] or manifest["source_snapshot"] != str(NATIVE)
            or manifest["sampling"] != GENERATION or manifest["engine"] != ENGINE):
        raise ValueError("complete native 235B calibration is required")
    pools = read_json(root / "isolation/pools.json")
    accepted = {row["id"]: row for row in manifest["attempts"] if row["accepted"]}
    splits = []
    all_ids = []
    for split in ("train", "val"):
        filename = root / f"calibration/{split}.pt"
        if sha256(filename) != manifest["calibration_sha256"][split]:
            raise ValueError("calibration changed after native generation")
        samples = load(filename)
        if len(samples) != 2 * COUNTS[split]:
            raise ValueError("incomplete calibration split")
        for domain in ("math", "code"):
            eligible = {row["id"] for row in pools[domain][split]}
            actual = [s for s in samples if s.meta["domain"] == domain]
            if len(actual) != COUNTS[split] or any(s.sample_id not in eligible for s in actual):
                raise ValueError("calibration domain balance or question pool differs")
        for sample in samples:
            item = accepted[sample.sample_id]
            raw = root / "generation" / item["path"]
            record = read_json(raw)
            expected = torch.tensor(record["prompt_token_ids"] + record["generated_token_ids"], dtype=torch.long)
            if (item["split"] != split or sha256(raw) != item["sha256"] or record["source_snapshot"] != str(NATIVE)
                    or not torch.equal(sample.input_ids, expected) or not sample.loss_mask.bool().all()
                    or not completed(record["text"], record["finish_reason"])):
                raise ValueError("calibration no longer preserves the complete native trace")
            all_ids.append(sample.sample_id)
        splits.append(samples)
    if len(set(all_ids)) != 160:
        raise ValueError("calibration train/validation IDs overlap")
    return tuple(splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--smoke", action="store_true")
    generate(**vars(parser.parse_args()))
