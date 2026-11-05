"""Match Ban's average expert budget on GSM8K train, without scoring answers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .artifact import sha256, write_json
from .engine import configure, create_engine, snapshot


TARGET = 6.0
TOLERANCE = 0.10
K_MIN = 5


def token_hash(ids):
    encoded = json.dumps(list(ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_inputs(args):
    from datasets import Dataset
    from lighteval.tasks.prompt_manager import PromptManager
    from lighteval.tasks.tasks.gsm8k import gsm8k_prompt
    from transformers import AutoTokenizer

    if not 1 <= args.nsamples <= 8:
        raise ValueError("budget matching uses 1 to 8 GSM8K train questions")
    path = Path(args.train_arrow).resolve(strict=True)
    if path.name != "gsm8k-train.arrow":
        raise ValueError("budget matching requires gsm8k-train.arrow, not the test split")
    train = Dataset.from_file(str(path))
    if len(train) < args.nsamples:
        raise ValueError("not enough GSM8K train questions")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError("budget matching requires the same chat template as formal inference")
    manager = PromptManager(use_chat_template=True, tokenizer=tokenizer, enable_thinking=None)
    rows = []
    for index in range(args.nsamples):
        doc = gsm8k_prompt(train[index], task_name="gsm8k|0")
        prompt = manager.prepare_prompt(doc)
        ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if len(ids) + 32768 > 40960:
            raise ValueError("GSM8K budget prompt would require truncation")
        rows.append({"train_index": index, "prompt": prompt, "input_ids": ids,
                     "input_token_sha256": token_hash(ids)})
    data = {"dataset": "openai/gsm8k", "subset": "main", "split": "train",
            "arrow_path": str(path), "arrow_sha256": sha256(path),
            "available_samples": len(train), "sample_indices": list(range(args.nsamples)),
            "formatter": "lighteval.tasks.tasks.gsm8k.gsm8k_prompt",
            "use_chat_template": True, "enable_thinking": None, "add_special_tokens": True,
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
    return rows, data


def run_budget(args):
    rows, data = prepare_inputs(args)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "inputs.json", {"data": data, "samples": rows})
    started = time.monotonic()
    trials = []
    result = {"status": "running", "model_path": str(Path(args.model_path).resolve()),
              "spec_path": str(Path(args.hif4_runtime_spec).resolve()),
              "artifact_path": str(Path(args.artifact_path).resolve()),
              "artifact_sha256": sha256(args.artifact_path), "data": data,
              "inputs_path": str(root / "inputs.json"), "target_average_experts": TARGET,
              "tolerance": TOLERANCE, "k_min": K_MIN, "max_bisection_steps": 6,
              "tensor_parallel_size": args.tensor_parallel_size, "seed": 1234,
              "max_num_seqs": 128, "max_num_batched_tokens": 4096,
              "max_model_len": 40960, "kv_cache_dtype": "auto", "dtype": "bfloat16",
              "gpu_memory_utilization": 0.9, "enable_chunked_prefill": None,
              "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                           "min_p": 0.0, "max_tokens": 32768, "n": 1,
                           "stop": [], "ignore_eos": False, "logprobs": 0},
              "command": sys.argv, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "created_at": datetime.now(timezone.utc).isoformat(),
              "code_sha256": sha256(__file__), "trials": trials}

    def save():
        best = min(trials, key=lambda row: abs(row["average_experts"] - TARGET)) if trials else None
        result.update(chosen_lambda=best["lambda"] if best else None,
                      chosen_average_experts=best["average_experts"] if best else None,
                      budget_matched=bool(best and abs(best["average_experts"] - TARGET) <= TOLERANCE),
                      elapsed_seconds=time.monotonic() - started)
        write_json(root / "budget.json", result)

    save()
    try:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        engine = create_engine(
            args.model_path, args.hif4_runtime_spec, args.tensor_parallel_size,
            artifact_path=args.artifact_path, k_min=K_MIN, execution="compact",
            max_model_len=40960, max_num_seqs=128, max_num_batched_tokens=4096,
            gpu_memory_utilization=0.9, seed=1234, enable_chunked_prefill=None,
            kv_cache_dtype="auto")
        engine.collective_rpc("ban_context")
        requests = [TokensPrompt(prompt_token_ids=row["input_ids"]) for row in rows]

        def probe(lam):
            configure(engine, mode="ban", capture=None, execution="compact", lam=lam, k_min=K_MIN)
            trial_started = time.monotonic()
            outputs = engine.generate(requests, sampling_params=SamplingParams(**result["sampling"]),
                                      use_tqdm=True)
            stats = snapshot(engine)
            if (len(outputs) != len(rows) or not stats["tokens"] or min(stats["tokens"]) <= 0
                    or stats["selected"] != stats["dispatched"]):
                raise RuntimeError("incomplete Ban generation or expert dispatch does not match selection")
            observations = []
            for row, output in zip(rows, outputs):
                if list(output.prompt_token_ids) != row["input_ids"] or len(output.outputs) != 1:
                    raise RuntimeError("budget generation input or sample count differs from the saved prompt")
                completion = output.outputs[0]
                observations.append({"train_index": row["train_index"],
                                     "input_tokens": len(output.prompt_token_ids),
                                     "input_token_sha256": row["input_token_sha256"],
                                     "output_tokens": len(completion.token_ids),
                                     "output_token_sha256": token_hash(completion.token_ids),
                                     "finish_reason": completion.finish_reason,
                                     "stop_reason": completion.stop_reason})
            trial = {"lambda": lam, "average_experts": stats["average_experts"],
                     "stats": stats, "requests": observations,
                     "length_terminated_requests": sum(row["finish_reason"] == "length" for row in observations),
                     "elapsed_seconds": time.monotonic() - trial_started}
            trials.append(trial)
            save()
            print(f"BAN_BUDGET lambda={lam:.6f} average_experts={stats['average_experts']:.6f} "
                  f"length_terminated={trial['length_terminated_requests']}", flush=True)
            return trial

        first = probe(0.7)
        if not result["budget_matched"]:
            lower = probe(0.0)
            upper = probe(1.0)
            if lower["average_experts"] <= TARGET <= upper["average_experts"]:
                if first["average_experts"] < TARGET:
                    lower = first
                else:
                    upper = first
                for _ in range(6):
                    if result["budget_matched"]:
                        break
                    trial = probe(0.5 * (lower["lambda"] + upper["lambda"]))
                    if trial["average_experts"] < TARGET:
                        lower = trial
                    else:
                        upper = trial
        result["status"] = "complete" if result["budget_matched"] else "failed"
        save()
        if not result["budget_matched"]:
            raise RuntimeError("Ban lambda search did not match average 6 experts within 0.10 on GSM8K train")
        print(f"BAN_BUDGET_MATCHED lambda={result['chosen_lambda']:.6f} "
              f"average_experts={result['chosen_average_experts']:.6f}", flush=True)
        return result
    except Exception as error:
        result.update(status="failed", error=str(error))
        save()
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model_path", required=True)
    result.add_argument("--hif4_runtime_spec", required=True)
    result.add_argument("--artifact_path", required=True)
    result.add_argument("--train_arrow", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--nsamples", type=int, default=8)
    result.add_argument("--tensor_parallel_size", type=int, default=2)
    return result


if __name__ == "__main__":
    run_budget(parser().parse_args())
