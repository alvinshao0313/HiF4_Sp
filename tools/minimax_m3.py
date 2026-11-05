#!/usr/bin/env python
"""Opt-in MiniMax M3.1 entrypoint; existing inference entrypoints stay intact."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "3rdparty/lighteval/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Run on multiprocessing spawn imports as well as in the parent. Registration
# is confined to processes launched through this dedicated entrypoint.
from vllm.model_executor.models.minimax_m3.integration import register
register()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "generate", "evaluate", "qualify"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tensor_parallel_size", type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_num_batched_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--prompt", default="What is 2 + 2?")
    parser.add_argument("--datasets", default="gsm8k")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--output_dir", default="results/minimax_m3")
    parser.add_argument("--dataset_manifest")
    args = parser.parse_args()
    args.repo_root = REPO_ROOT
    from vllm.config import ModelConfig
    config = ModelConfig(model=args.model_path, trust_remote_code=False,
        enforce_eager=True, max_model_len=args.max_model_len)
    if config.architecture != "MiniMaxM3SparseForConditionalGeneration":
        parser.error("This entrypoint expects MiniMax-M3.1-Flash")
    if args.command == "check":
        print(json.dumps({"architecture": config.architecture,
            "quantization": config.quantization, "config_check": "passed",
            "generation_verified": False}, indent=2))
        return
    if args.tensor_parallel_size is None or args.tensor_parallel_size < 1:
        parser.error("generate/evaluate require an explicit --tensor_parallel_size")
    from vllm.model_executor.models.minimax_m3.evaluation import (
        engine_kwargs, evaluate, require_complete_text_checkpoint,
    )
    require_complete_text_checkpoint(args.model_path)
    if args.command == "qualify":
        from tools.minimax_m3_eval.qualification import qualify
        qualify(args)
        return
    if args.command == "evaluate":
        evaluate(args)
        return
    from vllm import LLM, SamplingParams
    model = LLM(**engine_kwargs(args.model_path, args.tensor_parallel_size,
        args.max_model_len, args.gpu_memory_utilization, args.batch_size,
        args.max_num_batched_tokens, args.pipeline_parallel_size))
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_new_tokens)
    outputs = model.chat([{"role": "user", "content": args.prompt}], params)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
