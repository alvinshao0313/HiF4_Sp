"""Generate twice on a local TP2 checkpoint. Import registration in spawn children."""
import argparse
import json
import math
import os
from pathlib import Path

from . import entry  # register the paper formats before vLLM starts worker processes


def main():
    from vllm import LLM, SamplingParams
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = args.model.resolve(strict=True)
    config = json.loads((model / "config.json").read_text())
    quant = config.get("quantization_config", {}).get("quant_method")
    kwargs = {}
    if quant != "paper_fp4":
        kwargs["kernel_config"] = {"linear_backend": "emulation", "moe_backend": "emulation"}
    llm = LLM(model=str(model), tensor_parallel_size=2, dtype="bfloat16",
              max_model_len=256, max_num_seqs=2, max_num_batched_tokens=256,
              enforce_eager=True, seed=17, kv_cache_dtype="bfloat16",
              kv_cache_memory_bytes=512 * 1024**2, **kwargs)
    prompts = ["Write the next two integers after 10, 11:", "The capital city of France is"]
    results = []
    for prompt in prompts:
        response = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=6,
                                ignore_eos=True, logprobs=1), use_tqdm=False)[0].outputs[0]
        tokens = list(response.token_ids)
        if response.logprobs is None or len(tokens) != 6 or len(response.logprobs) != len(tokens):
            raise ValueError("Incomplete token or logprob output")
        logprobs = [row[token].logprob for row, token in zip(response.logprobs, tokens, strict=True)]
        if not all(math.isfinite(value) for value in logprobs):
            raise ValueError("Non-finite chosen-token logprob")
        results.append({"prompt": prompt, "tokens": tokens, "text": response.text,
                        "chosen_logprobs": logprobs})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump({"model": str(model), "tp": 2, "seed": 17, "kv_dtype": "bfloat16",
                   "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "answers": results},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
