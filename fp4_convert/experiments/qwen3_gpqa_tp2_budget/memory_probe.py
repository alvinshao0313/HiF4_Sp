"""Compare KV capacities on frozen real GPQA continuation prefixes, not accuracy."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from ..qwen3_mxfp4_dualscale_tp2.integration import register as register_g512
from ..qwen3_mxfp4_uos.integration import register as register_uos

register_g512()
register_uos()


class MemoryMetrics:
    """Use the supported named worker RPC, without serializing Python callables."""

    def gpqa_memory_stats(self):
        import torch
        return {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved()}


def run(args):
    import torch
    from vllm import LLM, SamplingParams
    from ..qwen3_mxfp4.artifacts import save
    from ..qwen3_mxfp4.validate_gpu import device_state
    torch.set_num_threads(8)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    payload = json.loads(Path(args.inputs).read_text())
    prompts = [{"prompt_token_ids": item["tokens"]} for item in payload["rows"]]
    if len(prompts) != 256 or any(len(p["prompt_token_ids"]) != 4096 for p in prompts):
        raise ValueError("Expected 256 frozen 4096-token GPQA continuations")
    result = {"status": "running", "model": args.model, "kv_gib": args.kv_gib,
              "devices": os.environ["CUDA_VISIBLE_DEVICES"], "input_sha256": hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
              "scope": "performance-only real continuation prefixes; 256 fixed output tokens, ignore_eos; no accuracy score",
              "gpu_before": device_state(), "samples": []}
    save(output, result)
    try:
        llm = LLM(model=args.model, tensor_parallel_size=2, dtype="bfloat16", seed=1234,
                  trust_remote_code=True, max_model_len=40960, gpu_memory_utilization=.55,
                  kv_cache_memory_bytes=args.kv_gib * 1024**3, kv_cache_dtype="bfloat16",
                  enforce_eager=True, max_num_seqs=256, max_num_batched_tokens=8192,
                  enable_prefix_caching=True, linear_backend="emulation", moe_backend="emulation",
                  disable_log_stats=False,
                  worker_extension_cls=__package__ + ".memory_probe.MemoryMetrics")
        warm = SamplingParams(max_tokens=16, min_tokens=16, ignore_eos=True, temperature=.6,
                              top_p=.95, top_k=20, min_p=0, seed=1234)
        llm.generate(prompts[:1], warm, use_tqdm=False)
        result["warmup_worker_memory"] = llm.collective_rpc("gpqa_memory_stats")
        save(output, result)
        params = SamplingParams(max_tokens=256, min_tokens=256, ignore_eos=True, temperature=.6,
                                top_p=.95, top_k=20, min_p=0, seed=1234)
        for repeat in range(2):
            if not llm.reset_prefix_cache():
                raise RuntimeError("Unable to clear prefix cache between measurements")
            start = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=True)
            seconds = time.perf_counter() - start
            lengths = [len(item.outputs[0].token_ids) for item in outputs]
            if len(lengths) != 256 or set(lengths) != {256}:
                raise ValueError("Incomplete fixed-work performance trial")
            sample = {"repeat": repeat, "seconds": seconds, "output_tokens": sum(lengths),
                      "output_tokens_per_second": sum(lengths) / seconds,
                      "metrics": [asdict(m) for m in llm.get_metrics()],
                      "worker_memory": llm.collective_rpc("gpqa_memory_stats"), "gpu_state": device_state()}
            result["samples"].append(sample)
            save(output, result)
            print(f"KV {args.kv_gib} GiB repeat {repeat}: {seconds:.3f}s", flush=True)
        result.update(status="passed", median_seconds=statistics.median(s["seconds"] for s in result["samples"]))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--kv-gib", required=True, type=int, choices=(32, 56))
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
