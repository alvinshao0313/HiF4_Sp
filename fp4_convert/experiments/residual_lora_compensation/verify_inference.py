"""Check exported residual adapters on real vLLM TP2 prefill and decode."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .artifact import load, sha256, tensor_digest, write_json


class SetAdapters:
    # vLLM's msgpack transport converts dataclasses to dicts; keep this callable
    # as an ordinary class so the worker receives executable code via pickle.
    def __init__(self, phase, spec_path):
        self.phase = phase
        self.spec_path = spec_path

    def __call__(self, model):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        payload = load(self.spec_path)["residual_lora"]
        layers = model.model.layers
        if not hasattr(model, "_residual_check_adapters"):
            saved = []
            for index, layer in enumerate(layers):
                for branch in ("attention", "moe"):
                    name = f"residual_{branch}_lora"
                    adapter = getattr(layer, name)
                    active = payload["mode"] in (branch, "both")
                    if (adapter is not None) != active:
                        raise RuntimeError("runtime adapter mode differs from export")
                    if not active:
                        continue
                    reference = payload["layers"][str(index)]
                    for part, value in (("A", adapter.a), ("B", adapter.b)):
                        if value.dtype != torch.float32 or not torch.equal(value.cpu(), reference[f"{branch}_lora_{part}"]):
                            raise RuntimeError(f"adapter load mismatch: layer {index} {branch} {part}")
                    if adapter.scale != payload["alpha"] / payload["rank"]:
                        raise RuntimeError("runtime adapter scale mismatch")
                    saved.append((layer, name, adapter, adapter.b.detach().clone()))
            model._residual_check_adapters = saved
        for layer, name, adapter, original_b in model._residual_check_adapters:
            setattr(layer, name, None if self.phase.startswith("disabled") else adapter)
            with torch.inference_mode():
                adapter.b.copy_(torch.zeros_like(adapter.b) if self.phase == "zero" else original_b)
            adapter._check_calls = 0
            adapter._check_nonzero = False
            if not hasattr(adapter, "_residual_check_hook"):
                def observe(module, args, output):
                    if not output.isfinite().all():
                        raise RuntimeError("nonfinite residual adapter output")
                    module._check_calls += 1
                    module._check_nonzero |= bool(output.count_nonzero())
                adapter._residual_check_hook = adapter.register_forward_hook(observe)
        return {"rank": get_tensor_model_parallel_rank(), "adapters": len(model._residual_check_adapters)}


class AdapterStats:
    def __call__(self, model):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        return {"rank": get_tensor_model_parallel_rank(),
                "calls": [a._check_calls for _, _, a, _ in model._residual_check_adapters],
                "nonzero": [a._check_nonzero for _, _, a, _ in model._residual_check_adapters],
                "all_cuda_fp32": all(a.a.is_cuda and a.b.is_cuda and a.a.dtype == a.b.dtype == torch.float32
                                     for _, _, a, _ in model._residual_check_adapters),
                "allocated_bytes": torch.cuda.memory_allocated()}


def output_signature(outputs):
    result = []
    for output in outputs:
        completion = output.outputs[0]
        result.append({"tokens": list(completion.token_ids),
                       "logprobs": [{str(k): float(v.logprob) for k, v in p.items()}
                                    for p in completion.logprobs]})
    return result


def verify(model_dir, output_dir):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model, out = Path(model_dir).resolve(), Path(output_dir).resolve()
    spec_path = model / "hif4_runtime_spec.pt"
    spec = load(spec_path)
    payload = spec["residual_lora"]
    expected_count = 48 * (2 if payload["mode"] == "both" else 1)
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["HIF4_RUNTIME_SPEC_PATH"] = str(spec_path)
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=True)
               for text in ("Calculate 17 * 23.", "Write a Python function that returns the sum of a list of integers.")]
    kwargs = dict(model=str(model), tensor_parallel_size=2, dtype="bfloat16", trust_remote_code=True,
                  gpu_memory_utilization=0.9, max_model_len=40960, max_num_seqs=128,
                  max_num_batched_tokens=4096, kv_cache_dtype="bfloat16", enforce_eager=True,
                  enable_prefix_caching=False, moe_backend="triton", seed=42,
                  additional_config={"hif4_runtime_spec_path": str(spec_path)})
    llm = LLM(**kwargs)
    core = llm.llm_engine.engine_core
    thread = core.output_queue_thread
    outputs, stats = {}, {}
    try:
        for phase in ("disabled", "disabled_repeat", "zero", "trained", "trained_repeat"):
            ranks = llm.apply_model(SetAdapters(phase, str(spec_path)))
            if sorted(r["rank"] for r in ranks) != [0, 1] or any(r["adapters"] != expected_count for r in ranks):
                raise RuntimeError("incomplete TP2 adapter coverage")
            # Isolate each identity probe from asynchronous request batching.
            # Engine capacity and TP2 remain the formal configuration; task
            # smoke separately exercises the actual benchmark batching.
            generated = [llm.generate([prompt], SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=5), use_tqdm=False)[0]
                         for prompt in prompts]
            outputs[phase] = output_signature(generated)
            stats[phase] = llm.apply_model(AdapterStats())
            for rank in stats[phase]:
                if phase.startswith("disabled"):
                    assert not any(rank["calls"])
                else:
                    assert rank["all_cuda_fp32"] and min(rank["calls"]) >= 8
                    assert all(rank["nonzero"]) if phase.startswith("trained") else not any(rank["nonzero"])
            print(f"actual TP2 {payload['mode']}: {phase} complete", flush=True)
        write_json({"outputs": outputs, "worker_stats": stats}, out / "diagnostics.json")
        if outputs["zero"] != outputs["disabled"]:
            raise RuntimeError("zero-adapter full-model output differs from disabled adapter")
        if outputs["trained"] == outputs["zero"]:
            raise RuntimeError("trained adapters did not change model logprobs")
        if outputs["trained"] != outputs["trained_repeat"]:
            raise RuntimeError("repeated deterministic generation differs")
    finally:
        core.shutdown(timeout=60)
        thread.join(timeout=60)
        if thread.is_alive():
            raise RuntimeError("verification engine did not terminate")
    result = {"complete": True, "model_dir": str(model), "mode": payload["mode"],
              "runtime_spec_sha256": sha256(spec_path), "runtime": kwargs,
              "adapter_hashes": {i: tensor_digest(v) for i, v in payload["layers"].items()},
              "zero_disabled_identical": True, "nonzero_changes_logprobs": True,
              "trained_repeat_identical": True, "worker_stats": stats,
              "outputs": outputs, "scope": "real TP2 short prefill and 8 incremental decode steps; downstream task smoke is separate"}
    write_json(result, out / "verification.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    verify(**vars(parser.parse_args()))
