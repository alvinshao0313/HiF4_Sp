#!/usr/bin/env python3
"""Task 11 helper: short vLLM LLM smoke (mirrors main.py backend kwargs).

Must be a real .py file (not stdin) so vLLM multiprocessing spawn can re-exec.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VLLM_ROOT = REPO_ROOT / "3rdparty" / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))

DEFAULT_CKPT = (
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4"
    / "snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3"
)

PROMPTS = [
    "Hello",
    "1+1=",
    "The capital of France is",
    "Write one word: ok",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--kv-cache-dtype", type=str, required=True)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    result = {
        "tag": args.tag,
        "checkpoint": str(args.checkpoint),
        "tp": args.tp,
        "kv_cache_dtype_arg": args.kv_cache_dtype,
        "enforce_eager": bool(args.enforce_eager),
        "linear_backend": "emulation",
        "moe_backend": "emulation",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "passed": False,
        "error": None,
        "oom": False,
    }
    llm = None
    try:
        # Defer torch.cuda queries until after import; do not touch CUDA before LLM
        # if possible. Device name is read after successful init.
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(args.checkpoint),
            trust_remote_code=True,
            tensor_parallel_size=args.tp,
            dtype="auto",
            enforce_eager=bool(args.enforce_eager),
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            linear_backend="emulation",
            moe_backend="emulation",
            kv_cache_dtype=args.kv_cache_dtype,
            enable_prefix_caching=False,
        )
        import torch

        result["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        result["gpu_count_visible"] = torch.cuda.device_count()

        vcfg = llm.llm_engine.vllm_config
        result["resolved_linear_backend"] = getattr(
            vcfg.kernel_config, "linear_backend", None
        )
        result["resolved_moe_backend"] = getattr(
            vcfg.kernel_config, "moe_backend", None
        )
        result["resolved_kv_cache_dtype"] = str(
            getattr(vcfg.cache_config, "cache_dtype", None)
        )

        marlin_hits = []
        samples = []
        try:
            model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        except Exception:
            model = None
        if model is not None:
            for name, mod in model.named_modules():
                qmethod = getattr(mod, "quant_method", None)
                if qmethod is None:
                    continue
                cls = type(qmethod).__name__
                if not any(x in cls for x in ("NvFp4", "NVFP4", "Fp4", "FP4")):
                    continue
                kernel = getattr(qmethod, "kernel", None)
                experts_cls = getattr(qmethod, "experts_cls", None)
                backend = getattr(qmethod, "nvfp4_backend", None)
                kname = type(kernel).__name__ if kernel is not None else None
                ename = experts_cls.__name__ if experts_cls is not None else None
                bname = str(backend) if backend is not None else None
                blob = f"{kname}|{ename}|{bname}|{cls}"
                if "Marlin" in blob or "MARLIN" in blob:
                    marlin_hits.append({"module": name, "detail": blob})
                if len(samples) < 8:
                    samples.append(
                        {
                            "module": name,
                            "quant_method": cls,
                            "kernel": kname,
                            "experts_cls": ename,
                            "nvfp4_backend": bname,
                        }
                    )
        result["nvfp4_layer_samples"] = samples
        result["marlin_fallback_hits"] = marlin_hits

        sp = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens
        )
        outputs = llm.generate(PROMPTS, sp)
        texts = []
        for o in outputs:
            t = o.outputs[0].text if o.outputs else ""
            texts.append({"prompt": o.prompt, "text": t})
        result["generations"] = texts
        result["backend_ok"] = (
            result["resolved_linear_backend"] == "emulation"
            and result["resolved_moe_backend"] == "emulation"
        )
        result["no_marlin"] = len(marlin_hits) == 0
        if args.kv_cache_dtype == "auto":
            cd = result["resolved_kv_cache_dtype"].lower()
            result["fp8_kv_resolved"] = "fp8" in cd
        else:
            result["fp8_kv_resolved"] = None

        result["passed"] = bool(
            result["backend_ok"]
            and result["no_marlin"]
            and (result["fp8_kv_resolved"] is not False)
            and len(texts) == len(PROMPTS)
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "tag",
                        "passed",
                        "resolved_linear_backend",
                        "resolved_moe_backend",
                        "resolved_kv_cache_dtype",
                        "fp8_kv_resolved",
                        "no_marlin",
                        "gpu_name",
                    )
                },
                indent=2,
            )
        )
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        result["error"] = msg
        result["traceback"] = traceback.format_exc()
        result["oom"] = ("out of memory" in msg.lower()) or ("OutOfMemory" in msg)
        print(result["traceback"], file=sys.stderr)
        print(
            json.dumps(
                {
                    "tag": args.tag,
                    "passed": False,
                    "error": msg,
                    "oom": result["oom"],
                },
                indent=2,
            )
        )
    finally:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        try:
            del llm
        except Exception:
            pass
        try:
            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
