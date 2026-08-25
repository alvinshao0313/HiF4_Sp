#!/usr/bin/env python3
"""Task 13 I helper: short vLLM TP2 smoke for HiF4 materialized checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VLLM_ROOT = REPO_ROOT / "3rdparty" / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))

PROMPTS = [
    "Hello",
    "1+1=",
    "The capital of France is",
    "Write one word: ok",
]


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--hif4-runtime-spec", type=Path, required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--kv-cache-dtype", type=str, default="bfloat16")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    trace_path = args.out_json.with_suffix(".trace.jsonl")
    if trace_path.exists():
        trace_path.unlink()
    os.environ["HIF4_RUNTIME_TRACE_JSONL"] = str(trace_path)
    os.environ["HIF4_RUNTIME_SPEC_PATH"] = str(args.hif4_runtime_spec.resolve())

    result: dict[str, Any] = {
        "tag": args.tag,
        "checkpoint": str(args.checkpoint),
        "hif4_runtime_spec": str(args.hif4_runtime_spec),
        "tp": int(args.tp),
        "kv_cache_dtype_arg": args.kv_cache_dtype,
        "enforce_eager": bool(args.enforce_eager),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "max_model_len": int(args.max_model_len),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "trace_jsonl": str(trace_path),
        "passed": False,
        "error": None,
        "oom": False,
    }
    llm = None
    try:
        if not args.checkpoint.is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {args.checkpoint}")
        if not args.hif4_runtime_spec.is_file():
            raise FileNotFoundError(f"HiF4 runtime sidecar not found: {args.hif4_runtime_spec}")

        from vllm import LLM, SamplingParams
        import torch

        llm = LLM(
            model=str(args.checkpoint),
            trust_remote_code=True,
            tensor_parallel_size=int(args.tp),
            dtype="auto",
            enforce_eager=bool(args.enforce_eager),
            max_model_len=int(args.max_model_len),
            max_num_batched_tokens=int(args.max_num_batched_tokens),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            kv_cache_dtype=str(args.kv_cache_dtype),
            enable_prefix_caching=False,
            async_scheduling=False,
            additional_config={
                "hif4_runtime_spec_path": str(args.hif4_runtime_spec.resolve()),
            },
        )
        result["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        result["gpu_count_visible"] = torch.cuda.device_count()
        vcfg = llm.llm_engine.vllm_config
        result["resolved_kv_cache_dtype"] = str(
            getattr(vcfg.cache_config, "cache_dtype", None)
        )
        result["resolved_async_scheduling"] = bool(
            getattr(vcfg.scheduler_config, "async_scheduling", True)
        )
        additional = getattr(vcfg, "additional_config", {}) or {}
        result["resolved_hif4_runtime_spec_path"] = additional.get(
            "hif4_runtime_spec_path"
        )

        sp = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(args.max_new_tokens),
        )
        outputs = llm.generate(PROMPTS, sp)
        result["generations"] = [
            {"prompt": out.prompt, "text": out.outputs[0].text if out.outputs else ""}
            for out in outputs
        ]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            result["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
            result["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))

        trace_rows = _read_trace(trace_path)
        trace_events = sorted({str(row.get("event")) for row in trace_rows})
        result["trace_event_counts"] = {
            event: sum(1 for row in trace_rows if row.get("event") == event)
            for event in trace_events
        }
        result["trace_samples"] = trace_rows[:12]
        result["sidecar_effective"] = (
            result["resolved_hif4_runtime_spec_path"]
            == str(args.hif4_runtime_spec.resolve())
            and "sidecar_load" in trace_events
            and "dense_apply" in trace_events
            and "moe_apply" in trace_events
        )
        result["passed"] = bool(
            len(result["generations"]) == len(PROMPTS)
            and result["resolved_kv_cache_dtype"] == "bfloat16"
            and result["resolved_async_scheduling"] is False
            and result["sidecar_effective"]
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "tag",
                        "passed",
                        "resolved_kv_cache_dtype",
                        "resolved_async_scheduling",
                        "sidecar_effective",
                        "trace_event_counts",
                        "gpu_name",
                        "cuda_peak_allocated_bytes",
                        "cuda_peak_reserved_bytes",
                    )
                    if k in result
                },
                indent=2,
            )
        )
    except BaseException as exc:  # noqa: BLE001 - smoke must serialize failures
        result["error"] = "".join(traceback.format_exception(exc))
        result["oom"] = "out of memory" in str(exc).lower()
        print(result["error"], file=sys.stderr)
    finally:
        result["trace_event_counts"] = result.get("trace_event_counts", {})
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if llm is not None:
            del llm
        gc.collect()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
