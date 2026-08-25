#!/usr/bin/env python3
"""Diagnose TP1/TP2 parity for Qwen3-30B-A3B ModelOpt NVFP4 emulation.

This is a diagnostic script, not the Task 13 generation smoke.  It compares
identical-input next-token top-N scores and stops analysis at the first greedy
token divergence for each prompt.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
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


def _score_rows(pos: dict[int, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for token_id, obj in pos.items():
        rows.append(
            {
                "token_id": int(token_id),
                "score": float(obj.logprob),
                "decoded_token": getattr(obj, "decoded_token", None),
                "rank": getattr(obj, "rank", None),
            }
        )
    rows.sort(key=lambda row: (-float(row["score"]), int(row["token_id"])))
    return rows


def _run_single(args: argparse.Namespace) -> int:
    result: dict[str, Any] = {
        "tp": int(args.single_tp),
        "checkpoint": str(args.checkpoint),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "kv_cache_dtype": "bfloat16",
        "linear_backend": "emulation",
        "moe_backend": "emulation",
        "enforce_eager": True,
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "max_model_len": int(args.max_model_len),
        "max_new_tokens": int(args.max_new_tokens),
        "top_logprobs": int(args.top_logprobs),
        "logprobs_mode": str(args.logprobs_mode),
        "passed": False,
        "error": None,
    }
    llm = None
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(args.checkpoint),
            trust_remote_code=True,
            tensor_parallel_size=int(args.single_tp),
            dtype="auto",
            enforce_eager=True,
            max_model_len=int(args.max_model_len),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            linear_backend="emulation",
            moe_backend="emulation",
            kv_cache_dtype="bfloat16",
            enable_prefix_caching=False,
            async_scheduling=False,
            max_logprobs=int(args.top_logprobs),
            logprobs_mode=str(args.logprobs_mode),
        )
        import torch

        result["gpu_count_visible"] = torch.cuda.device_count()
        result["gpu_name"] = torch.cuda.get_device_name(0)
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
        result["resolved_async_scheduling"] = bool(
            getattr(vcfg.scheduler_config, "async_scheduling", True)
        )

        sp = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(args.max_new_tokens),
            logprobs=int(args.top_logprobs),
        )
        outputs = llm.generate(PROMPTS, sp)
        generations: list[dict[str, Any]] = []
        for out in outputs:
            completion = out.outputs[0]
            scores = [_score_rows(pos) for pos in (completion.logprobs or [])]
            generations.append(
                {
                    "prompt": out.prompt,
                    "text": completion.text,
                    "token_ids": list(completion.token_ids),
                    "scores": scores,
                }
            )
        result["generations"] = generations
        result["passed"] = (
            result["resolved_linear_backend"] == "emulation"
            and result["resolved_moe_backend"] == "emulation"
            and result["resolved_kv_cache_dtype"] == "bfloat16"
            and result["resolved_async_scheduling"] is False
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
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
    print(
        json.dumps(
            {
                "tp": result["tp"],
                "passed": result["passed"],
                "cuda_visible_devices": result["cuda_visible_devices"],
                "resolved_async_scheduling": result.get("resolved_async_scheduling"),
                "out_json": str(args.out_json),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


def _by_token(rows: list[dict[str, Any]]) -> dict[int, float]:
    return {int(row["token_id"]): float(row["score"]) for row in rows}


def _top_tokens(rows: list[dict[str, Any]], k: int) -> list[int]:
    return [int(row["token_id"]) for row in rows[:k]]


def _score_metrics(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> dict[str, Any]:
    a = _by_token(rows_a)
    b = _by_token(rows_b)
    shared = sorted(set(a) & set(b))
    diffs = [a[tok] - b[tok] for tok in shared]
    abs_diffs = [abs(x) for x in diffs]
    norm_a = math.sqrt(sum(a[tok] * a[tok] for tok in shared))
    norm_b = math.sqrt(sum(b[tok] * b[tok] for tok in shared))
    diff_norm = math.sqrt(sum(x * x for x in diffs))
    dot = sum(a[tok] * b[tok] for tok in shared)
    cosine = dot / max(norm_a * norm_b, 1e-30) if shared else None
    top1 = rows_a[0] if rows_a else None
    top2 = rows_a[1] if len(rows_a) > 1 else None
    top1_b = rows_b[0] if rows_b else None
    top2_b = rows_b[1] if len(rows_b) > 1 else None
    return {
        "shared_count": len(shared),
        "tp1_only_count": len(set(a) - set(b)),
        "tp2_only_count": len(set(b) - set(a)),
        "max_abs_shared": max(abs_diffs) if abs_diffs else None,
        "mean_abs_shared": (sum(abs_diffs) / len(abs_diffs)) if abs_diffs else None,
        "rel_l2_shared": diff_norm / max(norm_a, 1e-30) if shared else None,
        "cosine_shared": cosine,
        "tp1_top1_token_id": None if top1 is None else int(top1["token_id"]),
        "tp1_top2_token_id": None if top2 is None else int(top2["token_id"]),
        "tp1_top1_score": None if top1 is None else float(top1["score"]),
        "tp1_top2_score": None if top2 is None else float(top2["score"]),
        "tp1_top1_top2_margin": None
        if top1 is None or top2 is None
        else float(top1["score"]) - float(top2["score"]),
        "tp2_top1_token_id": None if top1_b is None else int(top1_b["token_id"]),
        "tp2_top2_token_id": None if top2_b is None else int(top2_b["token_id"]),
        "tp2_top1_score": None if top1_b is None else float(top1_b["score"]),
        "tp2_top2_score": None if top2_b is None else float(top2_b["score"]),
        "tp2_top1_top2_margin": None
        if top1_b is None or top2_b is None
        else float(top1_b["score"]) - float(top2_b["score"]),
        "top1_equal": (
            top1 is not None
            and top1_b is not None
            and int(top1["token_id"]) == int(top1_b["token_id"])
        ),
        "top5_overlap": len(set(_top_tokens(rows_a, 5)) & set(_top_tokens(rows_b, 5))),
        "top10_overlap": len(set(_top_tokens(rows_a, 10)) & set(_top_tokens(rows_b, 10))),
    }


def _first_diff(ids_a: list[int], ids_b: list[int]) -> int | None:
    for idx, (a, b) in enumerate(zip(ids_a, ids_b)):
        if int(a) != int(b):
            return idx
    if len(ids_a) != len(ids_b):
        return min(len(ids_a), len(ids_b))
    return None


def _compare(tp1: dict[str, Any], tp2: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for gen1, gen2 in zip(tp1["generations"], tp2["generations"]):
        ids1 = [int(x) for x in gen1["token_ids"]]
        ids2 = [int(x) for x in gen2["token_ids"]]
        diff = _first_diff(ids1, ids2)
        prefill = _score_metrics(gen1["scores"][0], gen2["scores"][0])
        divergence = None
        if diff is not None and diff < len(gen1["scores"]) and diff < len(gen2["scores"]):
            divergence = _score_metrics(gen1["scores"][diff], gen2["scores"][diff])
        rows.append(
            {
                "prompt": gen1["prompt"],
                "tp1_text": gen1["text"],
                "tp2_text": gen2["text"],
                "tp1_token_ids": ids1,
                "tp2_token_ids": ids2,
                "first_divergence_step": diff,
                "prefill_next_token": prefill,
                "first_divergence": divergence,
            }
        )
    all_prefill_top1 = all(r["prefill_next_token"]["top1_equal"] for r in rows)
    any_divergence = any(r["first_divergence_step"] is not None for r in rows)
    return {
        "diagnostic_kind": "topN raw_logits/logprobs parity, not full-vocab logits",
        "score_mode": tp1["logprobs_mode"],
        "top_logprobs": tp1["top_logprobs"],
        "tp1_passed": tp1["passed"],
        "tp2_passed": tp2["passed"],
        "all_prefill_top1_equal": all_prefill_top1,
        "any_greedy_divergence": any_divergence,
        "prompts": rows,
    }


def _run_driver(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tp1_json = out_dir / "tp1.json"
    tp2_json = out_dir / "tp2.json"
    script = Path(__file__).resolve()
    common = [
        sys.executable,
        str(script),
        "--single-tp",
        "{tp}",
        "--checkpoint",
        str(args.checkpoint),
        "--max-model-len",
        str(args.max_model_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--top-logprobs",
        str(args.top_logprobs),
        "--logprobs-mode",
        str(args.logprobs_mode),
        "--out-json",
        "{out}",
    ]
    for tp, gpus, out in ((1, args.tp1_gpus, tp1_json), (2, args.tp2_gpus, tp2_json)):
        cmd = [str(tp) if x == "{tp}" else str(out) if x == "{out}" else x for x in common]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus
        print("running", " ".join(cmd), "CUDA_VISIBLE_DEVICES=" + gpus, flush=True)
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    tp1 = json.loads(tp1_json.read_text(encoding="utf-8"))
    tp2 = json.loads(tp2_json.read_text(encoding="utf-8"))
    summary = _compare(tp1, tp2)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["tp1_passed"] and summary["tp2_passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--tp1-gpus", type=str, default="0")
    parser.add_argument("--tp2-gpus", type=str, default="0,7")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--top-logprobs", type=int, default=100)
    parser.add_argument("--logprobs-mode", type=str, default="raw_logits")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/hif4_task13_tp_parity"))
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("/tmp/hif4_task13_tp_parity/summary.json"),
    )
    parser.add_argument("--single-tp", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=Path(""))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.single_tp:
        if not str(args.out_json):
            raise ValueError("--out-json is required with --single-tp")
        return _run_single(args)
    return _run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
