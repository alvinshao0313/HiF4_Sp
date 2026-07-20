#!/usr/bin/env python3
"""Run lm_eval (arc_easy / arc_challenge / mmlu) on a pruned HF checkpoint.

Works around transformers>=5 removing AutoModelForVision2Seq, which breaks
stock lm_eval imports. Uses Block_Sparse's Qwen3.5 CausalLM loader.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# --- must run before importing lm_eval ---
import transformers

_orig_tf_getattr = transformers.__class__.__getattr__


def _tf_getattr_patched(self, name):  # noqa: ANN001
    if name == "AutoModelForVision2Seq":
        return transformers.AutoModelForImageTextToText
    return _orig_tf_getattr(self, name)


transformers.__class__.__getattr__ = _tf_getattr_patched
setattr(
    transformers,
    "AutoModelForVision2Seq",
    transformers.AutoModelForImageTextToText,
)
# --- end patch ---

BLOCK_SPARSE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BLOCK_SPARSE_ROOT.parent
if str(BLOCK_SPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(BLOCK_SPARSE_ROOT))

from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402

from block_pruning.config import GradientBlockPruningConfig  # noqa: E402
from block_pruning.model_loader import load_model_and_tokenizer  # noqa: E402


DEFAULT_TASKS = ("arc_easy", "arc_challenge", "mmlu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated lm_eval task names / groups.",
    )
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--batch_size", type=str, default="8")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Where to write the compact metrics JSON.",
    )
    p.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional lm_eval limit (fraction or count) for smoke tests.",
    )
    return p.parse_args()


def _pick_metric(task_result: dict) -> tuple[str, float | None]:
    """Prefer acc_norm then acc (lm_eval key style: 'acc,none')."""
    for key in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
        if key in task_result and isinstance(task_result[key], (int, float)):
            return key, float(task_result[key])
    return "", None


def compact_metrics(results: dict) -> dict:
    out: dict[str, float | None] = {}
    metric_keys: dict[str, str] = {}
    for task, trez in results.items():
        if not isinstance(trez, dict):
            continue
        # Skip nested group dumps that are not leaf scores when present.
        key, val = _pick_metric(trez)
        if val is None:
            continue
        out[task] = val
        metric_keys[task] = key
    return {"scores": out, "metric_keys": metric_keys}


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    if not Path(model_path).is_absolute() and Path(model_path).exists():
        model_path = str(Path(model_path).resolve())

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(
        f"[lm_eval] model={model_path} tasks={tasks} "
        f"fewshot={args.num_fewshot} batch_size={args.batch_size}",
        flush=True,
    )

    cfg = GradientBlockPruningConfig(
        model_path=model_path,
        output_dir=str(REPO_ROOT / "Block_Sparse" / "results" / "lm_eval_tmp"),
        score_type="magnitude",
        dtype=args.dtype,
        device=args.device,
        gradient_checkpointing=False,
        trust_remote_code=True,
    )
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(cfg)
    print(f"[lm_eval] model loaded in {time.time() - t0:.1f}s", flush=True)

    # batch_size for HFLM: int or "auto"
    bs: int | str
    try:
        bs = int(args.batch_size)
    except ValueError:
        bs = args.batch_size

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bs)

    eval_kwargs = dict(
        model=hflm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=bs,
        log_samples=False,
    )
    if args.limit is not None:
        eval_kwargs["limit"] = args.limit

    t1 = time.time()
    raw = simple_evaluate(**eval_kwargs)
    elapsed = time.time() - t1
    results = raw["results"]
    compact = compact_metrics(results)

    payload = {
        "model_path": model_path,
        "tasks": tasks,
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "seconds": elapsed,
        "scores": compact["scores"],
        "metric_keys": compact["metric_keys"],
        "results_raw": {
            k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str, bool))}
            for k, v in results.items()
            if isinstance(v, dict)
        },
    }

    # Convenience aliases for the report.
    scores = compact["scores"]
    payload["arc_easy"] = scores.get("arc_easy")
    payload["arc_challenge"] = scores.get("arc_challenge")
    payload["mmlu"] = scores.get("mmlu")

    print(
        f"[lm_eval] DONE "
        f"arc_easy={payload['arc_easy']} "
        f"arc_challenge={payload['arc_challenge']} "
        f"mmlu={payload['mmlu']} "
        f"time={elapsed:.1f}s",
        flush=True,
    )

    if args.output_json:
        out = Path(args.output_json)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[lm_eval] wrote {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
