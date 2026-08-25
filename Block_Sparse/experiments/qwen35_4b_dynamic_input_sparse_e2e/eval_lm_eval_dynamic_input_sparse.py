#!/usr/bin/env python3
"""lm_eval ARC on HF Qwen3.5 with DynamicInputSparseMLPReference installed."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[2]
BLOCK_SPARSE_ROOT = REPO_ROOT / "Block_Sparse"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BLOCK_SPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(BLOCK_SPARSE_ROOT))

from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402

from block_pruning.config import GradientBlockPruningConfig  # noqa: E402
from block_pruning.model_loader import load_model_and_tokenizer  # noqa: E402
from Block_Sparse.dynamic_input_sparse.config import (  # noqa: E402
    DynamicInputMaskMethod,
    DynamicInputSparseConfig,
)
from Block_Sparse.dynamic_input_sparse.hf_reference import (  # noqa: E402
    install_dynamic_input_sparse_on_hf_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--tasks", type=str, default="arc_easy,arc_challenge")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--batch_size", type=str, default="8")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--dynamic_input_sparse_method",
        choices=["none", "m1_oracle", "m8_energy"],
        default="none",
    )
    p.add_argument("--dynamic_input_keep_ratio", type=float, default=1.0)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--limit", type=float, default=None)
    return p.parse_args()


def _pick_metric(task_result: dict) -> tuple[str, float | None]:
    for key in ("acc,none", "acc", "acc_norm,none", "acc_norm"):
        if key in task_result and isinstance(task_result[key], (int, float)):
            return key, float(task_result[key])
    return "", None


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    method = DynamicInputMaskMethod(args.dynamic_input_sparse_method)
    print(
        f"[lm_eval_dyn] model={model_path} tasks={tasks} method={method.value} "
        f"keep={args.dynamic_input_keep_ratio} fewshot={args.num_fewshot} "
        f"batch_size={args.batch_size}",
        flush=True,
    )

    cfg = GradientBlockPruningConfig(
        model_path=model_path,
        output_dir=str(EXP_DIR / "tmp" / "lm_eval_scratch"),
        score_type="magnitude",
        dtype=args.dtype,
        device=args.device,
        gradient_checkpointing=False,
        trust_remote_code=True,
    )
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(cfg)
    print(f"[lm_eval_dyn] model loaded in {time.time() - t0:.1f}s", flush=True)

    replaced = []
    if method != DynamicInputMaskMethod.NONE:
        dyn_cfg = DynamicInputSparseConfig(
            method=method, keep_ratio=float(args.dynamic_input_keep_ratio)
        )
        replaced = install_dynamic_input_sparse_on_hf_model(model, dyn_cfg)
        print(f"[lm_eval_dyn] wrapped {len(replaced)} MLP modules", flush=True)

    try:
        bs: int | str = int(args.batch_size)
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

    scores: dict[str, float | None] = {}
    metric_keys: dict[str, str] = {}
    secondary: dict[str, float | None] = {}
    for task, trez in results.items():
        if not isinstance(trez, dict):
            continue
        key, val = _pick_metric(trez)
        if val is not None:
            scores[task] = val
            metric_keys[task] = key
        if "acc_norm,none" in trez and isinstance(trez["acc_norm,none"], (int, float)):
            secondary[f"{task}_acc_norm"] = float(trez["acc_norm,none"])

    payload = {
        "model_path": model_path,
        "tasks": tasks,
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "dynamic_input_sparse_method": method.value,
        "dynamic_input_keep_ratio": float(args.dynamic_input_keep_ratio),
        "wrapped_mlps": len(replaced),
        "seconds": elapsed,
        "scores": scores,
        "metric_keys": metric_keys,
        "secondary": secondary,
        "arc_easy": scores.get("arc_easy"),
        "arc_challenge": scores.get("arc_challenge"),
        "results_raw": {
            k: {
                mk: mv
                for mk, mv in v.items()
                if isinstance(mv, (int, float, str, bool))
            }
            for k, v in results.items()
            if isinstance(v, dict)
        },
    }
    print(
        f"[lm_eval_dyn] DONE arc_easy={payload['arc_easy']} "
        f"arc_challenge={payload['arc_challenge']} time={elapsed:.1f}s",
        flush=True,
    )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[lm_eval_dyn] wrote {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
