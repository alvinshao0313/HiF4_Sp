#!/usr/bin/env python3
"""Paired lm_eval driver for the revalidation experiment.

Runs the SAME evaluator for every variant so BF16 identity/permuted and
W4A4 identity/permuted are strictly comparable. Tasks: arc_easy,
arc_challenge, mmlu (0-shot accuracy) + wikitext (word perplexity).

``--mode bf16``: no activation quant (weights are the given BF16 ckpt).
``--mode w4a4``: apply HiF4 activation fake quant via QLinear2 (weights must
already be an RTN HiF4 ckpt).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# --- must run before importing lm_eval (same patch as eval_lm_eval_hif4a) ---
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BLOCK_SPARSE_ROOT = REPO_ROOT / "Block_Sparse"
HIFLOAT4_ROOT = REPO_ROOT / "HiFloat4"
for p in (str(BLOCK_SPARSE_ROOT), str(HIFLOAT4_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402

from block_pruning.config import GradientBlockPruningConfig  # noqa: E402
from block_pruning.model_loader import load_model_and_tokenizer  # noqa: E402


def _load_hif4_main():
    spec = importlib.util.spec_from_file_location(
        "hif4_rtn_main", HIFLOAT4_ROOT / "main.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {HIFLOAT4_ROOT / 'main.py'}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pick_metric(task_result: dict) -> tuple[str, float | None]:
    for key in (
        "acc,none",
        "acc",
        "acc_norm,none",
        "acc_norm",
        "exact_match,none",
        "exact_match",
        "word_perplexity,none",
        "word_perplexity",
    ):
        if key in task_result and isinstance(task_result[key], (int, float)):
            return key, float(task_result[key])
    return "", None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["bf16", "w4a4"], required=True)
    p.add_argument(
        "--tasks", type=str, default="arc_easy,arc_challenge,mmlu,wikitext"
    )
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--fake_act_quant_exclude", type=str, default="lm_head")
    p.add_argument("--output_json", type=str, required=True)
    args = p.parse_args()

    model_path = args.model_path
    if not Path(model_path).is_absolute() and Path(model_path).exists():
        model_path = str(Path(model_path).resolve())
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(
        f"[eval_paired] model={model_path} mode={args.mode} tasks={tasks}",
        flush=True,
    )

    cfg = GradientBlockPruningConfig(
        model_path=model_path,
        output_dir=str(SCRIPT_DIR / "tmp" / "lm_eval_scratch"),
        score_type="magnitude",
        dtype=args.dtype,
        device="cuda",
        gradient_checkpointing=False,
        trust_remote_code=True,
    )
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(cfg)
    print(f"[eval_paired] model loaded in {time.time() - t0:.1f}s", flush=True)

    if args.mode == "w4a4":
        hif4_main = _load_hif4_main()
        excludes = [
            x.strip() for x in args.fake_act_quant_exclude.split(",") if x.strip()
        ]
        ns = SimpleNamespace(exclude_layers=excludes, disable_fast_forward=False)
        model = hif4_main.replace_linear_with_hif4_activation_quant(model, ns)
        model.eval()
        print("[eval_paired] applied HiF4 activation quant", flush=True)

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    t1 = time.time()
    raw = simple_evaluate(
        model=hflm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        log_samples=False,
    )
    elapsed = time.time() - t1
    results = raw.get("results", {}) if isinstance(raw, dict) else {}
    scores: dict[str, float] = {}
    metric_keys: dict[str, str] = {}
    for task, trez in results.items():
        if not isinstance(trez, dict):
            continue
        key, val = _pick_metric(trez)
        if val is None:
            continue
        scores[task] = val
        metric_keys[task] = key
    out = {
        "model_path": model_path,
        "mode": args.mode,
        "tasks": tasks,
        "num_fewshot": args.num_fewshot,
        "scores": scores,
        "metric_keys": metric_keys,
        "eval_elapsed_sec": elapsed,
        "unix_time": time.time(),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
