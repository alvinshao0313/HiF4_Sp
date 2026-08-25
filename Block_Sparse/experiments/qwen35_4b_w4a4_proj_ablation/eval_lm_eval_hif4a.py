#!/usr/bin/env python3
"""lm_eval (arc_easy/arc_challenge/mmlu) on a HiF4-RTN HF ckpt with optional HiF4 activation fake quant.

Reuses Block_Sparse/tools/eval_lm_eval loading path and HiFloat4 QLinear2 activation path.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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

EXP_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    """Locate HiF4_Sp even when this file is reached via HiF4_exp symlink."""
    for p in [start, *start.parents]:
        if (p / "Block_Sparse" / "block_pruning").is_dir() and (p / "HiFloat4").is_dir():
            return p
    # Inside Block_Sparse/experiments/... → parent-of-Block_Sparse.
    for p in start.parents:
        if p.name == "Block_Sparse" and (p.parent / "HiFloat4").is_dir():
            return p.parent
    raise RuntimeError(f"Cannot locate repo root from {start}")


REPO_ROOT = _find_repo_root(EXP_DIR)
BLOCK_SPARSE_ROOT = REPO_ROOT / "Block_Sparse"
HIFLOAT4_ROOT = REPO_ROOT / "HiFloat4"

if str(BLOCK_SPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(BLOCK_SPARSE_ROOT))

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
    # Ensure HiFloat4 package imports resolve.
    if str(HIFLOAT4_ROOT) not in sys.path:
        sys.path.insert(0, str(HIFLOAT4_ROOT))
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument(
        "--tasks",
        type=str,
        default="arc_easy,arc_challenge,mmlu",
    )
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--batch_size", type=str, default="8")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument(
        "--fake_act_quant",
        type=str,
        default="hif4",
        choices=["none", "hif4"],
        help="Apply HiF4 activation fake quant via QLinear2 after load.",
    )
    p.add_argument(
        "--fake_act_quant_exclude",
        type=str,
        default="lm_head",
        help="Comma-separated module suffixes to skip for activation quant.",
    )
    return p.parse_args()


def _pick_metric(task_result: dict) -> tuple[str, float | None]:
    for key in (
        "acc,none",
        "acc",
        "acc_norm,none",
        "acc_norm",
        "exact_match,custom-extract",
        "exact_match,none",
        "exact_match",
    ):
        if key in task_result and isinstance(task_result[key], (int, float)):
            return key, float(task_result[key])
    for key, val in task_result.items():
        if key.startswith("exact_match") and isinstance(val, (int, float)):
            return key, float(val)
    return "", None


def compact_metrics(results: dict) -> dict:
    out: dict[str, float | None] = {}
    metric_keys: dict[str, str] = {}
    for task, trez in results.items():
        if not isinstance(trez, dict):
            continue
        key, val = _pick_metric(trez)
        if val is None:
            continue
        out[task] = val
        metric_keys[task] = key
    return {"scores": out, "metric_keys": metric_keys}


def apply_hif4_activation(model, exclude_csv: str):
    hif4_main = _load_hif4_main()
    excludes = [x.strip() for x in exclude_csv.split(",") if x.strip()]
    if not excludes:
        excludes = ["lm_head"]
    args = SimpleNamespace(exclude_layers=excludes, disable_fast_forward=False)
    return hif4_main.replace_linear_with_hif4_activation_quant(model, args)


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    if not Path(model_path).is_absolute() and Path(model_path).exists():
        model_path = str(Path(model_path).resolve())

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(
        f"[lm_eval_hif4a] model={model_path} tasks={tasks} "
        f"fewshot={args.num_fewshot} act={args.fake_act_quant} "
        f"exclude={args.fake_act_quant_exclude}",
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
    print(f"[lm_eval_hif4a] model loaded in {time.time() - t0:.1f}s", flush=True)

    if args.fake_act_quant == "hif4":
        t_a = time.time()
        model = apply_hif4_activation(model, args.fake_act_quant_exclude)
        model.eval()
        print(
            f"[lm_eval_hif4a] applied HiF4 activation quant in {time.time() - t_a:.1f}s",
            flush=True,
        )

    try:
        bs: int | str = int(args.batch_size)
    except ValueError:
        bs = args.batch_size

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bs)
    t1 = time.time()
    raw = simple_evaluate(
        model=hflm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=bs,
        log_samples=False,
    )
    elapsed = time.time() - t1
    results = raw["results"]
    compact = compact_metrics(results)
    scores = compact["scores"]

    payload = {
        "model_path": model_path,
        "tasks": tasks,
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "fake_act_quant": args.fake_act_quant,
        "fake_act_quant_exclude": args.fake_act_quant_exclude,
        "seconds": elapsed,
        "scores": scores,
        "metric_keys": compact["metric_keys"],
        "arc_easy": scores.get("arc_easy"),
        "arc_challenge": scores.get("arc_challenge"),
        "mmlu": scores.get("mmlu"),
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

    out = Path(args.output_json)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(
        f"[lm_eval_hif4a] DONE arc_easy={payload['arc_easy']} "
        f"arc_challenge={payload['arc_challenge']} mmlu={payload['mmlu']} "
        f"time={elapsed:.1f}s wrote={out}",
        flush=True,
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
