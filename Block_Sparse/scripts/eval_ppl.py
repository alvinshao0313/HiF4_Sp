#!/usr/bin/env python3
"""Evaluate WikiText-2 perplexity for a (pruned) HF CausalLM checkpoint.

Uses non-overlapping windows (same convention as SparseGPT/Wanda/GPTQ PPL).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

BLOCK_SPARSE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BLOCK_SPARSE_ROOT.parent
if str(BLOCK_SPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(BLOCK_SPARSE_ROOT))

from block_pruning.config import GradientBlockPruningConfig  # noqa: E402
from block_pruning.model_loader import (  # noqa: E402
    load_model_and_tokenizer,
    resolve_model_input_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to write a one-line JSON result.",
    )
    p.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4"])
    p.add_argument("--sequence_length", type=int, default=2048)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, only evaluate this many windows (smoke test).",
    )
    return p.parse_args()


def build_eval_windows(
    model_path: str,
    dataset: str,
    sequence_length: int,
    seed: int,
) -> list[torch.Tensor]:
    """Return list of [1, seq_len] input_id tensors from the dataset test split."""
    sys.path.insert(0, str(REPO_ROOT / "HiFloat4"))
    from hif4gptq.brq.calib import get_loaders  # noqa: E402

    encoded = get_loaders(
        name=dataset,
        nsamples=1,
        seed=seed,
        seqlen=sequence_length,
        model=model_path,
        eval_mode=True,
    )
    ids = encoded.input_ids
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(f"Unexpected eval encoding shape: {tuple(ids.shape)}")
    total = ids.shape[1]
    n = total // sequence_length
    if n < 1:
        raise RuntimeError(
            f"Encoded length {total} < sequence_length {sequence_length}"
        )
    windows = [
        ids[:, i * sequence_length : (i + 1) * sequence_length].contiguous()
        for i in range(n)
    ]
    return windows


@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module,
    windows: list[torch.Tensor],
) -> dict:
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    device = resolve_model_input_device(model)
    nll_sum = 0.0
    tok_sum = 0
    t0 = time.time()

    for i, batch in enumerate(windows):
        batch = batch.to(device)
        out = model(input_ids=batch, use_cache=False)
        logits = out.logits[:, :-1, :].float()
        labels = batch[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="sum",
        )
        nll_sum += float(loss.item())
        tok_sum += int(labels.numel())
        if (i + 1) % 10 == 0 or (i + 1) == len(windows):
            print(
                f"[ppl] window {i + 1}/{len(windows)} "
                f"running_ppl={torch.exp(torch.tensor(nll_sum / tok_sum)).item():.4f}",
                flush=True,
            )

    mean_nll = nll_sum / tok_sum
    ppl = float(torch.exp(torch.tensor(mean_nll)).item())
    elapsed = time.time() - t0
    return {
        "ppl": ppl,
        "mean_nll": mean_nll,
        "num_tokens": tok_sum,
        "num_windows": len(windows),
        "seconds": elapsed,
        "input_device": str(device),
    }


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    if not Path(model_path).is_absolute() and Path(model_path).exists():
        model_path = str(Path(model_path).resolve())

    print(
        f"[ppl] model={model_path} dataset={args.dataset} "
        f"seq={args.sequence_length} dtype={args.dtype}",
        flush=True,
    )

    # Reuse pruning loader (device_map=auto over CUDA_VISIBLE_DEVICES).
    # score_type=magnitude avoids enabling gradient checkpointing.
    config = GradientBlockPruningConfig(
        model_path=model_path,
        output_dir=str(REPO_ROOT / "Block_Sparse" / "results" / "ppl_tmp"),
        score_type="magnitude",
        dtype=args.dtype,
        device=args.device,
        gradient_checkpointing=False,
        trust_remote_code=True,
    )
    model, _tokenizer = load_model_and_tokenizer(config)

    windows = build_eval_windows(
        model_path=model_path,
        dataset=args.dataset,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    if args.max_samples > 0:
        windows = windows[: args.max_samples]
    print(f"[ppl] num_windows={len(windows)}", flush=True)

    result = evaluate_ppl(model, windows)
    result.update(
        {
            "model_path": model_path,
            "dataset": args.dataset,
            "sequence_length": args.sequence_length,
            "dtype": args.dtype,
        }
    )

    print(
        f"[ppl] DONE ppl={result['ppl']:.4f} "
        f"tokens={result['num_tokens']} time={result['seconds']:.1f}s",
        flush=True,
    )

    if args.output_json:
        out = Path(args.output_json)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[ppl] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
