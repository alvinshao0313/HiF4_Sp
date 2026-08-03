"""Collect activation inputs for offline threshold calibration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_HIFLOAT4 = _ROOT.parent
for p in (_ROOT, _HIFLOAT4):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.model_hooks import collect_layer_inputs, iter_target_linears  # noqa: E402


_S1K_HF_ID = "simplescaling/s1K-1.1_tokenized"


def _calibration_batches(
    tokenizer, *, n_samples: int, seqlen: int, split: str, seed: int, dataset: str
):
    from datasets import load_dataset

    if dataset == "s1k":
        # Full-length reasoning traces; seqlen must be 0 (no truncation).
        if seqlen != 0:
            raise ValueError("s1k requires --seqlen 0 (no truncation)")
        import random

        ds = load_dataset(_S1K_HF_ID, split="train")
        if n_samples > len(ds):
            raise ValueError(f"requested {n_samples} s1k samples, dataset has {len(ds)}")
        indices = random.Random(seed).sample(range(len(ds)), k=n_samples)
        for i in indices:
            text = ds[i]["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"s1k sample {i} has empty or non-string text")
            enc = tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=False,
            )
            yield {
                "input_ids": enc["input_ids"],
                "attention_mask": torch.ones_like(enc["input_ids"]),
            }
        return

    # Calib = train, val collection can use validation — caller decides.
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [t for t in ds["text"] if t and t.strip()]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(texts), generator=g)[:n_samples].tolist()
    for i in idx:
        enc = tokenizer(
            texts[i],
            return_tensors="pt",
            truncation=True,
            max_length=seqlen,
            padding=False,
        )
        yield {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "s1k"])
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument(
        "--max-rows-per-batch",
        type=int,
        default=None,
        help="Cap tokens sampled per forward so long sequences cannot fill the whole budget",
    )
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=512, help="0 = no truncation (s1k only)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--layer-sample",
        action="store_true",
        help="Only collect early/mid/late layers to reduce cost",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "results" / f"{stamp}_act_stats_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(vars(args)), encoding="utf-8")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.device,
    )
    model.eval()

    layer_filter = None
    if args.layer_sample:
        targets = [n for n, _ in iter_target_linears(model)]
        picks = [targets[0], targets[len(targets) // 2], targets[-1]]
        # also a few module types
        layer_filter = list(dict.fromkeys(picks + targets[:: max(len(targets) // 6, 1)][:6]))
        print(f"Sampling {len(layer_filter)} layers")

    batches = _calibration_batches(
        tok,
        n_samples=args.n_samples,
        seqlen=args.seqlen,
        split=args.split,
        seed=args.seed,
        dataset=args.dataset,
    )
    store = collect_layer_inputs(
        model,
        batches,
        device=torch.device(args.device),
        max_rows=args.max_rows,
        seed=args.seed,
        layer_filter=layer_filter,
        max_rows_per_batch=args.max_rows_per_batch,
    )
    payload = {
        "inputs": store.inputs,
        "weight_col_energy": store.weight_col_energy,
    }
    torch.save(payload, out_dir / "activation_store.pt")
    meta = {
        "layers": list(store.inputs.keys()),
        "rows": {k: int(v.shape[0]) for k, v in store.inputs.items()},
        "in_features": {k: int(v.shape[1]) for k, v in store.inputs.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {len(store.inputs)} layers to {out_dir}")


if __name__ == "__main__":
    main()
