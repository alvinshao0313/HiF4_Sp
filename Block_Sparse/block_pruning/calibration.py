from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Reuse HiFloat4 calibration loaders without making Block_Sparse a package install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_HIF4GPTQ = _REPO_ROOT / "HiFloat4"
if str(_HIF4GPTQ) not in sys.path:
    sys.path.insert(0, str(_HIF4GPTQ))

from hif4gptq.brq.calib import get_loaders  # noqa: E402

_S1K_HF_ID = "simplescaling/s1K-1.1_tokenized"
_WINDOW_DATASETS = frozenset({"wikitext2", "c4", "ptb"})


def _batch_from_input_ids(inp: torch.Tensor) -> dict[str, torch.Tensor]:
    if inp.ndim != 2:
        raise ValueError(f"Unexpected calib tensor rank: {inp.shape}")
    if inp.shape[1] < 2:
        raise ValueError(
            f"Calibration sequence too short for LM loss: length={inp.shape[1]}"
        )
    labels = inp.clone()
    attention_mask = torch.ones_like(inp)
    return {
        "input_ids": inp.contiguous(),
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _build_s1k_batches(
    model_path: str,
    num_samples: int,
    sequence_length: int,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    """Load full s1K-1.1_tokenized text samples (no silent truncation)."""
    if sequence_length < 0:
        raise ValueError(
            f"sequence_length must be >= 0 (0 = no truncate), got {sequence_length}"
        )
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    ds = load_dataset(_S1K_HF_ID, split="train")
    if num_samples > len(ds):
        raise ValueError(
            f"Requested {num_samples} s1k samples but dataset only has {len(ds)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    indices = random.Random(seed).sample(range(len(ds)), k=num_samples)

    batches: list[dict[str, torch.Tensor]] = []
    for idx in indices:
        text = ds[idx]["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"s1k sample {idx} has empty or non-string text")

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
        )
        inp = encoded["input_ids"]
        seq_len = int(inp.shape[1])
        if sequence_length > 0 and seq_len > sequence_length:
            raise ValueError(
                f"s1k sample {idx} has length {seq_len} > sequence_length "
                f"{sequence_length}; refuse to truncate. "
                f"Use sequence_length=0 for no upper bound."
            )
        batches.append(_batch_from_input_ids(inp))

    if len(batches) != num_samples:
        raise RuntimeError(
            f"Expected {num_samples} calibration samples, got {len(batches)}"
        )
    return batches


def _build_window_batches(
    model_path: str,
    dataset_name: str,
    num_samples: int,
    sequence_length: int,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    """Build fixed-window batches via HiFloat4 get_loaders."""
    if sequence_length <= 0:
        raise ValueError(
            f"sequence_length must be > 0 for window datasets "
            f"({sorted(_WINDOW_DATASETS)}), got {sequence_length}. "
            f"Use sequence_length=0 only with s1k."
        )

    raw = get_loaders(
        name=dataset_name,
        nsamples=num_samples,
        seed=seed,
        seqlen=sequence_length,
        model=model_path,
        eval_mode=False,
    )
    batches: list[dict[str, torch.Tensor]] = []
    for inp, _tar in raw:
        batches.append(_batch_from_input_ids(inp))
    if len(batches) != num_samples:
        raise RuntimeError(
            f"Expected {num_samples} calibration samples, got {len(batches)}"
        )
    return batches


def build_calibration_batches(
    model_path: str,
    dataset_name: str,
    num_samples: int,
    sequence_length: int,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    """Build causal-LM calibration batches with full-token LM labels.

    - ``s1k``: full ``simplescaling/s1K-1.1_tokenized`` text samples.
      ``sequence_length=0`` means no upper bound; ``>0`` hard-fails if any
      sample is longer (never silent truncation).
    - ``wikitext2`` / ``c4`` / ``ptb``: fixed windows via HiFloat4 ``get_loaders``.
      Requires ``sequence_length > 0``.

    Labels are ``input_ids`` so HuggingFace CausalLM applies standard next-token
    loss on the whole sequence, not GPTQ last-token-only labels.
    """
    if dataset_name == "s1k":
        return _build_s1k_batches(
            model_path=model_path,
            num_samples=num_samples,
            sequence_length=sequence_length,
            seed=seed,
        )
    if dataset_name in _WINDOW_DATASETS:
        return _build_window_batches(
            model_path=model_path,
            dataset_name=dataset_name,
            num_samples=num_samples,
            sequence_length=sequence_length,
            seed=seed,
        )
    raise ValueError(f"Unsupported calibration dataset: {dataset_name}")


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}
