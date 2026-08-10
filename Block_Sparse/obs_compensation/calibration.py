from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset

from obs_compensation.config import OBSCompensationConfig


_S1K_HF_ID = "simplescaling/s1K-1.1_tokenized"
_WIKITEXT_HF_ID = "Salesforce/wikitext"
_WIKITEXT_CONFIG = "wikitext-2-raw-v1"


@dataclass(frozen=True)
class CalibrationSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def make_calibration_sample(input_ids: torch.Tensor) -> CalibrationSample:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a Tensor")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids shape must be [1, T], got {tuple(input_ids.shape)}")
    if input_ids.is_floating_point() or input_ids.dtype == torch.bool:
        raise TypeError(f"input_ids dtype must be integer, got {input_ids.dtype}")
    if int(input_ids.shape[1]) < 2:
        raise ValueError(
            f"Calibration sequence too short: length={int(input_ids.shape[1])}"
        )
    ids = input_ids.detach().to(device="cpu").contiguous().clone()
    if ids.dtype != torch.long:
        ids = ids.long()
    mask = torch.ones_like(ids)
    return CalibrationSample(input_ids=ids, attention_mask=mask)


def _build_s1k_samples(
    tokenizer: Any,
    num_samples: int,
    sequence_length: int,
    seed: int,
) -> list[CalibrationSample]:
    ds = load_dataset(_S1K_HF_ID, split="train")
    if num_samples > len(ds):
        raise ValueError(
            f"Requested {num_samples} s1k samples but dataset only has {len(ds)}"
        )
    indices = random.Random(seed).sample(range(len(ds)), k=num_samples)
    samples: list[CalibrationSample] = []
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
        if int(inp.shape[1]) < 2:
            raise ValueError(f"s1k sample {idx} tokenized to fewer than 2 tokens")
        if int(inp.shape[1]) > sequence_length:
            inp = inp[:, :sequence_length]
        samples.append(make_calibration_sample(inp))
    if len(samples) != num_samples:
        raise RuntimeError(
            f"Expected {num_samples} calibration samples, got {len(samples)}"
        )
    return samples


def _build_wikitext2_samples(
    tokenizer: Any,
    num_samples: int,
    sequence_length: int,
    seed: int,
) -> list[CalibrationSample]:
    ds = load_dataset(_WIKITEXT_HF_ID, _WIKITEXT_CONFIG, split="train")
    texts = [row["text"] for row in ds if isinstance(row.get("text"), str) and row["text"].strip()]
    if not texts:
        raise ValueError("WikiText-2 train split produced no non-empty texts")
    corpus = "\n\n".join(texts)
    encoded = tokenizer(
        corpus,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
    )
    tokens = encoded["input_ids"]
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError(f"Unexpected tokenizer output shape: {tuple(tokens.shape)}")
    total = int(tokens.shape[1])
    if total < sequence_length:
        raise ValueError(
            f"WikiText-2 corpus has {total} tokens < sequence_length={sequence_length}"
        )
    max_start = total - sequence_length
    rng = random.Random(seed)
    starts = [rng.randint(0, max_start) for _ in range(num_samples)]
    samples: list[CalibrationSample] = []
    for start in starts:
        window = tokens[:, start : start + sequence_length]
        if int(window.shape[1]) != sequence_length:
            raise RuntimeError("internal error: window length mismatch")
        samples.append(make_calibration_sample(window))
    if len(samples) != num_samples:
        raise RuntimeError(
            f"Expected {num_samples} calibration samples, got {len(samples)}"
        )
    return samples


def build_calibration_samples(
    tokenizer: Any,
    config: OBSCompensationConfig,
) -> list[CalibrationSample]:
    if config.calibration_dataset == "s1k":
        return _build_s1k_samples(
            tokenizer=tokenizer,
            num_samples=config.calibration_samples,
            sequence_length=config.sequence_length,
            seed=config.seed,
        )
    if config.calibration_dataset == "wikitext2":
        return _build_wikitext2_samples(
            tokenizer=tokenizer,
            num_samples=config.calibration_samples,
            sequence_length=config.sequence_length,
            seed=config.seed,
        )
    raise ValueError(
        f"Unsupported calibration_dataset={config.calibration_dataset!r}"
    )
