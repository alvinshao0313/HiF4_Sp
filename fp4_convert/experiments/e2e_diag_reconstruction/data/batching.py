"""Dynamic padding and length-bucket batching for reconstruction calibration."""

from __future__ import annotations

import random

import torch

from fp4_convert.experiments.e2e_diag_reconstruction.data.calibration import (
    CalibrationSample,
)


class DynamicCalibrationCollator:
    def __init__(self, pad_token_id: int) -> None:
        if pad_token_id is None:
            raise ValueError("pad_token_id is required")
        self.pad_token_id = int(pad_token_id)

    def __call__(self, samples: list[CalibrationSample]) -> dict[str, torch.Tensor | list[str]]:
        if not samples:
            raise ValueError("empty calibration batch")
        lengths = torch.tensor([int(s.input_ids.numel()) for s in samples], dtype=torch.long)
        tmax = int(lengths.max().item())
        bsz = len(samples)
        input_ids = torch.full((bsz, tmax), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, tmax), dtype=torch.long)
        loss_mask = torch.zeros((bsz, tmax), dtype=torch.long)
        for i, sample in enumerate(samples):
            n = int(sample.input_ids.numel())
            input_ids[i, :n] = sample.input_ids
            attention_mask[i, :n] = 1
            loss_mask[i, :n] = sample.loss_mask
            loss_mask[i, :n] = loss_mask[i, :n] * attention_mask[i, :n]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "lengths": lengths,
            "sample_ids": [s.sample_id for s in samples],
        }


def build_length_bucket_batches(
    samples: list[CalibrationSample],
    batch_size: int,
    seed: int,
) -> list[list[CalibrationSample]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    ordered = sorted(samples, key=lambda s: int(s.input_ids.numel()))
    bucket = 8 * batch_size
    rng = random.Random(seed)
    batches: list[list[CalibrationSample]] = []
    for i in range(0, len(ordered), bucket):
        group = list(ordered[i : i + bucket])
        rng.shuffle(group)
        for j in range(0, len(group), batch_size):
            chunk = group[j : j + batch_size]
            if chunk:
                batches.append(chunk)
    rng.shuffle(batches)
    return batches


def build_validation_batches(
    samples: list[CalibrationSample],
    batch_size: int,
) -> list[list[CalibrationSample]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    ordered = sorted(samples, key=lambda s: int(s.input_ids.numel()))
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]
