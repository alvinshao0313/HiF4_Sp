"""Deterministic S1K train/validation/holdout protocol and hidden cache."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator,
    build_length_bucket_batches,
    build_validation_batches,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    build_s1k_original_sample,
    load_s1k_dataset,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_trainer import (
    build_initial_moe_hidden_cache,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from .artifact import atomic_save, read_json, tensor_sha256, write_json


def _sample_row(sample):
    return {"id": sample.sample_id, "source_index": sample.source_index,
            "tokens": int(sample.input_ids.numel()),
            "token_sha256": tensor_sha256(sample.input_ids)}


def prepare(root, cfg):
    """Build all three disjoint splits in a new experiment directory."""
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("prepare requires a new empty experiment directory")
    root.mkdir(parents=True, exist_ok=True)
    snapshot = Path(cfg.model_path).resolve() if Path(cfg.model_path).is_dir() else Path(resolve_local_snapshot(cfg.model_path)).resolve()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    ds = load_s1k_dataset()
    count = cfg.calib_nsamples + cfg.calib_val_nsamples + cfg.calib_holdout_nsamples
    if len(ds) < count:
        raise ValueError(f"S1K has {len(ds)} rows, need {count}")
    permutation = random.Random(cfg.calib_seed).sample(range(len(ds)), k=len(ds))
    sizes = (cfg.calib_nsamples, cfg.calib_val_nsamples, cfg.calib_holdout_nsamples)
    names = ("train", "val", "holdout")
    entries = {}
    splits = {}
    cursor = 0
    seen = set()
    for name, size in zip(names, sizes):
        rows = []
        for source_index in permutation[cursor:cursor + size]:
            sample = build_s1k_original_sample(tokenizer, ds[source_index], source_index)
            row = _sample_row(sample)
            if row["token_sha256"] in seen:
                raise RuntimeError("duplicate token sequence across splits")
            seen.add(row["token_sha256"])
            entries[sample.sample_id] = sample
            rows.append(row)
        splits[name] = [r["id"] for r in rows]
        cursor += size
    batches = [[s.sample_id for s in sorted((entries[sid] for sid in splits["train"]),
                                             key=lambda x: (len(x.input_ids), x.sample_id))[i:i + cfg.batch_size]]
               for i in range(0, len(splits["train"]), cfg.batch_size)]
    protocol = {"status": "COMPLETE", "schema_version": 1,
                "model_path": str(snapshot), "seed": cfg.calib_seed,
                "splits": splits, "samples": {sid: _sample_row(s) for sid, s in entries.items()},
                "batches": batches, "config": cfg.to_dict()}
    for sid, sample in entries.items():
        atomic_save(sample, root / "data" / f"{sid}.pt")
    write_json(protocol, root / "protocol.json")
    return protocol


class Dataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.protocol = read_json(self.root / "protocol.json")
        if self.protocol.get("status") != "COMPLETE":
            raise RuntimeError("incomplete protocol")
        self.samples = {sid: torch.load(self.root / "data" / f"{sid}.pt", map_location="cpu", weights_only=False)
                        for sid in self.protocol["samples"]}
        for sid, sample in self.samples.items():
            row = self.protocol["samples"][sid]
            if tensor_sha256(sample.input_ids) != row["token_sha256"]:
                raise RuntimeError(f"sample hash changed: {sid}")

    def ids(self, split):
        return list(self.protocol["splits"][split])

    def split(self, name):
        return [self.samples[sid] for sid in self.ids(name)]

    @property
    def snapshot(self):
        return Path(self.protocol["model_path"])


def ensure_dataset(root, cfg):
    root = Path(root)
    if not (root / "protocol.json").is_file():
        prepare(root, cfg)
    return Dataset(root)


def initial_hidden(snapshot, samples, collator, device, batch_size):
    return build_initial_moe_hidden_cache(Path(snapshot), samples, collator, device, batch_size)


def assemble(values, samples, device):
    if not isinstance(values, dict):
        raise TypeError("hidden/target values must be a sample-id mapping")
    items = [values[s.sample_id] for s in samples]
    if any(x is None for x in items):
        raise RuntimeError("hidden cache missing sample")
    tmax = max(int(x.shape[0]) for x in items)
    out = torch.zeros(len(items), tmax, items[0].shape[-1], dtype=items[0].dtype, device=device)
    lengths = torch.tensor([int(x.shape[0]) for x in items], device=device)
    for i, value in enumerate(items):
        out[i, :value.shape[0]] = value.to(device)
    return out, lengths


def batches(samples, batch_size, *, training=False, seed=42):
    return (build_length_bucket_batches(samples, batch_size, seed) if training
            else build_validation_batches(samples, batch_size))


def valid_mask(samples, batch, device):
    lengths = torch.tensor([int(s.input_ids.numel()) for s in batch], device=device)
    positions = torch.arange(int(lengths.max()), device=device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)
