"""Calibration sample objects and source builders."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from fp4_convert.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
    S1K_CALIB_SOURCES,
    WINDOW_CALIB_SOURCES,
)
from fp4_convert.experiments.e2e_diag_reconstruction.data.teacher_traces import (
    generate_split_teacher_traces,
)
from fp4_convert.src.io_utils import ensure_dir, load_pt, save_pt, write_json

S1K_HF_ID = "simplescaling/s1K-1.1_tokenized"
S1K_REQUIRED_FIELDS = ("question", "solution", "text")


@dataclass
class CalibrationSample:
    sample_id: str
    source_index: int | None
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 1 or self.loss_mask.ndim != 1:
            raise ValueError("input_ids and loss_mask must be 1-D unpadded tensors")
        if int(self.input_ids.numel()) != int(self.loss_mask.numel()):
            raise ValueError("input_ids and loss_mask length mismatch")
        if self.input_ids.device.type != "cpu" or self.loss_mask.device.type != "cpu":
            raise ValueError("calibration tensors must live on CPU")


def require_s1k_fields(row: dict[str, Any], index: int) -> None:
    for key in S1K_REQUIRED_FIELDS:
        val = row.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                f"s1k sample {index} field {key!r} is missing, empty, or not a string"
            )


def load_s1k_dataset():
    from datasets import load_dataset

    return load_dataset(S1K_HF_ID, split="train")


def split_source_ids(n_rows: int, n_train: int, n_val: int, seed: int) -> tuple[list[int], list[int]]:
    if n_rows < n_train + n_val:
        raise ValueError(
            f"dataset has {n_rows} rows, need {n_train}+{n_val} for train/val"
        )
    perm = random.Random(seed).sample(range(n_rows), k=n_rows)
    return perm[:n_train], perm[n_train : n_train + n_val]


def _encode_text(tokenizer, text: str) -> torch.Tensor:
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=False)
    ids = enc["input_ids"].reshape(-1).to(dtype=torch.long, device="cpu")
    if ids.numel() == 0:
        raise ValueError("tokenizer produced empty input_ids")
    return ids.contiguous()


def sample_from_ids_and_mask(
    sample_id: str,
    source_index: int | None,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    meta: dict[str, Any],
) -> CalibrationSample:
    return CalibrationSample(
        sample_id=sample_id,
        source_index=source_index,
        input_ids=input_ids.to(device="cpu", dtype=torch.long).contiguous(),
        loss_mask=loss_mask.to(device="cpu", dtype=torch.long).contiguous(),
        meta=meta,
    )


def build_s1k_original_sample(tokenizer, row: dict[str, Any], source_index: int) -> CalibrationSample:
    require_s1k_fields(row, source_index)
    ids = _encode_text(tokenizer, row["text"])
    mask = torch.ones_like(ids)
    return sample_from_ids_and_mask(
        f"s1k_original_{source_index}",
        source_index,
        ids,
        mask,
        {"source": "s1k_original", "question": row["question"]},
    )


def build_s1k_question_sample(tokenizer, row: dict[str, Any], source_index: int) -> CalibrationSample:
    require_s1k_fields(row, source_index)
    ids = _encode_text(tokenizer, row["question"])
    mask = torch.ones_like(ids)
    return sample_from_ids_and_mask(
        f"s1k_question_{source_index}",
        source_index,
        ids,
        mask,
        {"source": "s1k_question", "question": row["question"]},
    )


def load_window_samples(
    *,
    model_path: str,
    dataset_name: str,
    nsamples: int,
    seqlen: int,
    seed: int,
    split_name: str,
) -> list[CalibrationSample]:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    hif4 = repo_root / "HiFloat4"
    if str(hif4) not in sys.path:
        sys.path.insert(0, str(hif4))
    from hif4gptq.brq.calib import get_loaders

    raw = get_loaders(
        name=dataset_name,
        nsamples=nsamples,
        seed=seed,
        seqlen=seqlen,
        model=model_path,
        eval_mode=False,
    )
    samples: list[CalibrationSample] = []
    for i, (inp, _tar) in enumerate(raw):
        ids = inp.reshape(-1).to(dtype=torch.long, device="cpu").contiguous()
        mask = torch.ones_like(ids)
        samples.append(
            sample_from_ids_and_mask(
                f"{dataset_name}_{split_name}_{i}",
                i,
                ids,
                mask,
                {"source": dataset_name, "window_index": i},
            )
        )
    if len(samples) != nsamples:
        raise RuntimeError(f"expected {nsamples} {dataset_name} samples, got {len(samples)}")
    return samples


def _sample_manifest(samples: list[CalibrationSample]) -> list[dict[str, Any]]:
    rows = []
    for s in samples:
        rows.append(
            {
                "sample_id": s.sample_id,
                "source_index": s.source_index,
                "n_tokens": int(s.input_ids.numel()),
                "n_loss_tokens": int(s.loss_mask.to(torch.bool).sum().item()),
                "meta": {
                    k: v
                    for k, v in s.meta.items()
                    if k in {"source", "truncated", "judge", "attempts", "question_source_index"}
                },
            }
        )
    return rows


def nearest_rank_percentile(values: list[int], p: float) -> int:
    if not values:
        raise ValueError("cannot compute percentile of an empty list")
    if not 0.0 < p <= 1.0:
        raise ValueError(f"percentile p must be in (0, 1], got {p}")
    ordered = sorted(int(x) for x in values)
    n = len(ordered)
    idx = min(n - 1, math.ceil(p * n) - 1)
    if idx < 0:
        idx = 0
    return ordered[idx]


def split_length_stats(samples: list[CalibrationSample]) -> dict[str, int]:
    lengths = [int(s.input_ids.numel()) for s in samples]
    return {
        "n_samples": len(samples),
        "total_tokens": int(sum(lengths)),
        "max_seqlen": int(max(lengths)) if lengths else 0,
        "p50_seqlen": nearest_rank_percentile(lengths, 0.50) if lengths else 0,
        "p95_seqlen": nearest_rank_percentile(lengths, 0.95) if lengths else 0,
    }


def _core_calib_fields(cfg: E2ETrainConfig) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "calib_source": cfg.calib_source,
        "calib_nsamples": cfg.calib_nsamples,
        "calib_val_nsamples": cfg.calib_val_nsamples,
        "calib_seed": cfg.calib_seed,
        "calib_seqlen": cfg.calib_seqlen if cfg.calib_source in WINDOW_CALIB_SOURCES else None,
    }
    if cfg.calib_source == "s1k_teacher_cot":
        fields["teacher_trace_policy"] = cfg.teacher_trace_policy
        fields["teacher_max_attempts"] = cfg.teacher_max_attempts
        fields["teacher_max_new_tokens"] = cfg.teacher_max_new_tokens
    return fields


def _check_cached_manifest(path: Path, cfg: E2ETrainConfig, split: str) -> dict[str, Any]:
    man = json.loads(path.read_text(encoding="utf-8"))
    core = _core_calib_fields(cfg)
    for key, val in core.items():
        if man.get(key) != val:
            raise ValueError(
                f"cached {split} calibration {key}={man.get(key)!r} != current {val!r}"
            )
    n_expected = cfg.calib_nsamples if split == "train" else cfg.calib_val_nsamples
    ids = man.get("sample_ids")
    if not isinstance(ids, list) or len(ids) != n_expected:
        raise ValueError(
            f"cached {split} sample_ids count {0 if not isinstance(ids, list) else len(ids)} "
            f"!= {n_expected}"
        )
    return man


def _write_split(
    out_dir: Path,
    split: str,
    samples: list[CalibrationSample],
    cfg: E2ETrainConfig,
    extra: dict[str, Any],
) -> None:
    calib_dir = ensure_dir(out_dir / "calibration")
    save_pt(calib_dir / f"{split}.pt", samples)
    man = _core_calib_fields(cfg)
    man.update(
        {
            "split": split,
            "sample_ids": [s.sample_id for s in samples],
            "source_indices": [s.source_index for s in samples],
            "samples": _sample_manifest(samples),
        }
    )
    man.update(extra)
    write_json(calib_dir / f"{split}_manifest.json", man)


def build_or_load_calibration(
    cfg: E2ETrainConfig,
    tokenizer,
    native_model,
    out_dir: str | Path,
) -> tuple[list[CalibrationSample], list[CalibrationSample]]:
    out = Path(out_dir)
    calib_dir = out / "calibration"
    train_pt = calib_dir / "train.pt"
    val_pt = calib_dir / "val.pt"
    train_man = calib_dir / "train_manifest.json"
    val_man = calib_dir / "val_manifest.json"
    if train_pt.is_file() or val_pt.is_file() or train_man.is_file() or val_man.is_file():
        if not (train_pt.is_file() and val_pt.is_file() and train_man.is_file() and val_man.is_file()):
            raise FileNotFoundError("incomplete calibration cache; delete the directory and rebuild")
        _check_cached_manifest(train_man, cfg, "train")
        _check_cached_manifest(val_man, cfg, "val")
        return load_pt(train_pt, map_location="cpu"), load_pt(val_pt, map_location="cpu")

    if cfg.calib_source in WINDOW_CALIB_SOURCES:
        train = load_window_samples(
            model_path=cfg.model_path,
            dataset_name=cfg.calib_source,
            nsamples=cfg.calib_nsamples,
            seqlen=cfg.calib_seqlen,
            seed=cfg.calib_seed,
            split_name="train",
        )
        val = load_window_samples(
            model_path=cfg.model_path,
            dataset_name=cfg.calib_source,
            nsamples=cfg.calib_val_nsamples,
            seqlen=cfg.calib_seqlen,
            seed=cfg.calib_seed + 1,
            split_name="val",
        )
        _write_split(out, "train", train, cfg, {})
        _write_split(out, "val", val, cfg, {})
        return train, val

    if cfg.calib_source not in S1K_CALIB_SOURCES:
        raise ValueError(cfg.calib_source)

    ds = load_s1k_dataset()
    train_ids, val_ids = split_source_ids(
        len(ds), cfg.calib_nsamples, cfg.calib_val_nsamples, cfg.calib_seed
    )
    unused = [
        i
        for i in random.Random(cfg.calib_seed).sample(range(len(ds)), k=len(ds))
        if i not in set(train_ids) and i not in set(val_ids)
    ]

    if cfg.calib_source == "s1k_original":
        train = [build_s1k_original_sample(tokenizer, ds[i], i) for i in train_ids]
        val = [build_s1k_original_sample(tokenizer, ds[i], i) for i in val_ids]
        _write_split(out, "train", train, cfg, {"base_source_ids": train_ids})
        _write_split(out, "val", val, cfg, {"base_source_ids": val_ids})
        return train, val

    if cfg.calib_source == "s1k_question":
        train = [build_s1k_question_sample(tokenizer, ds[i], i) for i in train_ids]
        val = [build_s1k_question_sample(tokenizer, ds[i], i) for i in val_ids]
        _write_split(out, "train", train, cfg, {"base_source_ids": train_ids})
        _write_split(out, "val", val, cfg, {"base_source_ids": val_ids})
        return train, val

    if native_model is None:
        raise ValueError("s1k_teacher_cot requires a native NVFP4 model")

    train, train_traces = generate_split_teacher_traces(
        cfg=cfg,
        tokenizer=tokenizer,
        native_model=native_model,
        dataset=ds,
        base_ids=train_ids,
        unused_ids=unused,
        split_name="train",
        split_id=0,
    )
    used_train = {s.source_index for s in train}
    unused_val = [i for i in unused if i not in used_train]
    val, val_traces = generate_split_teacher_traces(
        cfg=cfg,
        tokenizer=tokenizer,
        native_model=native_model,
        dataset=ds,
        base_ids=val_ids,
        unused_ids=unused_val,
        split_name="val",
        split_id=1,
    )
    trace_dir = ensure_dir(out / "calibration" / "teacher_traces")
    _write_jsonl(trace_dir / "train.jsonl", train_traces)
    _write_jsonl(trace_dir / "val.jsonl", val_traces)
    _write_split(out, "train", train, cfg, {"base_source_ids": train_ids})
    _write_split(out, "val", val, cfg, {"base_source_ids": val_ids})
    return train, val


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
