"""Fixed index sharding and merge for teacher_trace_policy=all only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fp4_convert.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
)
from fp4_convert.experiments.e2e_diag_reconstruction.data.calibration import (
    CalibrationSample,
    _write_jsonl,
    _write_split,
)
from fp4_convert.src.io_utils import ensure_dir, load_pt, save_pt, write_json


def require_all_policy(cfg: E2ETrainConfig) -> None:
    if cfg.teacher_trace_policy != "all":
        raise ValueError(
            "sharded teacher-cot cache only supports teacher_trace_policy=all, "
            f"got {cfg.teacher_trace_policy!r}"
        )
    if cfg.calib_source != "s1k_teacher_cot":
        raise ValueError(
            "sharded teacher-cot cache only supports calib_source=s1k_teacher_cot, "
            f"got {cfg.calib_source!r}"
        )


def shard_source_ids(ids: list[int], num_shards: int, shard_id: int) -> list[int]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id={shard_id} out of range for num_shards={num_shards}")
    return [ids[i] for i in range(len(ids)) if i % num_shards == shard_id]


def assert_shards_partition(full_ids: list[int], num_shards: int) -> list[list[int]]:
    shards = [shard_source_ids(full_ids, num_shards, k) for k in range(num_shards)]
    flat = [x for shard in shards for x in shard]
    if sorted(flat) != sorted(full_ids):
        raise RuntimeError("shard partition union != full_ids")
    seen: set[int] = set()
    for shard in shards:
        for x in shard:
            if x in seen:
                raise RuntimeError(f"source_index {x} appears in multiple shards")
            seen.add(x)
    return shards


def split_shard_dir(output_shard_dir: Path, split_name: str) -> Path:
    return Path(output_shard_dir) / split_name


def write_split_shard(
    *,
    output_shard_dir: Path,
    split_name: str,
    split_id: int,
    shard_id: int,
    num_shards: int,
    base_ids: list[int],
    samples: list[CalibrationSample],
    traces: list[dict[str, Any]],
) -> None:
    if len(samples) != len(base_ids):
        raise RuntimeError(
            f"shard {shard_id} {split_name}: len(samples)={len(samples)} != "
            f"len(base_ids)={len(base_ids)}"
        )
    for sample, expected in zip(samples, base_ids, strict=True):
        if sample.source_index != expected:
            raise RuntimeError(
                f"shard {shard_id} {split_name}: sample source_index="
                f"{sample.source_index} != expected {expected}"
            )
    out = ensure_dir(split_shard_dir(output_shard_dir, split_name))
    save_pt(out / "samples.pt", samples)
    _write_jsonl(out / "traces.jsonl", traces)
    write_json(
        out / "shard_meta.json",
        {
            "shard_id": int(shard_id),
            "num_shards": int(num_shards),
            "split_name": split_name,
            "split_id": int(split_id),
            "base_ids": [int(x) for x in base_ids],
            "n_samples": len(samples),
        },
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid jsonl {path}:{line_no}") from exc
    return rows


def _load_split_shard(
    shard_dir: Path,
    *,
    split_name: str,
    split_id: int,
    shard_id: int,
    num_shards: int,
    expected_ids: list[int],
) -> tuple[list[CalibrationSample], list[dict[str, Any]]]:
    split_dir = split_shard_dir(shard_dir, split_name)
    samples_path = split_dir / "samples.pt"
    traces_path = split_dir / "traces.jsonl"
    meta_path = split_dir / "shard_meta.json"
    if not samples_path.is_file():
        raise FileNotFoundError(f"missing {samples_path}")
    if not traces_path.is_file():
        raise FileNotFoundError(f"missing {traces_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, expected in (
        ("shard_id", shard_id),
        ("num_shards", num_shards),
        ("split_name", split_name),
        ("split_id", split_id),
    ):
        if meta.get(key) != expected:
            raise ValueError(
                f"{meta_path} {key}={meta.get(key)!r} != expected {expected!r}"
            )
    if meta.get("base_ids") != expected_ids:
        raise ValueError(
            f"{meta_path} base_ids mismatch for shard_id={shard_id} split={split_name}"
        )
    samples = load_pt(samples_path, map_location="cpu")
    if not isinstance(samples, list):
        raise TypeError(f"{samples_path} must contain a list of CalibrationSample")
    if len(samples) != len(expected_ids):
        raise RuntimeError(
            f"{samples_path}: got {len(samples)} samples, expected {len(expected_ids)}"
        )
    for sample, expected in zip(samples, expected_ids, strict=True):
        if sample.source_index != expected:
            raise RuntimeError(
                f"{samples_path}: source_index={sample.source_index} != expected {expected}"
            )
    traces = _load_jsonl(traces_path)
    if len(traces) != len(expected_ids):
        raise RuntimeError(
            f"{traces_path}: got {len(traces)} traces, expected {len(expected_ids)}"
        )
    for row, expected in zip(traces, expected_ids, strict=True):
        if int(row["source_index"]) != int(expected):
            raise RuntimeError(
                f"{traces_path}: source_index={row['source_index']} != expected {expected}"
            )
    return samples, traces


def merge_teacher_cot_shards(
    *,
    cfg: E2ETrainConfig,
    calib_cache_dir: Path | str,
    shard_dirs: list[Path | str],
    train_ids: list[int],
    val_ids: list[int],
) -> None:
    require_all_policy(cfg)
    out = Path(calib_cache_dir)
    dirs = [Path(p) for p in shard_dirs]
    num_shards = len(dirs)
    if num_shards < 1:
        raise ValueError("need at least one shard directory")
    for i, d in enumerate(dirs):
        if not d.is_dir():
            raise FileNotFoundError(f"missing shard directory: {d}")

    train_parts = assert_shards_partition(train_ids, num_shards)
    val_parts = assert_shards_partition(val_ids, num_shards)

    train_samples: list[CalibrationSample] = []
    train_traces: list[dict[str, Any]] = []
    val_samples: list[CalibrationSample] = []
    val_traces: list[dict[str, Any]] = []

    train_by_src: dict[int, CalibrationSample] = {}
    train_tr_by_src: dict[int, dict[str, Any]] = {}
    val_by_src: dict[int, CalibrationSample] = {}
    val_tr_by_src: dict[int, dict[str, Any]] = {}

    for shard_id, shard_dir in enumerate(dirs):
        ts, tt = _load_split_shard(
            shard_dir,
            split_name="train",
            split_id=0,
            shard_id=shard_id,
            num_shards=num_shards,
            expected_ids=train_parts[shard_id],
        )
        vs, vt = _load_split_shard(
            shard_dir,
            split_name="val",
            split_id=1,
            shard_id=shard_id,
            num_shards=num_shards,
            expected_ids=val_parts[shard_id],
        )
        for sample, row in zip(ts, tt, strict=True):
            src = int(sample.source_index)
            if src in train_by_src:
                raise RuntimeError(f"duplicate train source_index={src}")
            train_by_src[src] = sample
            train_tr_by_src[src] = row
        for sample, row in zip(vs, vt, strict=True):
            src = int(sample.source_index)
            if src in val_by_src:
                raise RuntimeError(f"duplicate val source_index={src}")
            val_by_src[src] = sample
            val_tr_by_src[src] = row

    for src in train_ids:
        if src not in train_by_src:
            raise RuntimeError(f"missing train sample source_index={src}")
        train_samples.append(train_by_src[src])
        train_traces.append(train_tr_by_src[src])
    for src in val_ids:
        if src not in val_by_src:
            raise RuntimeError(f"missing val sample source_index={src}")
        val_samples.append(val_by_src[src])
        val_traces.append(val_tr_by_src[src])

    if [s.source_index for s in train_samples] != train_ids:
        raise RuntimeError("merged train order != train_ids")
    if [s.source_index for s in val_samples] != val_ids:
        raise RuntimeError("merged val order != val_ids")

    calib_root = out / "calibration"
    if calib_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing calibration dir: {calib_root}"
        )

    trace_dir = ensure_dir(out / "calibration" / "teacher_traces")
    _write_jsonl(trace_dir / "train.jsonl", train_traces)
    _write_jsonl(trace_dir / "val.jsonl", val_traces)
    _write_split(out, "train", train_samples, cfg, {"base_source_ids": train_ids})
    _write_split(out, "val", val_samples, cfg, {"base_source_ids": val_ids})
