"""Prebuild non-Teacher calibration caches without loading the 8B native model."""

from __future__ import annotations

import argparse

from pathlib import Path

from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    DEFAULT_MODEL_PATH,
    E2ETrainConfig,
    validate_train_config,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    build_or_load_calibration,
    split_length_stats,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import write_json

PREPARE_CALIB_SOURCES = ("s1k_original", "s1k_question", "wikitext2", "c4")


def require_prepare_source(source: str) -> None:
    if source == "s1k_teacher_cot":
        raise ValueError("prepare_calibration rejects s1k_teacher_cot")
    if source not in PREPARE_CALIB_SOURCES:
        raise ValueError(f"unsupported calib_source={source!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prebuild non-Teacher calibration cache")
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--calib_source", type=str, required=True, choices=PREPARE_CALIB_SOURCES)
    p.add_argument("--calib_nsamples", type=int, default=128)
    p.add_argument("--calib_val_nsamples", type=int, default=32)
    p.add_argument("--calib_seed", type=int, default=42)
    p.add_argument("--calib_seqlen", type=int, default=1024)
    p.add_argument("--calib_cache_dir", type=str, required=True)
    return p


def _print_split(name: str, stats: dict[str, int]) -> None:
    print(
        f"{name}: n={stats['n_samples']} total_tokens={stats['total_tokens']} "
        f"max={stats['max_seqlen']} p50={stats['p50_seqlen']} p95={stats['p95_seqlen']}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    require_prepare_source(args.calib_source)
    cfg = E2ETrainConfig(
        model_path=str(args.model_path),
        output_dir=str(args.calib_cache_dir),
        calib_source=str(args.calib_source),
        calib_nsamples=int(args.calib_nsamples),
        calib_val_nsamples=int(args.calib_val_nsamples),
        calib_seed=int(args.calib_seed),
        calib_seqlen=int(args.calib_seqlen),
        calib_cache_dir=str(args.calib_cache_dir),
    )
    validate_train_config(cfg)

    snapshot = resolve_local_snapshot(cfg.model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    cache_dir = Path(cfg.calib_cache_dir)
    train, val = build_or_load_calibration(cfg, tokenizer, None, cache_dir)
    train_stats = split_length_stats(train)
    val_stats = split_length_stats(val)
    blob = {
        "calib_source": cfg.calib_source,
        "calib_nsamples": cfg.calib_nsamples,
        "calib_val_nsamples": cfg.calib_val_nsamples,
        "calib_seed": cfg.calib_seed,
        "calib_seqlen": cfg.calib_seqlen if cfg.calib_source in {"wikitext2", "c4"} else None,
        "train": train_stats,
        "val": val_stats,
    }
    write_json(cache_dir / "calibration" / "stats.json", blob)
    _print_split("train", train_stats)
    _print_split("val", val_stats)


if __name__ == "__main__":
    main()
