"""Build shared Teacher-CoT calibration cache via index shards (policy=all only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    DEFAULT_MODEL_PATH,
    E2ETrainConfig,
    TEACHER_TRACE_POLICIES,
    validate_train_config,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    load_s1k_dataset,
    split_source_ids,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.teacher_cot_shards import (
    merge_teacher_cot_shards,
    require_all_policy,
    shard_source_ids,
    write_split_shard,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.teacher_traces import (
    generate_split_teacher_traces,
    require_qwen3_chat_template,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import ensure_dir
from Native_NVFP4_HiF4_Linear_Puncture.src.semantic_model import load_native_nvfp4_semantic_model


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("teacher-cot shard worker requires CUDA")
    return torch.device("cuda")


def _cfg_from_args(args: argparse.Namespace) -> E2ETrainConfig:
    cfg = E2ETrainConfig(
        model_path=args.model_path,
        output_dir=str(args.output_dir),
        calib_source=args.calib_source,
        calib_nsamples=args.calib_nsamples,
        calib_val_nsamples=args.calib_val_nsamples,
        calib_seed=args.calib_seed,
        teacher_trace_policy=args.teacher_trace_policy,
        teacher_max_attempts=args.teacher_max_attempts,
        teacher_max_new_tokens=args.teacher_max_new_tokens,
        calib_cache_dir=str(args.calib_cache_dir),
    )
    validate_train_config(cfg)
    require_all_policy(cfg)
    return cfg


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--calib_source", type=str, default="s1k_teacher_cot")
    p.add_argument("--calib_nsamples", type=int, default=128)
    p.add_argument("--calib_val_nsamples", type=int, default=32)
    p.add_argument("--calib_seed", type=int, default=42)
    p.add_argument(
        "--teacher_trace_policy",
        type=str,
        choices=TEACHER_TRACE_POLICIES,
        default="all",
    )
    p.add_argument("--teacher_max_attempts", type=int, default=4)
    p.add_argument("--teacher_max_new_tokens", type=int, default=32768)
    p.add_argument("--calib_cache_dir", type=str, required=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build sharded Teacher-CoT calibration cache (policy=all only)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    worker = sub.add_parser("worker", help="Generate one shard on one GPU")
    _add_shared_args(worker)
    worker.add_argument("--shard_id", type=int, required=True)
    worker.add_argument("--num_shards", type=int, required=True)
    worker.add_argument("--output_shard_dir", type=str, required=True)
    worker.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="unused placeholder for E2ETrainConfig; defaults to output_shard_dir",
    )

    merge = sub.add_parser("merge", help="Merge shard directories into shared calibration cache")
    _add_shared_args(merge)
    merge.add_argument(
        "--shards_root",
        type=str,
        required=True,
        help="directory containing shard_0 .. shard_{N-1}",
    )
    merge.add_argument("--num_shards", type=int, required=True)
    merge.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="unused placeholder for E2ETrainConfig; defaults to calib_cache_dir",
    )
    return p


def run_worker(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(
            f"shard_id={args.shard_id} out of range for num_shards={args.num_shards}"
        )
    if not args.output_dir:
        args.output_dir = args.output_shard_dir
    cfg = _cfg_from_args(args)
    device = _require_cuda()
    out_shard = ensure_dir(args.output_shard_dir)

    snapshot = resolve_local_snapshot(cfg.model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no pad_token_id or eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    require_qwen3_chat_template(tokenizer)

    model, _index = load_native_nvfp4_semantic_model(snapshot, device=device)
    ds = load_s1k_dataset()
    train_ids, val_ids = split_source_ids(
        len(ds), cfg.calib_nsamples, cfg.calib_val_nsamples, cfg.calib_seed
    )
    train_shard = shard_source_ids(train_ids, args.num_shards, args.shard_id)
    val_shard = shard_source_ids(val_ids, args.num_shards, args.shard_id)
    print(
        f"[worker] shard_id={args.shard_id}/{args.num_shards} "
        f"train={len(train_shard)} val={len(val_shard)}",
        flush=True,
    )

    train_samples, train_traces = generate_split_teacher_traces(
        cfg=cfg,
        tokenizer=tokenizer,
        native_model=model,
        dataset=ds,
        base_ids=train_shard,
        unused_ids=[],
        split_name="train",
        split_id=0,
    )
    write_split_shard(
        output_shard_dir=out_shard,
        split_name="train",
        split_id=0,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        base_ids=train_shard,
        samples=train_samples,
        traces=train_traces,
    )

    val_samples, val_traces = generate_split_teacher_traces(
        cfg=cfg,
        tokenizer=tokenizer,
        native_model=model,
        dataset=ds,
        base_ids=val_shard,
        unused_ids=[],
        split_name="val",
        split_id=1,
    )
    write_split_shard(
        output_shard_dir=out_shard,
        split_name="val",
        split_id=1,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        base_ids=val_shard,
        samples=val_samples,
        traces=val_traces,
    )
    print(f"[worker] shard_id={args.shard_id} done -> {out_shard}", flush=True)


def run_merge(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}")
    if not args.output_dir:
        args.output_dir = args.calib_cache_dir
    cfg = _cfg_from_args(args)
    shards_root = Path(args.shards_root)
    shard_dirs = [shards_root / f"shard_{k}" for k in range(args.num_shards)]
    for d in shard_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"missing shard directory: {d}")

    ds = load_s1k_dataset()
    train_ids, val_ids = split_source_ids(
        len(ds), cfg.calib_nsamples, cfg.calib_val_nsamples, cfg.calib_seed
    )
    merge_teacher_cot_shards(
        cfg=cfg,
        calib_cache_dir=args.calib_cache_dir,
        shard_dirs=shard_dirs,
        train_ids=train_ids,
        val_ids=val_ids,
    )
    print(f"[merge] wrote shared cache -> {args.calib_cache_dir}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "worker":
        run_worker(args)
        return
    if args.command == "merge":
        run_merge(args)
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
