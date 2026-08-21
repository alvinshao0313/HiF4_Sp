#!/usr/bin/env python3
"""Dedicated staged runner for HiF4 deployment-equivalent scaling experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from Inference_Paradigm_Conversion.ipc_analysis.analysis.hif4_scaling_pipeline import (
    build_candidate_scales,
    build_refined_candidate_scales,
    instantiate_all_layer_policy,
    load_scaling_experiment_config,
    merge_scaling_eval,
    merge_scaling_full_eval,
    merge_scaling_refine_eval,
    merge_scaling_stats,
    run_representative_validation,
    run_scaling_eval_shard,
    run_scaling_full_eval_shard,
    run_scaling_refine_eval_shard,
    run_scaling_stats_shard,
    run_target_trajectory_check,
    select_full_eval_candidates,
    select_representative_recipe_and_policy,
)

REPO_ROOT = Path("/home/shaoyuantian/program/HiF4_Sp")
DEFAULT_SCALING_CONFIG = (
    REPO_ROOT
    / "Inference_Paradigm_Conversion"
    / "configs"
    / "qwen3_8b_hif4_equivalent_scaling.yaml"
)
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Inference_Paradigm_Conversion" / "results"


def _load_runtime_config(path: Path) -> tuple[Path, dict[str, Any], str]:
    formal, experiment, config_hash = load_scaling_experiment_config(path)
    experiment = dict(experiment)
    experiment["config_sha256"] = config_hash
    return formal.source_checkpoint_path(), experiment, config_hash


def _run_dir(args: argparse.Namespace) -> Path:
    out = Path(args.out_dir) / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint, config, config_hash = _load_runtime_config(Path(args.scaling_config))
    run_dir = _run_dir(args)
    stage = args.stage

    if stage == "stats":
        return run_scaling_stats_shard(
            checkpoint,
            run_dir,
            split=args.split,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            device=args.device,
            config=config,
        )
    if stage == "build-candidates":
        path = build_candidate_scales(run_dir / "es_stats_merged.pt", config=config)
        return {"candidate_scales": str(path), "config_sha256": config_hash}
    if stage == "eval":
        return run_scaling_eval_shard(
            checkpoint,
            run_dir,
            candidate_scales_path=run_dir / "candidate_scales.pt",
            split=args.split,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            device=args.device,
            config=config,
        )
    if stage == "select-full":
        path = select_full_eval_candidates(
            run_dir,
            candidate_scales_path=run_dir / "candidate_scales.pt",
            config=config,
        )
        return {"subset": str(path)}
    if stage == "eval-full":
        return run_scaling_full_eval_shard(
            checkpoint,
            run_dir,
            candidate_scales_path=run_dir / "candidate_scales.pt",
            subset_path=run_dir / "es3_candidate_subset.json",
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            device=args.device,
            config=config,
        )
    if stage == "build-refine":
        path = build_refined_candidate_scales(run_dir, config=config)
        return {"refined_scales": str(path)}
    if stage == "eval-refine":
        return run_scaling_refine_eval_shard(
            checkpoint,
            run_dir,
            refined_scales_path=run_dir / "es5_refined_candidate_scales.pt",
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            device=args.device,
            config=config,
        )
    if stage == "select-policy":
        config["selection_device"] = args.device
        policy, scales = select_representative_recipe_and_policy(run_dir, config=config)
        return {"policy": str(policy), "scales": str(scales)}
    if stage == "validate":
        return run_representative_validation(
            checkpoint,
            run_dir,
            device=args.device,
            config=config,
        )
    if stage == "all-layer":
        policy, scales = instantiate_all_layer_policy(
            checkpoint,
            run_dir / "best_scaling_policy.json",
            calibration_split="discovery",
            device=args.device,
            out_dir=run_dir,
            config=config,
        )
        return {"all_layer_policy": str(policy), "all_layer_scales": str(scales)}
    if stage == "trajectory":
        return run_target_trajectory_check(
            checkpoint,
            run_dir / "best_scaling_policy_all_layers.json",
            run_dir / "best_scaling_scales_all_layers.pt",
            split="validation",
            device=args.device,
            out_dir=run_dir,
            config=config,
        )
    if stage == "e2e":
        from Inference_Paradigm_Conversion.ipc_analysis.eval.semantic_e2e import (
            run_equalized_arc_e2e,
        )

        return run_equalized_arc_e2e(
            checkpoint,
            run_dir=run_dir,
            device=args.device,
            all_layer_policy_path=run_dir / "best_scaling_policy_all_layers.json",
            all_layer_scales_path=run_dir / "best_scaling_scales_all_layers.pt",
        )
    raise ValueError(stage)


def run_merge(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _run_dir(args)
    if args.stage == "stats":
        path = merge_scaling_stats(run_dir, expected_num_shards=args.num_shards)
        return {"merged_stats": str(path)}
    if args.stage == "eval":
        merged = merge_scaling_eval(run_dir, expected_num_shards=args.num_shards)
        return {"records": len(merged["records"]), "artifact": str(run_dir / "es_eval_merged.pt")}
    if args.stage == "eval-full":
        merged = merge_scaling_full_eval(run_dir, expected_num_shards=args.num_shards)
        return {"records": len(merged["records"]), "artifact": str(run_dir / "es_full_eval_merged.pt")}
    if args.stage == "eval-refine":
        merged = merge_scaling_refine_eval(run_dir, expected_num_shards=args.num_shards)
        return {"records": len(merged["records"]), "artifact": str(run_dir / "es5_refine_eval_merged.pt")}
    raise ValueError(args.stage)


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    from Inference_Paradigm_Conversion.ipc_analysis.reporting.hif4_scaling_report import (
        build_hif4_scaling_report,
    )

    return build_hif4_scaling_report(_run_dir(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HiF4 equivalent scaling experiment runner")
    parser.add_argument("--scaling-config", type=Path, default=DEFAULT_SCALING_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    stage = sub.add_parser("stage")
    stage.add_argument(
        "--stage",
        required=True,
        choices=[
            "stats",
            "build-candidates",
            "eval",
            "select-full",
            "eval-full",
            "build-refine",
            "eval-refine",
            "select-policy",
            "validate",
            "all-layer",
            "trajectory",
            "e2e",
        ],
    )
    stage.add_argument("--run-id", required=True)
    stage.add_argument("--device", default="cuda:0")
    stage.add_argument("--split", default="discovery", choices=["discovery", "validation"])
    stage.add_argument("--shard-id", type=int, default=0)
    stage.add_argument("--num-shards", type=int, default=1)

    merge = sub.add_parser("merge")
    merge.add_argument("--stage", required=True, choices=["stats", "eval", "eval-full", "eval-refine"])
    merge.add_argument("--run-id", required=True)
    merge.add_argument("--num-shards", type=int, required=True)

    report = sub.add_parser("report")
    report.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "stage":
        result = run_stage(args)
    elif args.command == "merge":
        result = run_merge(args)
    elif args.command == "report":
        result = run_report(args)
    else:
        raise SystemExit(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
