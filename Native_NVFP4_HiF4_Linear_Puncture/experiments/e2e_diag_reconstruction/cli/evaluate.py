"""CLI for fake-QDQ E2E evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    DEFAULT_MODEL_PATH,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.runner import (
    EVAL_VARIANTS,
    run_eval_groups,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate semantic NVFP4/HiF4 models")
    p.add_argument("--variant", required=True, choices=EVAL_VARIANTS)
    p.add_argument("--artifact_path", type=str, default="")
    p.add_argument(
        "--artifact_diag_variant",
        type=str,
        choices=("adopted", "candidate"),
        default="adopted",
        help="Replay the final adopted DIAG or the pre-rollback candidate from a schema-v3 artifact.",
    )
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--groups",
        type=str,
        default="arc",
        help="comma-separated: arc,mmlu_pro_300,aime25_avg5. "
        "Default is arc-only (MMLU-Pro temporarily suspended). "
        "mmlu_pro_300 and aime25_avg5 use repo-root main.py (vLLM TP=2); "
        "CUDA_VISIBLE_DEVICES must expose at least 2 GPUs",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval_seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    groups = [x.strip() for x in args.groups.split(",") if x.strip()]
    artifact = args.artifact_path or None
    if args.variant == "artifact" and not artifact:
        raise ValueError("--artifact_path is required when --variant artifact")
    run_eval_groups(
        variant=args.variant,
        groups=groups,
        output_dir=Path(args.output_dir),
        device=args.device,
        artifact_path=artifact,
        artifact_diag_variant=str(args.artifact_diag_variant),
        model_path=args.model_path,
        eval_seed=int(args.eval_seed),
    )


if __name__ == "__main__":
    main()
