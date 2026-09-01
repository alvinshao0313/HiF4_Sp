"""CLI for LiveCodeBench evaluation with the locked Qwen3 thinking protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import DEFAULT_MODEL_PATH
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.lcb_runner import run_livecodebench_vllm

EVAL_VARIANTS = ("native_nvfp4", "direct_hif4", "r64_only", "artifact")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate LiveCodeBench codegeneration_v6 for NVFP4/HiF4 variants")
    p.add_argument("--variant", required=True, choices=EVAL_VARIANTS)
    p.add_argument("--artifact_path", type=str, default="")
    p.add_argument(
        "--artifact_diag_variant",
        type=str,
        choices=("adopted", "candidate"),
        default="adopted",
    )
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    artifact = args.artifact_path or None
    if args.variant == "artifact" and not artifact:
        raise ValueError("--artifact_path is required when --variant artifact")
    run_livecodebench_vllm(
        variant=args.variant,
        output_dir=Path(args.output_dir),
        model_path=args.model_path,
        artifact_path=artifact,
        artifact_diag_variant=str(args.artifact_diag_variant),
        device=args.device,
    )


if __name__ == "__main__":
    main()
