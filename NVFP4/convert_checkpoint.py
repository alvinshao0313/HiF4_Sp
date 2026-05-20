#!/usr/bin/env python3
"""CLI for converting compressed-tensors NVFP4 checkpoints to BF16."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from NVFP4.dequantize import (  # noqa: E402
    DEFAULT_MAX_SHARD_SIZE,
    convert_nvfp4_checkpoint_to_bf16,
    estimate_converted_checkpoint_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a compressed-tensors NVFP4 checkpoint to BF16."
    )
    parser.add_argument("--input_dir", required=True, help="NVFP4 model directory")
    parser.add_argument("--output_dir", required=True, help="BF16 output directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output_dir if it already exists.",
    )
    parser.add_argument(
        "--drop_activation_scales",
        action="store_true",
        help="Do not save input_global_scale tensors to the sidecar file.",
    )
    parser.add_argument(
        "--max_shard_size_gb",
        type=float,
        default=DEFAULT_MAX_SHARD_SIZE / 1024**3,
        help="Maximum output safetensors shard size in GiB. Default: 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_activation_scales = not args.drop_activation_scales
    max_shard_size = int(args.max_shard_size_gb * 1024**3)
    if max_shard_size <= 0:
        raise ValueError("--max_shard_size_gb must be positive")

    estimate = estimate_converted_checkpoint_size(
        args.input_dir,
        keep_activation_scales=keep_activation_scales,
    )
    print(
        "Estimated output size: "
        f"main={estimate.main_bytes / 1024**3:.2f} GiB, "
        f"activation_sidecar={estimate.sidecar_bytes / 1024**3:.2f} GiB, "
        f"auxiliary={estimate.auxiliary_bytes / 1024**3:.2f} GiB"
    )
    output_path = convert_nvfp4_checkpoint_to_bf16(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        keep_activation_scales=keep_activation_scales,
        overwrite=args.overwrite,
        max_shard_size=max_shard_size,
    )
    print(f"BF16 checkpoint written to: {output_path}")


if __name__ == "__main__":
    main()
