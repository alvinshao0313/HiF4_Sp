"""Short training through the formal entrypoint, with untruncated cached samples."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("attention", "moe", "both"), default="both")
    parser.add_argument("--layers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    cfg = Config(
        output_dir=str(out), lora_mode=args.mode, epochs=2,
    )
    train(cfg, smoke_layers=args.layers, resume=args.resume)


if __name__ == "__main__":
    main()
