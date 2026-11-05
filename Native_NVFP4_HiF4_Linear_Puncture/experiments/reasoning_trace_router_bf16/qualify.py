"""Use the established real TP2 engine on isolated model views."""
import argparse
from pathlib import Path

from ..reasoning_trace_calibration.qualify import inference
from .artifacts import verify
from .config import CANDIDATE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--zero-step", action="store_true")
    args = parser.parse_args()
    root, _ = verify(args.run)
    if args.zero_step and not args.qualification:
        parser.error("zero-step is a qualification model")
    inference(root, "B_zero_step" if args.zero_step else CANDIDATE, qualification=args.qualification)
