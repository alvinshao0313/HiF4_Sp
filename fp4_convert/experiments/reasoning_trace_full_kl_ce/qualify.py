"""Actual TP2 prefill/decode of zero-step and updated independent model exports."""
import argparse
import gc
from pathlib import Path

import torch

from ..reasoning_trace_calibration.qualify import inference
from .artifacts import verify
from .config import CANDIDATES
from .export import materialize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--objective", choices=CANDIDATES, required=True)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--zero-step", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    root, _ = verify(args.run)
    if args.checkpoint is not None:
        if not args.qualification or args.zero_step:
            parser.error("checkpoint materialization is for updated qualification models")
        # Exercise development's export-to-inference transition in one process.
        materialize(root, args.objective, checkpoint=args.checkpoint, qualification=True)
        gc.collect()
        torch.cuda.empty_cache()
    inference(root, "B_zero_step" if args.zero_step else CANDIDATES[args.objective], qualification=args.qualification)
