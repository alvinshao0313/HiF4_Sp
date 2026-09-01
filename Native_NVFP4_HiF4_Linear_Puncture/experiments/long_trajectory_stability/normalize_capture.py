#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import normalize_capture


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capture_dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    path = normalize_capture(Path(args.capture_dir), Path(args.output))
    print(path)


if __name__ == "__main__":
    main()
