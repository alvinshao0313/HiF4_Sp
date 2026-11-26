"""Finite CPU conversion queue; fail on the first incomplete candidate."""
import argparse
import os
from pathlib import Path

from .artifacts import save
from .convert import SOURCE, convert
from .format import VARIANTS


def run(root, lane):
    root = Path(root)
    status = root / f"conversion_lane{lane}.json"
    if status.exists():
        raise FileExistsError(status)
    state = {"status": "running", "pid": os.getpid(), "completed": []}
    save(status, state)
    try:
        for variant in VARIANTS[lane::2]:
            state["current"] = variant
            save(status, state)
            convert(SOURCE, root / "weights" / variant, variant, 8)
            state["completed"].append(variant)
            save(status, state)
        state["status"] = "completed"
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(status, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--lane", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    run(args.root, args.lane)
