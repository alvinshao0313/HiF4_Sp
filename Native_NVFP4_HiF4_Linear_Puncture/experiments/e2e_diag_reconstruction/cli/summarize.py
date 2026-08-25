"""CLI to summarize one or more reconstruction runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.summarize import (
    summarize_runs,
    write_chinese_report,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import write_json


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize e2e diag reconstruction runs")
    p.add_argument(
        "--run_map",
        type=str,
        required=True,
        help="JSON object mapping method name -> run directory",
    )
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--output_md", type=str, default="")
    p.add_argument("--nvfp4_method", type=str, default="E0_native_nvfp4")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raw = args.run_map
    path = Path(raw)
    if path.is_file():
        run_map = json.loads(path.read_text(encoding="utf-8"))
    else:
        run_map = json.loads(raw)
    summary = summarize_runs(run_map, nvfp4_method=args.nvfp4_method)
    write_json(args.output_json, summary)
    if args.output_md:
        write_chinese_report(summary, Path(args.output_md))


if __name__ == "__main__":
    main()
