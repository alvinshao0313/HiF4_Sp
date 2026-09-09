#!/usr/bin/env python3
"""Run one clean MMLU-Pro300 variant into the replan tree (adopted artifacts)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner import (
    run_mmlu_pro_300_vllm,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--eval_variant", required=True, choices=["native_nvfp4", "direct_hif4", "r64_only", "artifact"])
    p.add_argument("--artifact_path", type=Path, default=None)
    p.add_argument("--model_path", default="nvidia/Qwen3-30B-A3B-NVFP4")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()
    metrics = args.output_dir / "eval/mmlu_pro/metrics.json"
    if metrics.exists():
        print(json.dumps({"status": "SKIP", "metrics": str(metrics)}, ensure_ascii=False))
        return
    if args.eval_variant == "artifact" and args.artifact_path is None:
        raise SystemExit("artifact variant requires --artifact_path")
    if args.artifact_path is not None and not args.artifact_path.exists():
        raise SystemExit(f"missing artifact: {args.artifact_path}")
    if int(args.batch_size) != 128:
        raise SystemExit(
            f"formal MMLU-Pro benchmark must use batch_size/max_num_seqs=128 to match Phase-A E0; got {args.batch_size}"
        )
    result = run_mmlu_pro_300_vllm(
        variant=args.eval_variant,
        output_dir=args.output_dir,
        model_path=args.model_path,
        artifact_path=args.artifact_path,
        artifact_diag_variant="adopted",
        batch_size=args.batch_size,
    )
    print(json.dumps({"status": "PASS", "metrics_keys": list(result.keys())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
