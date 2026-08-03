"""Master pipeline for threshold optimization experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PY = Path(sys.executable)
_SCRIPTS = _ROOT / "scripts"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--stages",
        type=str,
        default="synthetic,weight_sample,act,e2e",
        help="Comma list: synthetic,weight_sample,weight_all,act,e2e",
    )
    parser.add_argument("--skip-downstream", action="store_true")
    parser.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else _ROOT / "results" / f"{stamp}_pipeline"
    out_root.mkdir(parents=True, exist_ok=True)
    env = {"CUDA_VISIBLE_DEVICES": args.gpu}

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    meta = {"stages": stages, "model": args.model, "out_root": str(out_root)}
    (out_root / "pipeline_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    fixed_d, fixed_t8, fixed_t4 = 7.0, 3.75, 1.875
    act_map = None

    if "synthetic" in stages:
        d = out_root / "phase2_3"
        run(
            [
                str(_PY),
                str(_SCRIPTS / "run_synthetic.py"),
                "--rows",
                "64",
                "--cols",
                "2048",
                "--device",
                args.device,
                "--model",
                args.model,
                "--out-dir",
                str(d),
            ]
        )
        # Read best fixed from gaussian joint if present
        raw = json.loads((d / "raw_metrics.json").read_text(encoding="utf-8"))
        try:
            best = raw["phase3_grid"]["by_distribution"]["gaussian"]["joint_cal"]["best"]
            fixed_d = best["s0_divisor"]
            fixed_t8 = best["e8_threshold"]
            fixed_t4 = best["e4_threshold"]
            print(f"Using cal-best fixed params from gaussian: {(fixed_d, fixed_t8, fixed_t4)}")
        except Exception as e:
            print(f"Keep default fixed best; parse failed: {e}")

    if "weight_sample" in stages or "weight_all" in stages:
        d = out_root / "phase4_weight"
        layers = "all" if "weight_all" in stages else "sample"
        cmd = [
            str(_PY),
            str(_SCRIPTS / "run_weight_search.py"),
            "--model",
            args.model,
            "--device",
            args.device,
            "--budget",
            "fast",
            "--layers",
            layers,
            "--fixed-best-d",
            str(fixed_d),
            "--fixed-best-t8",
            str(fixed_t8),
            "--fixed-best-t4",
            str(fixed_t4),
            "--out-dir",
            str(d),
        ]
        if layers == "all":
            cmd.append("--save-state")
        run(cmd)

    if "act" in stages:
        d_stats = out_root / "phase5_act_stats"
        run(
            [
                str(_PY),
                str(_SCRIPTS / "collect_activation_stats.py"),
                "--model",
                args.model,
                "--device",
                args.device,
                "--max-rows",
                "256",
                "--n-samples",
                "16",
                "--seqlen",
                "512",
                "--split",
                "train",
                "--out-dir",
                str(d_stats),
            ]
        )
        d_cal = out_root / "phase5_act_calib"
        run(
            [
                str(_PY),
                str(_SCRIPTS / "calibrate_activation_params.py"),
                "--store",
                str(d_stats / "activation_store.pt"),
                "--granularity",
                "all",
                "--device",
                args.device,
                "--out-dir",
                str(d_cal),
            ]
        )
        act_map = str(d_cal / "param_map_per_layer.pt")

    if "e2e" in stages:
        d = out_root / "phase6_e2e"
        cmd = [
            str(_PY),
            str(_SCRIPTS / "evaluate_model.py"),
            "--model",
            args.model,
            "--device",
            args.device,
            "--fixed-best-d",
            str(fixed_d),
            "--fixed-best-t8",
            str(fixed_t8),
            "--fixed-best-t4",
            str(fixed_t4),
            "--out-dir",
            str(d),
            "--lm-batch-size",
            "2",
            "--mmlu-pro-limit",
            "300",
        ]
        if act_map:
            cmd.extend(["--act-param-map", act_map])
        if args.skip_downstream:
            cmd.append("--skip-downstream")
        run(cmd)

    print(f"Pipeline complete: {out_root}")


if __name__ == "__main__":
    main()
