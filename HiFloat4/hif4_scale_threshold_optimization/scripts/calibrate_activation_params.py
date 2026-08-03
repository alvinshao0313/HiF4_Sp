"""Calibrate per-layer / per-type / global activation (d,t8,t4)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.activation_calibration import calibrate_activations  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=str, required=True, help="activation_store.pt path")
    parser.add_argument(
        "--granularity",
        type=str,
        default="all",
        choices=["global", "per_module_type", "per_layer", "all"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "results" / f"{stamp}_act_calib"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(vars(args)), encoding="utf-8")

    blob = torch.load(args.store, map_location="cpu", weights_only=False)
    inputs = blob["inputs"]
    energy = blob["weight_col_energy"]

    gran_list = (
        ["global", "per_module_type", "per_layer"]
        if args.granularity == "all"
        else [args.granularity]
    )
    all_out = {}
    for g in gran_list:
        print(f"Calibrating granularity={g} ...")
        result = calibrate_activations(
            inputs, energy, granularity=g, device=args.device  # type: ignore[arg-type]
        )
        all_out[g] = result["summary"]
        # Save param map for per_layer as default for eval
        if g == "per_layer":
            torch.save(result["param_map"], out_dir / "param_map_per_layer.pt")
        (out_dir / f"summary_{g}.json").write_text(
            json.dumps(result["summary"], indent=2), encoding="utf-8"
        )

    (out_dir / "raw_metrics.json").write_text(json.dumps(all_out, indent=2), encoding="utf-8")

    # Markdown summary for per_layer
    if "per_layer" in all_out:
        layers = all_out["per_layer"]["layers"]
        improved = sum(1 for v in layers.values() if v["val_improvement"] > 0)
        mean_imp = sum(v["val_improvement"] for v in layers.values()) / max(len(layers), 1)
        lines = [
            "# Activation Calibration Summary",
            "",
            f"Layers: {len(layers)}",
            f"Layers improved on val vs standard: {improved}",
            f"Mean val output-MSE improvement: {mean_imp:.6e}",
            "",
        ]
        # show top/bottom
        ranked = sorted(layers.items(), key=lambda kv: kv[1]["val_improvement"], reverse=True)
        lines.append("## Top improvements")
        for n, v in ranked[:10]:
            b = v["best"]
            lines.append(
                f"- `{n}`: Δ={v['val_improvement']:.6e} "
                f"params=({b['s0_divisor']},{b['e8_threshold']},{b['e4_threshold']})"
            )
        (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
