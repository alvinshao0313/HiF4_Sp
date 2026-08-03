"""Phase 2+3: fixed baselines and (d,t8,t4) grid search on synthetic + optional weights."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.fixed_thresholds import FIXED_BASELINES, get_baseline_config  # noqa: E402
from src.metrics import detailed_quant_metrics, evaluate_config  # noqa: E402
from src.quantizer import HiF4QuantConfig, quantize_hif4  # noqa: E402
from src.synthetic import DISTRIBUTIONS, make_matrix  # noqa: E402


def _env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
    try:
        import subprocess

        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        info["git_commit"] = r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        info["git_commit"] = None
    return info


def run_fixed_baselines(
    rows: int,
    cols: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dist in DISTRIBUTIONS:
        x = make_matrix(dist, rows, cols, seed, device=device)
        std = quantize_hif4(x, config=get_baseline_config("standard"))
        dist_res = {}
        for name, cfg in FIXED_BASELINES.items():
            result = quantize_hif4(x, config=cfg)
            m = detailed_quant_metrics(x, result, reference_result=std)
            m["s0_divisor"] = cfg.s0_divisor
            m["e8_threshold"] = cfg.e8_threshold
            m["e4_threshold"] = cfg.e4_threshold
            dist_res[name] = m
        out[dist] = dist_res
    return out


def _grid_values(start: float, stop: float, step: float) -> list[float]:
    vals: list[float] = []
    x = start
    # inclusive stop with float-safe loop
    while x <= stop + 1e-9:
        vals.append(round(x, 10))
        x += step
    return vals


def run_divisor_sweep(
    x: torch.Tensor,
    *,
    t8: float = 4.0,
    t4: float = 2.0,
) -> list[dict[str, Any]]:
    rows = []
    for d in _grid_values(5.5, 7.5, 0.25):
        cfg = HiF4QuantConfig(s0_divisor=d, e8_threshold=t8, e4_threshold=t4)
        m = evaluate_config(x, cfg)
        rows.append(m)
    return rows


def run_joint_grid(x: torch.Tensor) -> dict[str, Any]:
    ds = _grid_values(5.5, 7.5, 0.25)
    t8s = _grid_values(3.4, 4.1, 0.1)
    t4s = _grid_values(1.70, 2.05, 0.05)
    best = None
    best_nmse = float("inf")
    all_rows: list[dict[str, Any]] = []
    # Vectorize over candidates by looping configs but reusing x on device.
    for d, t8, t4 in product(ds, t8s, t4s):
        cfg = HiF4QuantConfig(s0_divisor=d, e8_threshold=t8, e4_threshold=t4)
        m = evaluate_config(x, cfg)
        all_rows.append(m)
        if m["nmse"] < best_nmse:
            best_nmse = m["nmse"]
            best = m
    return {"best": best, "num_candidates": len(all_rows), "top5": sorted(all_rows, key=lambda r: r["nmse"])[:5]}


def maybe_load_weight_slices(model_name: str | None, device: str) -> dict[str, torch.Tensor]:
    if not model_name:
        return {}
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    slices: dict[str, torch.Tensor] = {}
    named = dict(model.named_parameters())
    # Sample early / mid / late linear weights.
    linear_names = [
        n
        for n, p in named.items()
        if p.ndim == 2 and p.shape[-1] % 64 == 0 and any(
            k in n
            for k in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj",
            )
        )
    ]
    if not linear_names:
        return {}
    picks = []
    picks.append(linear_names[0])
    picks.append(linear_names[len(linear_names) // 2])
    picks.append(linear_names[-1])
    for n in picks:
        w = named[n].detach().float()
        # Take first 64 rows for speed if huge.
        if w.shape[0] > 64:
            w = w[:64]
        slices[n] = w.to(device)
    del model
    return slices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", type=str, default=None, help="Optional HF model for real weight slices")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "results" / f"{stamp}_phase2_phase3_synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "rows": args.rows,
        "cols": args.cols,
        "seed": args.seed,
        "device": args.device,
        "model": args.model,
    }
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps(_env_info(), indent=2), encoding="utf-8")

    print("[phase2] fixed baselines on synthetic distributions...")
    phase2 = run_fixed_baselines(args.rows, args.cols, args.seed, args.device)

    print("[phase3] divisor sweep + joint grid per distribution...")
    phase3: dict[str, Any] = {"by_distribution": {}}
    for dist in DISTRIBUTIONS:
        x = make_matrix(dist, args.rows, args.cols, args.seed + 1, device=args.device)
        # Split calib/val by rows
        mid = args.rows // 2
        x_cal, x_val = x[:mid], x[mid:]
        cal_joint = run_joint_grid(x_cal)
        # Evaluate cal-best on val
        b = cal_joint["best"]
        assert b is not None
        val_cfg = HiF4QuantConfig(
            s0_divisor=b["s0_divisor"],
            e8_threshold=b["e8_threshold"],
            e4_threshold=b["e4_threshold"],
        )
        val_m = evaluate_config(x_val, val_cfg)
        std_val = evaluate_config(x_val, get_baseline_config("standard"))
        phase3["by_distribution"][dist] = {
            "divisor_sweep_cal": run_divisor_sweep(x_cal),
            "joint_cal": cal_joint,
            "val_with_cal_best": val_m,
            "val_standard": std_val,
            "generalization_nmse_delta": val_m["nmse"] - std_val["nmse"],
        }

    weight_results: dict[str, Any] = {}
    if args.model:
        print(f"[phase2/3] real weight slices from {args.model}...")
        slices = maybe_load_weight_slices(args.model, args.device)
        for name, w in slices.items():
            std = quantize_hif4(w, config=get_baseline_config("standard"))
            base = {}
            for bname, cfg in FIXED_BASELINES.items():
                r = quantize_hif4(w, config=cfg)
                base[bname] = detailed_quant_metrics(w, r, reference_result=std)
            joint = run_joint_grid(w)
            weight_results[name] = {"baselines": base, "joint": joint}

    raw = {
        "phase2_fixed_baselines": phase2,
        "phase3_grid": phase3,
        "real_weight_slices": weight_results,
    }
    (out_dir / "raw_metrics.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

    # Summary markdown
    lines = [
        "# Phase 2/3 Summary",
        "",
        f"Device: `{args.device}`",
        f"Shape: `{args.rows} x {args.cols}`",
        "",
        "## Fixed baselines (NMSE)",
        "",
        "| distribution | standard | scalar_mse | no_clip |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dist, res in phase2.items():
        lines.append(
            f"| {dist} | {res['standard']['nmse']:.6e} | {res['scalar_mse']['nmse']:.6e} | {res['no_clip']['nmse']:.6e} |"
        )
    lines.extend(["", "## Joint grid best (calib) vs standard on val", ""])
    lines.append("| distribution | best (d,t8,t4) | val_nmse_best | val_nmse_std | delta |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for dist, res in phase3["by_distribution"].items():
        b = res["joint_cal"]["best"]
        lines.append(
            f"| {dist} | ({b['s0_divisor']}, {b['e8_threshold']}, {b['e4_threshold']}) | "
            f"{res['val_with_cal_best']['nmse']:.6e} | {res['val_standard']['nmse']:.6e} | "
            f"{res['generalization_nmse_delta']:.6e} |"
        )
    if weight_results:
        lines.extend(["", "## Real weight slices (joint best NMSE)", ""])
        for name, res in weight_results.items():
            b = res["joint"]["best"]
            lines.append(
                f"- `{name}`: best NMSE={b['nmse']:.6e} at (d,t8,t4)=({b['s0_divisor']},{b['e8_threshold']},{b['e4_threshold']}); "
                f"standard={res['baselines']['standard']['nmse']:.6e}"
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
