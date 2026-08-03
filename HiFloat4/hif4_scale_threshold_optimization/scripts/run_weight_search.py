"""Run weight search on Qwen linear layers (sample or full)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_HIFLOAT4 = _ROOT.parent
for p in (_ROOT, _HIFLOAT4):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.fixed_thresholds import get_baseline_config  # noqa: E402
from src.metrics import detailed_quant_metrics, nmse  # noqa: E402
from src.model_hooks import iter_target_linears, module_type_of  # noqa: E402
from src.quantizer import HiF4QuantConfig, quantize_hif4  # noqa: E402
from src.weight_search import search_weight_groups, standard_rtn_quantize  # noqa: E402


def _env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def quantize_weight_tensor(w: torch.Tensor, device: str) -> tuple[torch.Tensor, int]:
    """Return contiguous float weight with group dim last, and whether transposed."""
    if w.shape[-1] % 64 == 0:
        return w.float().contiguous(), 0
    if w.shape[0] % 64 == 0:
        return w.float().T.contiguous(), 1
    raise ValueError(f"weight shape {tuple(w.shape)} not divisible by 64 on either dim")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--budget", type=str, default="fast", choices=["fast", "full"])
    parser.add_argument(
        "--layers",
        type=str,
        default="sample",
        help="sample | all | comma-separated names",
    )
    parser.add_argument("--fixed-best-d", type=float, default=None)
    parser.add_argument("--fixed-best-t8", type=float, default=None)
    parser.add_argument("--fixed-best-t4", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--save-state", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "results" / f"{stamp}_phase4_weight_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(vars(args)), encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps(_env_info(), indent=2), encoding="utf-8")

    from transformers import AutoModelForCausalLM

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu"
    )
    targets = iter_target_linears(model)
    if args.layers == "sample":
        # early / mid / late of each module type when possible
        by_type: dict[str, list] = {}
        for n, m in targets:
            by_type.setdefault(module_type_of(n) or "other", []).append((n, m))
        selected = []
        for mt, items in by_type.items():
            selected.append(items[0])
            if len(items) > 2:
                selected.append(items[len(items) // 2])
            if len(items) > 1:
                selected.append(items[-1])
        # unique
        seen = set()
        targets = []
        for n, m in selected:
            if n not in seen:
                targets.append((n, m))
                seen.add(n)
    elif args.layers != "all":
        allow = set(args.layers.split(","))
        targets = [(n, m) for n, m in targets if n in allow]

    print(f"Searching {len(targets)} layers, budget={args.budget}")
    rows: list[dict[str, Any]] = []
    state_dict_updates: dict[str, torch.Tensor] = {}

    fixed_cfg = None
    if args.fixed_best_d is not None:
        fixed_cfg = HiF4QuantConfig(
            s0_divisor=args.fixed_best_d,
            e8_threshold=args.fixed_best_t8 or 4.0,
            e4_threshold=args.fixed_best_t4 or 2.0,
        )

    t_all = time.perf_counter()
    for name, mod in targets:
        w = mod.weight.data
        try:
            wf, transposed = quantize_weight_tensor(w, args.device)
        except ValueError:
            continue
        wf = wf.to(args.device)
        std_recon = standard_rtn_quantize(wf)
        std_nmse = nmse(wf, std_recon)

        baselines = {}
        for bname in ("standard", "scalar_mse", "no_clip"):
            cfg = get_baseline_config(bname)
            r = quantize_hif4(wf, config=cfg)
            baselines[bname] = detailed_quant_metrics(wf, r)

        fixed_m = None
        if fixed_cfg is not None:
            fr = quantize_hif4(wf, config=fixed_cfg)
            fixed_m = detailed_quant_metrics(wf, fr)

        s0_only = search_weight_groups(
            wf, budget=args.budget, enumerate_e8_e4=False, device=args.device
        )
        full_search = search_weight_groups(
            wf, budget=args.budget, enumerate_e8_e4=True, device=args.device
        )
        full_budget = None
        if args.budget == "fast":
            # Also report full budget on this layer for ablation when doing sample
            full_budget = search_weight_groups(
                wf, budget="full", enumerate_e8_e4=True, device=args.device
            )

        recon = full_search.reconstruction
        if transposed:
            recon_w = recon.T.contiguous()
        else:
            recon_w = recon
        state_dict_updates[name] = recon_w.to(dtype=w.dtype).cpu()
        # write into model if saving
        mod.weight.data.copy_(recon_w.to(dtype=w.dtype, device=w.device))

        row = {
            "name": name,
            "module_type": module_type_of(name),
            "shape": list(w.shape),
            "standard_nmse": std_nmse,
            "baselines": {k: v["nmse"] for k, v in baselines.items()},
            "fixed_best_nmse": None if fixed_m is None else fixed_m["nmse"],
            "s0_only_nmse": s0_only.nmse,
            "search_nmse": full_search.nmse,
            "search_elapsed_s": full_search.elapsed_s,
            "search_groups_per_s": full_search.groups_per_second,
            "search_peak_mem": full_search.peak_memory_bytes,
            "search_chunk": full_search.group_chunk_size,
            "improvement_vs_standard": std_nmse - full_search.nmse,
            "improvement_from_e8e4": s0_only.nmse - full_search.nmse,
        }
        if full_budget is not None:
            row["full_budget_nmse"] = full_budget.nmse
            row["full_budget_elapsed_s"] = full_budget.elapsed_s
            row["fast_vs_full_nmse_delta"] = full_search.nmse - full_budget.nmse
        rows.append(row)
        print(
            f"  {name}: std={std_nmse:.6e} search={full_search.nmse:.6e} "
            f"gain={std_nmse - full_search.nmse:.6e} ({full_search.groups_per_second:.0f} g/s)"
        )

    elapsed = time.perf_counter() - t_all
    raw = {"layers": rows, "elapsed_s": elapsed, "budget": args.budget}
    (out_dir / "raw_metrics.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

    if args.save_state:
        torch.save(state_dict_updates, out_dir / "weight_recon_updates.pt")
        model.save_pretrained(out_dir / "model_weight_search")
        print(f"Saved model to {out_dir / 'model_weight_search'}")

    # summary
    if rows:
        avg_std = sum(r["standard_nmse"] for r in rows) / len(rows)
        avg_search = sum(r["search_nmse"] for r in rows) / len(rows)
        avg_s0 = sum(r["s0_only_nmse"] for r in rows) / len(rows)
        lines = [
            "# Phase 4 Weight Search Summary",
            "",
            f"Layers: {len(rows)}",
            f"Budget: {args.budget}",
            f"Device: {args.device}",
            f"Total time (incl. baselines): {elapsed:.2f}s",
            "",
            f"Mean NMSE standard: {avg_std:.6e}",
            f"Mean NMSE S0-only search: {avg_s0:.6e}",
            f"Mean NMSE S0+e8/e4 search: {avg_search:.6e}",
            f"Mean gain vs standard: {avg_std - avg_search:.6e}",
            f"Mean gain from e8/e4 given S0 search: {avg_s0 - avg_search:.6e}",
            "",
        ]
        (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
