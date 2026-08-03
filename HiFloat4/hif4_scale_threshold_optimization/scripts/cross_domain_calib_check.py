"""Cross-domain check: wikitext-calibrated vs s1k-calibrated per-layer params.

For each layer, evaluate on the validation split of BOTH stores:
  standard (7,4,2) / params calibrated on wikitext / params calibrated on s1k.
Report mean activation NMSE and col-energy weighted output MSE per domain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.activation_calibration import _split_rows  # noqa: E402
from src.metrics import nmse  # noqa: E402
from src.quantizer import HiF4QuantConfig, quantize_hif4  # noqa: E402


def eval_domain(
    inputs: dict[str, torch.Tensor],
    energy: dict[str, torch.Tensor],
    maps: dict[str, dict[str, HiF4QuantConfig] | None],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    acc: dict[str, dict[str, list[float]]] = {k: {"nmse": [], "wmse": []} for k in maps}
    for name in sorted(inputs):
        if name not in energy:
            continue
        x = inputs[name].float()
        e = energy[name].float()
        _, xv = _split_rows(x)
        xv = xv.to(device)
        ev = e.to(device)
        for tag, pm in maps.items():
            cfg = HiF4QuantConfig() if pm is None else pm.get(name, HiF4QuantConfig())
            xq = quantize_hif4(xv, config=cfg).reconstruction
            acc[tag]["nmse"].append(nmse(xv, xq))
            acc[tag]["wmse"].append(
                float(((xv - xq).pow(2) * ev.view(1, -1)).sum().item())
            )
    return {
        tag: {
            "nmse": sum(v["nmse"]) / max(len(v["nmse"]), 1),
            "wmse": sum(v["wmse"]) / max(len(v["wmse"]), 1),
            "layers": len(v["nmse"]),
        }
        for tag, v in acc.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt-store", type=str, required=True)
    parser.add_argument("--s1k-store", type=str, required=True)
    parser.add_argument("--wt-map", type=str, required=True)
    parser.add_argument("--s1k-map", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wt = torch.load(args.wt_store, map_location="cpu", weights_only=False)
    s1k = torch.load(args.s1k_store, map_location="cpu", weights_only=False)
    wt_map = torch.load(args.wt_map, map_location="cpu", weights_only=False)
    s1k_map = torch.load(args.s1k_map, map_location="cpu", weights_only=False)

    maps = {"standard": None, "wt_calib": wt_map, "s1k_calib": s1k_map}
    res_wt = eval_domain(wt["inputs"], wt["weight_col_energy"], maps, device)
    res_s1k = eval_domain(s1k["inputs"], s1k["weight_col_energy"], maps, device)

    raw = {"val_on_wikitext": res_wt, "val_on_s1k": res_s1k}
    (out_dir / "raw_metrics.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

    lines = [
        "# 跨域标定检验（wikitext 标定 vs s1k 标定）",
        "",
        "每个方案在两份验证激活上分别评估；数值为 128 层平均。",
        "",
        "## 在 WikiText 验证激活上",
        "",
        "| 参数来源 | mean NMSE | mean 加权输出 MSE |",
        "| --- | ---: | ---: |",
    ]
    for tag in ("standard", "wt_calib", "s1k_calib"):
        v = res_wt[tag]
        lines.append(f"| {tag} | {v['nmse']:.6e} | {v['wmse']:.6e} |")
    lines += [
        "",
        "## 在 S1K 验证激活上",
        "",
        "| 参数来源 | mean NMSE | mean 加权输出 MSE |",
        "| --- | ---: | ---: |",
    ]
    for tag in ("standard", "wt_calib", "s1k_calib"):
        v = res_s1k[tag]
        lines.append(f"| {tag} | {v['nmse']:.6e} | {v['wmse']:.6e} |")

    # param distribution comparison
    def dist(m):
        from collections import Counter

        c = Counter(
            (round(v.s0_divisor, 2), round(v.e8_threshold, 2), round(v.e4_threshold, 2))
            for v in m.values()
        )
        return c.most_common(8)

    lines += [
        "",
        "## 参数分布 top（(d,t8,t4): 层数）",
        "",
        f"- wikitext 标定：{dist(wt_map)}",
        f"- s1k 标定：{dist(s1k_map)}",
        "",
    ]
    (out_dir / "cross_domain.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
