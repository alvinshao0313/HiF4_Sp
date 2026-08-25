#!/usr/bin/env python3
"""Summarize whether accepted perms reduce HiF4 loss / down output NRMSE on calib activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    args = p.parse_args()

    rows = []
    for line in args.metrics.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"empty metrics: {args.metrics}")

    accepted = [r for r in rows if r.get("accepted")]
    rejected = [r for r in rows if not r.get("accepted")]

    def _rel(id_v: float, opt_v: float) -> float:
        if abs(id_v) <= 1e-12:
            return 0.0
        return (id_v - opt_v) / id_v * 100.0

    table = []
    for r in sorted(rows, key=lambda x: x.get("layer_name", "")):
        id_loss = float(r["identity_hif4_loss"])
        opt_loss = float(r["optimized_hif4_loss"])
        id_n = float(r["identity_output_nrmse"])
        opt_n = float(r["optimized_output_nrmse"])
        table.append(
            {
                "layer": r["layer_name"],
                "accepted": bool(r["accepted"]),
                "id_loss": id_loss,
                "opt_loss": opt_loss,
                "loss_rel_pct": _rel(id_loss, opt_loss),
                "id_nrmse": id_n,
                "opt_nrmse": opt_n,
                "nrmse_rel_pct": _rel(id_n, opt_n),
            }
        )

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {
        "n_layers": len(rows),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "accepted_mean_loss_rel_pct": _mean(
            [_rel(float(r["identity_hif4_loss"]), float(r["optimized_hif4_loss"])) for r in accepted]
        ),
        "accepted_mean_nrmse_rel_pct": _mean(
            [
                _rel(float(r["identity_output_nrmse"]), float(r["optimized_output_nrmse"]))
                for r in accepted
            ]
        ),
        "all_mean_loss_rel_pct": _mean([t["loss_rel_pct"] for t in table]),
        "all_mean_nrmse_rel_pct": _mean([t["nrmse_rel_pct"] for t in table]),
        "n_accepted_loss_improved": sum(1 for t in table if t["accepted"] and t["loss_rel_pct"] > 0),
        "n_accepted_nrmse_improved": sum(
            1 for t in table if t["accepted"] and t["nrmse_rel_pct"] > 0
        ),
        "layers": table,
        "note": (
            "Metrics are from the search pipeline on s1k-collected activations "
            "(search/val split inside optimize_layer_permutation). "
            "loss=full_layout_hif4_loss; nrmse=down_output_nrmse. "
            "Positive rel_pct means error decreased vs identity."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "error_audit.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# s1k 校准集上的量化误差审计",
        "",
        summary["note"],
        "",
        f"- 层数: {summary['n_layers']}",
        f"- accepted: {summary['n_accepted']} / {summary['n_layers']}",
        f"- accepted 层平均 loss 相对下降: {summary['accepted_mean_loss_rel_pct']:.2f}%",
        f"- accepted 层平均 down_output_nrmse 相对下降: {summary['accepted_mean_nrmse_rel_pct']:.2f}%",
        f"- accepted 且 loss 下降: {summary['n_accepted_loss_improved']}",
        f"- accepted 且 nrmse 下降: {summary['n_accepted_nrmse_improved']}",
        "",
        "| layer | accepted | id_loss | opt_loss | loss↓% | id_nrmse | opt_nrmse | nrmse↓% |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in table:
        lines.append(
            f"| {t['layer']} | {t['accepted']} | {t['id_loss']:.6f} | {t['opt_loss']:.6f} | "
            f"{t['loss_rel_pct']:.2f} | {t['id_nrmse']:.6f} | {t['opt_nrmse']:.6f} | "
            f"{t['nrmse_rel_pct']:.2f} |"
        )
    md = "\n".join(lines) + "\n"
    (args.output_dir / "error_audit.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
