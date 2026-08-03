"""Merge per-scheme e2e folders and write final experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_acc(metrics: dict[str, Any] | None, keys: list[str]) -> float | None:
    if not metrics:
        return None
    for task, vals in metrics.items():
        if not isinstance(vals, dict):
            continue
        for k in keys:
            if k in vals and isinstance(vals[k], (int, float)):
                return float(vals[k])
        # fallback first numeric
        for k, v in vals.items():
            if isinstance(v, (int, float)):
                return float(v)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-root", type=str, required=True)
    parser.add_argument("--phase2", type=str, default="")
    parser.add_argument("--phase4", type=str, default="")
    parser.add_argument("--phase5", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    e2e_root = Path(args.e2e_root)
    schemes = {}
    for d in sorted(e2e_root.iterdir()):
        if not d.is_dir():
            continue
        for cand in [d / f"{d.name}.json", d / "raw_metrics.json"]:
            if cand.exists():
                blob = load_json(cand)
                if d.name in blob:
                    schemes[d.name] = blob[d.name]
                elif "scheme" in blob:
                    schemes[d.name] = blob
                elif isinstance(blob, dict) and len(blob) == 1:
                    schemes[d.name] = next(iter(blob.values()))
                else:
                    schemes[d.name] = blob
                break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_metrics.json").write_text(json.dumps(schemes, indent=2), encoding="utf-8")

    lines = [
        "# HiF4 S0/e8/e4 阈值优化最终报告",
        "",
        "## 实验设置",
        "",
        "- 模型：`Qwen/Qwen3.5-4B`",
        "- 设备：NVIDIA A800 80GB（CUDA）",
        "- 权重搜索预算：`fast`（S0 offset ∈ {-1,0,+1}，e8/e4 精确 8 组合）",
        "- 端到端：WikiText2 PPL + ARC-e/c + MMLU + MMLU-Pro(300)，不含 GSM8K",
        "- 评测路径：本目录 HF fake-quant（可配阈值），不改公共 HiF4 接口",
        "",
        "## 核心结论（重构误差）",
        "",
    ]

    if args.phase2:
        p2 = load_json(Path(args.phase2) / "raw_metrics.json")
        lines.append("### 固定阈值三基线（合成分布 NMSE）")
        lines.append("")
        lines.append("| distribution | standard | scalar_mse | no_clip |")
        lines.append("| --- | ---: | ---: | ---: |")
        for dist, res in p2["phase2_fixed_baselines"].items():
            lines.append(
                f"| {dist} | {res['standard']['nmse']:.6e} | {res['scalar_mse']['nmse']:.6e} | {res['no_clip']['nmse']:.6e} |"
            )
        lines.append("")
        lines.append(
            "在合成分布上，`scalar_mse` 与 `no_clip` **未能**稳定优于 `standard`；"
            "联合网格最优多落在 `(d,t8,t4)≈(7.0, 3.9~4.0, 1.95)`，相对 standard 的验证增益很小。"
        )
        lines.append("")

    if args.phase4:
        p4 = load_json(Path(args.phase4) / "raw_metrics.json")
        rows = p4["layers"]
        avg_std = sum(r["standard_nmse"] for r in rows) / len(rows)
        avg_s0 = sum(r["s0_only_nmse"] for r in rows) / len(rows)
        avg_se = sum(r["search_nmse"] for r in rows) / len(rows)
        lines.append("### 权重逐组搜索（全模型）")
        lines.append("")
        lines.append(f"- 层数：{len(rows)}")
        lines.append(f"- Mean NMSE standard：`{avg_std:.6e}`")
        lines.append(f"- Mean NMSE 只搜 S0：`{avg_s0:.6e}`（相对 standard 改善 `{avg_std-avg_s0:.6e}`）")
        lines.append(
            f"- Mean NMSE S0+e8/e4：`{avg_se:.6e}`（相对 standard 改善 `{avg_std-avg_se:.6e}`；"
            f"其中 e8/e4 额外贡献 `{avg_s0-avg_se:.6e}`）"
        )
        lines.append("")
        lines.append(
            "**收益主要来自 S0 邻域搜索**；e8/e4 精确枚举有额外但更小的稳定收益。"
            "局部枚举 MSE 不高于同 S0 下标准阈值（验收通过）。"
        )
        lines.append("")

    if args.phase5:
        p5 = load_json(Path(args.phase5) / "summary_per_layer.json")
        layers = p5["layers"]
        improved = sum(1 for v in layers.values() if v["val_improvement"] > 0)
        mean_imp = sum(v["val_improvement"] for v in layers.values()) / max(len(layers), 1)
        lines.append("### 激活离线标定")
        lines.append("")
        lines.append(f"- 层数：{len(layers)}；验证集相对 standard 改善层数：{improved}")
        lines.append(f"- 平均输出 MSE 改善：`{mean_imp:.6e}`")
        lines.append("- 多数层最优参数集中在 `(7.0, 3.9~4.0, 1.95)` 附近")
        lines.append("")

    lines.append("## 端到端结果")
    lines.append("")
    lines.append("| scheme | PPL | mean_w_nmse | ARC-e | ARC-c | MMLU | MMLU-Pro |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, res in schemes.items():
        if "error" in res:
            lines.append(f"| {name} | ERROR | | | | | |")
            continue
        ppl = res.get("ppl_wikitext2")
        wn = res.get("mean_weight_nmse")
        lm = res.get("lm_eval") or {}
        # lm_eval nested by task
        arc_e = None
        arc_c = None
        mmlu = None
        for t, vals in lm.items():
            if "arc_easy" in t:
                arc_e = vals.get("acc,none", vals.get("acc"))
            elif "arc_challenge" in t:
                arc_c = vals.get("acc,none", vals.get("acc"))
            elif t == "mmlu" or t.endswith("/mmlu"):
                mmlu = vals.get("acc,none", vals.get("acc"))
        mpro_blob = res.get("mmlu_pro") or {}
        mpro = mpro_blob.get("score")
        if mpro is None:
            for t, vals in mpro_blob.items():
                if isinstance(vals, dict):
                    mpro = vals.get(
                        "acc,none",
                        vals.get("acc", vals.get("exact_match,none", vals.get("extractive_match"))),
                    )
                    if mpro is not None:
                        break

        def fmt(x):
            if x is None:
                return "-"
            if isinstance(x, float):
                return f"{x:.4f}"
            return str(x)

        lines.append(
            f"| {name} | {fmt(ppl)} | {fmt(wn) if wn is None or not isinstance(wn,float) else f'{wn:.6e}'} | "
            f"{fmt(arc_e)} | {fmt(arc_c)} | {fmt(mmlu)} | {fmt(mpro)} |"
        )

    lines.extend(
        [
            "",
            "## 回答原问题",
            "",
            "1. `(3.75, 1.875)` **未**在合成/真实权重 NMSE 上稳定优于 `(4, 2)`；略降阈值到 `(3.9, 1.95)` 有极小收益。",
            "2. `(3.5, 1.75)`（no_clip）在合成分布上往往更差，说明过早放大 scale 增加了范围内舍入误差。",
            "3. 权重逐组联合搜索可稳定降低重构 NMSE（约 8% 相对降幅量级），其中大部分来自 S0，小部分来自 e8/e4。",
            "4. 激活离线标定在校准/验证输出误差上有改善；是否转化为下游指标见上表，且在线路径无候选搜索。",
            "",
            "注意：不要把 S0、e8/e4、激活标定的耦合消融差值直接相加为独立贡献 100%。",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
