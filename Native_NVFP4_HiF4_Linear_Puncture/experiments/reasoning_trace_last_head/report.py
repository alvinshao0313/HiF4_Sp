"""Question-paired comparisons to original B, with four- and three-task losses."""
import argparse
from pathlib import Path

import numpy as np

from ..reasoning_trace_calibration.report import paired
from ..reasoning_trace_router_bf16.report import scores
from .artifacts import read_json, verify, write_json
from .config import BASELINE, CANDIDATE, ORIGIN


def without_gsm(left, right, repeats=10000):
    rng = np.random.default_rng(42)
    tasks = [k for k in BASELINE if k != "GSM8K"]
    differences = np.zeros(repeats)
    values = []
    for k in tasks:
        if left[k].keys() != right[k].keys():
            raise ValueError("three-task paired IDs differ")
        ids = sorted(left[k])
        a, b = np.array([left[k][i] for i in ids]), np.array([right[k][i] for i in ids])
        values.append((max(0., BASELINE[k] - a.mean()) * 100, max(0., BASELINE[k] - b.mean()) * 100))
        for start in range(0, repeats, 100):
            indices = rng.integers(0, len(ids), (min(100, repeats - start), len(ids)))
            differences[start:start + len(indices)] += (np.maximum(0, BASELINE[k] - b[indices].mean(1))
                                                        - np.maximum(0, BASELINE[k] - a[indices].mean(1))) * 100 / 3
    a, b = np.array(values).mean(0)
    return {"tasks": tasks, "B_mean_positive_loss_pp": float(a), "candidate_mean_positive_loss_pp": float(b),
            "mean_positive_loss_change_pp": float(b - a), "paired_95pct_pp": np.quantile(differences, [.025, .975]).tolist(),
            "at_most_0_5_pp": bool(b <= .5 + 1e-12)}


def summarize(root):
    root, m = verify(root)
    a, left = scores(Path(m["source_run"]) / "evaluation" / ORIGIN)
    b, right = scores(root / "evaluation" / CANDIDATE)
    result = {"complete": True, "baseline": BASELINE, "models": {ORIGIN: a, CANDIDATE: b},
              "candidate_minus_B": paired(left, right), "without_GSM8K": without_gsm(left, right), "protocol": m["protocol"]}
    write_json(result, root / "comparison.json")
    lines = ["# B 末层与输出端词表 KL 蒸馏", "", "完整 TP2 评测；AIME 为已复核最终提交 avg5。损失单位为百分点。", "",
             "| 模型 | MMLU-Pro | LCB | GSM8K | AIME25 avg5 | 四项平均正向损失 | ≤0.5pp |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    lines.append("| 旧原生基线 | " + " | ".join(f"{v * 100:.4f}%" for v in BASELINE.values()) + " | 0 | 是 |")
    for name, row in result["models"].items():
        lines.append(f"| {name} | " + " | ".join(f"{v * 100:.4f}%" for v in row["accuracy"].values())
                     + f" | {row['mean_positive_loss_pp']:.4f} | {'是' if row['at_most_0_5_pp'] else '否'} |")
    change, secondary = result["candidate_minus_B"], result["without_GSM8K"]
    lines += ["", f"四项损失变化 {change['mean_positive_loss_change_pp']:+.4f} pp，配对 95% 区间 {change['paired_95pct_pp']}。",
              f"去掉 GSM8K：B {secondary['B_mean_positive_loss_pp']:.4f} pp，新候选 {secondary['candidate_mean_positive_loss_pp']:.4f} pp；变化 {secondary['mean_positive_loss_change_pp']:+.4f} pp，区间 {secondary['paired_95pct_pp']}。",
              "", "区间跨 0 不宣称稳定改善。AIME 抽样单位为题目 avg5；区间不覆盖换种子及重新训练的方差。局部词表 KL 不代表下游收益。",
              "", "逐任务变化及区间、ARC 辅助指标见 [comparison.json](comparison.json)。", ""]
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    summarize(parser.parse_args().run)
