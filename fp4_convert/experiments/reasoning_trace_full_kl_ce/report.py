"""Both candidates versus B and each other, with corrected avg5 and paired intervals."""
import argparse
from pathlib import Path

from ..reasoning_trace_calibration.report import paired
from ..reasoning_trace_last_head.report import without_gsm
from ..reasoning_trace_router_bf16.report import scores
from .artifacts import read_json, verify, write_json
from .config import BASELINE, CANDIDATES, ORIGIN


def summarize(root):
    root, m = verify(root)
    origin, origin_rows = scores(Path(m["source_run"]) / "evaluation" / ORIGIN)
    models, rows, changes = {ORIGIN: origin}, {ORIGIN: origin_rows}, {}
    for objective, candidate in CANDIDATES.items():
        selection = read_json(root / "development" / objective / "selection.json")
        if not selection["complete"] or selection["formal_tests_used_for_selection"]:
            raise ValueError("final candidate was not locked independently of formal tests")
        models[candidate], rows[candidate] = scores(root / "evaluation" / candidate)
        changes[candidate] = {"minus_B": paired(origin_rows, rows[candidate]),
                              "without_GSM8K": without_gsm(origin_rows, rows[candidate])}
    full, ce = CANDIDATES.values()
    result = {"complete": True, "baseline": BASELINE, "models": models, "comparisons": changes,
              "hard_CE_minus_full_KL": paired(rows[full], rows[ce]),
              "hard_CE_minus_full_KL_without_GSM8K": without_gsm(rows[full], rows[ce]),
              "protocol": m["protocol"], "selection_policy": m["selection_policy"],
              "top_mass_reference": read_json(Path(m["cache_run"]) / "comparison.json")}
    write_json(result, root / "comparison.json")
    lines = ["# 原 B 的完整词表 KL 与轨迹硬 CE 对照", "", "完整 TP2，AIME 使用已复核的最终提交 avg5；平均正向损失单位为百分点。", "",
             "| 模型 | MMLU-Pro | LCB | GSM8K | AIME25 avg5 | 四项平均正向损失 | ≤0.5pp |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    lines.append("| 旧原生基线 | " + " | ".join(f"{v*100:.4f}%" for v in BASELINE.values()) + " | 0 | 是 |")
    for name, row in models.items():
        lines.append(f"| {name} | " + " | ".join(f"{v*100:.4f}%" for v in row["accuracy"].values())
                     + f" | {row['mean_positive_loss_pp']:.4f} | {'是' if row['at_most_0_5_pp'] else '否'} |")
    for candidate, value in changes.items():
        change, secondary = value["minus_B"], value["without_GSM8K"]
        lines += ["", f"{candidate} 减 B：四项平均损失变化 {change['mean_positive_loss_change_pp']:+.4f} pp，配对 95% 区间 {change['paired_95pct_pp']}。",
                  f"去掉 GSM8K：{secondary['candidate_mean_positive_loss_pp']:.4f} pp，变化 {secondary['mean_positive_loss_change_pp']:+.4f} pp，区间 {secondary['paired_95pct_pp']}。"]
    lines += ["", "区间跨 0 不宣称稳定改善。AIME 按题目 avg5 抽样；区间不覆盖换种子或重新训练方差。",
              "", "逐任务变化、辅助 ARC、两组直接配对比较、选择记录及原 top-mass 参考见 [comparison.json](comparison.json)。", ""]
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    summarize(parser.parse_args().run)
