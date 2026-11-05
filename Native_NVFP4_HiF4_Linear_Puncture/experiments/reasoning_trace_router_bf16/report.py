"""B versus its BF16 router-aligned successor; question-paired intervals."""
import argparse
from pathlib import Path

from ..reasoning_trace_calibration.report import loss, paired
from .artifacts import read_json, verify, write_json
from .config import BASELINE, CANDIDATE, ORIGIN


def scores(directory):
    values, samples = {}, {}
    for task, label, count in (("mmlu_pro", "MMLU-Pro", 12032), ("lcb", "LCB", 175), ("gsm8k", "GSM8K", 1319)):
        values[label] = read_json(directory / task / "verified_scores.json")[label]
        samples[label] = read_json(directory / task / "paired_scores.json")
        if len(samples[label]) != count or abs(sum(samples[label].values()) / count - values[label]) > 1e-12:
            raise ValueError("incomplete full paired reasoning results")
    aime = read_json(directory / "aime25/corrected_metrics.json")
    responses = read_json(directory / "aime25/corrected_responses.json")
    if (not aime["complete"] or aime["responses"] != 150 or len(aime["per_question"]) != 30
            or responses["unresolved"]):
        raise ValueError("AIME requires complete, reviewed avg5")
    values["AIME25 avg5"], samples["AIME25 avg5"] = aime["score"], aime["per_question"]
    return {"accuracy": values, "auxiliary": read_json(directory / "arc/verified_scores.json"), **loss(values)}, samples


def summarize(root):
    root, manifest = verify(root)
    a, left = scores(Path(manifest["source_run"]) / "evaluation" / ORIGIN)
    b, right = scores(root / "evaluation" / CANDIDATE)
    result = {"complete": True, "baseline": BASELINE, "models": {ORIGIN: a, CANDIDATE: b},
              "B_router_minus_B": paired(left, right), "protocol": manifest["protocol"]}
    write_json(result, root / "comparison.json")
    lines = ["# B 的 BF16 Router 对齐结果", "", "AIME 使用纠正后的最终提交 avg5；四项 reasoning 平均正向损失单位为百分点。", "",
             "| 模型 | MMLU-Pro | LCB | GSM8K | AIME25 avg5 | 平均正向损失 | ≤0.5pp |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    lines.append("| 旧原生基线 | " + " | ".join(f"{x * 100:.4f}%" for x in BASELINE.values()) + " | 0 | 是 |")
    for name, row in result["models"].items():
        lines.append(f"| {name} | " + " | ".join(f"{x * 100:.4f}%" for x in row["accuracy"].values())
                     + f" | {row['mean_positive_loss_pp']:.4f} | {'是' if row['at_most_0_5_pp'] else '否'} |")
    change = result["B_router_minus_B"]
    lines += ["", f"平均损失变化：{change['mean_positive_loss_change_pp']:+.4f} pp；配对 95% 区间：{change['paired_95pct_pp']}。",
              "", "区间跨 0 不宣称稳定改善；AIME 按题目 avg5 抽样，未覆盖换 seed 或重新训练的方差。旧 C 的精度和教师输入方式不同，不作单变量对照。",
              "", "逐任务变化、辅助 ARC 和配对区间见 [comparison.json](comparison.json)。", ""]
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    summarize(parser.parse_args().run)
