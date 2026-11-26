"""Same-RUN raw four-task comparison with separate final-submission AIME results."""
from __future__ import annotations

import argparse
import math

from ..reasoning_trace_calibration.evaluation import AimeReviewRequired, correct_aime
from .artifacts import read_json, sha256, verify_inputs, verify_sources, write_json
from .config import MODELS, TASKS
from .evaluate import validate_result


def compare(native, converted):
    if set(native) != set(TASKS) or set(converted) != set(TASKS):
        raise ValueError("the comparison requires exactly the four complete tasks")
    values = list(native.values()) + list(converted.values())
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("scores must be finite fractions in [0,1]")
    changes = {task: (converted[task] - native[task]) * 100 for task in TASKS}
    return {"delta_pp": changes, "positive_loss_pp": {k: max(0., -v) for k, v in changes.items()},
            "mean_positive_loss_pp": math.fsum(max(0., -v) for v in changes.values()) / len(TASKS)}


def report(root):
    root = verify_inputs(root)
    verify_sources(root)
    audit = read_json(root / "prompt_audit.json")
    rows, corrected, pending = {}, {}, []
    for candidate in MODELS:
        kind = "native_nvfp4" if candidate == "native_nvfp4" else "hif4"
        rows[candidate] = {}
        for task in TASKS:
            target = root / "evaluation" / candidate / task
            result = read_json(target / "metrics.json")
            checked = validate_result(result, task, audit["tasks"][task], model_kind=kind)
            if checked != read_json(target / "verified_scores.json"):
                raise ValueError("verified score artifact differs from raw results")
            cache = read_json(target / "raw_cache_validation.json")
            if not cache["complete"] or sha256(cache["path"]) != cache["sha256"]:
                raise ValueError("raw generations changed after validation")
            rows[candidate][task] = checked["score"]
        target = root / "evaluation" / candidate / "aime25"
        try:
            correct_aime(target, read_json(target / "metrics.json"))
        except AimeReviewRequired as error:
            pending.append({"model": candidate, "reason": str(error), "responses": str(target / "corrected_responses.json")})
        else:
            corrected[candidate] = {**rows[candidate], "aime25": read_json(target / "corrected_metrics.json")["score"]}
    result = {"complete": not pending, "raw_complete": True, "baseline": "same-RUN original 235B NVFP4",
              "scores": rows, "raw_comparison": compare(rows[MODELS[0]], rows[MODELS[1]]),
              "aime_final_submission_scores": {key: value["aime25"] for key, value in corrected.items()},
              "pending_aime_reviews": pending,
              "definition": "equal mean of max(0, native-candidate)*100 across four tasks; no acceptance threshold"}
    if not pending:
        result["final_submission_comparison"] = compare(corrected[MODELS[0]], corrected[MODELS[1]])
    write_json(result, root / "comparison.json")
    lines = ["# Qwen3-235B matrix_norm 转换前后对照", "",
             "本轮原生 NVFP4 与本轮转换模型使用同一固定数据和 TP8 协议；分数单位为 %。", "",
             "| 任务 | 原生 NVFP4 | matrix_norm HiF4 | 变化（百分点） |", "| --- | ---: | ---: | ---: |"]
    for task in TASKS:
        lines.append(f"| {task} | {rows[MODELS[0]][task] * 100:.4f} | {rows[MODELS[1]][task] * 100:.4f} | {result['raw_comparison']['delta_pp'][task]:+.4f} |")
    lines += ["", f"原评分器口径的四项平均正向下降：{result['raw_comparison']['mean_positive_loss_pp']:.4f} 个百分点。",
              "", "AIME 为一轮 30 题×5 次的 Avg@5；原评分器结果保留，最终提交答案复核另列：", ""]
    for candidate in MODELS:
        value = corrected.get(candidate)
        lines.append(f"- {candidate}：" + (f"{value['aime25'] * 100:.4f}%" if value else "待冲突回答复核"))
    if not pending:
        lines += ["", f"以双方最终提交答案 Avg@5 代入后的平均正向下降：{result['final_submission_comparison']['mean_positive_loss_pp']:.4f} 个百分点。"]
    lines += ["", "HiF4 权重以 BF16 保存 QDQ 数值，配合运行时 W4A4 激活路径；本报告不构成打包 4-bit 存储或推理加速实测。",
              "", "- [完整配置](manifest.json)", "- [逐项分数及待复核项](comparison.json)",
              "- [原始输出和逐题指标](evaluation)", "- [源码摘要](source_hashes.json)", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    result = report(parser.parse_args().run)
    raise SystemExit(0 if result["complete"] else 21)
