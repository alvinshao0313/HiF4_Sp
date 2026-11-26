"""Fixed-baseline positive losses and question-paired bootstrap intervals."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .artifacts import read_json, write_json
from .config import BASELINE, MODELS, PRIMARY


def loss(scores):
    if set(scores) != set(PRIMARY) or not all(np.isfinite(v) and 0 <= v <= 1 for v in scores.values()):
        raise ValueError("all four finite reasoning accuracies are required")
    individual = {k: max(0., BASELINE[k] - scores[k]) * 100 for k in PRIMARY}
    return {"per_task_positive_loss_pp": individual, "mean_positive_loss_pp": sum(individual.values()) / 4,
            "at_most_0_5_pp": sum(individual.values()) / 4 <= .5 + 1e-12}


def paired(left, right, *, repeats=10000, seed=42):
    rng = np.random.default_rng(seed)
    deltas, task_draws, left_means, right_means = {}, {}, {}, {}
    ldraw, rdraw = {}, {}
    for task in PRIMARY:
        if left[task].keys() != right[task].keys() or not left[task]:
            raise ValueError(f"paired coverage differs: {task}")
        ids = sorted(left[task])
        a, b = np.array([left[task][i] for i in ids]), np.array([right[task][i] for i in ids])
        if not np.isfinite(a).all() or not np.isfinite(b).all() or ((a < 0) | (a > 1) | (b < 0) | (b > 1)).any():
            raise ValueError("invalid paired scores")
        left_means[task], right_means[task] = float(a.mean()), float(b.mean())
        # Bounded chunks avoid allocating 10k x 12k MMLU indices at once.
        am, bm = [], []
        for start in range(0, repeats, 100):
            indices = rng.integers(0, len(ids), (min(100, repeats - start), len(ids)))
            am.extend(a[indices].mean(1))
            bm.extend(b[indices].mean(1))
        ldraw[task], rdraw[task] = np.asarray(am), np.asarray(bm)
        draws = (rdraw[task] - ldraw[task]) * 100
        deltas[task] = {"delta_accuracy_pp": (right_means[task] - left_means[task]) * 100,
                        "paired_95pct_pp": np.quantile(draws, [.025, .975]).tolist(), "paired_questions": len(ids)}
    changes = sum((np.maximum(0, BASELINE[k] - rdraw[k]) - np.maximum(0, BASELINE[k] - ldraw[k])) * 100
                  for k in PRIMARY) / 4
    interval = np.quantile(changes, [.025, .975]).tolist()
    delta_loss = loss(right_means)["mean_positive_loss_pp"] - loss(left_means)["mean_positive_loss_pp"]
    return {"per_task": deltas, "mean_positive_loss_change_pp": delta_loss, "paired_95pct_pp": interval,
            "observed_gain": delta_loss < 0, "interval_supports_gain": interval[1] < 0,
            "uncertainty_scope": "paired question sampling; AIME unit is the question avg5; excludes new seeds/training variance"}


def summarize(root):
    root = Path(root)
    winner = read_json(root / "development/selection.json")["winner"]
    per_model, rows = {}, {}
    tasks = {"mmlu_pro": "MMLU-Pro", "lcb": "LCB", "gsm8k": "GSM8K"}
    for model in MODELS:
        directory = root / "evaluation" / model
        scores, samples = {}, {}
        for task, label in tasks.items():
            scores[label] = read_json(directory / task / "verified_scores.json")[label]
            samples[label] = read_json(directory / task / "paired_scores.json")
            if len(samples[label]) != {"mmlu_pro": 12032, "lcb": 175, "gsm8k": 1319}[task]:
                raise ValueError("full downstream coverage is required")
        aime = read_json(directory / "aime25/corrected_metrics.json")
        if not aime["complete"] or len(aime["per_question"]) != 30 or aime["responses"] != 150:
            raise ValueError("corrected AIME25 avg5 is incomplete")
        scores["AIME25 avg5"], samples["AIME25 avg5"] = aime["score"], aime["per_question"]
        auxiliary = read_json(directory / "arc/verified_scores.json")
        rows[model] = {"accuracy": scores, "auxiliary": auxiliary, **loss(scores)}
        per_model[model] = samples
    result = {"baseline": BASELINE, "models": rows, "winner": winner,
              "B_minus_A": paired(per_model[MODELS[0]], per_model[MODELS[1]]),
              "C_minus_initialization": paired(per_model[winner], per_model[MODELS[2]])}
    from .training import samples_for, training_dir
    from ..e2e_diag_reconstruction.data.batching import build_length_bucket_batches
    calibration = {}
    for name in MODELS[:2]:
        train, val, hashes = samples_for(root, name)
        stats = {}
        for split, samples in (("train", train), ("val", val)):
            lengths = [s.input_ids.numel() for s in samples]
            stats[split] = {"count": len(lengths), "total_tokens": sum(lengths), "min": min(lengths),
                            "median": float(np.median(lengths)), "max": max(lengths), "lengths": lengths}
        stats["training_valid_token_visits"] = sum(int(s.loss_mask.bool().sum()) for s in train) * 20 * 48
        stats["training_padded_token_visits"] = sum(max(s.input_ids.numel() for s in batch) * len(batch)
                for epoch in range(20) for batch in build_length_bucket_batches(train, 4, 42 + epoch)) * 48
        stats["training_seconds"] = read_json(training_dir(root, name) / "manifest.json").get("seconds")
        stats["input_sha256"] = hashes
        calibration[name] = stats
    result["calibration_compute"] = calibration
    result["native_trace_generation_seconds"] = read_json(root / "generation/manifest.json")["seconds"]
    write_json(result, root / "comparison.json")
    lines = ["# 校准数据与追加 Router 对齐结果", "", "四项 reasoning 等权平均正向损失，单位为百分点。AIME 使用纠正后的最终提交答案 avg5。", "",
             "| 候选 | MMLU-Pro | LCB | GSM8K | AIME25 avg5 | 平均正向损失 | ≤0.5pp |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    lines.append("| 原生历史基线 | " + " | ".join(f"{BASELINE[k] * 100:.4f}%" for k in PRIMARY) + " | 0 | 是 |")
    for model, row in rows.items():
        lines.append(f"| {model} | " + " | ".join(f"{row['accuracy'][k] * 100:.4f}%" for k in PRIMARY)
                     + f" | {row['mean_positive_loss_pp']:.4f} | {'是' if row['at_most_0_5_pp'] else '否'} |")
    lines += ["", "| 候选 | ARC-E | ARC-C |", "| --- | ---: | ---: |"]
    for model, row in rows.items():
        lines.append(f"| {model} | {row['auxiliary']['ARC-E'] * 100:.4f}% | {row['auxiliary']['ARC-C'] * 100:.4f}% |")
    lines += ["", f"Router 起点：`{winner}`。逐任务变化、配对区间及辅助 ARC 结果见 [comparison.json](comparison.json)。",
              "", "配对区间覆盖题目抽样不确定性；AIME 按 30 个题目簇抽样，不把 150 次回答视为独立题。区间跨 0 时不宣称稳定改善；单轮实验不覆盖重新训练或换种子的方差。", ""]
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    summarize(parser.parse_args().run)
