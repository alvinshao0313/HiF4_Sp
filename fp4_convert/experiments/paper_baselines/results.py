"""Validate complete QA artifacts and render the two-model paper tables."""
import csv
import hashlib
import json
import math
from pathlib import Path


QA_COUNTS = {"arc_challenge": 1172, "arc_easy": 2376, "openbookqa": 500, "boolq": 3270}
METHOD_NAMES = {"brq_fused": "BRQ（融合）", "gptq": "GPTQ", "awq": "AWQ", "smoothquant": "SmoothQuant"}
REASONING_TASKS = ("math500", "gpqa", "gsm8k", "lcb", "mmlu_pro")


def validate_reasoning(output, settings, seed, expected=None, max_tokens=32768):
    output = Path(output)
    expected = settings["count"] if expected is None else expected
    results = list((output / "model/results").glob("results_*.json"))
    details = list((output / "model/details").rglob("details_*.json"))
    if len(results) != 1 or len(details) != 1:
        raise ValueError("Missing or ambiguous reasoning results/details")
    rows = json.loads(details[0].read_text())
    if len(rows) != expected or len({str(row["doc"]["id"]) for row in rows}) != expected:
        raise ValueError("Incomplete reasoning question coverage")
    metric_names = set(rows[0]["metric"])
    if len(metric_names) != 1:
        raise ValueError(f"Unexpected metric set: {metric_names}")
    metric = next(iter(metric_names))
    values = [float(row["metric"][metric]) for row in rows]
    if any(not math.isfinite(x) for x in values):
        raise ValueError("Nonfinite reasoning metric")
    if any(len(row["model_response"]["text_post_processed"]) != 1 for row in rows):
        raise ValueError("Expected one answer per question")
    summary = json.loads(results[0].read_text())
    score = summary["results"][settings["task_name"] + "|0"][metric]
    if abs(score - sum(values) / expected) > 1e-12:
        raise ValueError("Reasoning summary disagrees with per-question scores")
    generated = 0
    for path in sorted((output / "generation_observations").glob("batch_*.json")):
        data = json.loads(path.read_text())
        actual = data["sampling"]
        wanted = {**settings["sampling"], "seed": seed, "n": 1, "max_tokens": max_tokens}
        # lighteval/vLLM accepts top_k=0 as disabling top-k, equivalent to -1.
        for name, value in wanted.items():
            if name == "top_k" and value == 0 and actual[name] in (0, -1):
                continue
            if actual[name] != value:
                raise ValueError(f"Effective sampler mismatch: {name}: {actual[name]} != {value}")
        generated += len(data["responses"])
        if any(row["finish_reason"] not in ("stop", "length") or row["output_tokens"] > max_tokens
               for row in data["responses"]):
            raise ValueError("Invalid generation completion")
    if generated != expected:
        raise ValueError(f"Fresh generation coverage: {generated} != {expected}")
    return {"score": score, "metric": metric, "seed": seed, "count": expected,
            "results": str(results[0]), "details": str(details[0]),
            "details_sha256": hashlib.sha256(details[0].read_bytes()).hexdigest()}


def validate_qa(output, counts=QA_COUNTS):
    output = Path(output)
    summaries = list(output.rglob("results_*.json"))
    if len(summaries) != 1:
        raise ValueError(f"Expected one QA summary in {output}")
    summary = json.loads(summaries[0].read_text())
    scores, evidence = {}, {}
    for task, expected in counts.items():
        files = list(output.rglob(f"samples_{task}_*.jsonl"))
        if len(files) != 1:
            raise ValueError(f"Missing or duplicate samples for {task}")
        raw = files[0].read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != expected or len({row["doc_id"] for row in rows}) != expected:
            raise ValueError(f"Incomplete {task}: expected {expected}, found {len(rows)}")
        if any(not math.isfinite(float(item[0][0])) for row in rows for item in row["resps"]):
            raise ValueError(f"Nonfinite choice loglikelihood in {task}")
        scores[task] = {}
        for metric in ("acc",) if task == "boolq" else ("acc", "acc_norm"):
            score = summary["results"][task][metric + ",none"]
            observed = sum(row[metric] for row in rows) / expected
            if not math.isfinite(score) or abs(score - observed) > 1e-12:
                raise ValueError(f"Summary/sample disagreement: {task}/{metric}")
            scores[task][metric] = score
        evidence[task] = {"count": expected, "samples": str(files[0]),
                          "sha256": hashlib.sha256(raw).hexdigest()}
    return {"metrics": str(summaries[0]), "scores": scores, "evidence": evidence}


def render(spec, root):
    """Only full, validated task results enter the table; absent cells stay empty."""
    records = []
    for config in spec["configs"]:
        report = root / "verified" / (config["id"] + ".json")
        reasoning = {}
        for task in REASONING_TASKS:
            rounds = []
            for seed in (1234, 1235, 1236):
                path = root / "verified/reasoning" / f"{config['id']}--{task}--{seed}.json"
                if path.exists():
                    rounds.append(json.loads(path.read_text()))
            reasoning[task] = rounds
        records.append({**config, "qa": json.loads(report.read_text()) if report.exists() else None,
                        "reasoning": reasoning})
    with (root / "scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["config", "model", "target", "method", "task", "seed", "metric", "score", "samples", "source"])
        for row in records:
            for task, rounds in row["reasoning"].items():
                for result in rounds:
                    writer.writerow([row["id"], row["model"], row["target"], row["method"], task,
                                     result["seed"], result["metric"], result["score"], result["count"], result["results"]])
                if len(rounds) == 3:
                    writer.writerow([row["id"], row["model"], row["target"], row["method"], task,
                                     "mean3", rounds[0]["metric"], sum(x["score"] for x in rounds) / 3,
                                     rounds[0]["count"], "three validated seeds"])
            if row["qa"] is None:
                continue
            for task, metrics in row["qa"]["scores"].items():
                for metric, score in metrics.items():
                    writer.writerow([row["id"], row["model"], row["target"], row["method"], task,
                                     1234, metric, score, QA_COUNTS[task], row["qa"]["metrics"]])
    lines = ["% Generated from verified complete results. Do not fill partial-task scores by hand."]
    for target, title in (("hif4", "HiF4"), ("mxfp4", "MXFP4")):
        lines += [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
                  r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
                  r"& \multicolumn{5}{c}{推理任务：三轮均值} & \multicolumn{4}{c}{普通 QA：一轮} \\",
                  r"\cmidrule(lr){2-6}\cmidrule(lr){7-10}",
                  r"方法 & MATH500 & GPQA & GSM8K & LCB & MMLU-Pro & ARC-C & ARC-E & OBQA & BoolQ \\",
                  r"\midrule"]
        for model, name in (("n9", "Nemotron Nano 9B"), ("q4q", "QUASAR Qwen3.5-4B")):
            lines += [r"\multicolumn{10}{l}{\textbf{" + name + r"}}\\", r"\addlinespace[2pt]"]
            for row in records:
                if row["model"] != model or row["target"] != target:
                    continue
                cells = [METHOD_NAMES[row["method"]]]
                for task in REASONING_TASKS:
                    rounds = row["reasoning"][task]
                    cells.append(f"{100 * sum(x['score'] for x in rounds) / 3:.2f}" if len(rounds) == 3 else r"\pending")
                for task in QA_COUNTS:
                    scores = row["qa"]["scores"][task] if row["qa"] else None
                    if scores is None:
                        cells.append(r"\pending")
                    else:
                        value = f"{scores['acc'] * 100:.2f}"
                        if "acc_norm" in scores:
                            value += f"/{scores['acc_norm'] * 100:.2f}"
                        cells.append(value)
                lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\addlinespace[3pt]")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\caption{" + title +
                  r" 全量格式转换的单项方法结果（\%）。普通 QA 使用 lm-eval、0-shot、不套聊天模板、seed=1234；"
                  r"ARC-C、ARC-E、OBQA 暂同时列出 acc/acc\_norm，BoolQ 列 acc，待与外机基线确定主表统计列。"
                  r"推理任务须完成三个种子后填入均值；破折号表示尚无完整结果。}",
                  r"\label{tab:single-" + target + "}", r"\end{table*}", ""]
    table = Path(spec["paper_table"])
    table.write_text("\n".join(lines) + "\n")
