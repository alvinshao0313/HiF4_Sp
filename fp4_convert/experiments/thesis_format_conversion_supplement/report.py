"""Audit original results and recompute only the five retained tasks."""
import argparse
import math
from pathlib import Path

import numpy as np
from ..downstream_accuracy_recovery.protocol import protocol_for, reasoning_details
from ..downstream_accuracy_recovery.reporting import _check_reasoning_config, validate_metrics
from .artifacts import read_json, write_json, sha256, digest
from .config import *


def historical_metrics(source, candidate, task):
    directory = source / "evaluation" / candidate / task
    files = list(directory.rglob("metrics.json"))
    return files[0] if len(files) == 1 else None


def verify_reasoning(path, task, audit, kind):
    result = read_json(path)
    protocol = protocol_for(task)
    if result["protocol"] != protocol:
        raise ValueError("saved protocol differs")
    raw = _check_reasoning_config(result, protocol, kind)
    details_path, details = reasoning_details(result, task)
    scores = {}
    for row in details:
        key = str(row["doc"]["id"])
        ref = audit["documents"][key]
        response = row["model_response"]
        if (key in scores or digest(response["input"]) != ref["prompt_sha256"]
                or digest(row["gold"]) != ref["gold_sha256"] or len(response["text_post_processed"]) != 1):
            raise ValueError("actual prompt/gold/multiplicity differs")
        value = row["metric"][protocol["metric"]]
        if value not in (0, 1):
            raise ValueError("nonbinary per-question score")
        scores[key] = value
    score = result["results"][protocol["task_name"]][protocol["metric"]]
    if (set(scores) != set(audit["documents"]) or not math.isclose(np.mean(list(scores.values())), score, abs_tol=1e-12)
            or raw["results"][protocol["task_name"]][protocol["metric"]] != score):
        raise ValueError("coverage/aggregation differs")
    # Raw token IDs are required for new reuse, not inferred from equal question counts.
    model = Path(raw["config_general"]["model_config"]["model_name"])
    caches = list(model.rglob("GENERATIVE.parquet")) if model.exists() else []
    token_evidence = []
    import pyarrow.parquet as pq
    for cache in caches:
        seen = set()
        compatible = True
        for batch in pq.ParquetFile(cache).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                key, sample = str(row["sample_id"]), row["sample"]
                ref = audit["documents"].get(key)
                if (ref is None or key in seen or digest(sample["input"]) != ref["prompt_sha256"]
                        or digest(sample["input_tokens"]) != ref["token_sha256"]
                        or sample["truncated_tokens_count"] != 0 or len(sample["text"]) != 1):
                    compatible = False; break
                seen.add(key)
            if not compatible:
                break
        if compatible and seen == set(scores):
            token_evidence.append({"path": str(cache), "sha256": sha256(cache)})
    # Some historical model views have been cleaned; existing explicit full-token audit remains evidence.
    if not token_evidence:
        for parent in (path.parent, path.parents[2] if len(path.parents) > 2 else path.parent):
            saved = parent / "raw_cache_validation.json"
            if saved.exists():
                record = read_json(saved)
                if record.get("complete") and record.get("actual_input_tokens_match_audit") and record.get("questions") == COUNTS[task]:
                    token_evidence.append({"record": str(saved), "sha256": sha256(saved), "raw_cache_retained": Path(record["path"]).exists()})
    if not token_evidence:
        raise ValueError("raw input-token evidence missing; score retained as historical only")
    return {"score": score, "paired_scores": scores, "metrics": str(path), "metrics_sha256": sha256(path),
            "details": str(details_path), "details_sha256": sha256(details_path), "tokens": token_evidence}


def paired_interval(left, right):
    ids = sorted(left)
    if set(ids) != set(right):
        raise ValueError("paired IDs differ")
    d = np.array([right[k] - left[k] for k in ids], dtype=float)
    rng = np.random.default_rng(42)
    means = np.concatenate([d[rng.integers(0, len(ids), (100, len(ids)))].mean(1) for _ in range(100)])
    return {"delta_pp": float(d.mean() * 100), "95pct_pp": (np.quantile(means, [.025, .975]) * 100).tolist(),
            "positive_loss_vs_native_95pct_pp": np.quantile(np.maximum(0,-means)*100,[.025,.975]).tolist(),
            "questions": len(ids), "scope": "question means across five complete runs; excludes training-seed variance"}


def collect(root):
    root = Path(root)
    audit = read_json(root / "evaluation/prompt_audit.json")["tasks"]
    reuse, rows = {}, {}
    for role in MAIN + EFFICIENCY_ROLES + ("orthogonal_a20",):
        row, evidence, missing = {}, {}, []
        source, candidate = HISTORY.get(role, (root, role))
        for task in REASONING + ("arc",):
            fresh = historical_metrics(root, role + "/round_01", task)
            path = fresh or historical_metrics(source, candidate, task)
            if path is None:
                missing.extend(("arc_easy", "arc_challenge") if task == "arc" else (task,)); continue
            try:
                if task == "arc":
                    values = validate_metrics(read_json(path), "arc", smoke=False,
                                              model_kind="native_nvfp4" if role == "native" else "hif4")
                    row.update(arc_easy=values["ARC-E"], arc_challenge=values["ARC-C"])
                    evidence[task] = {"metrics": str(path), "sha256": sha256(path), "paired": False,
                                      "scope": "historical aggregate; original input contract, no per-question interval"}
                else:
                    checked = verify_reasoning(path, task, audit[task], "native_nvfp4" if role == "native" else "hif4")
                    row[task] = checked.pop("score")
                    scores = checked.pop("paired_scores")
                    write_json(scores, root / "paired" / role / f"{task}.json")
                    evidence[task] = checked
            except (ValueError, KeyError, FileNotFoundError) as error:
                evidence[task] = {"status": "not_reused", "reason": str(error), "metrics": str(path)}
                missing.extend(("arc_easy", "arc_challenge") if task == "arc" else (task,))
        rows[role] = {"accuracy": row, "missing": missing}
        reuse[role] = evidence
        print(f"results {role}: reusable={list(row)} missing={missing}", flush=True)
    write_json({"scope": "existing first-run evidence, NOT five-run means", "rows": rows}, root / "first_round_inventory.json")
    for role, row in rows.items():
        rounds = {t: ({"1":v} if t in row["accuracy"] else {}) for t,v in row["accuracy"].items()}
        rounds.update({t:{} for t in TASKS if t not in rounds})
        question_runs = {t: ([read_json(root/f"paired/{role}/{t}.json")] if t in row["accuracy"] else []) for t in REASONING}
        additional = read_json(root/"additional_native_reuse.json") if role=="native" and (root/"additional_native_reuse.json").exists() else {}
        for repetition in range(2, 6):
            for task in REASONING + ("arc",):
                path = historical_metrics(root, role+f"/round_{repetition:02d}", task)
                if path is None and repetition==2 and task in additional:
                    entry=additional[task]
                    if sha256(entry["metrics"])!=entry["metrics_sha256"]: raise ValueError("extra historical round changed")
                    rounds[task]["2"]=entry["score"]
                    question_runs[task].append(entry["paired_scores"])
                    reuse[role][task+"_round_2"]={k:v for k,v in entry.items() if k!="paired_scores"}
                    continue
                if path is None: continue
                kind = "native_nvfp4" if role == "native" else "hif4"
                if task == "arc":
                    checked = validate_metrics(read_json(path), "arc", smoke=False, model_kind=kind)
                    for key,label in (("arc_easy","ARC-E"),("arc_challenge","ARC-C")):
                        rounds[key][str(repetition)] = checked[label]
                else:
                    checked = verify_reasoning(path,task,audit[task],kind)
                    rounds[task][str(repetition)] = checked["score"]
                    question_runs[task].append(checked["paired_scores"])
        row["round_scores"] = rounds
        row["completed_repetitions"] = {t:len(values) for t,values in rounds.items()}
        row["accuracy"] = {t:float(np.mean(list(values.values()))) for t,values in rounds.items() if len(values)==5}
        row["missing"] = [t for t in TASKS if len(rounds[t]) != 5]
        for task, values in question_runs.items():
            if len(values)==5:
                if any(set(v)!=set(values[0]) for v in values): raise ValueError("five-run question IDs differ")
                write_json({key:float(np.mean([v[key] for v in values])) for key in values[0]},
                           root/f"paired_mean5/{role}/{task}.json")
    native = rows["native"]["accuracy"]
    for role, row in rows.items():
        row["delta_pp"] = {t: (v-native[t])*100 for t,v in row["accuracy"].items() if t in native}
        row["L5_pp"] = (sum(max(0, native[t]-row["accuracy"][t])*100 for t in TASKS)/5
                        if set(native) == set(TASKS) and not row["missing"] else None)
        row["paired_intervals"] = {}
        if role != "native":
            for task in REASONING:
                if task in native and task in row["accuracy"]:
                    row["paired_intervals"][task] = paired_interval(read_json(root / f"paired_mean5/native/{task}.json"),
                                                                   read_json(root / f"paired_mean5/{role}/{task}.json"))
        row["L5_interval"] = None  # Historical ARC aggregates cannot supply paired samples.
    write_json(reuse, root / "reuse.json")
    write_json({"tasks": TASKS, "rows": rows, "complete": all(not rows[r]["missing"] for r in MAIN),
                "bootstrap": PROTOCOL["bootstrap"]}, root / "summary.json")
    inventory = read_json(root / "inventory.json")
    costs = {r: {"training_seconds": e.get("training", {}).get("seconds"),
                 "source": e.get("training_manifest"), "boundary": "original training timer; export/evaluation separate"}
             for r,e in inventory.items() if "training" in e}
    costs["existing_efficiency_ledger"] = {"path": str(EFFICIENCY / "costs.json"), "sha256": sha256(EFFICIENCY / "costs.json")}
    costs["new_jobs"] = {str(p.parent.name):read_json(p) for p in (root/"jobs").glob("*/status.json")}
    costs["new_conversion_measurements"] = {p.name:read_json(p/"conversion.json") for p in (root/"models").glob("*") if (p/"conversion.json").exists()}
    write_json(costs, root / "costs.json")
    lines = ["# 格式转换补充实验：五次均值", "", "五个任务均须独立完整执行五次，每轮保持原协议和 seed，分别保存缓存。未满五次的单轮结果仅在 first_round_inventory.json / summary.json 中记录，不冒充五次均值。历史 ARC 无逐题记录，不报告五任务配对区间。", "",
             "| 方法 | MMLU-Pro | LCB | GSM8K | ARC-E | ARC-C | L5 (pp) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for role, row in rows.items():
        values = [f"{row['accuracy'][t]*100:.4f}" if t in row["accuracy"] else f"待补 ({row['completed_repetitions'][t]}/5)" for t in TASKS]
        values.append(f"{row['L5_pp']:.4f}" if row["L5_pp"] is not None else "—")
        lines.append(f"| {role} | " + " | ".join(values) + " |")
    lines += ["", "来源、拒绝复用原因：[reuse.json](reuse.json)；各任务差值及题目配对区间：[summary.json](summary.json)。",
              "", "效率副实验优先补 full_e5 / coverage50_e5 的 LCB、MMLU-Pro；其余缺项不默认为已授权重训。",
              "", "仅为已有证据整理；新强基线、正交消融与机制结论以各自完成记录为准。无新增方法效果经验。", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__); p.add_argument("--run", type=Path, required=True)
    collect(p.parse_args().run)
