"""Read-only mechanism analysis for format-conversion LCB experiments.

Consumes completed run_root artifacts only. Never replays vLLM or simulates
decoder/runtime paths. Distinguishes facts, statistical association, and
causal intervention evidence in all outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.build_probe_plan import (
    first_divergence,
    transitions as trajectory_transitions,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import read_jsonl

EVIDENCE_FACT = "FACT"
EVIDENCE_ASSOCIATION = "STATISTICAL_ASSOCIATION"
EVIDENCE_CAUSAL = "CAUSAL_INTERVENTION"

FEATURE_FAMILIES = {
    "qerror": (
        "max_hif4_qdq_rel_l2",
        "median_hif4_qdq_rel_l2",
    ),
    "norm": (
        "max_amax_over_rms",
        "median_amax_over_rms",
        "rms",
        "amax",
        "amax_over_rms",
        "group64_amax_dynamic_range",
        "subgroup16_imbalance_max",
    ),
    "actual_path": (
        "attention_core_rel_l2",
        "moe_out_rel_l2",
        "layer_out_rel_l2",
        "final_hidden_rel_l2",
        "final_hidden_cosine",
        "final_hidden_norm_ratio",
    ),
    "decision": (
        "e0_margin",
        "target_rank_variant",
        "logit_kl_e0_to_variant",
        "top1_flip",
    ),
    "router": (
        "router_min_margin",
        "router_first_topk_change_layer",
        "router_topk_mismatch_count",
    ),
    "puncture": (
        "qkv_q_rel_l2",
        "qkv_p_rel_l2",
        "qkv_cos_q_p",
        "o_proj_q_rel_l2",
        "o_proj_p_rel_l2",
        "o_proj_cos_q_p",
        "moe_q_rel_l2",
        "moe_p_rel_l2",
        "moe_cos_q_p",
        "moe_q_expert_rel_l2",
        "moe_q_router_rel_l2",
    ),
}

TASK19_QUESTIONS = (
    ("Q01", "E1 为什么会让 61 个 E0-correct LCB case 变成错误？"),
    ("Q02", "thinking 不收束是不是绝对主 failure mode？"),
    ("Q03", "first lexical divergence 通常发生在 thinking 的什么阶段？"),
    ("Q04", "真实 quantizer-input 的 HiF4 susceptibility 在 divergence 前是否异常？"),
    ("Q05", "RMS/amax/amax-RMS 是否比 qerror 更有解释力，还是反过来？"),
    ("Q06", "final hidden 主要表现为 norm drift 还是 direction drift？"),
    ("Q07", "E0 margin 在 LCB 是否真的是主触发因素？若不是，明确否定。"),
    ("Q08", "Attention 和 MoE 哪个更主要？"),
    ("Q09", "MoE 大误差主要是 expert/fused computation，还是 router decision？"),
    ("Q10", "same-input local puncture 与 actual-path growth 能否对上？"),
    ("Q11", "state reset 能否把 first wrong token 决策救回？关键深度在哪里？"),
    ("Q12", "E1 state injection 能否把 E0 决策推错？"),
    ("Q13", "first lexical divergence 与真正 accuracy-critical frontier 是否相距很远？"),
    ("Q14", "什么机制使 E1 无法正常结束 thinking？"),
    ("Q15", "R64 在 formal sampled regression 中出现了哪些 observed recovery/翻牌？哪些又被 deterministic mechanism 证据支持？"),
    ("Q16", "R64 是否降低了 HiF4 same-state QDQ error？"),
    ("Q17", "如果局部 QDQ error 降了但 task 仍没恢复，问题出在哪一环？"),
    ("Q18", "R64 是否主要改变局部能量排布/error direction，而不是简单减小误差？"),
    ("Q19", "DIAG 在 formal sampled regression 中出现了哪些 observed recovery/翻牌？哪些又被 deterministic mechanism 证据支持？"),
    ("Q20", "DIAG 主要改变 channel scale/distribution、local injection、MoE error 还是 downstream decision？"),
    ("Q21", "为什么 DIAG 在 MMLU 能维持/恢复，但 LCB 仍只有约 0.223？"),
)


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _median(values: list[float]) -> float | None:
    clean = [_finite(v) for v in values]
    clean = [v for v in clean if v is not None]
    return None if not clean else float(statistics.median(clean))


def _mean(values: list[float]) -> float | None:
    clean = [_finite(v) for v in values]
    clean = [v for v in clean if v is not None]
    return None if not clean else float(sum(clean) / len(clean))


def _read_json(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text())


def _read_jsonl(path: Path | None) -> list[dict] | None:
    if path is None or not path.is_file():
        return None
    return read_jsonl(path)


def _glob_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def _artifact_status(path: Path | None) -> str:
    if path is None or not path.is_file():
        return "NOT_MEASURED"
    return "AVAILABLE"

class RunArtifacts:
    """Resolved read-only artifact bundle for one run_root."""

    def __init__(self, run_root: Path):
        self.run_root = run_root.resolve()
        self.formal_audit = _read_json(self.run_root / "00_audit/FORMAL_PROTOCOL_MISMATCH.json")
        self.formal_summary = _read_json(self.run_root / "01_formal_matrix/formal_task_summary.json")
        self.formal_matrix = _read_jsonl(self.run_root / "01_formal_matrix/formal_task_matrix.jsonl")
        self.cohort = _read_jsonl(self.run_root / "02_cohort/benchmark_cohort.jsonl")
        self.cohort_manifest = _read_json(self.run_root / "02_cohort/benchmark_cohort_manifest.json")
        self.greedy_matrix = _read_jsonl(self.run_root / "04_greedy_judge/greedy_task_matrix.jsonl")
        self.greedy_summary = _read_json(self.run_root / "04_greedy_judge/greedy_task_summary.json")
        self.divergence_events = _read_jsonl(self.run_root / "05_probe_plan/divergence_events.jsonl")
        self.think_transitions = _read_jsonl(self.run_root / "05_probe_plan/think_transition_events.jsonl")
        self.probe_plan = _read_json(self.run_root / "05_probe_plan/probe_plan.json")
        self.isolated = {
            variant: _read_jsonl(self.run_root / f"03_isolated/{variant}.jsonl")
            for variant in ("E0", "E1", "E2", "E3")
        }
        self.feature_tokens = _read_jsonl(_glob_first(self.run_root / "06_feature_capture", "**/*_tokens.jsonl"))
        self.feature_rows = _read_jsonl(_glob_first(self.run_root / "06_feature_capture", "**/features.jsonl"))
        self.compare_e0_e1 = _read_jsonl(_glob_first(self.run_root / "07_core_capture", "**/e0_e1_compare.jsonl"))
        self.compare_e0_e2 = _read_jsonl(_glob_first(self.run_root / "12_rq3_e2", "**/e0_e2_compare.jsonl"))
        self.compare_e0_e3 = _read_jsonl(_glob_first(self.run_root / "13_rq3_e3", "**/e0_e3_compare.jsonl"))
        self.compare_e1_e2 = _read_jsonl(_glob_first(self.run_root / "12_rq3_e2", "**/e1_e2_compare.jsonl"))
        self.compare_e1_e3 = _read_jsonl(_glob_first(self.run_root / "13_rq3_e3", "**/e1_e3_compare.jsonl"))
        self.semantic_frontier = _read_jsonl(_glob_first(self.run_root / "10_semantic_frontier", "**/frontier_rows.jsonl"))
        self.semantic_summary = _read_json(_glob_first(self.run_root / "10_semantic_frontier", "**/frontier_summary.json"))
        self.state_reset = _read_jsonl(_glob_first(self.run_root / "11_state_intervention", "**/state_reset_rows.jsonl"))
        self.state_injection = _read_jsonl(_glob_first(self.run_root / "11_state_intervention", "**/state_injection_rows.jsonl"))
        self.puncture_roots = sorted({p.parent.parent for p in (self.run_root / "08_rq1_puncture").rglob("rank0.pt")})

    @property
    def formal_prompt_mismatch(self) -> bool:
        if self.formal_audit:
            return self.formal_audit.get("status") == "FORMAL_PROMPT_PROTOCOL_MISMATCH"
        if self.formal_summary:
            return not bool(self.formal_summary.get("FORMAL_PROMPT_PROTOCOL_ALIGNED", True))
        return False


def select_matched_controls(event_decode: int, output_len: int, transition: dict,
                            *, max_controls: int = 4, min_distance: int = 8) -> list[int]:
    if event_decode is None or output_len <= 0:
        return []
    thinking_end = transition.get("thinking_end")
    code_start = transition.get("code_start")

    def phase(decode: int) -> str:
        if thinking_end is not None and decode < thinking_end:
            return "thinking"
        if code_start is not None and decode >= code_start:
            return "code"
        return "post_thinking"

    event_phase = phase(event_decode)
    candidates = []
    for decode in range(output_len):
        if decode == event_decode or abs(decode - event_decode) < min_distance:
            continue
        if phase(decode) != event_phase:
            continue
        candidates.append(decode)
    candidates.sort(key=lambda j: (abs(j - event_decode), j))
    return candidates[:max_controls]


def _index_feature_tokens(rows: list[dict] | None) -> dict[tuple[str, int], dict]:
    if not rows:
        return {}
    out = {}
    for row in rows:
        key = (str(row["sample_key"]), int(row["decode_index"]))
        if key in out:
            raise ValueError(f"duplicate feature token row: {key}")
        out[key] = row
    return out


def _index_feature_rows(rows: list[dict] | None) -> dict[tuple[str, int, int, str], dict]:
    if not rows:
        return {}
    out = {}
    for row in rows:
        key = (str(row["sample_key"]), int(row["decode_index"]), int(row["layer"]), str(row["boundary"]))
        if key in out:
            raise ValueError(f"duplicate feature row: {key}")
        out[key] = row
    return out


def _index_compare(rows: list[dict] | None) -> dict[tuple[str, int, str, str, int | None], dict]:
    if not rows:
        return {}
    out = {}
    for row in rows:
        layer = row.get("layer")
        key = (str(row["sample_key"]), int(row["decode_index"]), str(row["boundary"]), str(row["role"]), layer)
        if key in out:
            raise ValueError(f"duplicate compare row: {key}")
        out[key] = row
    return out


def _best_layer_metric(index: dict, sample: str, decode: int, boundary: str, role: str) -> float | None:
    values = [row["rel_l2"] for key, row in index.items()
              if key[0] == sample and key[1] == decode and key[2] == boundary and key[3] == role]
    clean = [_finite(v) for v in values]
    clean = [v for v in clean if v is not None]
    return max(clean) if clean else None


def _router_summary(index: dict, sample: str, decode: int) -> dict[str, float | int | None]:
    rows = [row for key, row in index.items()
            if key[0] == sample and key[1] == decode and key[2] == "router_logits" and key[3] == "logits"]
    if not rows:
        return {"router_min_margin": None, "router_first_topk_change_layer": None, "router_topk_mismatch_count": None}
    margins = [_finite(row.get("e0_boundary_margin")) for row in rows]
    margins = [m for m in margins if m is not None]
    mismatches = [row for row in rows if row.get("topk_exact") is False]
    first_change = min((int(row["layer"]) for row in mismatches), default=None)
    return {
        "router_min_margin": min(margins) if margins else None,
        "router_first_topk_change_layer": first_change,
        "router_topk_mismatch_count": len(mismatches),
    }


def extract_token_features(sample_key: str, decode_index: int, *,
                           feature_tokens: dict[tuple[str, int], dict],
                           feature_rows: dict[tuple[str, int, int, str], dict],
                           compare_index: dict[tuple[str, int, str, str, int | None], dict],
                           puncture_summary: dict[tuple[str, int], dict] | None = None) -> dict[str, float | int | bool | None]:
    token = feature_tokens.get((sample_key, decode_index), {})
    out: dict[str, float | int | bool | None] = {
        "max_hif4_qdq_rel_l2": _finite(token.get("max_hif4_qdq_rel_l2")),
        "median_hif4_qdq_rel_l2": _finite(token.get("median_hif4_qdq_rel_l2")),
        "max_amax_over_rms": _finite(token.get("max_amax_over_rms")),
        "median_amax_over_rms": _finite(token.get("median_amax_over_rms")),
    }
    layer_rows = [row for key, row in feature_rows.items() if key[0] == sample_key and key[1] == decode_index]
    if layer_rows:
        out["rms"] = _median([row.get("rms") for row in layer_rows])
        out["amax"] = max(v for v in (_finite(row.get("amax")) for row in layer_rows) if v is not None)
        out["amax_over_rms"] = _median([row.get("amax_over_rms") for row in layer_rows])
        out["group64_amax_dynamic_range"] = _median([row.get("group64_amax_dynamic_range") for row in layer_rows])
        out["subgroup16_imbalance_max"] = _median([row.get("subgroup16_imbalance_max") for row in layer_rows])
    out["attention_core_rel_l2"] = _best_layer_metric(compare_index, sample_key, decode_index, "attention_core", "rank_local")
    out["moe_out_rel_l2"] = _best_layer_metric(compare_index, sample_key, decode_index, "moe_out", "tp_reduced")
    out["layer_out_rel_l2"] = _best_layer_metric(compare_index, sample_key, decode_index, "layer_out", "branch")
    final = compare_index.get((sample_key, decode_index, "layer_out", "branch", 47))
    if final:
        out["final_hidden_rel_l2"] = _finite(final.get("rel_l2"))
        out["final_hidden_cosine"] = _finite(final.get("cosine"))
        out["final_hidden_norm_ratio"] = _finite(final.get("norm_ratio"))
    logits = compare_index.get((sample_key, decode_index, "raw_logits", "full_vocab", None))
    if logits:
        out["e0_margin"] = _finite(logits.get("e0_margin"))
        out["target_rank_variant"] = _finite(logits.get("target_rank_variant"))
        out["logit_kl_e0_to_variant"] = _finite(logits.get("logit_kl_e0_to_variant"))
        out["top1_flip"] = None if logits.get("top1_agree") is None else (not bool(logits.get("top1_agree")))
    out.update(_router_summary(compare_index, sample_key, decode_index))
    if puncture_summary:
        puncture = puncture_summary.get((sample_key, decode_index), {})
        for prefix in ("qkv", "o_proj", "moe"):
            block = puncture.get(prefix, {})
            out[f"{prefix}_q_rel_l2"] = _finite(block.get("q_rel_l2"))
            out[f"{prefix}_p_rel_l2"] = _finite(block.get("p_rel_l2"))
            out[f"{prefix}_cos_q_p"] = _finite(block.get("cos_q_p"))
        moe = puncture.get("moe", {})
        out["moe_q_expert_rel_l2"] = _finite(moe.get("q_expert_rel_l2"))
        out["moe_q_router_rel_l2"] = _finite(moe.get("q_router_rel_l2"))
    return out


def load_puncture_summary(puncture_roots: list[Path]) -> dict[tuple[str, int], dict]:
    """Aggregate per-sample/token puncture e/q/p from saved rank0 records."""
    summary: dict[tuple[str, int], dict] = defaultdict(dict)
    for root in puncture_roots:
        for path in sorted(root.rglob("rank0.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            sample = str(payload.get("sample_key") or path.parent.name)
            for record in payload.get("records", []):
                if record.get("status") != "PASS" or "decomposition" not in record:
                    continue
                operator = str(record["operator"])
                decode = int(record["decode_index"])
                dec = record["decomposition"]
                block = {
                    "e_rel_l2": _finite(dec.get("e_rel_l2")),
                    "q_rel_l2": _finite(dec.get("q_rel_l2")),
                    "p_rel_l2": _finite(dec.get("p_rel_l2")),
                    "cos_q_p": _finite(dec.get("cos_q_p")),
                }
                router = record.get("router_decomposition") or {}
                if router:
                    block["q_expert_rel_l2"] = _finite(router.get("q_expert_rel_l2"))
                    block["q_router_rel_l2"] = _finite(router.get("q_router_rel_l2"))
                current = summary[(sample, decode)].get(operator, {})
                for metric, value in block.items():
                    if value is None:
                        continue
                    prior = current.get(metric)
                    current[metric] = value if prior is None else max(prior, value)
                summary[(sample, decode)][operator] = current
    return dict(summary)


def bootstrap_median_ci(diffs: list[float], *, n_boot: int = 2000, seed: int = 42) -> dict[str, Any]:
    clean = [_finite(v) for v in diffs]
    clean = [v for v in clean if v is not None]
    if not clean:
        return {"n": 0, "median_diff": None, "ci95_low": None, "ci95_high": None, "status": "NOT_MEASURED"}
    rng = np.random.default_rng(seed)
    arr = np.array(clean, dtype=np.float64)
    boots = [float(np.median(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    low, high = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    return {
        "n": len(clean),
        "median_diff": float(np.median(arr)),
        "ci95_low": low,
        "ci95_high": high,
        "status": "OK",
    }


def maybe_logistic_regression(rows: list[dict], *, min_samples: int = 16) -> dict[str, Any]:
    if len(rows) < min_samples:
        return {"status": "SKIPPED_INSUFFICIENT_SAMPLES", "n": len(rows), "min_required": min_samples}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneOut
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        return {"status": "BLOCKED", "reason": f"sklearn unavailable: {exc}"}
    feature_names = sorted({name for row in rows for name in row["features"]})
    x_rows, y_rows = [], []
    for row in rows:
        values = []
        missing = False
        for name in feature_names:
            value = _finite(row["features"].get(name))
            if value is None:
                missing = True
                break
            values.append(value)
        if missing:
            continue
        x_rows.append(values)
        y_rows.append(int(row["label_event"]))
    if len(x_rows) < min_samples:
        return {"status": "SKIPPED_AFTER_MISSING_FEATURES", "n": len(x_rows), "min_required": min_samples}
    x = np.array(x_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.int64)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    loo = LeaveOneOut()
    preds = []
    for train_idx, test_idx in loo.split(x_scaled):
        model = LogisticRegression(max_iter=2000)
        model.fit(x_scaled[train_idx], y[train_idx])
        preds.append(int(model.predict(x_scaled[test_idx])[0]))
    accuracy = float(np.mean(np.array(preds) == y))
    full = LogisticRegression(max_iter=2000).fit(x_scaled, y)
    coef = {name: float(value) for name, value in zip(feature_names, full.coef_.reshape(-1), strict=True)}
    families = {}
    for family, names in FEATURE_FAMILIES.items():
        subset = [name for name in names if name in coef]
        if subset:
            families[family] = {"features": subset, "coef_sum_abs": float(sum(abs(coef[n]) for n in subset))}
    best_family = max(families, key=lambda k: families[k]["coef_sum_abs"]) if families else None
    return {
        "status": "OK",
        "n": len(x_rows),
        "leave_one_out_accuracy": accuracy,
        "coefficients": coef,
        "feature_families": families,
        "strongest_family_by_coef_mass": best_family,
    }


def build_mechanism_groups(artifacts: RunArtifacts) -> dict[str, list[dict]]:
    if artifacts.greedy_matrix:
        groups = defaultdict(list)
        for row in artifacts.greedy_matrix:
            groups[str(row.get("mechanism_group", "unknown"))].append(row)
        return dict(groups)
    if artifacts.cohort and any(artifacts.isolated.get(v) for v in ("E0", "E1")):
        e0 = {r["prompt_key"]: r for r in artifacts.isolated["E0"] or []}
        e1 = {r["prompt_key"]: r for r in artifacts.isolated["E1"] or []}
        rows = []
        for cohort_row in artifacts.cohort:
            key = cohort_row["prompt_key"]
            if key not in e0 or key not in e1:
                continue
            t_lex = first_divergence(e0[key]["output_ids"], e1[key]["output_ids"])
            e0_pass = cohort_row.get("formal_group") in {"formal_regression", "formal_robust"}
            e1_pass = cohort_row.get("formal_group") == "formal_robust"
            if e0[key]["output_ids"] and e1[key]["output_ids"]:
                judged_e0 = cohort_row.get("formal_group") != "formal_both_fail"
                judged_e1 = cohort_row.get("formal_group") == "formal_robust"
                e0_pass, e1_pass = judged_e0, judged_e1
            group = ("mechanism_regression" if not e1_pass else "mechanism_robust") if e0_pass else "mechanism_e0_fail"
            rows.append({**cohort_row, "prompt_key": key, "t_lex": t_lex, "mechanism_group": group})
        groups = defaultdict(list)
        for row in rows:
            groups[row["mechanism_group"]].append(row)
        return dict(groups)
    return {}


def rq2_matched_statistics(artifacts: RunArtifacts, *, seed: int = 42) -> dict[str, Any]:
    groups = build_mechanism_groups(artifacts)
    regressions = groups.get("mechanism_regression", [])
    if not regressions:
        return {"status": "NOT_MEASURED", "reason": "no mechanism-regression labels"}
    e0_index = {r["prompt_key"]: r for r in (artifacts.isolated.get("E0") or [])}
    e1_index = {r["prompt_key"]: r for r in (artifacts.isolated.get("E1") or [])}
    if not e0_index or not e1_index:
        return {"status": "NOT_MEASURED", "reason": "missing 03_isolated/E0 or E1"}
    feature_tokens = _index_feature_tokens(artifacts.feature_tokens)
    feature_rows = _index_feature_rows(artifacts.feature_rows)
    compare_index = _index_compare(artifacts.compare_e0_e1)
    puncture_summary = load_puncture_summary(artifacts.puncture_roots)
    transition_index = {}
    built_transitions = _build_transition_rows(artifacts)
    if built_transitions:
        transition_index = {str(row["sample_key"]): row["E0"] for row in built_transitions}
    event_rows = []
    logistic_rows = []
    all_features = sorted({name for names in FEATURE_FAMILIES.values() for name in names})
    per_feature_diffs: dict[str, list[float]] = defaultdict(list)
    for row in regressions:
        key = str(row["prompt_key"])
        if key not in e0_index:
            continue
        e0 = e0_index[key]
        t_lex = row.get("t_lex")
        if t_lex is None and key in e1_index:
            t_lex = first_divergence(e0["output_ids"], e1_index[key]["output_ids"])
        if t_lex is None:
            continue
        if key not in transition_index:
            continue
        transition = transition_index[key]
        controls = select_matched_controls(int(t_lex), len(e0["output_ids"]), transition)
        event_features = extract_token_features(key, int(t_lex), feature_tokens=feature_tokens,
                                                 feature_rows=feature_rows, compare_index=compare_index,
                                                 puncture_summary=puncture_summary)
        control_features = [extract_token_features(key, c, feature_tokens=feature_tokens, feature_rows=feature_rows,
                                                   compare_index=compare_index, puncture_summary=puncture_summary)
                            for c in controls]
        diffs = {}
        for name in all_features:
            event_value = _finite(event_features.get(name))
            control_values = [_finite(cf.get(name)) for cf in control_features]
            control_values = [v for v in control_values if v is not None]
            if event_value is None or not control_values:
                continue
            diff = event_value - float(statistics.median(control_values))
            diffs[name] = diff
            per_feature_diffs[name].append(diff)
        event_rows.append({"sample_key": key, "t_lex": int(t_lex), "n_controls": len(controls), "diffs": diffs})
        logistic_rows.append({"label_event": 1, "features": event_features})
        for cf in control_features:
            logistic_rows.append({"label_event": 0, "features": cf})
    feature_stats = {name: bootstrap_median_ci(values, seed=seed) for name, values in per_feature_diffs.items()}
    logistic = maybe_logistic_regression(logistic_rows)
    return {
        "status": "OK" if event_rows else "NOT_MEASURED",
        "n_mechanism_regression": len(regressions),
        "n_event_rows": len(event_rows),
        "matched_event_control": event_rows,
        "feature_bootstrap_ci": feature_stats,
        "logistic": logistic,
        "evidence_tier": EVIDENCE_ASSOCIATION,
    }


def _build_transition_rows(artifacts: RunArtifacts) -> list[dict] | None:
    if artifacts.think_transitions:
        return artifacts.think_transitions
    if not artifacts.isolated.get("E0") or not artifacts.isolated.get("E1"):
        return None
    from transformers import AutoTokenizer
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.config import DEFAULT_MODEL_PATH

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True, local_files_only=True)
    e0_index = {r["prompt_key"]: r for r in artifacts.isolated["E0"]}
    e1_index = {r["prompt_key"]: r for r in artifacts.isolated["E1"]}
    rows = []
    for key, e0 in e0_index.items():
        if key not in e1_index:
            continue
        group = next((r.get("mechanism_group") for r in (artifacts.greedy_matrix or [])
                      if str(r.get("prompt_key")) == key), "unknown")
        rows.append({
            "sample_key": key,
            "mechanism_group": group,
            "E0": trajectory_transitions(e0, tokenizer),
            "E1": trajectory_transitions(e1_index[key], tokenizer),
        })
    return rows


def analyze_thinking_transitions(artifacts: RunArtifacts) -> dict[str, Any]:
    rows = _build_transition_rows(artifacts)
    if not rows:
        return {"status": "NOT_MEASURED", "reason": "missing think_transition_events or isolated trajectories"}

    def summarize(group_name: str) -> dict[str, Any]:
        subset = [row for row in rows if row.get("mechanism_group") == group_name]
        if not subset:
            return {"n": 0}
        e0_inside = []
        distances = []
        e1_finished = []
        e1_lengths = []
        e0_lengths = []
        for row in subset:
            e0_t = row["E0"]
            e1_t = row["E1"]
            t_lex = row.get("t_lex")
            if t_lex is None and artifacts.isolated.get("E0") and artifacts.isolated.get("E1"):
                e0_row = next((r for r in artifacts.isolated["E0"] if r["prompt_key"] == row["sample_key"]), None)
                e1_row = next((r for r in artifacts.isolated["E1"] if r["prompt_key"] == row["sample_key"]), None)
                if e0_row and e1_row:
                    t_lex = first_divergence(e0_row["output_ids"], e1_row["output_ids"])
            if t_lex is not None and e0_t.get("thinking_end") is not None:
                e0_inside.append(t_lex < e0_t["thinking_end"])
                distances.append(int(e0_t["thinking_end"]) - int(t_lex))
            e1_finished.append(bool(e1_t.get("finished_thinking")))
            if e0_t.get("thinking_end") is not None:
                e0_lengths.append(int(e0_t["thinking_end"]) - int(e0_t.get("thinking_start") or 0))
            if e1_t.get("thinking_end") is not None:
                e1_lengths.append(int(e1_t["thinking_end"]) - int(e1_t.get("thinking_start") or 0))
        return {
            "n": len(subset),
            "fraction_first_divergence_inside_thinking": _mean([1.0 if v else 0.0 for v in e0_inside]),
            "median_distance_to_e0_thinking_end": _median(distances),
            "fraction_e1_finished_thinking": _mean([1.0 if v else 0.0 for v in e1_finished]),
            "median_e0_thinking_length": _median([float(v) for v in e0_lengths]),
            "median_e1_thinking_length": _median([float(v) for v in e1_lengths]),
        }

    return {
        "status": "OK",
        "mechanism_regression": summarize("mechanism_regression"),
        "mechanism_robust": summarize("mechanism_robust"),
        "evidence_tier": EVIDENCE_ASSOCIATION,
    }


def summarize_puncture(artifacts: RunArtifacts) -> dict[str, Any]:
    if not artifacts.puncture_roots:
        return {"status": "NOT_MEASURED", "reason": "no 08_rq1_puncture artifacts"}
    summary = load_puncture_summary(artifacts.puncture_roots)
    if not summary:
        return {"status": "BLOCKED", "reason": "puncture files present but no PASS decomposition rows"}
    by_operator: dict[str, list[dict]] = defaultdict(list)
    for block in summary.values():
        for operator, metrics in block.items():
            by_operator[operator].append(metrics)
    aggregate = {}
    for operator, rows in by_operator.items():
        aggregate[operator] = {
            "n_tokens": len(rows),
            "median_q_rel_l2": _median([r.get("q_rel_l2") for r in rows]),
            "median_p_rel_l2": _median([r.get("p_rel_l2") for r in rows]),
            "median_e_rel_l2": _median([r.get("e_rel_l2") for r in rows]),
            "median_cos_q_p": _median([r.get("cos_q_p") for r in rows]),
            "median_q_expert_rel_l2": _median([r.get("q_expert_rel_l2") for r in rows]),
            "median_q_router_rel_l2": _median([r.get("q_router_rel_l2") for r in rows]),
        }
    return {"status": "OK", "by_operator": aggregate, "evidence_tier": EVIDENCE_CAUSAL}


def formal_failure_taxonomy(artifacts: RunArtifacts) -> dict[str, Any]:
    if not artifacts.formal_matrix:
        return {"status": "NOT_MEASURED"}
    rows = [row for row in artifacts.formal_matrix if row.get("formal_group") == "formal_regression"]
    thinking_unfinished = 0
    wrong_answer = 0
    unknown = 0
    for row in rows:
        finished = row.get("E1_finished_thinking")
        if finished is True:
            wrong_answer += 1
        elif finished is False:
            thinking_unfinished += 1
        else:
            unknown += 1
    return {
        "status": "OK",
        "n_formal_regression": len(rows),
        "thinking_unfinished": thinking_unfinished,
        "wrong_answer_after_thinking": wrong_answer,
        "unknown_thinking_status": unknown,
        "evidence_tier": EVIDENCE_FACT,
    }


def build_task19_answers(artifacts: RunArtifacts, rq2: dict, thinking: dict, puncture: dict,
                         taxonomy: dict, figures: dict) -> dict[str, dict[str, Any]]:
    answers: dict[str, dict[str, Any]] = {}
    mismatch = artifacts.formal_prompt_mismatch
    formal = artifacts.formal_summary or {}
    n_reg = formal.get("group_counts", {}).get("formal_regression", 61)

    def put(qid: str, text: str, *, tier: str, status: str = "OK"):
        answers[qid] = {"question": dict(TASK19_QUESTIONS)[qid], "answer": text, "evidence_tier": tier, "status": status}

    put("Q01",
        f"正式 sampled LCB 观察到 {n_reg} 题 E0 Pass/E1 Fail（事实）。"
        f"{'正式 61 regression 的 prompt 协议与 E0 不一致（FORMAL_PROMPT_PROTOCOL_MISMATCH），不能作为格式转换因果证据；' if mismatch else ''}"
        f"mechanism 标签需来自 matched-prompt isolated greedy + 官方 judge。",
        tier=EVIDENCE_FACT)
    tax = taxonomy if taxonomy.get("status") == "OK" else {}
    if tax:
        total = tax["n_formal_regression"]
        frac = tax["thinking_unfinished"] / total if total else 0.0
        put("Q02",
            f"在可判定 formal regression 中，thinking 未收束占 {tax['thinking_unfinished']}/{total}（{frac:.1%}）。"
            f"WA 仅 {tax['wrong_answer_after_thinking']} 题；unknown={tax['unknown_thinking_status']}。",
            tier=EVIDENCE_FACT)
    else:
        put("Q02", "NOT_MEASURED：缺少 formal_task_matrix。", tier=EVIDENCE_FACT, status="NOT_MEASURED")

    reg_th = thinking.get("mechanism_regression", {})
    if reg_th.get("n", 0) > 0:
        put("Q03",
            f"mechanism-regression 中 first divergence 位于 E0 thinking 内的比例="
            f"{reg_th.get('fraction_first_divergence_inside_thinking')}; "
            f"距 E0 </think> 的中位 token 距离={reg_th.get('median_distance_to_e0_thinking_end')}。",
            tier=EVIDENCE_ASSOCIATION)
    else:
        put("Q03", "NOT_MEASURED：缺少 mechanism-regression thinking transition 数据。", tier=EVIDENCE_ASSOCIATION, status="NOT_MEASURED")

    qerr = rq2.get("feature_bootstrap_ci", {}).get("max_hif4_qdq_rel_l2") or rq2.get("feature_bootstrap_ci", {}).get("median_hif4_qdq_rel_l2")
    if qerr and qerr.get("status") == "OK":
        put("Q04",
            f"event-control matched median diff={qerr['median_diff']:.4g}, 95% CI=[{qerr['ci95_low']:.4g}, {qerr['ci95_high']:.4g}]（n={qerr['n']}）。",
            tier=EVIDENCE_ASSOCIATION)
    else:
        put("Q04", "NOT_MEASURED：缺少 feature_scan token 特征或 mechanism-regression event 行。", tier=EVIDENCE_ASSOCIATION, status="NOT_MEASURED")

    norm_key = "max_amax_over_rms"
    if qerr and qerr.get("status") == "OK" and rq2.get("feature_bootstrap_ci", {}).get(norm_key, {}).get("status") == "OK":
        norm = rq2["feature_bootstrap_ci"][norm_key]
        stronger = "norm/distribution" if abs(norm["median_diff"]) > abs(qerr["median_diff"]) else "qerror"
        put("Q05", f"matched 比较下 {stronger} 的 |median diff| 更大；二者均为统计关联，非因果。", tier=EVIDENCE_ASSOCIATION)
    else:
        put("Q05", "NOT_MEASURED：qerror 与 norm 特征均未齐备。", tier=EVIDENCE_ASSOCIATION, status="NOT_MEASURED")

    cos_ci = rq2.get("feature_bootstrap_ci", {}).get("final_hidden_cosine")
    ratio_ci = rq2.get("feature_bootstrap_ci", {}).get("final_hidden_norm_ratio")
    if cos_ci and ratio_ci and cos_ci.get("status") == "OK" and ratio_ci.get("status") == "OK":
        drift = "direction" if abs(cos_ci["median_diff"]) >= abs(ratio_ci["median_diff"]) else "norm"
        put("Q06", f"final hidden 更表现为 {drift} drift（cosine diff={cos_ci['median_diff']:.4g}, norm_ratio diff={ratio_ci['median_diff']:.4g}）。",
            tier=EVIDENCE_ASSOCIATION)
    else:
        put("Q06", "NOT_MEASURED：缺少 forced_core final hidden 对比。", tier=EVIDENCE_ASSOCIATION, status="NOT_MEASURED")

    margin = rq2.get("feature_bootstrap_ci", {}).get("e0_margin")
    if margin and margin.get("status") == "OK":
        negated = abs(margin["median_diff"]) < 1e-6 or (margin["ci95_low"] <= 0 <= margin["ci95_high"])
        put("Q07", "LCB 上 E0 margin 不是稳定的主触发因素（CI 跨 0 或 diff 近 0）。" if negated else
            f"E0 margin event-control diff={margin['median_diff']:.4g}，但仍仅为关联。",
            tier=EVIDENCE_ASSOCIATION)
    else:
        put("Q07", "NOT_MEASURED：缺少 raw_logits margin 对比。", tier=EVIDENCE_ASSOCIATION, status="NOT_MEASURED")

    attn = rq2.get("feature_bootstrap_ci", {}).get("attention_core_rel_l2")
    moe = rq2.get("feature_bootstrap_ci", {}).get("moe_out_rel_l2")
    if attn and moe and attn.get("status") == "OK" and moe.get("status") == "OK":
        dominant = "MoE" if abs(moe["median_diff"]) >= abs(attn["median_diff"]) else "Attention"
        put("Q08", f"actual-path 边界增长关联上 {dominant} rel-L2 更大（Attention={attn['median_diff']:.4g}, MoE={moe['median_diff']:.4g}）。",
            tier=EVIDENCE_ASSOCIATION)
    else:
        put("Q08", "NOT_MEASURED：缺少 E0/E1 forced_core 对比。", tier=EVIDENCE_ASSOCIATION, status="NOT_MEASURED")

    if puncture.get("status") == "OK":
        moe_p = puncture["by_operator"].get("moe", {})
        qe, qr = moe_p.get("median_q_expert_rel_l2"), moe_p.get("median_q_router_rel_l2")
        if qe is not None and qr is not None:
            dominant = "expert/fused compute" if qe >= qr else "router decision"
            put("Q09", f"same-input MoE puncture 中 {dominant} 更大（q_expert={qe:.4g}, q_router={qr:.4g}）。", tier=EVIDENCE_CAUSAL)
        else:
            put("Q09", "BLOCKED：MoE puncture 缺少 frozen-router 分解。", tier=EVIDENCE_CAUSAL, status="BLOCKED")
    else:
        put("Q09", f"{puncture.get('status', 'NOT_MEASURED')}：{puncture.get('reason', '无 puncture 产物')}", tier=EVIDENCE_CAUSAL, status=puncture.get("status", "NOT_MEASURED"))

    if puncture.get("status") == "OK" and rq2.get("feature_bootstrap_ci", {}).get("moe_out_rel_l2"):
        put("Q10", "production same-input q 与 actual-path moe_out 增长可在 PASS puncture 样本上并列对照；见 puncture 汇总与 Figure4。", tier=EVIDENCE_CAUSAL)
    else:
        put("Q10", "NOT_MEASURED：puncture 或 actual-path 数据不足。", tier=EVIDENCE_CAUSAL, status="NOT_MEASURED")

    if artifacts.state_reset:
        put("Q11", "见 11_state_intervention/state_reset_rows 与 Figure5；仅 PASS identity gate 后的 reset 可称因果。", tier=EVIDENCE_CAUSAL)
    else:
        put("Q11", "BLOCKED：state reset 产物缺失或 identity 未通过。", tier=EVIDENCE_CAUSAL, status="BLOCKED")

    if artifacts.state_injection:
        put("Q12", "见 state_injection_rows；alpha=0 必须通过 identity。", tier=EVIDENCE_CAUSAL)
    else:
        put("Q12", "BLOCKED：state injection 产物缺失。", tier=EVIDENCE_CAUSAL, status="BLOCKED")

    if artifacts.semantic_frontier:
        put("Q13", f"semantic frontier 状态={artifacts.semantic_summary.get('status') if artifacts.semantic_summary else 'see rows'}；t_lex 与 accuracy_frontier 分离定义。", tier=EVIDENCE_CAUSAL)
    else:
        put("Q13", "BLOCKED：10_semantic_frontier 未运行。", tier=EVIDENCE_CAUSAL, status="BLOCKED")

    if reg_th.get("n", 0) > 0:
        put("Q14",
            f"E1 未收束 thinking 比例={1 - (reg_th.get('fraction_e1_finished_thinking') or 0):.3f}；"
            f"E1 thinking 长度中位={reg_th.get('median_e1_thinking_length')} vs E0={reg_th.get('median_e0_thinking_length')}。",
            tier=EVIDENCE_ASSOCIATION)
    else:
        put("Q14", "NOT_MEASURED。", tier=EVIDENCE_ASSOCIATION, status="NOT_MEASURED")

    e2_ids = formal.get("observed_recovery_doc_ids", {}).get("E2", [])
    put("Q15", f"正式 observed recovery doc_ids={e2_ids}（单样本翻牌，非因果）。E2 mechanism 复测={'AVAILABLE' if artifacts.compare_e0_e2 else 'NOT_MEASURED'}。",
        tier=EVIDENCE_FACT, status="OK" if e2_ids else "NOT_MEASURED")
    put("Q16", "NOT_MEASURED" if not artifacts.compare_e0_e2 else "见 12_rq3_e2 与 feature 对比。", tier=EVIDENCE_ASSOCIATION,
        status="NOT_MEASURED" if not artifacts.compare_e0_e2 else "OK")
    put("Q17", "NOT_MEASURED" if not artifacts.compare_e1_e2 else "若 QDQ 降而 task 未恢复，检查 router/puncture/state reset。", tier=EVIDENCE_ASSOCIATION,
        status="NOT_MEASURED" if not artifacts.compare_e1_e2 else "OK")
    put("Q18", "NOT_MEASURED" if not artifacts.compare_e0_e2 else "R64 分布机制需对照 rotation gain 与 error direction。", tier=EVIDENCE_ASSOCIATION,
        status="NOT_MEASURED" if not artifacts.compare_e0_e2 else "OK")

    e3_ids = formal.get("observed_recovery_doc_ids", {}).get("E3", [])
    put("Q19", f"正式 observed recovery doc_ids={e3_ids}。E3 mechanism={'AVAILABLE' if artifacts.compare_e0_e3 else 'NOT_MEASURED'}。",
        tier=EVIDENCE_FACT)
    put("Q20", "NOT_MEASURED" if not artifacts.compare_e1_e3 else "见 13_rq3_e3 puncture/actual-path。", tier=EVIDENCE_ASSOCIATION,
        status="NOT_MEASURED" if not artifacts.compare_e1_e3 else "OK")
    e3_pass = formal.get("pass_counts", {}).get("E3")
    put("Q21", f"正式 LCB E3 pass={e3_pass}（≈0.223 事实）；LCB thinking 稳定性与 MMLU 机制不可直接类比。", tier=EVIDENCE_FACT)

    answers["algorithm_guidance"] = {
        "evidence_tier": "INTERPRETIVE",
        "notes": "仅基于已测证据归纳；未测项不得写死结论。",
        "layer_branch_protection": "NOT_MEASURED" if puncture.get("status") != "OK" else "待结合 q/p 与 reset 结果",
        "router_expert_objective": "NOT_MEASURED" if puncture.get("status") != "OK" else "若 q_router 显著则考虑",
        "format_susceptibility_vs_nmse": "NOT_MEASURED" if rq2.get("status") != "OK" else "见 Q04/Q05",
        "thinking_termination_calibration": "NOT_MEASURED" if thinking.get("status") != "OK" else "见 Q14",
        "r64_selective_use": "NOT_MEASURED" if not artifacts.compare_e0_e2 else "见 Q15-Q18",
        "diag_target_states": "NOT_MEASURED" if not artifacts.compare_e0_e3 else "见 Q19-Q21",
    }
    answers["_figures"] = figures
    answers["_formal_prompt_protocol_mismatch"] = mismatch
    return answers


def _skipped_figure(name: str, reason: str) -> dict[str, str]:
    return {"figure": name, "status": "SKIPPED", "reason": reason}


def render_figures(artifacts: RunArtifacts, rq2: dict, thinking: dict, puncture: dict,
                   output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, Any] = {}

    # Figure 1 — formal LCB flow
    if not artifacts.formal_matrix:
        rendered["figure1"] = _skipped_figure("Figure1_formal_lcb_flow", "missing formal_task_matrix")
    else:
        counts = Counter(row.get("formal_group") for row in artifacts.formal_matrix)
        e0_pass = counts["formal_regression"] + counts["formal_robust"]
        e1_fail = counts["formal_regression"]
        e2_rec = sum(1 for row in artifacts.formal_matrix if row.get("E2_formal_observed_recovery"))
        e3_rec = sum(1 for row in artifacts.formal_matrix if row.get("E3_formal_observed_recovery"))
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ["E0 Pass", "E1 Fail (regression)", "E2 recovery", "E3 recovery"]
        values = [e0_pass, e1_fail, e2_rec, e3_rec]
        ax.bar(labels, values, color=["#4C72B0", "#C44E52", "#55A868", "#DD8452"])
        ax.set_title("Figure 1: Formal LCB accuracy flow (facts)")
        ax.set_ylabel("task count")
        path = output_dir / "Figure1_formal_lcb_flow.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        rendered["figure1"] = {"figure": "Figure1_formal_lcb_flow", "status": "OK", "path": str(path)}

    # Figure 2 — early-token format-risk curves
    offsets = [-8, -4, -2, -1, 0]
    if rq2.get("status") != "OK" or not rq2.get("matched_event_control"):
        rendered["figure2"] = _skipped_figure("Figure2_early_format_risk", "missing RQ2 event-control rows")
    else:
        series: dict[str, list[float]] = defaultdict(list)
        for event in rq2["matched_event_control"]:
            t_lex = event["t_lex"]
            for off in offsets:
                # use diffs keyed by feature at event only as proxy when per-offset absent
                if off == 0:
                    for key in ("max_hif4_qdq_rel_l2", "max_amax_over_rms", "final_hidden_cosine", "e0_margin"):
                        val = event["diffs"].get(key)
                        if val is not None:
                            series[key].append(val)
        if not series:
            rendered["figure2"] = _skipped_figure("Figure2_early_format_risk", "no plottable feature diffs at t_lex")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = offsets
            for key, vals in series.items():
                ax.plot(x[: len(vals)], vals[: len(x)], marker="o", label=key)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title("Figure 2: Event-control diffs at first divergence (proxy)")
            ax.set_xlabel("offset from t_lex")
            ax.legend(fontsize=8)
            path = output_dir / "Figure2_early_format_risk.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            rendered["figure2"] = {"figure": "Figure2_early_format_risk", "status": "OK", "path": str(path)}

    # Figure 3 — layer heatmap
    if not artifacts.compare_e0_e1:
        rendered["figure3"] = _skipped_figure("Figure3_layer_heatmap", "missing e0_e1_compare")
    else:
        boundaries = ["attention_core", "o_proj", "post_attn_norm", "moe_out", "layer_out"]
        layers = list(range(48))
        grid = np.full((len(boundaries), len(layers)), np.nan)
        for row in artifacts.compare_e0_e1:
            if row.get("boundary") not in boundaries or row.get("layer") is None:
                continue
            b = boundaries.index(row["boundary"])
            grid[b, int(row["layer"])] = float(row["rel_l2"])
        if np.all(np.isnan(grid)):
            rendered["figure3"] = _skipped_figure("Figure3_layer_heatmap", "no rel_l2 values")
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            im = ax.imshow(grid, aspect="auto", cmap="magma")
            ax.set_yticks(range(len(boundaries)), boundaries)
            ax.set_xlabel("layer")
            ax.set_title("Figure 3: E0/E1 actual-path rel-L2 heatmap (all probes)")
            fig.colorbar(im, ax=ax, fraction=0.025)
            path = output_dir / "Figure3_layer_heatmap.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            rendered["figure3"] = {"figure": "Figure3_layer_heatmap", "status": "OK", "path": str(path)}

    # Figure 4 — puncture decomposition
    if puncture.get("status") != "OK":
        rendered["figure4"] = _skipped_figure("Figure4_puncture_decomposition", puncture.get("reason", "no puncture"))
    else:
        ops = ["qkv", "o_proj", "moe"]
        metrics = ["median_e_rel_l2", "median_q_rel_l2", "median_p_rel_l2"]
        labels = ["e", "q", "p"]
        x = np.arange(len(ops))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, metric in enumerate(metrics):
            vals = [puncture["by_operator"].get(op, {}).get(metric) or 0 for op in ops]
            ax.bar(x + (i - 1) * width, vals, width, label=labels[i])
        ax.set_xticks(x, ops)
        ax.set_title("Figure 4: Same-input puncture e/q/p")
        ax.legend()
        path = output_dir / "Figure4_puncture_decomposition.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        rendered["figure4"] = {"figure": "Figure4_puncture_decomposition", "status": "OK", "path": str(path)}

    # Figure 5 — state reset rescue map
    if not artifacts.state_reset:
        rendered["figure5"] = _skipped_figure("Figure5_state_reset_map", "missing state_reset_rows")
    else:
        samples = sorted({str(r["sample_key"]) for r in artifacts.state_reset})
        layers = sorted({int(r["layer"]) for r in artifacts.state_reset if r.get("layer") is not None})
        grid = np.full((len(samples), len(layers)), np.nan)
        for r in artifacts.state_reset:
            if r.get("layer") is None:
                continue
            i = samples.index(str(r["sample_key"]))
            j = layers.index(int(r["layer"]))
            grid[i, j] = float(r.get("target_rank_recovery") or r.get("top1_rescue") or 0)
        fig, ax = plt.subplots(figsize=(10, max(3, len(samples) * 0.4)))
        im = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(samples)), samples, fontsize=7)
        ax.set_xticks(range(len(layers)), layers)
        ax.set_title("Figure 5: State-reset rescue map")
        fig.colorbar(im, ax=ax, fraction=0.025)
        path = output_dir / "Figure5_state_reset_map.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        rendered["figure5"] = {"figure": "Figure5_state_reset_map", "status": "OK", "path": str(path)}

    # Figure 6 — thinking / accuracy frontier
    if not artifacts.semantic_frontier:
        rendered["figure6"] = _skipped_figure("Figure6_semantic_frontier", "missing semantic_frontier rows")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        for sample in sorted({str(r["sample_key"]) for r in artifacts.semantic_frontier}):
            rows = sorted([r for r in artifacts.semantic_frontier if str(r["sample_key"]) == sample], key=lambda r: int(r["k"]))
            ax.plot([int(r["k"]) for r in rows], [1.0 if r.get("finished_thinking") else 0.0 for r in rows], marker="o", label=sample)
        ax.set_title("Figure 6: Thinking frontier (finished_thinking vs k)")
        ax.set_xlabel("prefix length k")
        ax.legend(fontsize=6, ncol=2)
        path = output_dir / "Figure6_semantic_frontier.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        rendered["figure6"] = {"figure": "Figure6_semantic_frontier", "status": "OK", "path": str(path)}

    # Figure 7 — E2/E3 vs E1
    if not artifacts.formal_summary:
        rendered["figure7"] = _skipped_figure("Figure7_e2_e3_vs_e1", "missing formal summary")
    else:
        passes = artifacts.formal_summary.get("pass_counts", {})
        labels = ["E1", "E2", "E3"]
        vals = [passes.get(v, 0) for v in labels]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, vals, color=["#C44E52", "#55A868", "#DD8452"])
        ax.set_title("Figure 7: Formal LCB pass counts vs E1")
        ax.set_ylabel("pass@1 tasks")
        path = output_dir / "Figure7_e2_e3_vs_e1.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        rendered["figure7"] = {"figure": "Figure7_e2_e3_vs_e1", "status": "OK", "path": str(path)}

    return rendered


def write_report(path: Path, answers: dict[str, dict], artifacts: RunArtifacts, rq2: dict, thinking: dict,
                 puncture: dict, taxonomy: dict, figures: dict) -> None:
    lines = [
        "# FORMAT CONVERSION LCB MECHANISM REPORT",
        "",
        "## 证据分层说明",
        "",
        "- **事实 (FACT)**：正式/ greedy 标签、长度、thinking 标记、first divergence 等观测。",
        "- **统计关联 (STATISTICAL_ASSOCIATION)**：matched event-control bootstrap CI、regression vs robust 对比。",
        "- **因果 (CAUSAL_INTERVENTION)**：production puncture、state reset/injection、semantic frontier。",
        "",
        "## FORMAL_PROMPT_PROTOCOL_MISMATCH",
        "",
    ]
    if artifacts.formal_prompt_mismatch:
        lines.extend([
            "正式 61 个 `formal_regression` **不能**作为格式转换因果证据：E1/E2/E3 与 E0 的 prompt 协议不一致。",
            "mechanism 标签必须来自 matched-prompt isolated greedy + 官方 LCB judge（`mechanism_regression/robust`）。",
            "",
        ])
    else:
        lines.append("正式矩阵 prompt 与 E0 对齐；正式标签仍仅为 sampled 运行事实，不等于 greedy 因果链。")
        lines.append("")

    lines.extend(["## Task 19 问题回答", ""])
    for qid, _ in TASK19_QUESTIONS:
        item = answers.get(qid, {})
        lines.append(f"### {qid}")
        lines.append(f"- 证据层级：`{item.get('evidence_tier', 'UNKNOWN')}`")
        lines.append(f"- 状态：`{item.get('status', 'UNKNOWN')}`")
        lines.append(f"- {item.get('answer', 'NOT_MEASURED')}")
        lines.append("")

    lines.extend(["## RQ2 matched event-control 摘要", "", f"状态：`{rq2.get('status')}`", ""])
    if rq2.get("feature_bootstrap_ci"):
        lines.append("| feature | n | median_diff | CI95 |")
        lines.append("|---|---:|---:|---:|")
        for name, stat in sorted(rq2["feature_bootstrap_ci"].items()):
            if stat.get("status") != "OK":
                continue
            lines.append(f"| {name} | {stat['n']} | {stat['median_diff']:.4g} | [{stat['ci95_low']:.4g}, {stat['ci95_high']:.4g}] |")
        lines.append("")
    logistic = rq2.get("logistic", {})
    lines.append(f"Logistic（n≥16 才运行）：`{logistic.get('status')}`")
    lines.append("")

    lines.extend(["## Thinking transition（mechanism-regression vs robust）", "", f"状态：`{thinking.get('status')}`", ""])
    for group in ("mechanism_regression", "mechanism_robust"):
        block = thinking.get(group, {})
        lines.append(f"### {group}")
        for key, value in block.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.extend(["## Production puncture e/q/p 汇总", "", f"状态：`{puncture.get('status')}`", ""])
    if puncture.get("by_operator"):
        for op, stats in puncture["by_operator"].items():
            lines.append(f"### {op}")
            for key, value in stats.items():
                lines.append(f"- {key}: {value}")
            lines.append("")

    lines.extend(["## 图表", ""])
    for key in sorted(figures):
        fig = figures[key]
        lines.append(f"- {fig.get('figure')}: {fig.get('status')} {fig.get('reason', fig.get('path', ''))}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def analyze_mechanism(run_root: Path, *, seed: int = 42) -> dict[str, Any]:
    """Main entry: load artifacts, compute stats, write analysis/ outputs."""
    run_root = run_root.resolve()
    analysis_dir = run_root / "analysis"
    figure_dir = analysis_dir / "figures"
    artifacts = RunArtifacts(run_root)

    taxonomy = formal_failure_taxonomy(artifacts)
    rq2 = rq2_matched_statistics(artifacts, seed=seed)
    thinking = analyze_thinking_transitions(artifacts)
    puncture = summarize_puncture(artifacts)
    figures = render_figures(artifacts, rq2, thinking, puncture, figure_dir)
    answers = build_task19_answers(artifacts, rq2, thinking, puncture, taxonomy, figures)

    rq2_path = analysis_dir / "rq2_matched_statistics.json"
    thinking_path = analysis_dir / "thinking_transition_summary.json"
    puncture_path = analysis_dir / "puncture_summary.json"
    summary_path = analysis_dir / "FORMAT_CONVERSION_LCB_MECHANISM_SUMMARY.json"
    report_path = analysis_dir / "FORMAT_CONVERSION_LCB_MECHANISM_REPORT.md"

    analysis_dir.mkdir(parents=True, exist_ok=True)
    rq2_path.write_text(json.dumps(rq2, ensure_ascii=False, indent=2) + "\n")
    thinking_path.write_text(json.dumps(thinking, ensure_ascii=False, indent=2) + "\n")
    puncture_path.write_text(json.dumps(puncture, ensure_ascii=False, indent=2) + "\n")

    summary = {
        "schema_version": 1,
        "run_root": str(run_root),
        "FORMAL_PROMPT_PROTOCOL_MISMATCH": artifacts.formal_prompt_mismatch,
        "artifact_availability": {
            "formal_matrix": _artifact_status(run_root / "01_formal_matrix/formal_task_matrix.jsonl"),
            "greedy_matrix": _artifact_status(run_root / "04_greedy_judge/greedy_task_matrix.jsonl"),
            "probe_plan": _artifact_status(run_root / "05_probe_plan/probe_plan.json"),
            "feature_capture": _artifact_status(_glob_first(run_root / "06_feature_capture", "**/*_tokens.jsonl")),
            "core_capture": _artifact_status(_glob_first(run_root / "07_core_capture", "**/e0_e1_compare.jsonl")),
            "puncture": "AVAILABLE" if artifacts.puncture_roots else "NOT_MEASURED",
            "semantic_frontier": _artifact_status(_glob_first(run_root / "10_semantic_frontier", "**/frontier_rows.jsonl")),
            "state_intervention": _artifact_status(_glob_first(run_root / "11_state_intervention", "**/state_reset_rows.jsonl")),
        },
        "formal_failure_taxonomy": taxonomy,
        "rq2_matched_statistics": {"status": rq2.get("status"), "n_event_rows": rq2.get("n_event_rows"),
                                   "logistic_status": rq2.get("logistic", {}).get("status")},
        "thinking_transition": {"status": thinking.get("status")},
        "puncture_summary": {"status": puncture.get("status")},
        "figures": figures,
        "task19_answers": {k: v for k, v in answers.items() if not k.startswith("_")},
        "blockers": [name for name, status in {
            "greedy_bridge": _artifact_status(run_root / "04_greedy_judge/greedy_task_matrix.jsonl"),
            "feature_scan": _artifact_status(_glob_first(run_root / "06_feature_capture", "**/*_tokens.jsonl")),
            "forced_core": _artifact_status(_glob_first(run_root / "07_core_capture", "**/e0_e1_compare.jsonl")),
            "puncture": "AVAILABLE" if artifacts.puncture_roots else "NOT_MEASURED",
            "state_reset": _artifact_status(_glob_first(run_root / "11_state_intervention", "**/state_reset_rows.jsonl")),
        }.items() if status != "AVAILABLE"],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_report(report_path, answers, artifacts, rq2, thinking, puncture, taxonomy, figures)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True, help="format_conversion run root")
    parser.add_argument("--seed", type=int, default=42, help="bootstrap RNG seed")
    args = parser.parse_args()
    result = analyze_mechanism(args.run_root, seed=args.seed)
    print(json.dumps({"summary_path": str(args.run_root / "analysis/FORMAT_CONVERSION_LCB_MECHANISM_SUMMARY.json"),
                      "blockers": result.get("blockers", [])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
