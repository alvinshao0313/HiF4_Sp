from __future__ import annotations

from statistics import median
from typing import Any

from Block_Sparse.input_mask_proxy_study.config import MethodId

_CANDIDATES = [
    MethodId.XPROXY_EXACT_OWN_OUTPUT.value,
    MethodId.XPROXY_ENERGY_OWN_OUTPUT.value,
    MethodId.FULL_ENERGY_REF_OUTPUT.value,
    MethodId.XWPROXY_EXACT_REF_OUTPUT.value,
    MethodId.XWPROXY_EXACT_OWN_OUTPUT.value,
    MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT.value,
    MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT.value,
]


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _agg_method_metrics(condition_summary: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in condition_summary:
        by_method.setdefault(row["method_id"], []).append(row)
    out: dict[str, dict[str, float]] = {}
    for mid, rows in by_method.items():
        out[mid] = {
            "mean_input_overlap_to_m1": _mean(
                [float(r["input_overlap_to_m1_median"]) for r in rows]
            ),
            "mean_real_output_nrmse_regret": _mean(
                [float(r["nrmse_regret_vs_m1_median"]) for r in rows]
            ),
        }
    return out


def _latency_medians(
    latency_rows: list[dict[str, Any]],
    scope: str,
) -> dict[str, float]:
    by_method: dict[str, list[float]] = {}
    for row in latency_rows:
        if row["timing_scope"] != scope:
            continue
        if row["output_keep_ratio"] == "" or row["input_keep_ratio"] == "":
            continue
        by_method.setdefault(row["method_id"], []).append(float(row["median_ms"]))
    return {m: float(median(vs)) for m, vs in by_method.items() if vs}


def select_winners(
    condition_summary: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = _agg_method_metrics(condition_summary)
    recovery_ms = _latency_medians(latency_rows, "input_recovery_ms")
    online_ms = _latency_medians(latency_rows, "online_total_ms")

    def fidelity_key(mid: str) -> tuple:
        m = metrics[mid]
        return (
            -m["mean_input_overlap_to_m1"],
            m["mean_real_output_nrmse_regret"],
            recovery_ms.get(mid, float("inf")),
            mid,
        )

    fidelity_winner = sorted(_CANDIDATES, key=fidelity_key)[0]

    def recovery_speed_key(mid: str) -> tuple:
        return (
            recovery_ms.get(mid, float("inf")),
            -metrics[mid]["mean_input_overlap_to_m1"],
            mid,
        )

    recovery_speed_winner = sorted(_CANDIDATES, key=recovery_speed_key)[0]

    def pipeline_speed_key(mid: str) -> tuple:
        return (
            online_ms.get(mid, float("inf")),
            -metrics[mid]["mean_input_overlap_to_m1"],
            mid,
        )

    pipeline_speed_winner = sorted(_CANDIDATES, key=pipeline_speed_key)[0]

    # Recommendation gate
    overlaps = {m: metrics[m]["mean_input_overlap_to_m1"] for m in _CANDIDATES}
    o_best = max(overlaps.values())
    eligible = [
        m
        for m in _CANDIDATES
        if overlaps[m] >= o_best - 0.02
        and metrics[m]["mean_real_output_nrmse_regret"] <= 0.01
    ]
    if not eligible:
        recommended = "no_method_meets_recommendation_gate"
    else:
        recommended = sorted(
            eligible,
            key=lambda m: (
                recovery_ms.get(m, float("inf")),
                online_ms.get(m, float("inf")),
                m,
            ),
        )[0]

    return {
        "candidate_mask_fidelity_winner": fidelity_winner,
        "recovery_speed_winner": recovery_speed_winner,
        "prototype_pipeline_speed_winner": pipeline_speed_winner,
        "recommended_method": recommended,
        "method_aggregates": {
            m: {
                **metrics[m],
                "input_recovery_ms_median": recovery_ms.get(m),
                "online_total_ms_median": online_ms.get(m),
            }
            for m in [MethodId.FULL_EXACT_REF.value, *_CANDIDATES]
            if m in metrics
        },
    }


def pareto_front(
    points: list[tuple[str, float, float]],
) -> list[dict[str, Any]]:
    """Minimize x, maximize y. Return non-dominated points."""
    front: list[tuple[str, float, float]] = []
    for mid, x, y in points:
        dominated = False
        for _, x2, y2 in points:
            if (x2 <= x and y2 >= y) and (x2 < x or y2 > y):
                dominated = True
                break
        if not dominated:
            front.append((mid, x, y))
    front.sort(key=lambda t: (t[1], -t[2], t[0]))
    return [{"method_id": m, "x": x, "y": y} for m, x, y in front]


def build_aggregate_summary(
    condition_summary: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    m7_vs_m3: dict[str, Any] | None = None,
    m8_vs_m3: dict[str, Any] | None = None,
) -> dict[str, Any]:
    winners = select_winners(condition_summary, latency_rows)
    recovery_ms = _latency_medians(latency_rows, "input_recovery_ms")
    online_ms = _latency_medians(latency_rows, "online_total_ms")
    metrics = _agg_method_metrics(condition_summary)
    recovery_points = [
        (m, recovery_ms[m], metrics[m]["mean_input_overlap_to_m1"])
        for m in _CANDIDATES
        if m in recovery_ms
    ]
    online_points = [
        (m, online_ms[m], metrics[m]["mean_input_overlap_to_m1"])
        for m in _CANDIDATES
        if m in online_ms
    ]
    out = {
        **winners,
        "pareto_recovery_ms_vs_overlap": pareto_front(recovery_points),
        "pareto_online_total_ms_vs_overlap_pytorch_prototype": pareto_front(online_points),
        "m1_reference_point": {
            "method_id": MethodId.FULL_EXACT_REF.value,
            "input_recovery_ms_median": recovery_ms.get(MethodId.FULL_EXACT_REF.value),
            "online_total_ms_median": online_ms.get(MethodId.FULL_EXACT_REF.value),
            "mean_input_overlap_to_m1": 1.0,
        },
    }
    if m7_vs_m3 is not None:
        out["m7_vs_m3"] = m7_vs_m3
    if m8_vs_m3 is not None:
        out["m8_vs_m3"] = m8_vs_m3
    return out


def render_report(
    *,
    aggregate: dict[str, Any],
    condition_summary: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    m7 = aggregate.get("m7_vs_m3", {})
    m8 = aggregate.get("m8_vs_m3", {})
    lines = [
        "# Qwen3.5-4B Proxy Input Mask Recovery Ablation Report",
        "",
        "## 1. 实验问题",
        "比较八种从输出块 mask 反推输入 K 块 mask 的方法，区分输出 mask 误差、输入反推误差与在线开销。",
        "",
        "## 2. 八种方法定义",
        "- M1 `full_exact_ref`: MY_ref + (X,W) exact",
        "- M2 `xproxy_exact_own_output`: MY_xp + (Xp,W) exact",
        "- M3 `xproxy_energy_own_output`: MY_xp + mean(Xp²)×(MY@offline mean(W²)) energy",
        "- M4 `full_energy_ref_output`: MY_ref + mean(X²)×offline mean(W²) energy",
        "- M5 `xwproxy_exact_ref_output`: MY_ref + (Xp,Wp) exact",
        "- M6 `xwproxy_exact_own_output`: MY_xpwp + (Xp,Wp) exact",
        "- M7 `xproxy_s0mean_energy_own_output`: MY_xp + mean(S0)×(MY@offline mean(W²))",
        "- M8 `xproxy_energy_unconditioned_own_output`: MY_xp 最终输出 + mean(Xp²)×sum_j E_W（输入评分不乘 MY）",
        "",
        "## 3. 数据与分块",
        f"- module: `{manifest.get('module_name')}`",
        f"- X shape: {manifest.get('activation_shape')}",
        f"- W shape: {manifest.get('weight_shape')}",
        f"- sample_indices: {manifest.get('sample_indices')}",
        "",
        "## 4. HiF4 三值代理",
        "激活/权重沿 K 每 64 分组，使用 S0/e8/e4 与 payload 零点决定的三值 code；proxy = local_scale × code。",
        "M7 复用同一次 `build_hif4_ternary_proxy(X)` 产生的 S0，不额外构建代理。",
        "",
        "## 5. 输出 mask 准确率",
        "见 `condition_summary.csv` 中 `output_overlap_to_ref_*`。M1/M4/M5 应对 MY_ref 精确为 1。",
        "M2/M3/M7/M8 共享同一份 MY_xp。",
        "",
        "## 6. 输入 mask 与 M1",
        "见 `input_overlap_to_m1_*`。",
        "",
        "## 7. 条件 oracle",
        "见 `input_overlap_to_conditional_oracle_*`。M2/M3/M7/M8 共用 MX_cond_xp。",
        "",
        "## 8. 真实输出重构",
        "所有方法统一用真实 X/W 评价 `real_output_nrmse_*` 与 `nrmse_regret_vs_m1_*`。",
        "M8 最终 joint sparse 仍使用 MY_xp；去掉 MY 只作用于输入 recovery score。",
        "",
        "## 9. 在线/离线耗时",
        "见 `latency.csv`。`input_recovery_ms` 是主速度指标；`online_total_ms` 是 PyTorch 原型流水线指标，不等于融合 kernel 延迟。",
        "权重块能量离线预计算；M3/M4/M7/M8 的 `input_recovery_ms` 不再重算 W energy。",
        "诊断 scope `activation_statistic_ms` 比较 M3/M4/M7/M8。",
        "",
        "## S0 mean vs proxy block energy",
        f"- M7 vs M3 input mask overlap mean: `{m7.get('input_mask_overlap_mean')}`",
        f"- input_recovery_speedup (M3/M7): `{m7.get('input_recovery_speedup')}`",
        "M7 只优化反推阶段的激活 statistic，不消除 proxy-output GEMM 主开销。",
        "",
        "## MY-conditioning ablation (M8 vs M3)",
        f"- M8 vs M3 input mask overlap mean: `{m8.get('input_mask_overlap_mean')}`",
        f"- overlap_to_m1_delta_mean (M8-M3): `{m8.get('overlap_to_m1_delta_mean')}`",
        f"- conditional_overlap_delta_mean (M8-M3): `{m8.get('conditional_overlap_delta_mean')}`",
        f"- real_output_nrmse_delta_mean (M8-M3): `{m8.get('real_output_nrmse_delta_mean')}`",
        f"- input_recovery_speedup (M3/M8): `{m8.get('input_recovery_speedup')}`",
        f"- online_total_speedup (M3/M8): `{m8.get('online_total_speedup')}`",
        f"- by_output_keep_ratio: `{m8.get('by_output_keep_ratio')}`",
        "M8 不宣称删除 proxy-output GEMM；online_total 仍包含 MY_xp 生成。",
        "",
        "## 10. Candidate mask fidelity winner",
        f"- `{aggregate['candidate_mask_fidelity_winner']}`",
        "",
        "## 11. Recovery speed winner",
        f"- `{aggregate['recovery_speed_winner']}`",
        "",
        "## 12. PyTorch prototype pipeline speed winner",
        f"- `{aggregate['prototype_pipeline_speed_winner']}`",
        "",
        "## 13. Recommended method",
        f"- `{aggregate['recommended_method']}`",
        "",
        "## 14. Pareto frontiers",
        f"- recovery: {aggregate['pareto_recovery_ms_vs_overlap']}",
        f"- online prototype: {aggregate['pareto_online_total_ms_vs_overlap_pytorch_prototype']}",
        "",
        "## 15. 原型限制",
        "- M1 是 reference，不是部署方法。",
        "- PyTorch 原型延迟 ≠ 最终融合 HiF4 sparse kernel 延迟。",
        "- 本实验不实现最终稀疏 kernel，不做完整模型下游评测。",
        "- M7/M8 都不解决 proxy-output GEMM 主开销。",
        "",
        f"condition_summary rows: {len(condition_summary)}",
    ]
    return "\n".join(lines) + "\n"
