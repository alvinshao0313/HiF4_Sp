"""Analyze H4 experiment CSVs: Spearman, figures, summary, report."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import read_json, write_json

NMSE_EPS = 1e-30
CASE_COLORS = {
    "Identity": "#4C72B0",
    "DIAG": "#DD8452",
    "H4_FP32": "#55A868",
    "H4_BF16": "#C44E52",
}


def _finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"{name} is not finite: {value}")
    return float(value)


def _median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
    if values.size == 0:
        raise RuntimeError("median of empty series")
    return _finite("median", float(np.median(values)))


def _short_module(name: str) -> str:
    parts = name.split(".")
    layer = parts[parts.index("layers") + 1]
    return f"L{int(layer):02d}.{parts[-1]}"


def _spearman(x: np.ndarray, y: np.ndarray, label: str) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        raise RuntimeError(f"Spearman {label}: fewer than 3 finite pairs (n={n})")
    xs = x[mask]
    ys = y[mask]
    if float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return {
            "label": label,
            "rho": float("nan"),
            "pvalue": float("nan"),
            "n": n,
            "undefined_reason": "zero_variance",
        }
    rho, pvalue = spearmanr(xs, ys)
    return {
        "label": label,
        "rho": float(rho),
        "pvalue": float(pvalue),
        "n": n,
    }


def _verdict(median_ratio: float, fraction_improved: float) -> str:
    if median_ratio <= 0.90 and fraction_improved >= 0.70:
        return "明显有效"
    if 0.90 < median_ratio < 0.98 and fraction_improved > 0.50:
        return "小幅有效"
    if 0.98 <= median_ratio <= 1.02:
        return "基本中性"
    if median_ratio > 1.02:
        return "负收益"
    return "未达任一判定阈值（median 在有效区间但改善层比例不足）"


def _h4_vs_diag_label(median_ratio: float) -> str:
    if median_ratio < 0.98:
        return "更好"
    if median_ratio > 1.02:
        return "更差"
    return "相近"


def _fmt(value: float, digits: int = 6) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:.{digits}g}"


def _answers(summary: dict[str, Any]) -> list[str]:
    act = summary["activation"]
    wgt = summary["weight"]
    out = summary["output"]
    mech = summary["mechanism"]
    vs_diag = _h4_vs_diag_label(out["median_layer_ratio_h4_vs_diag"])
    rho_cf4 = mech["spearman"]["delta_log_nmse_vs_delta_cf4"]["rho"]
    rho_amax = mech["spearman"]["delta_log_nmse_vs_delta_log_amax64"]["rho"]
    rho_s0 = mech["spearman"]["delta_log_nmse_vs_delta_s0"]["rho"]
    act_verb = "降低了" if act["median_layer_ratio_h4_fp32"] < 1.0 else "没有降低"
    wgt_verb = "降低了" if wgt["median_layer_ratio_h4_fp32"] < 1.0 else "没有降低"
    out_verb = "下降了" if out["median_layer_ratio_h4_fp32"] < 1.0 else "没有下降"
    return [
        (
            f"1. 用户给出的 H4 {act_verb}保存 activation 的 HiF4 NMSE："
            f"median layer ratio={_fmt(act['median_layer_ratio_h4_fp32'])}，"
            f"改善层比例={_fmt(act['fraction_layers_improved'])}。"
        ),
        (
            f"2. H4 {wgt_verb}对应 weight 的 HiF4 NMSE："
            f"median layer ratio={_fmt(wgt['median_layer_ratio_h4_fp32'])}，"
            f"改善层比例={_fmt(wgt['fraction_layers_improved'])}。"
        ),
        (
            f"3. 激活和权重同步旋转后，Linear output error {out_verb}："
            f"median layer ratio={_fmt(out['median_layer_ratio_h4_fp32'])}，"
            f"改善层比例={_fmt(out['fraction_layers_improved'])}，"
            f"判定={out['verdict']}。"
        ),
        (
            f"4. 相比现有 DIAG，H4 的 Linear output 是{vs_diag}："
            f"median(H4/DIAG)={_fmt(out['median_layer_ratio_h4_vs_diag'])}。"
        ),
        (
            f"5. group 级 Spearman(Δlog NMSE, ΔCF4)={_fmt(rho_cf4)}，"
            f"与 Δlog amax64={_fmt(rho_amax)}，与 ΔS0={_fmt(rho_s0)}。"
            f"{mech['statement']}"
        ),
    ]


def _mechanism_statement(spearman: dict[str, dict[str, Any]]) -> str:
    rho_cf4 = spearman["delta_log_nmse_vs_delta_cf4"]["rho"]
    rho_amax = spearman["delta_log_nmse_vs_delta_log_amax64"]["rho"]
    rho_s0 = spearman["delta_log_nmse_vs_delta_s0"]["rho"]
    if any(math.isnan(v) for v in (rho_cf4, rho_amax, rho_s0)):
        return "部分 Spearman 因零方差未定义，不能把收益归因到单一 G4/G64 结构变化。"
    if rho_cf4 > 0.2 and rho_amax > 0.2:
        return (
            "Δlog NMSE 与 ΔCF4、Δlog amax64 同向相关，"
            "与“G4 摊平 → peak/amax/S0 更合理”的机制解释一致；这是相关不是因果。"
        )
    if rho_cf4 <= 0.05 and rho_amax <= 0.05:
        return (
            "Δlog NMSE 与 ΔCF4 / Δlog amax64 几乎无相关，"
            "不能把误差变化主要归因于 G4 摊平和 64-group amax。"
        )
    return (
        "机制相关存在但并不强；H4 的误差变化不能只用 CF4/amax64/S0 中的单一项解释。"
    )


def _special_cases(layer: pd.DataFrame, group: pd.DataFrame, summary: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    act_imp = float(summary["activation"]["median_layer_ratio_h4_fp32"]) < 0.98
    wgt_worse = float(summary["weight"]["median_layer_ratio_h4_fp32"]) > 1.02
    out_flat = 0.98 <= float(summary["output"]["median_layer_ratio_h4_fp32"]) <= 1.02
    if act_imp and wgt_worse:
        notes.append(
            "情况 A：activation 改善、weight 恶化。同一个 R4 对激活和权重的最佳方向不一致。"
        )
        if out_flat or float(summary["output"]["median_layer_ratio_h4_fp32"]) >= 0.98:
            notes.append(
                "因此不能写“H4 有效”。H4 能改善激活的 HiF4-friendly 分布，"
                "但激活收益被权重量化损失抵消，当前等价 Linear 方案无净收益。"
            )
    act_ratio = float(summary["activation"]["median_layer_ratio_h4_fp32"])
    out_ratio = float(summary["output"]["median_layer_ratio_h4_fp32"])
    if 0.98 <= act_ratio <= 1.02 and out_ratio < 0.98:
        notes.append(
            "情况 B：activation NMSE 几乎不变，但 output 改善。"
            "量化误差方向可能与 Linear 敏感方向一起变了，不能只看 raw activation NMSE。"
        )
    cf4_drop = float(layer["cf4_mean_after"].median()) < float(layer["cf4_mean_before"].median())
    if cf4_drop and act_ratio >= 0.98:
        notes.append(
            "情况 C：G4 crest 下降但 HiF4 NMSE 不降。"
            "需要看 S0 rounding、e8/e4 跳变和 payload=1.75 clipping，而不是只看 CF4。"
        )
    ratios = layer["output_ratio_h4_fp32"].to_numpy(dtype=np.float64)
    strong = int(np.sum(ratios <= 0.90))
    if strong <= max(1, int(0.2 * len(ratios))) and float(np.median(ratios)) >= 0.98:
        notes.append(
            "情况 D：少数层收益很大、大部分层中性。本实验不实现 layer-selective H4。"
        )
    clip_before = float(group["clip_rate_before"].mean())
    clip_after = float(group["clip_rate_after"].mean())
    e8_before = float(group["e8_rate_before"].mean())
    e8_after = float(group["e8_rate_after"].mean())
    notes.append(
        f"HiF4 metadata 均值：clip_rate { _fmt(clip_before)} → {_fmt(clip_after)}，"
        f"e8_rate {_fmt(e8_before)} → {_fmt(e8_after)}，"
        f"e4_rate {_fmt(float(group['e4_rate_before'].mean()))} → "
        f"{_fmt(float(group['e4_rate_after'].mean()))}。"
    )
    return notes


def _bar_by_module(
    path: Path,
    layer: pd.DataFrame,
    columns: dict[str, str],
    ylabel: str,
    title: str,
) -> None:
    labels = [_short_module(n) for n in layer["module_name"]]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(18, 6))
    offsets = np.linspace(-1.5, 1.5, num=len(columns)) * width
    for off, (legend, col) in zip(offsets, columns.items()):
        ax.bar(
            x + off,
            layer[col].to_numpy(dtype=np.float64),
            width=width,
            label=legend,
            color=CASE_COLORS[legend],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_ratio_hist(path: Path, ratios: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    finite = ratios[np.isfinite(ratios)]
    ax.hist(finite, bins=60, color="#4C72B0", alpha=0.85)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="ratio=1")
    ax.set_xlabel("group NMSE_H4 / NMSE_Identity")
    ax.set_ylabel("count")
    ax.set_title("Group-level H4 / Identity NMSE ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cf4_hist(path: Path, hist: pd.DataFrame) -> None:
    centers = 0.5 * (hist["bin_left"].to_numpy() + hist["bin_right"].to_numpy())
    width = float((hist["bin_right"] - hist["bin_left"]).mean())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        centers - 0.15 * width,
        hist["count_before"].to_numpy(),
        width=0.7 * width,
        label="before H4",
        color="#4C72B0",
        alpha=0.8,
    )
    ax.bar(
        centers + 0.15 * width,
        hist["count_after"].to_numpy(),
        width=0.7 * width,
        label="after H4",
        color="#55A868",
        alpha=0.8,
    )
    ax.set_xlabel("G4 crest factor")
    ax.set_ylabel("count")
    ax.set_title("G4 crest factor before vs after H4")
    ax.set_xlim(1.0, 2.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _hexbin(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(x[mask], y[mask], gridsize=50, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="count")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    layer_path = run_dir / "layer_metrics.csv"
    group_path = run_dir / "group_metrics.csv"
    hist_path = run_dir / "cf4_hist.csv"
    config = read_json(run_dir / "config.json")
    if not layer_path.is_file() or not group_path.is_file():
        raise FileNotFoundError(f"missing metrics under {run_dir}")
    layer = pd.read_csv(layer_path)
    group = pd.read_csv(group_path)
    hist = pd.read_csv(hist_path)
    if layer.empty or group.empty:
        raise RuntimeError("empty layer_metrics or group_metrics")

    n_layers = int(len(layer))
    act_ratio = layer["act_ratio_h4_fp32"].to_numpy(dtype=np.float64)
    wgt_ratio = layer["weight_ratio_h4_fp32"].to_numpy(dtype=np.float64)
    out_ratio = layer["output_ratio_h4_fp32"].to_numpy(dtype=np.float64)
    out_ratio_diag = layer["output_ratio_h4_vs_diag"].to_numpy(dtype=np.float64)
    group_ratio = group["nmse_ratio"].to_numpy(dtype=np.float64)

    activation = {
        "median_layer_ratio_h4_fp32": _median(layer["act_ratio_h4_fp32"]),
        "median_layer_ratio_h4_bf16": _median(layer["act_ratio_h4_bf16"]),
        "median_layer_ratio_diag": _median(layer["act_ratio_diag"]),
        "fraction_layers_improved": _finite(
            "act_frac", float(np.mean(act_ratio < 1.0))
        ),
        "best_layer_ratio": _finite("act_best", float(np.min(act_ratio))),
        "worst_layer_ratio": _finite("act_worst", float(np.max(act_ratio))),
    }
    weight = {
        "median_layer_ratio_h4_fp32": _median(layer["weight_ratio_h4_fp32"]),
        "median_layer_ratio_h4_bf16": _median(layer["weight_ratio_h4_bf16"]),
        "median_layer_ratio_diag": _median(layer["weight_ratio_diag"]),
        "fraction_layers_improved": _finite(
            "wgt_frac", float(np.mean(wgt_ratio < 1.0))
        ),
        "best_layer_ratio": _finite("wgt_best", float(np.min(wgt_ratio))),
        "worst_layer_ratio": _finite("wgt_worst", float(np.max(wgt_ratio))),
    }
    fraction_out_improved = _finite("out_frac", float(np.mean(out_ratio < 1.0)))
    median_out = _median(layer["output_ratio_h4_fp32"])
    output = {
        "median_layer_ratio_h4_fp32": median_out,
        "median_layer_ratio_h4_bf16": _median(layer["output_ratio_h4_bf16"]),
        "median_layer_ratio_diag": _median(layer["output_ratio_diag"]),
        "median_layer_ratio_h4_vs_diag": _median(layer["output_ratio_h4_vs_diag"]),
        "fraction_layers_improved": fraction_out_improved,
        "best_layer_ratio": _finite("out_best", float(np.min(out_ratio))),
        "worst_layer_ratio": _finite("out_worst", float(np.max(out_ratio))),
        "verdict": _verdict(median_out, fraction_out_improved),
        "h4_vs_diag": _h4_vs_diag_label(_median(layer["output_ratio_h4_vs_diag"])),
    }

    pos_mask = (group["nmse_identity"] > 0) & (group["nmse_h4"] > 0)
    amax_mask = (group["amax64_before"] > 0) & (group["amax64_after"] > 0)
    dlog_nmse = np.log(group.loc[pos_mask, "nmse_h4"].to_numpy(dtype=np.float64)) - np.log(
        group.loc[pos_mask, "nmse_identity"].to_numpy(dtype=np.float64)
    )
    d_cf4 = (
        group.loc[pos_mask, "cf4_mean_after"].to_numpy(dtype=np.float64)
        - group.loc[pos_mask, "cf4_mean_before"].to_numpy(dtype=np.float64)
    )
    d_cf64 = (
        group.loc[pos_mask, "cf64_after"].to_numpy(dtype=np.float64)
        - group.loc[pos_mask, "cf64_before"].to_numpy(dtype=np.float64)
    )
    d_s0 = (
        group.loc[pos_mask, "s0_after"].to_numpy(dtype=np.float64)
        - group.loc[pos_mask, "s0_before"].to_numpy(dtype=np.float64)
    )
    d_e8 = (
        group.loc[pos_mask, "e8_rate_after"].to_numpy(dtype=np.float64)
        - group.loc[pos_mask, "e8_rate_before"].to_numpy(dtype=np.float64)
    )
    d_e4 = (
        group.loc[pos_mask, "e4_rate_after"].to_numpy(dtype=np.float64)
        - group.loc[pos_mask, "e4_rate_before"].to_numpy(dtype=np.float64)
    )
    dlog_amax = np.log(
        group.loc[pos_mask & amax_mask, "amax64_after"].to_numpy(dtype=np.float64)
    ) - np.log(group.loc[pos_mask & amax_mask, "amax64_before"].to_numpy(dtype=np.float64))
    dlog_nmse_amax = np.log(
        group.loc[pos_mask & amax_mask, "nmse_h4"].to_numpy(dtype=np.float64)
    ) - np.log(group.loc[pos_mask & amax_mask, "nmse_identity"].to_numpy(dtype=np.float64))

    spearman = {
        "delta_log_nmse_vs_delta_cf4": _spearman(dlog_nmse, d_cf4, "dlogNMSE vs dCF4"),
        "delta_log_nmse_vs_delta_cf64": _spearman(dlog_nmse, d_cf64, "dlogNMSE vs dCF64"),
        "delta_log_nmse_vs_delta_log_amax64": _spearman(
            dlog_nmse_amax, dlog_amax, "dlogNMSE vs dlog amax64"
        ),
        "delta_log_nmse_vs_delta_s0": _spearman(dlog_nmse, d_s0, "dlogNMSE vs dS0"),
        "delta_log_nmse_vs_delta_e8_rate": _spearman(dlog_nmse, d_e8, "dlogNMSE vs de8"),
        "delta_log_nmse_vs_delta_e4_rate": _spearman(dlog_nmse, d_e4, "dlogNMSE vs de4"),
        "n_groups_total": int(len(group)),
        "n_groups_positive_nmse": int(pos_mask.sum()),
    }
    mechanism = {
        "fraction_groups_improved": _finite(
            "gfrac", float(np.mean(group_ratio < 1.0))
        ),
        "group_nmse_ratio_median": _finite("gmed", float(np.median(group_ratio))),
        "group_nmse_ratio_p90": _finite("gp90", float(np.quantile(group_ratio, 0.9))),
        "group_nmse_ratio_p99": _finite("gp99", float(np.quantile(group_ratio, 0.99))),
        "cf4_mean_before": _finite("cf4b", float(group["cf4_mean_before"].mean())),
        "cf4_mean_after": _finite("cf4a", float(group["cf4_mean_after"].mean())),
        "amax64_ratio_median": _finite(
            "amaxr", float(np.median(group["amax64_ratio"].to_numpy(dtype=np.float64)))
        ),
        "spearman": spearman,
        "statement": "",
    }
    mechanism["statement"] = _mechanism_statement(spearman)

    summary = {
        "experiment": "hif4_h4_block_rotation",
        "run_id": config["run_id"],
        "capture_run_id": config["capture_run_id"],
        "smoke": bool(config.get("smoke", False)),
        "split": config["split"],
        "num_layers": n_layers,
        "num_groups": int(len(group)),
        "activation": activation,
        "weight": weight,
        "output": output,
        "mechanism": mechanism,
        "headline": {
            "median_layer_activation_ratio": activation["median_layer_ratio_h4_fp32"],
            "median_layer_weight_ratio": weight["median_layer_ratio_h4_fp32"],
            "median_layer_output_ratio": output["median_layer_ratio_h4_fp32"],
            "fraction_layers_activation_improved": activation["fraction_layers_improved"],
            "fraction_layers_output_improved": output["fraction_layers_improved"],
            "fraction_groups_improved": mechanism["fraction_groups_improved"],
            "worst_layer_output_ratio": output["worst_layer_ratio"],
            "best_layer_output_ratio": output["best_layer_ratio"],
        },
        "note": {
            "main_case": "H4_FP32",
            "h4_bf16": "deployment sanity only; not mixed into the main verdict",
            "y_ref": "Linear(A_N, W_N, bias) from the existing Linear puncture path",
        },
    }
    summary["answers"] = _answers(summary)
    summary["special_cases"] = _special_cases(layer, group, summary)

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _bar_by_module(
        fig_dir / "fig01_activation_nmse_by_layer.png",
        layer,
        {
            "Identity": "act_nmse_identity",
            "DIAG": "act_nmse_diag",
            "H4_FP32": "act_nmse_h4_fp32",
            "H4_BF16": "act_nmse_h4_bf16",
        },
        ylabel="activation HiF4 NMSE",
        title="Activation HiF4 NMSE by Linear",
    )
    _bar_by_module(
        fig_dir / "fig02_output_nmse_by_layer.png",
        layer,
        {
            "Identity": "output_nmse_identity",
            "DIAG": "output_nmse_diag",
            "H4_FP32": "output_nmse_h4_fp32",
            "H4_BF16": "output_nmse_h4_bf16",
        },
        ylabel="Linear output NMSE vs Y_NN",
        title="Linear output NMSE by Linear",
    )
    _plot_ratio_hist(fig_dir / "fig03_group_nmse_ratio_hist.png", group_ratio)
    _plot_cf4_hist(fig_dir / "fig04_cf4_before_after.png", hist)
    _hexbin(
        fig_dir / "fig05_amax64_before_after.png",
        group["amax64_before"].to_numpy(dtype=np.float64),
        group["amax64_after"].to_numpy(dtype=np.float64),
        xlabel="amax64 before H4",
        ylabel="amax64 after H4",
        title="64-group amax before vs after H4",
    )
    gain = 1.0 - group["nmse_ratio"].to_numpy(dtype=np.float64)
    dcf4_all = (
        group["cf4_mean_after"].to_numpy(dtype=np.float64)
        - group["cf4_mean_before"].to_numpy(dtype=np.float64)
    )
    _hexbin(
        fig_dir / "fig06_nmse_gain_vs_cf4_change.png",
        dcf4_all,
        gain,
        xlabel="Δ CF4 mean (after - before)",
        ylabel="1 - NMSE_H4 / NMSE_Identity",
        title="Group NMSE gain vs G4 crest-factor change",
    )

    write_json(run_dir / "summary.json", summary)
    _write_report(run_dir / "report.md", summary, layer)
    return summary


def _write_report(path: Path, summary: dict[str, Any], layer: pd.DataFrame) -> None:
    answers = "\n".join(summary["answers"])
    special = "\n".join(f"- {s}" for s in summary["special_cases"])
    act = summary["activation"]
    wgt = summary["weight"]
    out = summary["output"]
    mech = summary["mechanism"]
    sp = mech["spearman"]
    rows = [
        "| module | act ratio | weight ratio | output Identity | output DIAG | output H4_FP32 | output ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in layer.iterrows():
        rows.append(
            f"| {row['module_name']} | {_fmt(float(row['act_ratio_h4_fp32']))} | "
            f"{_fmt(float(row['weight_ratio_h4_fp32']))} | "
            f"{_fmt(float(row['output_nmse_identity']))} | "
            f"{_fmt(float(row['output_nmse_diag']))} | "
            f"{_fmt(float(row['output_nmse_h4_fp32']))} | "
            f"{_fmt(float(row['output_ratio_h4_fp32']))} |"
        )
    text = f"""# H4 4维 Hadamard 块旋转实验报告

{answers}

主判定只用 H4_FP32 的 **median layer output ratio**，不把 H4_BF16 混进主结论。
当前判定：**{out['verdict']}**。

## 数据语义

- 保存激活：`X_rot`（checkpoint 在线 16×16 block rotation 之后、NVFP4 量化之前的 BF16）
- 转换源：对 `X_rot` 和反量化 `W_N` 做 HiF4，不撤销在线旋转，不用另一套 BF16 reference 当 conversion source
- Linear 对照：`Y_NN = Linear(A_N, W_N, bias)`，`A_N = Q_NVFP4(X_rot)`，与现有 Linear puncture / E4 / E5 相同
- DIAG：复用已有 `diagonal_scales/*.pt`，不重新搜索
- 评估 split：`{summary['split']}`；smoke={summary['smoke']}
- capture run：`{summary['capture_run_id']}`
- 本实验 run：`{summary['run_id']}`
- 层数：{summary['num_layers']}；64-group 行数：{summary['num_groups']}

## Activation HiF4 NMSE

- Identity / DIAG / H4_FP32 / H4_BF16 都是量化域自身 NMSE：`||Q(X')-X'||^2 / ||X'||^2`
- H4_FP32 median layer ratio = `{_fmt(act['median_layer_ratio_h4_fp32'])}`
- H4_BF16 median layer ratio = `{_fmt(act['median_layer_ratio_h4_bf16'])}`
- DIAG median layer ratio = `{_fmt(act['median_layer_ratio_diag'])}`
- 改善层比例 = `{_fmt(act['fraction_layers_improved'])}`

## Weight HiF4 NMSE

- H4_FP32 median layer ratio = `{_fmt(wgt['median_layer_ratio_h4_fp32'])}`
- H4_BF16 median layer ratio = `{_fmt(wgt['median_layer_ratio_h4_bf16'])}`
- DIAG median layer ratio = `{_fmt(wgt['median_layer_ratio_diag'])}`
- 改善层比例 = `{_fmt(wgt['fraction_layers_improved'])}`

## Linear output NMSE（相对 Y_NN）

- Identity→H4_FP32 median ratio = `{_fmt(out['median_layer_ratio_h4_fp32'])}`
- Identity→H4_BF16 median ratio = `{_fmt(out['median_layer_ratio_h4_bf16'])}`
- Identity→DIAG median ratio = `{_fmt(out['median_layer_ratio_diag'])}`
- H4_FP32 vs DIAG median ratio = `{_fmt(out['median_layer_ratio_h4_vs_diag'])}`（{out['h4_vs_diag']}）
- 改善层比例 = `{_fmt(out['fraction_layers_improved'])}`
- best / worst layer output ratio = `{_fmt(out['best_layer_ratio'])}` / `{_fmt(out['worst_layer_ratio'])}`

## 机制统计

- 64-group 改善比例 = `{_fmt(mech['fraction_groups_improved'])}`
- group NMSE ratio median / p90 / p99 = `{_fmt(mech['group_nmse_ratio_median'])}` / `{_fmt(mech['group_nmse_ratio_p90'])}` / `{_fmt(mech['group_nmse_ratio_p99'])}`
- mean CF4 before → after = `{_fmt(mech['cf4_mean_before'])}` → `{_fmt(mech['cf4_mean_after'])}`
- median amax64_ratio = `{_fmt(mech['amax64_ratio_median'])}`
- Spearman Δlog NMSE vs ΔCF4 = `{_fmt(sp['delta_log_nmse_vs_delta_cf4']['rho'])}`
- Spearman Δlog NMSE vs ΔCF64 = `{_fmt(sp['delta_log_nmse_vs_delta_cf64']['rho'])}`
- Spearman Δlog NMSE vs Δlog amax64 = `{_fmt(sp['delta_log_nmse_vs_delta_log_amax64']['rho'])}`
- Spearman Δlog NMSE vs ΔS0 = `{_fmt(sp['delta_log_nmse_vs_delta_s0']['rho'])}`
- Spearman Δlog NMSE vs Δe8_rate = `{_fmt(sp['delta_log_nmse_vs_delta_e8_rate']['rho'])}`
- Spearman Δlog NMSE vs Δe4_rate = `{_fmt(sp['delta_log_nmse_vs_delta_e4_rate']['rho'])}`

{mech['statement']}

相关不是因果。

## 特殊现象

{special}

## 逐层表（H4_FP32）

{chr(10).join(rows)}

## 图

- `figures/fig01_activation_nmse_by_layer.png`
- `figures/fig02_output_nmse_by_layer.png`
- `figures/fig03_group_nmse_ratio_hist.png`
- `figures/fig04_cf4_before_after.png`
- `figures/fig05_amax64_before_after.png`
- `figures/fig06_nmse_gain_vs_cf4_change.png`

原始表：`layer_metrics.csv`、`group_metrics.csv`、`cf4_hist.csv`、`resolved_inputs.json`。
"""
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze H4 block-rotation results")
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args(argv)
    summary = analyze_run(Path(args.run_dir))
    print("\n".join(summary["answers"]))


if __name__ == "__main__":
    main()
