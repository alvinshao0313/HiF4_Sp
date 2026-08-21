"""Offline HiF4 K=64 group activation stats before/after DIAG.

Reuses saved X_rot captures and searched diagonal d. Does not load the 8B
model, weights, or re-run diagonal search.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from Native_NVFP4_HiF4_Linear_Puncture.src.config import AppConfig, load_config, results_dir
from Native_NVFP4_HiF4_Linear_Puncture.src.grid_scale_validation import (
    REQUIRED_FORMAL_LAYERS,
    REQUIRED_MODULE_COUNT,
    SPLITS,
    assert_distinct_run_ids,
    capture_file_path,
    validate_capture_manifest,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import (
    ensure_dir,
    load_pt,
    module_capture_stem,
    read_json,
    write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.metrics import compute_recovery

GROUP_SIZE = 64
METRIC_NAMES = ("variance", "kurtosis", "max_mid", "divergence")
SUBSETS = ("all", "kept", "rollback")
HIGHLIGHT_MODULES = (
    "model.layers.2.mlp.gate_proj",
    "model.layers.2.mlp.up_proj",
    "model.layers.2.mlp.down_proj",
)
HIGHLIGHT_LABELS = {
    "model.layers.2.mlp.gate_proj": "L2.gate_proj",
    "model.layers.2.mlp.up_proj": "L2.up_proj",
    "model.layers.2.mlp.down_proj": "L2.down_proj",
}

PER_MODULE_COLUMNS = [
    "run_id",
    "source_capture_run_id",
    "split",
    "layer_idx",
    "projection",
    "module_name",
    "subset",
    "num_tokens",
    "num_k_groups",
    "num_k_groups_in_subset",
    "num_token_groups",
    "e4_error_energy",
    "e5_error_energy",
    "e4_nmse",
    "e5_nmse",
    "e5_recovery",
    "variance_before_median",
    "variance_after_median",
    "variance_delta_median",
    "variance_before_mean",
    "variance_after_mean",
    "variance_delta_mean",
    "kurtosis_before_median",
    "kurtosis_after_median",
    "kurtosis_delta_median",
    "kurtosis_before_mean",
    "kurtosis_after_mean",
    "kurtosis_delta_mean",
    "max_mid_before_median",
    "max_mid_after_median",
    "max_mid_delta_median",
    "max_mid_before_mean",
    "max_mid_after_mean",
    "max_mid_delta_mean",
    "divergence_before_median",
    "divergence_after_median",
    "divergence_delta_median",
    "divergence_before_mean",
    "divergence_after_mean",
    "divergence_delta_mean",
]

PER_KGROUP_COLUMNS = [
    "run_id",
    "source_capture_run_id",
    "split",
    "layer_idx",
    "projection",
    "module_name",
    "k_group_idx",
    "kept",
    "num_tokens",
    "variance_before_median",
    "variance_after_median",
    "variance_delta_median",
    "kurtosis_before_median",
    "kurtosis_after_median",
    "kurtosis_delta_median",
    "max_mid_before_median",
    "max_mid_after_median",
    "max_mid_delta_median",
    "divergence_before_median",
    "divergence_after_median",
    "divergence_delta_median",
]


def _median_scalar(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.to(torch.float64), 0.5).item())


def _mean_scalar(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.to(torch.float64).mean().item())


def reshape_to_groups(x: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected 2D [N,K] activation, got ndim={x.ndim}")
    n, k = x.shape
    if k % group_size != 0:
        raise ValueError(f"K={k} is not divisible by group_size={group_size}")
    return x.reshape(n, k // group_size, group_size)


def compute_group64_metrics(
    groups: torch.Tensor,
    *,
    module_name: str = "",
    split: str = "",
) -> dict[str, torch.Tensor]:
    """Population moments on the last axis. ``groups`` must be [..., 64]."""
    if groups.shape[-1] != GROUP_SIZE:
        raise ValueError(f"last dim must be {GROUP_SIZE}, got {groups.shape[-1]}")
    g = groups.to(torch.float64)
    mean = g.mean(dim=-1, keepdim=True)
    centered = g - mean
    var = (centered * centered).mean(dim=-1)
    m4 = (centered ** 4).mean(dim=-1)

    abs_g = g.abs()
    max_abs = abs_g.amax(dim=-1)
    sorted_abs = abs_g.sort(dim=-1).values
    median_abs = 0.5 * (sorted_abs[..., 31] + sorted_abs[..., 32])
    mean_abs = abs_g.mean(dim=-1)
    centered_abs = abs_g - mean_abs.unsqueeze(-1)
    std_abs = (centered_abs * centered_abs).mean(dim=-1).sqrt()

    bad = (var == 0) | (median_abs == 0) | (mean_abs == 0)
    if bool(bad.any().item()):
        if bad.ndim == 0:
            token_idx = 0
            group_idx = 0
            var_bad = float(var.item()) == 0.0
            med_bad = float(median_abs.item()) == 0.0
            mean_bad = float(mean_abs.item()) == 0.0
        elif bad.ndim == 1:
            group_idx = int(bad.nonzero(as_tuple=False)[0].item())
            token_idx = 0
            var_bad = float(var[group_idx].item()) == 0.0
            med_bad = float(median_abs[group_idx].item()) == 0.0
            mean_bad = float(mean_abs[group_idx].item()) == 0.0
        else:
            loc = bad.nonzero(as_tuple=False)[0]
            token_idx = int(loc[0].item())
            group_idx = int(loc[1].item())
            var_bad = float(var[token_idx, group_idx].item()) == 0.0
            med_bad = float(median_abs[token_idx, group_idx].item()) == 0.0
            mean_bad = float(mean_abs[token_idx, group_idx].item()) == 0.0
        kinds: list[str] = []
        if var_bad:
            kinds.append("Var==0")
        if med_bad:
            kinds.append("median(|g|)==0")
        if mean_bad:
            kinds.append("mean(|g|)==0")
        raise RuntimeError(
            f"{', '.join(kinds)} at module={module_name} split={split} "
            f"token={token_idx} group={group_idx} "
            f"(num_bad={int(bad.sum().item())})"
        )

    return {
        "variance": var,
        "kurtosis": m4 / (var * var),
        "max_mid": max_abs / median_abs,
        "divergence": std_abs / mean_abs,
    }


def apply_channel_diagonal(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected X [N,K], got shape {tuple(x.shape)}")
    if d.ndim != 1:
        raise ValueError(f"expected d [K], got shape {tuple(d.shape)}")
    if d.numel() != x.shape[1]:
        raise ValueError(f"d.numel()={d.numel()} != K={x.shape[1]}")
    return x.to(torch.float64) / d.to(torch.float64).reshape(1, -1)


def assert_rollback_identity(
    x: torch.Tensor,
    x_d: torch.Tensor,
    d: torch.Tensor,
    kept_mask: torch.Tensor,
    *,
    module_name: str,
    split: str,
) -> None:
    g = x.shape[1] // GROUP_SIZE
    d_g = d.to(torch.float64).reshape(g, GROUP_SIZE)
    rollback = ~kept_mask.to(dtype=torch.bool)
    if not bool(rollback.any().item()):
        return
    ones = torch.ones_like(d_g[rollback])
    if not torch.equal(d_g[rollback], ones):
        raise RuntimeError(
            f"rollback groups must have d=1 at module={module_name} split={split}"
        )
    x_g = reshape_to_groups(x.to(torch.float64))
    xd_g = reshape_to_groups(x_d.to(torch.float64))
    if not torch.equal(x_g[:, rollback, :], xd_g[:, rollback, :]):
        raise RuntimeError(
            f"rollback X_D must equal X_rot at module={module_name} split={split}"
        )


def _subset_mask(kept_mask: torch.Tensor, subset: str) -> torch.Tensor:
    if subset == "all":
        return torch.ones_like(kept_mask, dtype=torch.bool)
    if subset == "kept":
        return kept_mask.to(dtype=torch.bool)
    if subset == "rollback":
        return ~kept_mask.to(dtype=torch.bool)
    raise ValueError(f"unknown subset {subset!r}")


def _aggregate_metric_pair(
    before: torch.Tensor, after: torch.Tensor
) -> dict[str, float]:
    delta = after - before
    return {
        "before_median": _median_scalar(before),
        "after_median": _median_scalar(after),
        "delta_median": _median_scalar(delta),
        "before_mean": _mean_scalar(before),
        "after_mean": _mean_scalar(after),
        "delta_mean": _mean_scalar(delta),
    }


def load_e5_recovery(linear_results_csv: Path) -> dict[str, dict[str, float]]:
    if not linear_results_csv.is_file():
        raise FileNotFoundError(f"missing linear results: {linear_results_csv}")
    df = pd.read_csv(linear_results_csv)
    out: dict[str, dict[str, float]] = {}
    for module_name, sub in df.groupby("module_name"):
        e4 = sub[(sub["variant_id"] == "E4_WH_AH_RTN") & (sub["split"] == "val")]
        e5 = sub[(sub["variant_id"] == "E5_WH_AH_DIAG") & (sub["split"] == "val")]
        if len(e4) != 1 or len(e5) != 1:
            raise RuntimeError(
                f"{module_name} must have exactly one val E4 and one val E5 row, "
                f"got E4={len(e4)} E5={len(e5)}"
            )
        e4_err = float(e4["error_energy"].iloc[0])
        e5_err = float(e5["error_energy"].iloc[0])
        out[str(module_name)] = {
            "e4_error_energy": e4_err,
            "e5_error_energy": e5_err,
            "e4_nmse": float(e4["nmse"].iloc[0]),
            "e5_nmse": float(e5["nmse"].iloc[0]),
            "e5_recovery": compute_recovery(e4_err, e5_err),
        }
    if len(out) != REQUIRED_MODULE_COUNT:
        raise RuntimeError(
            f"linear_results must cover {REQUIRED_MODULE_COUNT} modules, got {len(out)}"
        )
    return out


def diagonal_scale_path(capture_dir: Path, module_name: str) -> Path:
    return capture_dir / "diagonal_scales" / f"{module_capture_stem(module_name)}.pt"


def evaluate_module_split(
    x_rot: torch.Tensor,
    d: torch.Tensor,
    kept_mask: torch.Tensor,
    *,
    module_name: str,
    split: str,
) -> dict[str, Any]:
    x = x_rot.to(torch.float64)
    d64 = d.to(torch.float64)
    if x.ndim != 2:
        raise ValueError(f"{module_name}/{split}: X_rot must be [N,K], got {tuple(x.shape)}")
    n, k = x.shape
    if k % GROUP_SIZE != 0:
        raise ValueError(f"{module_name}/{split}: K={k} not divisible by {GROUP_SIZE}")
    if d64.numel() != k:
        raise ValueError(f"{module_name}/{split}: d.numel()={d64.numel()} != K={k}")
    num_groups = k // GROUP_SIZE
    if kept_mask.numel() != num_groups:
        raise ValueError(
            f"{module_name}/{split}: group_kept_mask length {kept_mask.numel()} "
            f"!= K/64={num_groups}"
        )
    kept_mask = kept_mask.to(dtype=torch.bool)
    x_d = apply_channel_diagonal(x, d64)
    assert_rollback_identity(
        x, x_d, d64, kept_mask, module_name=module_name, split=split
    )

    x_g = reshape_to_groups(x)
    xd_g = reshape_to_groups(x_d)
    before = compute_group64_metrics(x_g, module_name=module_name, split=f"{split}:before")
    after = compute_group64_metrics(xd_g, module_name=module_name, split=f"{split}:after")

    rollback = ~kept_mask
    if bool(rollback.any().item()):
        for name in METRIC_NAMES:
            delta = after[name][:, rollback] - before[name][:, rollback]
            if not torch.equal(delta, torch.zeros_like(delta)):
                raise RuntimeError(
                    f"rollback {name} delta must be 0 at module={module_name} split={split}"
                )

    subset_rows: list[dict[str, Any]] = []
    for subset in SUBSETS:
        mask = _subset_mask(kept_mask, subset)
        row: dict[str, Any] = {
            "subset": subset,
            "num_tokens": n,
            "num_k_groups": num_groups,
            "num_k_groups_in_subset": int(mask.sum().item()),
            "num_token_groups": int(n * int(mask.sum().item())),
        }
        for name in METRIC_NAMES:
            b = before[name][:, mask].reshape(-1)
            a = after[name][:, mask].reshape(-1)
            agg = _aggregate_metric_pair(b, a)
            row[f"{name}_before_median"] = agg["before_median"]
            row[f"{name}_after_median"] = agg["after_median"]
            row[f"{name}_delta_median"] = agg["delta_median"]
            row[f"{name}_before_mean"] = agg["before_mean"]
            row[f"{name}_after_mean"] = agg["after_mean"]
            row[f"{name}_delta_mean"] = agg["delta_mean"]
        subset_rows.append(row)

    kgroup_rows: list[dict[str, Any]] = []
    for g_idx in range(num_groups):
        krow: dict[str, Any] = {
            "k_group_idx": g_idx,
            "kept": bool(kept_mask[g_idx].item()),
            "num_tokens": n,
        }
        for name in METRIC_NAMES:
            b = before[name][:, g_idx]
            a = after[name][:, g_idx]
            krow[f"{name}_before_median"] = _median_scalar(b)
            krow[f"{name}_after_median"] = _median_scalar(a)
            krow[f"{name}_delta_median"] = _median_scalar(a - b)
        kgroup_rows.append(krow)

    highlight: dict[str, dict[str, torch.Tensor]] | None = None
    if module_name in HIGHLIGHT_MODULES:
        mask = kept_mask
        highlight = {
            "before": {name: before[name][:, mask].reshape(-1).cpu() for name in METRIC_NAMES},
            "after": {name: after[name][:, mask].reshape(-1).cpu() for name in METRIC_NAMES},
        }

    return {
        "subset_rows": subset_rows,
        "kgroup_rows": kgroup_rows,
        "highlight": highlight,
    }


def _fmt(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


def _pct(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{100.0 * value:+.2f}%"


def _short_module(module_name: str) -> str:
    if module_name in HIGHLIGHT_LABELS:
        return HIGHLIGHT_LABELS[module_name]
    parts = module_name.split(".")
    layer = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            layer = parts[i + 1]
            break
    return f"L{layer}.{parts[-1]}"


def write_report_cn(
    path: Path,
    *,
    summary: dict[str, Any],
    per_module: pd.DataFrame,
) -> None:
    val_kept = per_module[
        (per_module["split"] == "val") & (per_module["subset"] == "kept")
    ].copy()
    val_kept = val_kept.sort_values("e5_recovery", ascending=False)
    val_all = per_module[
        (per_module["split"] == "val") & (per_module["subset"] == "all")
    ].copy()
    val_rb = per_module[
        (per_module["split"] == "val") & (per_module["subset"] == "rollback")
    ].copy()

    rb_delta_cols = [f"{n}_delta_median" for n in METRIC_NAMES]
    rb_max = float(val_rb[rb_delta_cols].abs().max().max())
    if rb_max != 0.0:
        raise RuntimeError(f"validation rollback median deltas must be 0, got max_abs={rb_max}")

    lines: list[str] = [
        "# DIAG 前后 HiF4 K=64 组激活分布统计",
        "",
        "## 1. 问题",
        "",
        "同一个 HiF4 量化组里的 64 个激活，做 `X_D = X_rot / d` 之后，",
        "峰度、方差、max/mid、组内散度怎么变，这和 E5 相对 E4 的输出损失下降是否对得上。",
        "",
        f"- source capture：`{summary['source_capture_run_id']}`",
        f"- 本实验 run：`{summary['run_id']}`",
        "- 统计对象是 **pre-quant** `X_rot` 与 `X_D`，不是量化后的 `A_H` / `A_H_D`。",
        "- 主表看 **validation + kept 组** 的中位数。cal 只对照，不选参数。",
        "",
        "## 2. 指标定义",
        "",
        "每个 `(token, K-group)` 是一组 64 元，对应一次 HiF4 group quant。矩都是 population（除以 64）。",
        "",
        "- 方差：`mean((g-μ)^2)`，有符号值",
        "- 峰度：Pearson 原峰度 `mean((g-μ)^4) / Var^2`（正态 = 3）",
        "- max/mid：`max(|g|) / median(|g|)`，中位数是绝对值的两个正中数平均",
        "- 散度：组内 `|g|` 的变异系数 `std(|g|) / mean(|g|)`，`std` 也是 population",
        "",
        "DIAG 是逐通道正缩放。单通道上看峰度 / max/mid 不变；这里看的是共享 S0 的 64 元联合分布。",
        "",
        "## 3. 主表：validation kept 组中位数",
        "",
        "delta = after − before。负数表示 DIAG 后该指标变小。",
        "没有 kept 组的模块，kept 中位数为 NA，分布变化按 0 理解。",
        "",
        "| Linear | kept 组 | E5 恢复 | 峰度前 | 峰度后 | Δ峰度 | 方差前 | 方差后 | Δ方差 | max/mid 前 | max/mid 后 | Δmax/mid | 散度前 | 散度后 | Δ散度 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in val_kept.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    _short_module(str(row["module_name"])),
                    f"{int(row['num_k_groups_in_subset'])}/{int(row['num_k_groups'])}",
                    _pct(float(row["e5_recovery"])),
                    _fmt(float(row["kurtosis_before_median"])),
                    _fmt(float(row["kurtosis_after_median"])),
                    _fmt(float(row["kurtosis_delta_median"])),
                    _fmt(float(row["variance_before_median"])),
                    _fmt(float(row["variance_after_median"])),
                    _fmt(float(row["variance_delta_median"])),
                    _fmt(float(row["max_mid_before_median"])),
                    _fmt(float(row["max_mid_after_median"])),
                    _fmt(float(row["max_mid_delta_median"])),
                    _fmt(float(row["divergence_before_median"])),
                    _fmt(float(row["divergence_after_median"])),
                    _fmt(float(row["divergence_delta_median"])),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 4. 三个真正有 DIAG 收益的 Linear",
            "",
            "E5 全局 38.4% 恢复几乎全部来自下面三个模块。下表同时给出 validation 的 `all` 与 `kept`。",
            "",
            "| Linear | 子集 | kept | E5 恢复 | Δ峰度中位 | Δ方差中位 | Δmax/mid 中位 | Δ散度中位 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for module_name in HIGHLIGHT_MODULES:
        for subset, frame in (("kept", val_kept), ("all", val_all)):
            sub = frame[frame["module_name"] == module_name]
            if len(sub) != 1:
                raise RuntimeError(f"missing val {subset} row for {module_name}")
            row = sub.iloc[0]
            lines.append(
                "| "
                + " | ".join(
                    [
                        _short_module(module_name),
                        subset,
                        f"{int(row['num_k_groups_in_subset'])}/{int(row['num_k_groups'])}",
                        _pct(float(row["e5_recovery"])),
                        _fmt(float(row["kurtosis_delta_median"])),
                        _fmt(float(row["variance_delta_median"])),
                        _fmt(float(row["max_mid_delta_median"])),
                        _fmt(float(row["divergence_delta_median"])),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## 5. 和输出损失的对齐方式",
            "",
            "- 搜索目标仍是 calibration 上的 group Gram **输出**误差，不是这四个激活分布指标。",
            "- 权重侧同时做了 `W_D = W_N * d`。激活组分布变化只是半边事实，不能单独解释输出损失。",
            "- 只陈述 kept 组四指标的中位 delta，以及该模块的 E5 恢复。不预设这些指标必须下降。",
            f"- validation rollback 组四指标 delta 中位数的最大绝对值为 `{_fmt(rb_max)}`，必须为 0。",
            "",
            "## 6. 边界",
            "",
            "- 没有重跑 Qwen3-8B，没有重搜 D，没有做 GEMM。",
            "- 32 个几乎 `d=1` 的 Linear，kept 很少或为 0，组指标不应有可见变化。",
            "- 禁止把 35 个模块的峰度做算术平均当主结论。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_l2_kept_boxes(
    path: Path,
    highlight: dict[str, dict[str, dict[str, torch.Tensor]]],
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=False)
    metric_titles = {
        "variance": "variance",
        "kurtosis": "kurtosis",
        "max_mid": "max/mid",
        "divergence": "divergence (CV of |g|)",
    }
    for row_i, module_name in enumerate(HIGHLIGHT_MODULES):
        if module_name not in highlight:
            raise RuntimeError(f"missing highlight metrics for {module_name}")
        data = highlight[module_name]
        for col_i, name in enumerate(METRIC_NAMES):
            ax = axes[row_i, col_i]
            before = data["before"][name].numpy()
            after = data["after"][name].numpy()
            if before.size == 0 or after.size == 0:
                raise RuntimeError(f"{module_name} val kept {name} is empty")
            ax.boxplot(
                [before, after],
                tick_labels=["before", "after"],
                showfliers=False,
            )
            if row_i == 0:
                ax.set_title(metric_titles[name])
            if col_i == 0:
                ax.set_ylabel(HIGHLIGHT_LABELS[module_name])
    fig.suptitle("L2 kept HiF4 groups, validation, before vs after DIAG")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_delta_maxmid_vs_recovery(path: Path, val_kept: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    xs: list[float] = []
    ys: list[float] = []
    colors: list[str] = []
    labels: list[str] = []
    for _, row in val_kept.iterrows():
        rec = float(row["e5_recovery"])
        if int(row["num_k_groups_in_subset"]) == 0:
            delta = 0.0
        else:
            delta = float(row["max_mid_delta_median"])
        xs.append(delta)
        ys.append(100.0 * rec)
        module = str(row["module_name"])
        if module in HIGHLIGHT_MODULES:
            colors.append("tab:red")
            labels.append(_short_module(module))
        else:
            colors.append("0.55")
            labels.append("")
    ax.scatter(xs, ys, c=colors, s=36, zorder=2)
    for x, y, lab in zip(xs, ys, labels, strict=True):
        if lab:
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("val kept median Δmax/mid (after − before)")
    ax.set_ylabel("E5 recovery vs E4 (%)")
    ax.set_title("35 Linears: group max/mid change vs output recovery")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def validate_outputs(
    run_dir: Path,
    *,
    config: AppConfig,
    source_capture_run_id: str,
) -> None:
    per_path = run_dir / "diag_group_stats_per_module.csv"
    k_path = run_dir / "diag_group_stats_per_kgroup.csv"
    summary_path = run_dir / "diag_group_stats_summary.json"
    report_path = run_dir / "diag_group_stats_report_cn.md"
    fig1 = run_dir / "figures" / "fig01_l2_kept_group_box_before_after.png"
    fig2 = run_dir / "figures" / "fig02_delta_maxmid_vs_e5_recovery.png"
    for path in (per_path, k_path, summary_path, report_path, fig1, fig2):
        if not path.is_file():
            raise FileNotFoundError(f"missing output: {path}")

    per_df = pd.read_csv(per_path)
    k_df = pd.read_csv(k_path)
    if len(per_df) != REQUIRED_MODULE_COUNT * len(SPLITS) * len(SUBSETS):
        raise RuntimeError(
            f"per-module CSV must have {REQUIRED_MODULE_COUNT * 2 * 3} rows, got {len(per_df)}"
        )
    expected_k = len(REQUIRED_FORMAL_LAYERS) * (6 * 64 + 192) * len(SPLITS)
    if len(k_df) != expected_k:
        raise RuntimeError(f"per-kgroup CSV must have {expected_k} rows, got {len(k_df)}")

    expected_modules = set(config.formal_module_names)
    for split in SPLITS:
        sub = per_df[per_df["split"] == split]
        if set(sub["module_name"].tolist()) != expected_modules:
            raise RuntimeError(f"{split} module set mismatch")
        rb = sub[sub["subset"] == "rollback"]
        for name in METRIC_NAMES:
            delta = rb[f"{name}_delta_median"].astype(float)
            if not (delta.fillna(0.0) == 0.0).all():
                raise RuntimeError(f"{split} rollback {name} delta_median must be 0")


def run_diag_group_stats(
    config: AppConfig,
    *,
    capture_run_id: str,
    run_id: str,
) -> dict[str, Any]:
    assert_distinct_run_ids(capture_run_id, run_id)
    if tuple(int(x) for x in config.experiment.formal_layers) != REQUIRED_FORMAL_LAYERS:
        raise ValueError(
            f"config formal_layers must be {list(REQUIRED_FORMAL_LAYERS)}, "
            f"got {list(config.experiment.formal_layers)}"
        )
    if config.hif4.group_size != GROUP_SIZE:
        raise ValueError(f"HiF4 group_size must be {GROUP_SIZE}, got {config.hif4.group_size}")

    capture_dir = results_dir(capture_run_id)
    manifest_path = capture_dir / "capture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing capture manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    validate_capture_manifest(config, capture_dir, manifest)

    recovery = load_e5_recovery(capture_dir / "linear_results.csv")
    missing_d = [
        diagonal_scale_path(capture_dir, name)
        for name in config.formal_module_names
        if not diagonal_scale_path(capture_dir, name).is_file()
    ]
    if missing_d:
        raise FileNotFoundError(f"missing {len(missing_d)} diagonal scale files; first={missing_d[0]}")

    module_rows: list[dict[str, Any]] = []
    kgroup_rows: list[dict[str, Any]] = []
    highlight_val: dict[str, dict[str, dict[str, torch.Tensor]]] = {}

    for module_name in config.formal_module_names:
        diag = load_pt(diagonal_scale_path(capture_dir, module_name), map_location="cpu")
        d = diag["d"]
        kept_mask = diag["group_kept_mask"]
        rec = recovery[module_name]
        for split in SPLITS:
            print(f"[diag_group_stats] {split} {module_name}", flush=True)
            capture = load_pt(
                capture_file_path(capture_dir, module_name, split),
                map_location="cpu",
            )
            if capture["module_name"] != module_name or capture["split"] != split:
                raise RuntimeError(
                    f"capture metadata mismatch: expected {module_name}/{split}, "
                    f"got {capture['module_name']}/{capture['split']}"
                )
            evaluated = evaluate_module_split(
                capture["x_rot_bf16"],
                d,
                kept_mask,
                module_name=module_name,
                split=split,
            )
            for subset_row in evaluated["subset_rows"]:
                module_rows.append(
                    {
                        "run_id": run_id,
                        "source_capture_run_id": capture_run_id,
                        "split": split,
                        "layer_idx": int(capture["layer_idx"]),
                        "projection": str(capture["projection"]),
                        "module_name": module_name,
                        **subset_row,
                        **rec,
                    }
                )
            for krow in evaluated["kgroup_rows"]:
                kgroup_rows.append(
                    {
                        "run_id": run_id,
                        "source_capture_run_id": capture_run_id,
                        "split": split,
                        "layer_idx": int(capture["layer_idx"]),
                        "projection": str(capture["projection"]),
                        "module_name": module_name,
                        **krow,
                    }
                )
            if split == "val" and evaluated["highlight"] is not None:
                highlight_val[module_name] = evaluated["highlight"]
            del capture

    if set(highlight_val) != set(HIGHLIGHT_MODULES):
        raise RuntimeError(
            f"highlight modules incomplete: {sorted(highlight_val)} vs {list(HIGHLIGHT_MODULES)}"
        )

    run_dir = ensure_dir(results_dir(run_id))
    per_df = pd.DataFrame(module_rows, columns=PER_MODULE_COLUMNS)
    per_df.to_csv(run_dir / "diag_group_stats_per_module.csv", index=False)
    k_df = pd.DataFrame(kgroup_rows, columns=PER_KGROUP_COLUMNS)
    k_df.to_csv(run_dir / "diag_group_stats_per_kgroup.csv", index=False)

    val_kept = per_df[(per_df["split"] == "val") & (per_df["subset"] == "kept")]
    summary = {
        "run_id": run_id,
        "source_capture_run_id": capture_run_id,
        "group_size": GROUP_SIZE,
        "num_modules": REQUIRED_MODULE_COUNT,
        "headline_split": "val",
        "headline_subset": "kept",
        "metric_definitions": {
            "variance": "population mean((g-mu)^2) on signed 64-group",
            "kurtosis": "Pearson population mean((g-mu)^4)/Var^2",
            "max_mid": "max(|g|) / median(|g|), median = mean of order stats 32 and 33",
            "divergence": "population std(|g|) / mean(|g|)",
        },
        "highlight_val_kept": {},
    }
    for module_name in HIGHLIGHT_MODULES:
        row = val_kept[val_kept["module_name"] == module_name]
        if len(row) != 1:
            raise RuntimeError(f"missing val kept summary row for {module_name}")
        r = row.iloc[0]
        summary["highlight_val_kept"][module_name] = {
            "e5_recovery": float(r["e5_recovery"]),
            "num_k_groups_in_subset": int(r["num_k_groups_in_subset"]),
            "num_k_groups": int(r["num_k_groups"]),
            "kurtosis_delta_median": float(r["kurtosis_delta_median"]),
            "variance_delta_median": float(r["variance_delta_median"]),
            "max_mid_delta_median": float(r["max_mid_delta_median"]),
            "divergence_delta_median": float(r["divergence_delta_median"]),
        }
    write_json(run_dir / "diag_group_stats_summary.json", summary)
    write_report_cn(
        run_dir / "diag_group_stats_report_cn.md",
        summary=summary,
        per_module=per_df,
    )
    plot_l2_kept_boxes(
        run_dir / "figures" / "fig01_l2_kept_group_box_before_after.png",
        highlight_val,
    )
    plot_delta_maxmid_vs_recovery(
        run_dir / "figures" / "fig02_delta_maxmid_vs_e5_recovery.png",
        val_kept,
    )
    validate_outputs(run_dir, config=config, source_capture_run_id=capture_run_id)
    print(f"DIAG GROUP STATS DONE -> {run_dir}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HiF4 K=64 group activation stats before/after DIAG"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--capture-run-id", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    run_diag_group_stats(
        config,
        capture_run_id=args.capture_run_id,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
