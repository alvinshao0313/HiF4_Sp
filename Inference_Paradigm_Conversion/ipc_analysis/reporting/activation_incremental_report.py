"""Chinese markdown logs, figures, and phase report for activation incremental experiments."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (
    atomic_write_json,
    ensure_dir,
    write_text,
)

_FONT_CANDIDATES = [
    Path("/home/shaoyuantian/program/HiF4_Sp/Inference_Paradigm_Conversion/assets/fonts/SourceHanSansSC-Regular.otf"),
    Path("/home/shaoyuantian/program/HiF4_Sp/Inference_Paradigm_Conversion/assets/fonts/NotoSansCJK-Regular.ttc"),
]


def _setup_chinese_font() -> str | None:
    for p in _FONT_CANDIDATES:
        if p.is_file():
            font_manager.fontManager.addfont(str(p))
            prop = font_manager.FontProperties(fname=str(p))
            name = prop.get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


_setup_chinese_font()


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    import csv

    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x in (None, ""):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[i]


def _fig_s0_divisor(ax1_rows: list[dict[str, Any]], out: Path) -> str | None:
    if not ax1_rows:
        return None
    by_proj: dict[str, list[float]] = defaultdict(list)
    for r in ax1_rows:
        by_proj[str(r.get("projection", "?"))].append(_f(r.get("alpha_oracle_nvfp4"), 7.0))
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = sorted(by_proj)
    data = [by_proj[k] for k in labels]
    ax.boxplot(data, tick_labels=labels)
    ax.axhline(7.0, color="r", linestyle="--", label="当前 alpha=7")
    ax.set_title("图 AX1-1：各投影最优 S0 除数分布")
    ax.set_xlabel("投影类型")
    ax.set_ylabel("最优 alpha（逼近 NVFP4 Source）")
    ax.legend()
    fig.tight_layout()
    path = out / "fig_ax1_s0_divisor_by_projection.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _fig_s0_recovery(ax1_rows: list[dict[str, Any]], out: Path) -> str | None:
    if not ax1_rows:
        return None
    by_proj: dict[str, list[float]] = defaultdict(list)
    for r in ax1_rows:
        by_proj[str(r.get("projection", "?"))].append(_f(r.get("output_recovery")))
    labels = sorted(by_proj)
    ys = [_mean(by_proj[k]) for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, ys)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_title("图 AX1-2：当前 /7 相对 Oracle S0 的 Linear 输出恢复率")
    ax.set_xlabel("投影类型")
    ax.set_ylabel("平均 R_Y")
    fig.tight_layout()
    path = out / "fig_ax1_output_recovery_by_projection.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _fig_group_size(ax2_rows: list[dict[str, Any]], out: Path) -> str | None:
    if not ax2_rows:
        return None
    by_gs: dict[str, list[float]] = defaultdict(list)
    for r in ax2_rows:
        gs = str(r.get("group_size", ""))
        if gs:
            by_gs[gs].append(_f(r.get("R_Y")))
    if not by_gs:
        return None
    xs = sorted(by_gs, key=lambda z: int(z))
    ys = [_mean(by_gs[k]) for k in xs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(xs, ys)
    ax.set_title("图 AX2-1：HiF4 G16/G32/G64 Linear 输出恢复率")
    ax.set_xlabel("Group Size")
    ax.set_ylabel("R_Y（相对标准 HiF4→NVFP4 误差）")
    fig.tight_layout()
    path = out / "fig_ax2_group_size_ry.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _fig_dispersion(ax2_disp: list[dict[str, Any]], out: Path) -> str | None:
    if not ax2_disp:
        return None
    series: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in ax2_disp:
        gs = str(r.get("group_size", r.get("variant", "?")))
        d = _f(r.get("dispersion_d", r.get("d")))
        series[gs][d].append(_f(r.get("R_Y", r.get("output_error_energy"))))
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for gs, mp in sorted(series.items(), key=lambda kv: kv[0]):
        xs = sorted(mp)
        ys = [_mean(mp[x]) for x in xs]
        ax.plot(xs, ys, marker="o", label=f"G{gs}" if gs.isdigit() else gs)
    ax.set_title("图 AX2-2：4×16 动态范围离散度扫描")
    ax.set_xlabel("离散度 d")
    ax.set_ylabel("平均指标")
    ax.legend()
    fig.tight_layout()
    path = out / "fig_ax2_dispersion_sweep.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _fig_occupancy(ax3_rows: list[dict[str, Any]], out: Path) -> list[str]:
    if not ax3_rows:
        return []
    paths: list[str] = []
    by_proj_z: dict[str, list[float]] = defaultdict(list)
    by_proj_b: dict[str, list[float]] = defaultdict(list)
    by_proj_h: dict[str, list[float]] = defaultdict(list)
    for r in ax3_rows:
        p = str(r.get("projection", "?"))
        by_proj_z[p].append(_f(r.get("hf_occ_zero_rate")))
        by_proj_b[p].append(_f(r.get("hf_occ_boundary_rate")))
        by_proj_h[p].append(_f(r.get("hf_occ_entropy")))
    for title, data, ylab, fname in [
        ("图 AX3-7：各投影 HiF4 零点占用率", by_proj_z, "零点占用率", "fig_ax3_zero_occ_by_proj.png"),
        ("图 AX3-8：各投影 HiF4 边界占用率", by_proj_b, "边界占用率", "fig_ax3_boundary_occ_by_proj.png"),
        ("图 AX3-9：各投影 HiF4 占用熵", by_proj_h, "占用熵", "fig_ax3_entropy_by_proj.png"),
    ]:
        labels = sorted(data)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, [_mean(data[k]) for k in labels])
        ax.set_title(title)
        ax.set_xlabel("投影类型")
        ax.set_ylabel(ylab)
        fig.tight_layout()
        path = out / fname
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _fig_hybrid(ax4_rows: list[dict[str, Any]], out: Path) -> str | None:
    if not ax4_rows:
        return None
    by_h: dict[str, list[float]] = defaultdict(list)
    for r in ax4_rows:
        key = str(r.get("hybrid", r.get("variant", "?")))
        by_h[key].append(_f(r.get("R_Y")))
    labels = sorted(by_h)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, [_mean(by_h[k]) for k in labels])
    ax.set_title("图 AX4-1：交叉格式 Hybrid 的 Linear 输出恢复率")
    ax.set_xlabel("组合")
    ax.set_ylabel("平均 R_Y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    path = out / "fig_ax4_hybrid_ry.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _fig_ranking(ranking: list[dict[str, Any]], out: Path) -> str | None:
    if not ranking:
        return None
    top = ranking[:12]
    labels = [str(r.get("root_cause")) for r in top][::-1]
    vals = [_f(r.get("R_Y")) for r in top][::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, vals)
    ax.set_title("图 AX5-1：激活量化误差根因恢复率总排名")
    ax.set_xlabel("聚合 R_Y / R_cf_output")
    fig.tight_layout()
    path = out / "fig_ax5_root_cause_ranking.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _load_full_grids(theory: dict[str, Any]) -> tuple[Any, Any] | None:
    import torch

    nv_path = theory.get("nvfp4_full_internal_grid_path")
    hf_path = theory.get("hif4_full_internal_grid_path")
    if nv_path and hf_path and Path(nv_path).is_file() and Path(hf_path).is_file():
        return torch.load(nv_path, map_location="cpu", weights_only=True), torch.load(
            hf_path, map_location="cpu", weights_only=True
        )
    nv = theory.get("nvfp4_full_internal_grid")
    hf = theory.get("hif4_full_internal_grid")
    if not nv or not hf:
        return None
    return torch.tensor(nv, dtype=torch.float32), torch.tensor(hf, dtype=torch.float32)


def _symlog_bin_edges(xmax: float, n_decades: int = 48, linthresh: float = 1.0) -> "np.ndarray":
    """Symmetric log-spaced edges on [-xmax, xmax]; linear only in [-linthresh, linthresh]."""
    import numpy as np

    if xmax <= 0:
        raise ValueError(f"xmax must be positive, got {xmax}")
    linthresh = float(min(linthresh, xmax))
    # Positive side: dense linear core + geometric decades out to xmax.
    pos_lin = np.linspace(0.0, linthresh, 17)
    if xmax > linthresh:
        pos_log = np.geomspace(linthresh, xmax, n_decades + 1)
        pos = np.unique(np.concatenate([pos_lin, pos_log]))
    else:
        pos = pos_lin
    return np.unique(np.concatenate([-pos[::-1], pos]))


def _fig_theory_grid(theory: dict[str, Any], out: Path) -> list[str]:
    """AX3 theory figures: payload codebook (legacy rename) + full internal grids."""
    import numpy as np
    import torch

    if not theory:
        return []
    paths: list[str] = []

    # Legacy payload codebook — renamed, not called "完整理论网格".
    nv_p = [float(x) for x in theory.get("nvfp4_payload_grid", theory.get("nvfp4_grid", []))]
    hf_p = [float(x) for x in theory.get("hif4_payload_grid", theory.get("hif4_grid", []))]
    if nv_p and hf_p:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.scatter(nv_p, [1] * len(nv_p), label="NVFP4 E2M1", alpha=0.85)
        ax.scatter(hf_p, [0] * len(hf_p), label="HiF4 S1P2", alpha=0.85)
        ax.set_yticks([0, 1], ["HiF4 S1P2", "NVFP4 E2M1"])
        ax.set_title("图 AX3-payload：NVFP4 E2M1 与 HiF4 S1P2 Payload Codebook")
        ax.set_xlabel("Payload codebook value")
        ax.legend()
        fig.tight_layout()
        p_cb = out / "fig_ax3_payload_codebook.png"
        fig.savefig(p_cb, dpi=140)
        plt.close(fig)
        paths.append(str(p_cb))

    grids = _load_full_grids(theory)
    if grids is None:
        return paths
    nv_full, hf_full = grids
    nv_full = nv_full.to(torch.float32).reshape(-1)
    hf_full = hf_full.to(torch.float32).reshape(-1)
    if nv_full.numel() < 16 or hf_full.numel() < 16:
        raise RuntimeError("full internal grids unexpectedly small; enumeration bug")

    # FP-like grids are exponentially spaced: equal-width linear hist collapses to 0.
    nv_abs = nv_full[nv_full != 0].abs()
    hf_abs = hf_full[hf_full != 0].abs()
    ov_lo = float(max(nv_abs.min().item(), hf_abs.min().item()))
    ov_hi = float(min(nv_abs.max().item(), hf_abs.max().item()))

    nv_z = nv_full[nv_full != 0].abs().log2().numpy()
    hf_z = hf_full[hf_full != 0].abs().log2().numpy()
    z_lo = float(np.floor(min(nv_z.min(), hf_z.min())))
    z_hi = float(np.ceil(max(nv_z.max(), hf_z.max())))
    z_edges = np.arange(z_lo, z_hi + 1.0, 1.0)

    z_ov_lo = float(np.floor(np.log2(ov_lo)))
    z_ov_hi = float(np.ceil(np.log2(ov_hi)))
    z_ov_edges = np.arange(z_ov_lo, z_ov_hi + 1.0, 1.0)
    n_ov_bins = int(z_ov_edges.size - 1)
    nv_z_ov = nv_z[(nv_z >= z_ov_lo) & (nv_z < z_ov_hi)]
    hf_z_ov = hf_z[(hf_z >= z_ov_lo) & (hf_z < z_ov_hi)]

    lin_w = 1.0e4
    linthresh = 1.0
    sym_edges = _symlog_bin_edges(lin_w, n_decades=64, linthresh=linthresh)
    nv_lin = nv_full[(nv_full >= -lin_w) & (nv_full <= lin_w)].numpy()
    hf_lin = hf_full[(hf_full >= -lin_w) & (hf_full <= lin_w)].numpy()

    # Main: log2 density + symlog-binned hist (shows structure across decades).
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.4), constrained_layout=True)
    axes[0].hist(
        nv_z_ov,
        bins=z_ov_edges,
        histtype="stepfilled",
        alpha=0.35,
        label=f"NVFP4（重叠区，{n_ov_bins} bins，Δlog2=1）",
    )
    axes[0].hist(
        hf_z_ov,
        bins=z_ov_edges,
        histtype="stepfilled",
        alpha=0.35,
        label=f"HiF4（重叠区，{n_ov_bins} bins，Δlog2=1）",
    )
    axes[0].set_title(
        f"图 AX3-1（上）：数量级密度  log2|x|∈[{z_ov_lo:.0f},{z_ov_hi:.0f}]"
    )
    axes[0].set_xlabel("log2(|representable value|)")
    axes[0].set_ylabel("Number of unique representable values")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, axis="y", alpha=0.25)

    # Drop exact 0 so the dense origin does not flatten every other bin.
    nv_lin_nz = nv_lin[nv_lin != 0.0]
    hf_lin_nz = hf_lin[hf_lin != 0.0]
    axes[1].hist(
        nv_lin_nz,
        bins=sym_edges,
        histtype="stepfilled",
        alpha=0.4,
        label=f"NVFP4（非零 {nv_lin_nz.size}/{nv_lin.size}）",
    )
    axes[1].hist(
        hf_lin_nz,
        bins=sym_edges,
        histtype="stepfilled",
        alpha=0.4,
        label=f"HiF4（非零 {hf_lin_nz.size}/{hf_lin.size}）",
    )
    axes[1].set_xscale("symlog", linthresh=linthresh, base=10)
    axes[1].set_yscale("log")
    axes[1].set_xlim(-lin_w, lin_w)
    axes[1].set_title(
        f"图 AX3-1（下）：symlog 分箱（去 0）x∈[{-lin_w:.0f},{lin_w:.0f}]，"
        f"|x|<{linthresh:g} 线性；y 轴 log 露出尾部疏密"
    )
    axes[1].set_xlabel("Representable value (symlog)")
    axes[1].set_ylabel("Number of unique representable values (log)")
    axes[1].legend()
    axes[1].grid(True, axis="both", which="both", alpha=0.25)
    p1 = out / "fig_ax3_full_internal_grid_hist.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    paths.append(str(p1))

    # Standalone log2 full-range.
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.hist(nv_z, bins=z_edges, histtype="stepfilled", alpha=0.35, label="NVFP4")
    ax.hist(hf_z, bins=z_edges, histtype="stepfilled", alpha=0.35, label="HiF4")
    ax.set_title("图 AX3-2：非零可表示点的数量级分布（每 1 个 log2 一桶）")
    ax.set_xlabel("log2(|representable value|)")
    ax.set_ylabel("Number of unique representable values")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    p2 = out / "fig_ax3_full_internal_grid_log2_hist.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    paths.append(str(p2))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.hist(nv_z_ov, bins=z_ov_edges, histtype="step", linewidth=2.0, label="NVFP4")
    ax.hist(hf_z_ov, bins=z_ov_edges, histtype="step", linewidth=2.0, label="HiF4")
    ax.set_title(
        f"图 AX3-2b：重叠动态范围放大（log2|x|∈[{z_ov_lo:.0f},{z_ov_hi:.0f}]，"
        f"{n_ov_bins} bins，Δlog2=1）"
    )
    ax.set_xlabel("log2(|representable value|)")
    ax.set_ylabel("Number of unique representable values")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p2b = out / "fig_ax3_full_internal_grid_log2_overlap_zoom.png"
    fig.savefig(p2b, dpi=160)
    plt.close(fig)
    paths.append(str(p2b))

    # Annulus zooms: each panel only keeps w/16 < |x| ≤ w so inner denser shells
    # cannot collapse the histogram into a single center bin.
    zoom_halfs = (1.0, 16.0, 256.0, 1.0e4)
    fig, axes = plt.subplots(len(zoom_halfs), 1, figsize=(11, 11.5), constrained_layout=True)
    for ax, w in zip(axes, zoom_halfs):
        lo = w / 16.0
        edges = np.linspace(-w, w, 257)
        nv_mask = (nv_full.abs() > lo) & (nv_full.abs() <= w)
        hf_mask = (hf_full.abs() > lo) & (hf_full.abs() <= w)
        nv_w = nv_full[nv_mask].numpy()
        hf_w = hf_full[hf_mask].numpy()
        ax.hist(nv_w, bins=edges, histtype="stepfilled", alpha=0.45, label=f"NVFP4（{nv_w.size}）")
        ax.hist(hf_w, bins=edges, histtype="stepfilled", alpha=0.45, label=f"HiF4（{hf_w.size}）")
        ax.set_xlim(-w, w)
        ax.set_title(f"线性环带 {lo:g}<|x|≤{w:g}，256 bins（去掉更内层点）")
        ax.set_ylabel("count")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xlabel("Representable value")
    fig.suptitle("图 AX3-1c：多级线性环带放大（只画当前量级外壳）", y=1.01)
    p1c = out / "fig_ax3_full_internal_grid_hist_linear_zoom.png"
    fig.savefig(p1c, dpi=160)
    plt.close(fig)
    paths.append(str(p1c))

    # Dedicated windows: |x| hist + signed rugs. Avoid signed equal-width hist
    # (half-open bins make ± look staggered even when the set is symmetric).
    for w_win, tag, n_abs_bins in (
        (100.0, "pm100", 201),
        (10.0, "pm10", 201),
        (1.0, "pm1", 201),
        (0.1, "pm0p1", 201),
        (0.01, "pm0p01", 201),
        (0.001, "pm0p001", 201),
    ):
        nv_w = nv_full[(nv_full >= -w_win) & (nv_full <= w_win)]
        hf_w = hf_full[(hf_full >= -w_win) & (hf_full <= w_win)]
        n_nv_pos = int((nv_w > 0).sum().item())
        n_nv_neg = int((nv_w < 0).sum().item())
        n_hf_pos = int((hf_w > 0).sum().item())
        n_hf_neg = int((hf_w < 0).sum().item())
        if n_nv_pos != n_nv_neg or n_hf_pos != n_hf_neg:
            raise RuntimeError(
                f"grid not sign-symmetric in [-{w_win:g},{w_win:g}]: "
                f"NV ±={n_nv_neg}/{n_nv_pos}, HiF4 ±={n_hf_neg}/{n_hf_pos}"
            )

        # Bin by |x|, then mirror counts onto ±x so the top panel shares the same
        # signed x-axis as the rugs (avoids “0 on left vs 0 in center” mismatch).
        abs_edges = np.linspace(0.0, w_win, n_abs_bins)
        abs_centers = 0.5 * (abs_edges[:-1] + abs_edges[1:])
        abs_widths = np.diff(abs_edges)
        nv_abs_hist, _ = np.histogram(nv_w[nv_w != 0].abs().numpy(), bins=abs_edges)
        hf_abs_hist, _ = np.histogram(hf_w[hf_w != 0].abs().numpy(), bins=abs_edges)

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 9.5),
            constrained_layout=True,
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]},
        )
        for centers_sign in (abs_centers, -abs_centers):
            axes[0].bar(
                centers_sign,
                nv_abs_hist,
                width=abs_widths,
                align="center",
                alpha=0.35,
                color="#1f77b4",
                label="NVFP4" if centers_sign is abs_centers else None,
            )
            axes[0].bar(
                centers_sign,
                hf_abs_hist,
                width=abs_widths,
                align="center",
                alpha=0.35,
                color="#ff7f0e",
                label="HiF4" if centers_sign is abs_centers else None,
            )
        axes[0].set_yscale("log")
        axes[0].set_title(
            f"图 AX3-1f（上）：按 |x| 分箱后左右镜像到 ±x（与下方同横轴）；"
            f"{n_abs_bins - 1} bins，y=log；NV ±={n_nv_neg}/{n_nv_pos}，"
            f"HiF4 ±={n_hf_neg}/{n_hf_pos}"
        )
        axes[0].set_ylabel("count (log)")
        axes[0].legend()
        axes[0].grid(True, axis="both", which="both", alpha=0.25)

        for ax, vals, name, color in (
            (axes[1], nv_w[nv_w != 0].numpy(), "NVFP4", "#1f77b4"),
            (axes[2], hf_w[hf_w != 0].numpy(), "HiF4", "#ff7f0e"),
        ):
            ax.vlines(vals, 0.0, 1.0, colors=color, alpha=0.55, linewidth=0.7)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([])
            ax.set_ylabel(name)
            ax.grid(True, axis="x", alpha=0.25)
        axes[1].set_title(
            f"图 AX3-1f（中/下）：x∈[{-w_win:g},{w_win:g}] 非零唯一点位置（rug）"
        )
        axes[2].set_xlabel("Representable value")
        axes[2].set_xlim(-w_win, w_win)
        p_win = out / f"fig_ax3_full_internal_grid_hist_{tag}.png"
        fig.savefig(p_win, dpi=170)
        plt.close(fig)
        paths.append(str(p_win))

    # Decade shells on +x: local equal-width hist inside each decade (signed mirror skipped).
    decades = list(range(0, 5))  # [1,10), [10,100), ..., [1e3,1e4]
    fig, axes = plt.subplots(len(decades), 1, figsize=(11, 12.0), constrained_layout=True)
    for ax, k in zip(axes, decades):
        a0, a1 = 10.0**k, 10.0 ** (k + 1)
        edges = np.linspace(a0, a1, 65)
        nv_d = nv_full[(nv_full >= a0) & (nv_full < a1)].numpy()
        hf_d = hf_full[(hf_full >= a0) & (hf_full < a1)].numpy()
        ax.hist(nv_d, bins=edges, histtype="step", linewidth=1.8, label=f"NVFP4（{nv_d.size}）")
        ax.hist(hf_d, bins=edges, histtype="step", linewidth=1.8, label=f"HiF4（{hf_d.size}）")
        ax.set_xlim(a0, a1)
        ax.set_title(f"正值 decade x∈[{a0:g},{a1:g})")
        ax.set_ylabel("count")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xlabel("Representable value")
    fig.suptitle("图 AX3-1e：按十进制 decade 展开的局部线性细节（正半轴）", y=1.01)
    p1e = out / "fig_ax3_full_internal_grid_hist_decade_zoom.png"
    fig.savefig(p1e, dpi=160)
    plt.close(fig)
    paths.append(str(p1e))

    # Rug / strip: every unique non-zero point on symlog x (gaps & clusters visible).
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True, constrained_layout=True)
    for ax, vals, name, color in (
        (axes[0], nv_lin_nz, "NVFP4", "#1f77b4"),
        (axes[1], hf_lin_nz, "HiF4", "#ff7f0e"),
    ):
        ax.vlines(vals, 0.0, 1.0, colors=color, alpha=0.45, linewidth=0.7)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_ylabel(name)
        ax.grid(True, axis="x", which="both", alpha=0.25)
    axes[0].set_title(
        f"图 AX3-1d：窗内非零唯一可表示点位置（x∈[{-lin_w:.0f},{lin_w:.0f}]，symlog；去 0）"
    )
    axes[1].set_xlabel("Representable value (symlog)")
    axes[1].set_xscale("symlog", linthresh=linthresh, base=10)
    axes[1].set_xlim(-lin_w, lin_w)
    p1d = out / "fig_ax3_full_internal_grid_rug_symlog.png"
    fig.savefig(p1d, dpi=170)
    plt.close(fig)
    paths.append(str(p1d))

    # Near-zero density: shared real xlim from 1% quantile inside overlapping |x| range.
    common_lo = float(max(nv_abs.min().item(), hf_abs.min().item()))
    common_hi = float(min(nv_abs.max().item(), hf_abs.max().item()))
    in_common = torch.cat(
        [
            nv_abs[(nv_abs >= common_lo) & (nv_abs <= common_hi)],
            hf_abs[(hf_abs >= common_lo) & (hf_abs <= common_hi)],
        ]
    )
    q = float(torch.quantile(in_common, 0.01).item())
    q = max(q, common_lo)
    if q <= 0 or not math.isfinite(q):
        raise RuntimeError(f"invalid near-zero window half-width: {q}")
    near_edges = np.linspace(-q, q, 65)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(nv_full.numpy(), bins=near_edges, alpha=0.5, label="NVFP4")
    ax.hist(hf_full.numpy(), bins=near_edges, alpha=0.5, label="HiF4")
    ax.set_xlim(-q, q)
    ax.set_title(f"图 AX3-3：0 附近完整内部网格密度（x ∈ [{-q:.6g}, {q:.6g}]）")
    ax.set_xlabel("Representable value")
    ax.set_ylabel("Number of unique representable values")
    ax.legend()
    fig.tight_layout()
    p3 = out / "fig_ax3_full_grid_density_near_zero.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    paths.append(str(p3))
    return paths


def build_experiment_logs(run_dir: Path, reports_dir: Path) -> list[str]:
    logs_dir = ensure_dir(reports_dir / "experiment_logs")
    ax1 = _read_csv_rows(run_dir / "ax1_s0_divisor_oracle.csv")
    ax2 = _read_csv_rows(run_dir / "ax2_group_size_ablation.csv")
    ax2d = _read_csv_rows(run_dir / "ax2_sub16_dispersion.csv")
    ax3 = _read_csv_rows(run_dir / "ax3_grid_occupancy.csv")
    ax4 = _read_csv_rows(run_dir / "ax4_cross_format_factorization.csv")
    ax5 = _read_csv_rows(run_dir / "ax5_root_cause_ranking.csv")
    rules = _read_csv_rows(run_dir / "ax5_rule_validation.csv")
    theory_path = run_dir / "ax3_theoretical_grid.json"
    theory = json.loads(theory_path.read_text(encoding="utf-8")) if theory_path.is_file() else {}

    written: list[str] = []

    # AX1
    alphas = [_f(r.get("alpha_oracle_nvfp4"), 7.0) for r in ax1]
    rys = [_f(r.get("output_recovery")) for r in ax1]
    by_proj_a: dict[str, list[float]] = defaultdict(list)
    by_proj_y: dict[str, list[float]] = defaultdict(list)
    for r in ax1:
        p = str(r.get("projection"))
        by_proj_a[p].append(_f(r.get("alpha_oracle_nvfp4"), 7.0))
        by_proj_y[p].append(_f(r.get("output_recovery")))
    proj_lines = "\n".join(
        f"- {p}: alpha 中位数={_pct(by_proj_a[p], 0.5):.4f}, R_Y 均值={_mean(by_proj_y[p]):.4f}"
        for p in sorted(by_proj_a)
    )
    s0_major = (
        "S0 位置是主要问题"
        if _mean(rys) >= 0.15
        else ("S0 位置是次要问题" if _mean(rys) >= 0.05 else "S0 位置不是主要问题")
    )
    ax1_md = f"""# AX1：HiF4 S0 除数 Oracle

## 1. 实验目标

判断固定 `amax/7` 是否适合真实激活；量化把 S0 除数调到 Oracle 后，相对 NVFP4 Source 能恢复多少 Linear 输出误差。

## 2. 实验设置

- 模型 checkpoint：`Qwen3-8B-FPQuant-QAT-NVFP4-Dequant-BF16-NoHadamard`
- run_id：`{run_dir.name}`
- 搜索：alpha∈[4,10]，步长 0.125，再对最优邻域精搜 33 点
- 样本行数：{len(ax1)}
- 指标：逼近 NVFP4 的激活恢复率 / Linear 输出恢复率 R_Y

## 3. 实验结果

- 当前 alpha=7
- 最优 alpha：中位数={_pct(alphas, 0.5):.4f}，p10={_pct(alphas, 0.1):.4f}，p90={_pct(alphas, 0.9):.4f}
- 平均 R_Y={_mean(rys):.4f}，p50={_pct(rys, 0.5):.4f}，p90={_pct(rys, 0.9):.4f}
- 分投影：
{proj_lines}

## 4. 实验分析

若最优 alpha 稳定靠近 7 且 R_Y 很小，说明「除数位置」不是主因；若某些投影显著偏离 7 且 R_Y 较大，则存在 projection-specific 的 S0 定位问题。alpha_A*（逼近 NVFP4）与 alpha_X*（逼近 BF16）若不一致，说明转换目标与重建目标要求不同的 S0。

## 5. 实验结论

- {s0_major}
- 全样本平均输出恢复率约为 {_mean(rys):.2%}
- 最优除数中位数相对 7 的偏移为 {_pct(alphas, 0.5) - 7.0:+.3f}

## 6. 对算法设计的启示

只有在 R_Y 足够大时，才值得做低开销在线 S0 规则（AX5-R）；否则应优先其他机制（group / scale system / payload）。
"""
    write_text(logs_dir / "AX1_s0_divisor_oracle.md", ax1_md)
    written.append(str(logs_dir / "AX1_s0_divisor_oracle.md"))

    # AX2
    by_gs: dict[str, list[float]] = defaultdict(list)
    for r in ax2:
        by_gs[str(r.get("group_size", "?"))].append(_f(r.get("R_Y")))
    gs_lines = "\n".join(f"- G{k}: 平均 R_Y={_mean(v):.4f} (n={len(v)})" for k, v in sorted(by_gs.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0))
    g16 = _mean(by_gs.get("16", []))
    g64_is_main = "64-group 共享范围是主要根因" if g16 >= 0.15 else ("64-group 共享范围有一定影响" if g16 >= 0.05 else "64-group 共享范围影响有限")
    ax2_md = f"""# AX2：HiF4 Group Size 消融

## 1. 实验目标

判断 64 元素共享一套 S0/层级尺度是否过大，是否因 4×16 子块动态范围耦合造成额外误差。

## 2. 实验设置

- 变体：G16 / G32 / G64（仅 G64 为标准 HiF4）
- 离散度 dose：d∈{{0,0.5,1.0,1.5,2.0}}
- 样本行数：group-size={len(ax2)}，dispersion={len(ax2d)}

## 3. 实验结果

{gs_lines}

## 4. 实验分析

比较 G16/G32 相对标准 G64 的输出恢复率。若 d 增大时 G64 恶化更快，则 4×16 动态范围差异会放大 64-group 耦合损失。

## 5. 实验结论

- {g64_is_main}
- G16 平均输出恢复率约为 {g16:.2%}

## 6. 对算法设计的启示

若 G16 恢复显著，可考虑更细的激活 scale 共享粒度；否则不必优先改 group size。
"""
    write_text(logs_dir / "AX2_group_size_ablation.md", ax2_md)
    written.append(str(logs_dir / "AX2_group_size_ablation.md"))

    # AX3
    z_hf = [_f(r.get("hf_occ_zero_rate")) for r in ax3]
    z_nv = [_f(r.get("nv_occ_zero_rate")) for r in ax3]
    b_hf = [_f(r.get("hf_occ_boundary_rate")) for r in ax3]
    b_nv = [_f(r.get("nv_occ_boundary_rate")) for r in ax3]
    ax3_md = f"""# AX3：NVFP4 与 HiF4 网格及真实占用

## 1. 实验目标

比较两种格式的理论 payload 网格、有效局部尺度，以及真实激活占用方式是否错配。

## 2. 实验设置

- 理论网格从 quantizer 枚举，边界取 `grid.abs().max()`
- 真实占用来自 sidecar / HiF4 metadata payload
- 样本行数：{len(ax3)}
- 理论网格点数：NVFP4={theory.get('nvfp4_stats', {}).get('num_positive_levels')}，HiF4={theory.get('hif4_stats', {}).get('num_positive_levels')}

## 3. 实验结果

- HiF4 零点占用率均值={_mean(z_hf):.4f}；NVFP4={_mean(z_nv):.4f}
- HiF4 边界占用率均值={_mean(b_hf):.4f}；NVFP4={_mean(b_nv):.4f}
- NVFP4 最大合法幅值={theory.get('nvfp4_max')}；HiF4={theory.get('hif4_max')}

## 4. 实验分析

理论网格只说明“能表示哪些点”；真实占用说明“激活实际落在哪里”。若 HiF4 零点/边界堆积更高，且 Oracle S0 能同时改善占用与输出误差，则形成「S0 定位 → 占用错配 → 输出误差」机制链。

## 5. 实验结论

- HiF4 是否更容易堆零点：{"是" if _mean(z_hf) > _mean(z_nv) + 0.02 else "证据不足/不明显"}
- HiF4 是否更容易堆边界：{"是" if _mean(b_hf) > _mean(b_nv) + 0.02 else "证据不足/不明显"}

## 6. 对算法设计的启示

占用错配本身不是直接算法旋钮，但能解释为何改 S0/Scale System 会连带改善误差。
"""
    write_text(logs_dir / "AX3_grid_occupancy.md", ax3_md)
    written.append(str(logs_dir / "AX3_grid_occupancy.md"))

    # AX4
    by_h: dict[str, list[float]] = defaultdict(list)
    for r in ax4:
        by_h[str(r.get("hybrid", r.get("variant", "?")))].append(_f(r.get("R_Y")))
    h_lines = "\n".join(f"- {k}: 平均 R_Y={_mean(v):.4f}" for k, v in sorted(by_h.items()))
    nh = _mean([v for k, vs in by_h.items() if "NH" in k.upper() for v in vs])
    hn = _mean([v for k, vs in by_h.items() if "HN" in k.upper() for v in vs])
    if nh > hn + 0.05:
        verdict = "Scale System 主导"
    elif hn > nh + 0.05:
        verdict = "Payload 主导"
    elif max(nh, hn) < 0.05:
        verdict = "证据不足"
    else:
        verdict = "两者交互"
    ax4_md = f"""# AX4：Scale System 与 Payload 交叉因子

## 1. 实验目标

区分 HiF4 激活损失主要来自 Scale System 还是 Payload Grid。

## 2. 实验设置

- 真实组合：NN / HH
- 诊断 Hybrid：HN / NH，并做 raw 与 range-matched
- Hybrid 标记 `is_valid_hardware_format=false`
- 样本行数：{len(ax4)}

## 3. 实验结果

{h_lines}

## 4. 实验分析

若 NH（NVFP4 Scale + HiF4 Payload）明显更好，说明 Scale System 是主因；若 HN 更好，则 Payload Grid 更关键。raw 与 range-matched 若结论不同，需把动态范围与网格形状分开讨论。

## 5. 实验结论

- {verdict}
- NH 聚合 R_Y≈{nh:.4f}；HN 聚合 R_Y≈{hn:.4f}

## 6. 对算法设计的启示

下一步应优先优化结论指向的部件，而不是同时大改 Scale 与 Payload。
"""
    write_text(logs_dir / "AX4_scale_payload_factorization.md", ax4_md)
    written.append(str(logs_dir / "AX4_scale_payload_factorization.md"))

    # AX5
    top = ax5[:5]
    top_lines = "\n".join(
        f"- #{r.get('rank')}: {r.get('root_cause')} | 来源={r.get('evidence_source')} | R_Y={r.get('R_Y')} | n={r.get('n','')}"
        for r in top
    )
    rule = rules[0] if rules else {}
    ax5_md = f"""# AX5：根因排序与低开销规则

## 1. 实验目标

统一主计划 A2 与增量 AX1–AX4 的输出侧可恢复误差，给出激活误差前三根因；并在 S0 足够重要时反推低开销规则。

## 2. 实验设置

- 聚合方式：按机制求平均 R_Y（禁止把每个 sample 当独立根因行）
- A2 指标为相对 BF16 的 R_cf_output，AX 指标为相对 NVFP4 Source 的转换 R_Y；报告中注明来源
- 规则状态：{rule.get('status', 'n/a')}

## 3. 实验结果

排名前五：
{top_lines}

规则摘要：`{json.dumps(rule, ensure_ascii=False)[:800]}`

## 4. 实验分析

排序服务于“下一步改什么”，不是把不同分母的恢复率硬加总。AX 转换误差与 A2 内部消融要分开阅读，再综合决策。

## 5. 实验结论

- 前三根因见阶段报告
- candidate_for_e2e={rule.get('candidate_for_e2e')}

## 6. 对算法设计的启示

只把验证集稳定、且有可实现旋钮的根因带入后续算法实现；本阶段不启动完整 E2E。
"""
    write_text(logs_dir / "AX5_rule_validation.md", ax5_md)
    written.append(str(logs_dir / "AX5_rule_validation.md"))
    return written


def build_phase_report(run_dir: Path, reports_dir: Path) -> dict[str, Any]:
    phase_dir = ensure_dir(reports_dir / "phase_reports")
    fig_dir = ensure_dir(run_dir / "figures")
    ax1 = _read_csv_rows(run_dir / "ax1_s0_divisor_oracle.csv")
    ax2 = _read_csv_rows(run_dir / "ax2_group_size_ablation.csv")
    ax2d = _read_csv_rows(run_dir / "ax2_sub16_dispersion.csv")
    ax3 = _read_csv_rows(run_dir / "ax3_grid_occupancy.csv")
    ax4 = _read_csv_rows(run_dir / "ax4_cross_format_factorization.csv")
    ranking = _read_csv_rows(run_dir / "ax5_root_cause_ranking.csv")
    rules = _read_csv_rows(run_dir / "ax5_rule_validation.csv")
    theory_path = run_dir / "ax3_theoretical_grid.json"
    theory = json.loads(theory_path.read_text(encoding="utf-8")) if theory_path.is_file() else {}

    figures: list[str] = []
    for p in [
        _fig_s0_divisor(ax1, fig_dir),
        _fig_s0_recovery(ax1, fig_dir),
        _fig_group_size(ax2, fig_dir),
        _fig_dispersion(ax2d, fig_dir),
        _fig_hybrid(ax4, fig_dir),
        _fig_ranking(ranking, fig_dir),
    ]:
        if p:
            figures.append(p)
    figures.extend(_fig_occupancy(ax3, fig_dir))
    figures.extend(_fig_theory_grid(theory, fig_dir))

    logs = build_experiment_logs(run_dir, reports_dir)
    top3 = ranking[:3]
    rule = rules[0] if rules else {}
    ry1 = _mean([_f(r.get("output_recovery")) for r in ax1])

    md = f"""# 02 激活量化误差定位（增量 AX1–AX5）

## 1. 研究问题

在主计划已确认 HiF4 激活转换误差较大的前提下，误差主要来自 S0 位置、64-group 粒度、层级指数、payload 网格，还是 NVFP4/HiF4 尺度系统差异？各机制在真实 Linear 输出上最多可恢复多少？

## 2. 已有三格式基线

复用主计划 A1 / repr-al 结果，不重跑。

## 3. 已有 HiF4 内部步骤消融

复用 A2：`continuous_s0` / `oracle_e8_e4_joint` / `continuous_payload_clipped` 等。

## 4. AX1：S0 位置是否合理

平均 Oracle 输出恢复率 R_Y={ry1:.4f}。详见 `experiment_logs/AX1_s0_divisor_oracle.md`。

## 5. AX2：64-group 是否过大

见 `ax2_group_size_ablation.csv` 与 AX2 日志。

## 6. AX3：网格与真实占用

见 `ax3_grid_occupancy.csv`、`ax3_theoretical_grid.json`。

## 7. AX4：Scale 与 Payload 谁主导

见 `ax4_cross_format_factorization.csv`。

## 8. NVFP4 Source 自身误差

引用主计划 A2 NVFP4 内部消融，不重跑。

## 9. 所有机制 Linear 输出可恢复误差排名

"""
    for r in ranking[:10]:
        md += (
            f"- #{r.get('rank')}: **{r.get('root_cause')}** "
            f"(来源 {r.get('evidence_source')}, R={r.get('R_Y')}, n={r.get('n','')}, note={r.get('note','')})\n"
        )

    md += f"""

## 10. Prefill/Decode、Layer、Projection 差异

本 run phases 以结果 CSV 中 `phase/projection/layer_idx` 字段为准；若仅含 prefill，需补跑 decode 后再更新。

## 11. 低开销规则可行性

状态：`{rule.get('status')}`；candidate_for_e2e=`{rule.get('candidate_for_e2e')}`。

## 12. 验证集结论

若本 run 仅为 discovery，请在 validation run 上复验前三根因是否反转。

## 13. 激活量化前三根因

"""
    for i, r in enumerate(top3, start=1):
        md += f"""### 根因 #{i}：{r.get('root_cause')}

- 机制：{r.get('root_cause')}
- 观察/反事实证据来源：{r.get('evidence_source')}
- Linear 输出可恢复误差（聚合）：{r.get('R_Y')}
- 备注：{r.get('note', '')}

"""

    md += f"""
## 图表

共生成 {len(figures)} 张图，目录：`{fig_dir}`。

## 实验日志

"""
    for p in logs:
        md += f"- `{p}`\n"

    report_path = phase_dir / "02_activation_error_localization.md"
    write_text(report_path, md)
    summary = {
        "run_id": run_dir.name,
        "figures": figures,
        "phase_report": str(report_path),
        "experiment_logs": logs,
        "top3_root_causes": top3,
        "rule_selection": rule,
    }
    atomic_write_json(run_dir / "activation_incremental_summary.json", summary)
    return summary


def build_activation_incremental_report(run_dir: Path, reports_root: Path | None = None) -> dict[str, Any]:
    reports_root = reports_root or Path(
        "/home/shaoyuantian/program/HiF4_Sp/Inference_Paradigm_Conversion/reports"
    )
    return build_phase_report(run_dir, ensure_dir(reports_root))
