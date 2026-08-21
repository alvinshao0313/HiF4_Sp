"""Figures + Chinese report for Linear puncture runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Native_NVFP4_HiF4_Linear_Puncture.src.config import load_config, results_dir
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import ensure_dir, read_json

HEADLINE = [
    "E1_WN_AM",
    "E2_WH_AM_RTN",
    "E3_WH_AM_GREEDY",
    "E4_WH_AH_RTN",
    "E5_WH_AH_DIAG",
    "E6_WH_AH_GREEDY",
]


def _heatmap(ax, df: pd.DataFrame, variant: str, title: str) -> None:
    sub = df[df["variant_id"] == variant]
    layers = sorted(sub["layer_idx"].unique())
    projs = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    mat = np.full((len(layers), len(projs)), np.nan)
    for i, layer in enumerate(layers):
        for j, proj in enumerate(projs):
            hit = sub[(sub["layer_idx"] == layer) & (sub["projection"] == proj)]
            if len(hit):
                mat[i, j] = float(hit["nmse"].iloc[0])
    im = ax.imshow(np.log10(np.clip(mat, 1e-12, None)), aspect="auto")
    ax.set_xticks(range(len(projs)))
    ax.set_xticklabels(projs, rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(x) for x in layers])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)


def build_figures(run_dir: Path) -> list[Path]:
    fig_dir = ensure_dir(run_dir / "figures")
    df = pd.read_csv(run_dir / "linear_results.csv")
    gdf = pd.read_csv(run_dir / "global_summary.csv")
    diag = pd.read_csv(run_dir / "diagonal_search.csv")
    wdf = pd.read_csv(run_dir / "weight_variants.csv")
    paths: list[Path] = []

    # fig01
    fig, ax = plt.subplots(figsize=(8, 4))
    sub = gdf[gdf["variant_id"].isin(HEADLINE)]
    ax.bar(sub["variant_id"], sub["global_nmse"])
    ax.set_yscale("log")
    ax.set_ylabel("global NMSE")
    ax.set_title("Headline variants (reference=WN+AN; baseline bar omitted)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = fig_dir / "fig01_global_nmse_by_variant.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # fig02
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sub["variant_id"], sub["global_sqnr_db"])
    ax.set_ylabel("global SQNR (dB)")
    ax.set_title("Headline SQNR (higher is better)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = fig_dir / "fig02_sqnr_by_variant.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # fig03-05 heatmaps
    for name, vid, title in [
        ("fig03_layer_projection_nmse_heatmap_e1.png", "E1_WN_AM", "E1 WN+AM NMSE (log10)"),
        ("fig04_layer_projection_nmse_heatmap_e2.png", "E2_WH_AM_RTN", "E2 WH+AM RTN NMSE (log10)"),
        ("fig05_layer_projection_nmse_heatmap_e4.png", "E4_WH_AH_RTN", "E4 WH+AH RTN NMSE (log10)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        _heatmap(ax, df, vid, title)
        fig.tight_layout()
        p = fig_dir / name
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    # fig06 greedy recovery
    fig, ax = plt.subplots(figsize=(10, 4))
    mods = sorted(df["module_name"].unique())
    r_hm, r_hh = [], []
    for m in mods:
        e2 = float(df[(df.module_name == m) & (df.variant_id == "E2_WH_AM_RTN")]["error_energy"].iloc[0])
        e3 = float(df[(df.module_name == m) & (df.variant_id == "E3_WH_AM_GREEDY")]["error_energy"].iloc[0])
        e4 = float(df[(df.module_name == m) & (df.variant_id == "E4_WH_AH_RTN")]["error_energy"].iloc[0])
        e6 = float(df[(df.module_name == m) & (df.variant_id == "E6_WH_AH_GREEDY")]["error_energy"].iloc[0])
        r_hm.append((e2 - e3) / e2 if e2 else np.nan)
        r_hh.append((e4 - e6) / e4 if e4 else np.nan)
    x = np.arange(len(mods))
    ax.plot(x, r_hm, label="HM greedy recovery (E2→E3)")
    ax.plot(x, r_hh, label="HH greedy recovery (E4→E6)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(".")[-1] + "@L" + m.split(".")[2] for m in mods], rotation=90)
    ax.set_ylabel("recovery")
    ax.legend()
    ax.set_title("Greedy recovery by module")
    fig.tight_layout()
    p = fig_dir / "fig06_greedy_recovery_by_module.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # fig07 diagonal recovery
    fig, ax = plt.subplots(figsize=(10, 4))
    r_diag = []
    for m in mods:
        e4 = float(df[(df.module_name == m) & (df.variant_id == "E4_WH_AH_RTN")]["error_energy"].iloc[0])
        e5 = float(df[(df.module_name == m) & (df.variant_id == "E5_WH_AH_DIAG")]["error_energy"].iloc[0])
        r_diag.append((e4 - e5) / e4 if e4 else np.nan)
    ax.bar(range(len(mods)), r_diag)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(range(len(mods)))
    ax.set_xticklabels([m.split(".")[-1] + "@L" + m.split(".")[2] for m in mods], rotation=90)
    ax.set_ylabel("recovery")
    ax.set_title("Diagonal-search recovery (E4→E5)")
    fig.tight_layout()
    p = fig_dir / "fig07_diagonal_recovery_by_module.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # fig08 diagonal scale heatmap median(|log2 d|)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    layers = sorted(df["layer_idx"].unique())
    projs = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    mat = np.full((len(layers), len(projs)), np.nan)
    for i, layer in enumerate(layers):
        for j, proj in enumerate(projs):
            name_hits = df[(df.layer_idx == layer) & (df.projection == proj)]["module_name"]
            if len(name_hits) == 0:
                continue
            mname = name_hits.iloc[0]
            hit = diag[diag.module_name == mname]
            if len(hit):
                # approximate median(|log2 d|) from percentiles of d
                dmed = float(hit["diag_scale_median"].iloc[0])
                mat[i, j] = abs(np.log2(max(dmed, 1e-12)))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(projs)))
    ax.set_xticklabels(projs, rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(x) for x in layers])
    ax.set_title("median(|log2 d|) by module")
    plt.colorbar(im, ax=ax, fraction=0.046)
    note = (
        f"p10/p50/p90 of diag_scale over modules: "
        f"{diag['diag_scale_p10'].median():.3g}/"
        f"{diag['diag_scale_median'].median():.3g}/"
        f"{diag['diag_scale_p90'].median():.3g}; "
        f"bound hit mean lo/hi="
        f"{diag['fraction_d_at_lower_bound'].mean():.3g}/"
        f"{diag['fraction_d_at_upper_bound'].mean():.3g}"
    )
    fig.text(0.01, 0.01, note, fontsize=8)
    fig.tight_layout()
    p = fig_dir / "fig08_diagonal_scale_distribution_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # fig09 weight nmse vs output nmse
    fig, ax = plt.subplots(figsize=(5, 5))
    out_rtn = []
    out_g = []
    for m in mods:
        out_rtn.append(
            float(df[(df.module_name == m) & (df.variant_id == "C2_WH_AN_RTN")]["nmse"].iloc[0])
        )
        out_g.append(
            float(df[(df.module_name == m) & (df.variant_id == "C3_WH_AN_GREEDY")]["nmse"].iloc[0])
        )
    ax.scatter(wdf["rtn_weight_nmse"], out_rtn, label="direct")
    ax.scatter(wdf["greedy_weight_nmse"], out_g, label="greedy")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("weight NMSE")
    ax.set_ylabel("output NMSE (C2/C3 vs NN)")
    ax.legend()
    ax.set_title("Weight NMSE vs output NMSE")
    fig.tight_layout()
    p = fig_dir / "fig09_weight_nmse_vs_output_nmse.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # fig10 activation-only
    fig, ax = plt.subplots(figsize=(6, 4))
    for vid, label in [("E1_WN_AM", "WN+AM"), ("C1_WN_AH", "WN+AH")]:
        vals = df[df.variant_id == vid]["nmse"].astype(float)
        ax.plot(range(len(vals)), vals.values, marker="o", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("module index")
    ax.set_ylabel("NMSE")
    ax.legend()
    ax.set_title("Activation-only loss vs source A_N")
    fig.tight_layout()
    p = fig_dir / "fig10_activation_only_loss.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # fig11 controls
    fig, ax = plt.subplots(figsize=(6, 4))
    for vid in ["C1_WN_AH", "C2_WH_AN_RTN", "E4_WH_AH_RTN"]:
        row = gdf[gdf.variant_id == vid].iloc[0]
        ax.bar(vid, row["global_nmse"])
    ax.set_yscale("log")
    ax.set_ylabel("global NMSE")
    ax.set_title("Controls: act-only / weight-only / total HH (not additive)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    p = fig_dir / "fig11_error_decomposition_controls.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # fig12 boxplot by projection
    fig, ax = plt.subplots(figsize=(10, 4))
    data = []
    labels = []
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        for vid in HEADLINE:
            vals = df[(df.projection == proj) & (df.variant_id == vid)]["nmse"].astype(float)
            data.append(vals.values)
            labels.append(f"{proj[:1]}:{vid.split('_')[0]}")
    ax.boxplot(data, showfliers=False)
    ax.set_yscale("log")
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title("Per-projection NMSE distributions across headline variants")
    fig.tight_layout()
    p = fig_dir / "fig12_per_projection_boxplot.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    return paths


def build_report_cn(run_dir: Path) -> Path:
    summary = read_json(run_dir / "summary.json")
    gdf = pd.read_csv(run_dir / "global_summary.csv")
    diag = pd.read_csv(run_dir / "diagonal_search.csv")
    q = summary["questions"]

    def gnmse(vid: str) -> str:
        return f"{float(gdf.loc[gdf.variant_id==vid,'global_nmse'].iloc[0]):.6g}"

    lines = [
        "# Qwen3-8B 原生 NVFP4 → HiF4 Linear 穿刺实验报告",
        "",
        "## 1. 实验目的",
        "在原生 packed NVFP4 QAT checkpoint 的 Linear 语义下，比较 MXFP8/HiF4 激活、HiF4 权重直接转换、",
        "三级层级 scale 联合搜索，以及逐通道对角等价缩放搜索对 Linear 输出相对 `Y_NN=Linear(A_N,W_N)` 的损失。",
        "",
        "## 2. 原生 NVFP4 QAT checkpoint 的真实 Linear 推理语义",
        "每个目标 Linear：`X_pre → block rotation → X_rot → NVFP4 A4 QDQ → A_N → GEMM(W_N)`。",
        "本实验用纯 PyTorch semantic oracle 复现该顺序；252 个目标 Linear 全部执行，不只测代表层。",
        "",
        "## 3. 为什么必须捕获 rotation 后、quantization 前 X_rot",
        "离线 MXFP8/HiF4 都作用在同一旋转坐标系；捕获点固定为 post-rotation / pre-quant，避免 double rotation。",
        "",
        "## 4. representative layers / prompts / split / token sampling",
        "正式层 `[2,10,18,26,34]` × 7 projections = 35；prompt 4 类×8，前4 cal / 后4 val；",
        "token：去 pad，T≤64 全存，否则 `linspace(0,T-1,64).round().long()`。",
        "",
        "## 5. source semantic oracle 与 native kernel cross-check",
        "正式结果基于 semantic oracle（rotate + fake NVFP4 QDQ + BF16 GEMM）。",
        "若硬件不支持官方 fused NV kernel，则不是 kernel bit-exact benchmark。",
        "",
        "## 6. WN+AM：只替换 activation 为 MXFP8 的损失",
        f"全局 NMSE = {gnmse('E1_WN_AM')}；摘要：{q['wn_am_loss_vs_wn_an']}",
        "",
        "## 7. WH+AM direct：weight 转 HiF4 后总损失",
        f"全局 NMSE = {gnmse('E2_WH_AM_RTN')}；摘要：{q['wh_am_rtn_loss_vs_wn_an']}",
        "",
        "## 8. WH+AM greedy：S0/e8/e4 三级层级 scale 联合搜索能恢复多少",
        f"E3 全局 NMSE = {gnmse('E3_WH_AM_GREEDY')}；recovery={q['wh_am_greedy_recovery'].get('recovery')}",
        "",
        "## 9. WH+AH direct：完整 HiF4 W4A4 local loss",
        f"全局 NMSE = {gnmse('E4_WH_AH_RTN')}；摘要：{q['wh_ah_rtn_loss_vs_wn_an']}",
        "",
        "## 10. WH+AH DIAG：逐通道对角等价缩放搜索能恢复多少",
        f"E5 全局 NMSE = {gnmse('E5_WH_AH_DIAG')}；recovery={q['wh_ah_diagonal_recovery'].get('recovery')}",
        "搜索只在 calibration 上进行；validation 只评估固定 D。这不是 SmoothQuant alpha。",
        "",
        "## 11. WH+AH greedy：三级 weight-scale 搜索在 HiF4 A4 下恢复多少",
        f"E6 全局 NMSE = {gnmse('E6_WH_AH_GREEDY')}；recovery={q['wh_ah_greedy_recovery'].get('recovery')}",
        "",
        "## 12. activation-only / weight-only controls",
        f"C1(WN+AH)={gnmse('C1_WN_AH')}, C2(WH+AN RTN)={gnmse('C2_WH_AN_RTN')}, "
        f"C3(WH+AN greedy)={gnmse('C3_WH_AN_GREEDY')}, C0(FP act)={gnmse('C0_FP')}。",
        "C2/C3 仍使用 source `A_N`，因此是固定激活下的权重转换 recovery。",
        "",
        "## 13. layer / projection 敏感性",
        "见 figures 中 heatmap / boxplot；headline 以 energy-weighted global NMSE 为准，不用模块 NMSE 简单平均。",
        "",
        "## 14. 对角阵 d_j 分布、边界命中率与 group rollback 统计",
        f"kept 均值={diag['num_groups_kept'].mean():.3g}, "
        f"rollback 均值={diag['num_groups_rolled_back'].mean():.3g}, "
        f"d 边界命中 lo/hi 均值="
        f"{diag['fraction_d_at_lower_bound'].mean():.3g}/"
        f"{diag['fraction_d_at_upper_bound'].mean():.3g}。",
        "",
        "## 15. 最终回答与下一步建议",
        f"- WN+AM 相对 WN+AN 的全局损失：NMSE={q['wn_am_loss_vs_wn_an']['global_nmse']}",
        f"- WH+AM RTN 全局损失：NMSE={q['wh_am_rtn_loss_vs_wn_an']['global_nmse']}",
        f"- WH+AM greedy recovery：{q['wh_am_greedy_recovery'].get('recovery')}",
        f"- WH+AH RTN 全局损失：NMSE={q['wh_ah_rtn_loss_vs_wn_an']['global_nmse']}",
        f"- WH+AH 对角搜索 recovery：{q['wh_ah_diagonal_recovery'].get('recovery')}",
        f"- WH+AH greedy recovery：{q['wh_ah_greedy_recovery'].get('recovery')}",
        "",
        "## 16. 限制",
        "- 这是 **Linear-local semantic oracle**，不是端到端部署精度，也不是低比特 kernel 数值/时延基准。",
        "- 正式比较保留 checkpoint 指定的 **online block rotation**；离线格式转换不再二次旋转。",
        "- 对角阵搜索只给出 Linear-local 可恢复上界，不证明可跨 RMSNorm / residual fold。",
        "",
    ]
    out = run_dir / "report_cn.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_all(run_id: str) -> None:
    run_dir = results_dir(run_id)
    figs = build_figures(run_dir)
    report = build_report_cn(run_dir)
    print(f"REPORT DONE figures={len(figs)} report={report}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build figures and Chinese report")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args(argv)
    _ = load_config(args.config)
    build_all(args.run_id)


if __name__ == "__main__":
    main()
