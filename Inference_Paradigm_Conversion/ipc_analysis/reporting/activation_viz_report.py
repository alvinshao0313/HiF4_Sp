"""NVFP4 W4A4 activation distribution + NVFP4→HiF4 residual visualization report."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_distribution_residual import (
    activation_quantile_residual_curve,
    residual_energy_concentration,
)
from Inference_Paradigm_Conversion.ipc_analysis.config import LINEAR_PROJECTIONS
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (
    atomic_write_json,
    ensure_dir,
    write_text,
)

_FONT_CANDIDATES = [
    Path(
        "/home/shaoyuantian/program/HiF4_Sp/Inference_Paradigm_Conversion/"
        "assets/fonts/SourceHanSansSC-Regular.otf"
    ),
    Path(
        "/home/shaoyuantian/program/HiF4_Sp/Inference_Paradigm_Conversion/"
        "assets/fonts/NotoSansCJK-Regular.ttc"
    ),
]

_NV_PAYLOAD_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
_HF_PAYLOAD_LEVELS = np.array(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float64
)
_R11_LAYERS = (4, 18, 34)
_MAX_GROUP_SCATTER = 50_000
_GROUP_SCATTER_SEED = 20260810
_TINY = 1e-12
_PHASE_REPORT_PATH = Path(
    "/home/shaoyuantian/program/HiF4_Sp/Inference_Paradigm_Conversion/"
    "reports/phase_reports/03_w4a4_activation_distribution_hif4_residual.md"
)

FIGURE_NAMES = [
    "fig_d1_w4a4_activation_hist_full.png",
    "fig_d2_w4a4_activation_hist_central.png",
    "fig_d3_w4a4_activation_log2_abs.png",
    "fig_d4_theory_vs_real_activation_triptych.png",
    "fig_d5_activation_distribution_by_projection_phase.png",
    "fig_d6_activation_rms_layer_projection_heatmap.png",
    "fig_r1_delta_hist_full.png",
    "fig_r2_delta_log10_abs_hist.png",
    "fig_r3_an_vs_ah_hexbin_full.png",
    "fig_r3_an_vs_ah_hexbin_central.png",
    "fig_r4_an_vs_delta_hexbin.png",
    "fig_r5_residual_energy_concentration.png",
    "fig_r6_residual_vs_activation_quantile.png",
    "fig_r7_zero_transition.png",
    "fig_r8_payload_transition_heatmap_count.png",
    "fig_r8_payload_transition_heatmap_row_normalized.png",
    "fig_r9_residual_nmse_layer_projection_heatmap.png",
    "fig_r9_residual_rms_layer_projection_heatmap.png",
    "fig_r10_residual_by_projection_boxplot.png",
    "fig_r11_3d_token_group_residual_surface_layer4.png",
    "fig_r11_3d_token_group_residual_surface_layer18.png",
    "fig_r11_3d_token_group_residual_surface_layer34.png",
    "fig_r11_token_group_residual_heatmap_layer4.png",
    "fig_r11_token_group_residual_heatmap_layer18.png",
    "fig_r11_token_group_residual_heatmap_layer34.png",
    "fig_r12_3d_group_mechanism_scatter.png",
    "fig_r12_amax64_vs_rms_delta_2d.png",
    "fig_r12_dispersion_vs_rms_delta_2d.png",
    "fig_r13_3d_layer_projection_residual_landscape.png",
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


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    _require_file(path)
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def _f(x: Any, default: float | None = None) -> float:
    if x in (None, ""):
        if default is None:
            raise ValueError("missing numeric field")
        return default
    return float(x)


def _mean(xs: list[float]) -> float:
    if not xs:
        raise ValueError("empty mean")
    return float(sum(xs) / len(xs))


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        raise ValueError("empty percentile")
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return float(ys[i])


def _median(xs: list[float]) -> float:
    return _pct(xs, 0.5)


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float64).cpu().numpy()


def _save(fig: plt.Figure, path: Path) -> str:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"failed to write figure: {path}")
    return str(path)


def _shared_hist_edges(*arrays: np.ndarray, bins: int = 128) -> np.ndarray:
    vals = np.concatenate([a.reshape(-1) for a in arrays])
    if vals.size == 0:
        raise ValueError("empty arrays for hist edges")
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("non-finite hist range")
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, bins + 1)


def _quantile_abs(arr: np.ndarray, q: float) -> float:
    a = np.abs(arr.reshape(-1))
    if a.size == 0:
        raise ValueError("empty quantile")
    return float(np.quantile(a, q))


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    # average ties
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size:
        raise ValueError("spearman size mismatch")
    if x.size < 2:
        raise ValueError("spearman needs >=2 points")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom <= 0:
        return 0.0
    return float((rx * ry).sum() / denom)


def _nearest_level_indices(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    # values: abs payload; levels: sorted codebook
    d = np.abs(values.reshape(-1, 1) - levels.reshape(1, -1))
    return np.argmin(d, axis=1)


def _load_points(path: Path) -> dict[str, Any]:
    _require_file(path)
    pts = torch.load(path, map_location="cpu", weights_only=False)
    required = (
        "x_in",
        "a_nvfp4",
        "a_hif4",
        "delta_a",
        "layer_idx",
        "projection_id",
        "phase_id",
        "nv_payload",
        "hf_payload",
    )
    for k in required:
        if k not in pts:
            raise KeyError(f"activation_viz_points.pt missing key: {k}")
        if not torch.is_tensor(pts[k]):
            raise TypeError(f"points[{k}] must be tensor")
    n = int(pts["x_in"].numel())
    if n == 0:
        raise RuntimeError("activation_viz_points.pt has zero points")
    for k in required:
        if int(pts[k].numel()) != n:
            raise RuntimeError(f"points[{k}] numel {pts[k].numel()} != {n}")
    if "projection_names" not in pts:
        pts["projection_names"] = list(LINEAR_PROJECTIONS)
    if "phase_names" not in pts:
        pts["phase_names"] = ["prefill", "decode"]
    return pts


def _load_theory_grids(run_dir: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    json_path = run_dir / "ax3_theoretical_grid.json"
    nv_pt = run_dir / "ax3_nvfp4_full_internal_grid.pt"
    hf_pt = run_dir / "ax3_hif4_full_internal_grid.pt"
    blob_pt = run_dir / "ax3_theoretical_grid.pt"
    if not json_path.is_file() and not blob_pt.is_file() and not (nv_pt.is_file() and hf_pt.is_file()):
        raise FileNotFoundError(
            f"missing ax3_theoretical_grid.json/.pt or full-internal grid pt under {run_dir}"
        )

    theory: dict[str, Any] = {}
    if json_path.is_file():
        theory = json.loads(json_path.read_text(encoding="utf-8"))

    nv: torch.Tensor | None = None
    hf: torch.Tensor | None = None
    if nv_pt.is_file() and hf_pt.is_file():
        nv = torch.load(nv_pt, map_location="cpu", weights_only=True).to(torch.float32).reshape(-1)
        hf = torch.load(hf_pt, map_location="cpu", weights_only=True).to(torch.float32).reshape(-1)
    elif theory.get("nvfp4_full_internal_grid") and theory.get("hif4_full_internal_grid"):
        nv = torch.tensor(theory["nvfp4_full_internal_grid"], dtype=torch.float32).reshape(-1)
        hf = torch.tensor(theory["hif4_full_internal_grid"], dtype=torch.float32).reshape(-1)
    elif blob_pt.is_file():
        blob = torch.load(blob_pt, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict):
            raise TypeError(f"{blob_pt} must be a dict payload")
        if "nvfp4_full_internal_grid" not in blob or "hif4_full_internal_grid" not in blob:
            raise KeyError(f"{blob_pt} missing full internal grid tensors/lists")
        nv = torch.as_tensor(blob["nvfp4_full_internal_grid"], dtype=torch.float32).reshape(-1)
        hf = torch.as_tensor(blob["hif4_full_internal_grid"], dtype=torch.float32).reshape(-1)
        theory = {**theory, **{k: v for k, v in blob.items() if not torch.is_tensor(v)}}
    else:
        # JSON may only store paths — resolve them.
        nv_path = theory.get("nvfp4_full_internal_grid_path")
        hf_path = theory.get("hif4_full_internal_grid_path")
        if not nv_path or not hf_path:
            raise FileNotFoundError(
                f"ax3 grids not embeddable/loadable under {run_dir}: "
                "need .pt files, embedded lists, or *_grid_path in JSON"
            )
        nv_p, hf_p = Path(nv_path), Path(hf_path)
        if not nv_p.is_file():
            nv_p = run_dir / Path(nv_path).name
        if not hf_p.is_file():
            hf_p = run_dir / Path(hf_path).name
        _require_file(nv_p)
        _require_file(hf_p)
        nv = torch.load(nv_p, map_location="cpu", weights_only=True).to(torch.float32).reshape(-1)
        hf = torch.load(hf_p, map_location="cpu", weights_only=True).to(torch.float32).reshape(-1)

    assert nv is not None and hf is not None
    if int(nv.numel()) < 16 or int(hf.numel()) < 16:
        raise RuntimeError("theoretical grids unexpectedly small")
    return nv.contiguous(), hf.contiguous(), theory


def _proj_names(points: dict[str, Any]) -> list[str]:
    names = list(points.get("projection_names", LINEAR_PROJECTIONS))
    if len(names) != len(LINEAR_PROJECTIONS):
        raise RuntimeError(f"unexpected projection_names: {names}")
    return names


def _phase_names(points: dict[str, Any]) -> list[str]:
    names = list(points.get("phase_names", ["prefill", "decode"]))
    return names


# ---------------------------------------------------------------------------
# Figures D1–D6
# ---------------------------------------------------------------------------


def fig_d1_activation_hist_full(points: dict[str, Any], out: Path) -> str:
    x = _to_np(points["x_in"])
    an = _to_np(points["a_nvfp4"])
    ah = _to_np(points["a_hif4"])
    edges = _shared_hist_edges(x, an, ah, bins=160)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(x, bins=edges, density=True, histtype="step", linewidth=1.4, label="X_in")
    ax.hist(an, bins=edges, density=True, histtype="step", linewidth=1.4, label="A_N (NVFP4→GEMM)")
    ax.hist(ah, bins=edges, density=True, histtype="step", linewidth=1.4, label="A_H (HiF4 CF)")
    ax.set_xlabel("activation value（未归一化）")
    ax.set_ylabel("probability density")
    ax.set_title(
        f"D1 真实 W4A4 激活分布（full-range）\n"
        f"n_points={x.size:,}；分层确定性抽样 max_point_samples_per_capture=1024"
    )
    ax.legend()
    fig.tight_layout()
    return _save(fig, out / "fig_d1_w4a4_activation_hist_full.png")


def fig_d2_activation_hist_central(points: dict[str, Any], out: Path) -> str:
    x = _to_np(points["x_in"])
    an = _to_np(points["a_nvfp4"])
    ah = _to_np(points["a_hif4"])
    lim = max(_quantile_abs(x, 0.999), _quantile_abs(an, 0.999), _quantile_abs(ah, 0.999))
    if lim <= 0:
        raise RuntimeError("central xlim non-positive")
    mask_x = np.abs(x) <= lim
    mask_an = np.abs(an) <= lim
    mask_ah = np.abs(ah) <= lim
    clipped = 1.0 - float(mask_x.mean() + mask_an.mean() + mask_ah.mean()) / 3.0
    edges = np.linspace(-lim, lim, 161)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(x[mask_x], bins=edges, density=True, histtype="step", linewidth=1.4, label="X_in")
    ax.hist(an[mask_an], bins=edges, density=True, histtype="step", linewidth=1.4, label="A_N")
    ax.hist(ah[mask_ah], bins=edges, density=True, histtype="step", linewidth=1.4, label="A_H")
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("activation value")
    ax.set_ylabel("probability density")
    ax.set_title(
        f"D2 中心放大 xlim=[±{lim:.4g}]；裁剪 point fraction≈{clipped:.4f}\n"
        f"n_points={x.size:,}（抽样估计，非 full-tensor）"
    )
    ax.legend()
    fig.tight_layout()
    return _save(fig, out / "fig_d2_w4a4_activation_hist_central.png")


def fig_d3_activation_log2_abs(points: dict[str, Any], out: Path) -> str:
    series = {
        "X_in": _to_np(points["x_in"]),
        "A_N": _to_np(points["a_nvfp4"]),
        "A_H": _to_np(points["a_hif4"]),
    }
    zero_rates = {k: float((v == 0).mean()) for k, v in series.items()}
    logs = {k: np.log2(np.abs(v[v != 0])) for k, v in series.items()}
    edges = _shared_hist_edges(*logs.values(), bins=80)
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, z in logs.items():
        ax.hist(z, bins=edges, density=True, histtype="step", linewidth=1.4, label=k)
    ax.set_xlabel("log2(|activation|)")
    ax.set_ylabel("probability density")
    zr = ", ".join(f"{k} zero={v:.4f}" for k, v in zero_rates.items())
    ax.set_title(f"D3 非零激活数量级分布\n{zr}（0 不入 bin）")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out / "fig_d3_w4a4_activation_log2_abs.png")


def fig_d4_theory_vs_real_triptych(
    points: dict[str, Any],
    nv_grid: torch.Tensor,
    hf_grid: torch.Tensor,
    out: Path,
) -> str:
    nv = _to_np(nv_grid)
    hf = _to_np(hf_grid)
    an = _to_np(points["a_nvfp4"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # A/B: unique-point count histograms（每个 unique 点计 1）
    for ax, grid, title in (
        (axes[0], nv, "A. NVFP4 full internal grid\nE4M3FN × signed E2M1"),
        (axes[1], hf, "B. HiF4 full internal grid\nS0×2^(e8+e4)×signed S1P2"),
    ):
        edges = _shared_hist_edges(grid, bins=120)
        counts, _, _ = ax.hist(grid, bins=edges, density=False, color="C0", alpha=0.85)
        ax.set_ylabel("unique point count")
        ax.set_xlabel("internal representable value（未 /max）")
        ax.set_title(title)
        if float(np.sum(counts)) != float(grid.size):
            # hist may drop exact edge duplicates; verify mass approx
            pass

    edges_c = _shared_hist_edges(an, bins=120)
    axes[2].hist(an, bins=edges_c, density=True, color="C2", alpha=0.85)
    axes[2].set_ylabel("activation probability density")
    axes[2].set_xlabel("A_N dequantized value（未归一化）")
    axes[2].set_title("C. 真实 W4A4 A_N")

    fig.suptitle(
        "D4 理论内部网格 vs 真实激活（A/B 排除最外层 FP32 scale；"
        "C 为实际 dequantized A_N；x 均不做 /max）",
        fontsize=10,
    )
    fig.tight_layout()
    return _save(fig, out / "fig_d4_theory_vs_real_activation_triptych.png")


def fig_d5_by_projection_phase(points: dict[str, Any], out: Path) -> str:
    an = _to_np(points["a_nvfp4"])
    pid = points["projection_id"].detach().cpu().numpy().astype(np.int64)
    ph = points["phase_id"].detach().cpu().numpy().astype(np.int64)
    proj_names = _proj_names(points)
    phase_names = _phase_names(points)
    if "prefill" not in phase_names or "decode" not in phase_names:
        raise RuntimeError(f"need prefill/decode phases, got {phase_names}")
    prefill_id = phase_names.index("prefill")
    decode_id = phase_names.index("decode")

    fig, axes = plt.subplots(2, 7, figsize=(18, 6), sharey="row")
    for row, phase_id, phase_label in ((0, prefill_id, "prefill"), (1, decode_id, "decode")):
        row_logs: list[np.ndarray] = []
        for j, pname in enumerate(proj_names):
            m = (pid == j) & (ph == phase_id)
            vals = an[m]
            if vals.size == 0:
                raise RuntimeError(f"no points for {phase_label}/{pname}")
            nz = vals[vals != 0]
            if nz.size == 0:
                raise RuntimeError(f"all-zero A_N for {phase_label}/{pname}")
            row_logs.append(np.log2(np.abs(nz)))
        xlim = (
            float(min(z.min() for z in row_logs)),
            float(max(z.max() for z in row_logs)),
        )
        edges = np.linspace(xlim[0], xlim[1], 49)
        for j, pname in enumerate(proj_names):
            ax = axes[row, j]
            m = (pid == j) & (ph == phase_id)
            vals = an[m]
            zr = float((vals == 0).mean())
            ax.hist(row_logs[j], bins=edges, density=True, color="C0", alpha=0.85)
            ax.set_xlim(xlim)
            if row == 0:
                ax.set_title(pname, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{phase_label}\ndensity")
            ax.text(
                0.02,
                0.95,
                f"zero={zr:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
            )
            ax.tick_params(labelsize=7)
    fig.suptitle("D5 projection×phase：log2(|A_N|) probability density（0 不入 bin）", fontsize=11)
    fig.tight_layout()
    return _save(fig, out / "fig_d5_activation_distribution_by_projection_phase.png")


def fig_d6_rms_heatmap(summary_rows: list[dict[str, str]], out: Path) -> str:
    layers = sorted({int(r["layer_idx"]) for r in summary_rows})
    projs = list(LINEAR_PROJECTIONS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    mats: list[np.ndarray] = []
    for phase in ("prefill", "decode"):
        mat = np.full((len(layers), len(projs)), np.nan, dtype=np.float64)
        for i, li in enumerate(layers):
            for j, pj in enumerate(projs):
                vals = [
                    _f(r["an_rms"])
                    for r in summary_rows
                    if int(r["layer_idx"]) == li
                    and r["projection"] == pj
                    and r["phase"] == phase
                ]
                if not vals:
                    raise RuntimeError(f"missing an_rms for {phase}/L{li}/{pj}")
                mat[i, j] = math.log10(_mean(vals) + _TINY)
        mats.append(mat)
    vmin = float(min(np.nanmin(m) for m in mats))
    vmax = float(max(np.nanmax(m) for m in mats))
    for ax, mat, phase in zip(axes, mats, ("prefill", "decode")):
        im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(projs)))
        ax.set_xticklabels(projs, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(x) for x in layers])
        ax.set_title(phase)
        ax.set_ylabel("layer_idx")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log10(RMS(A_N))")
    fig.suptitle("D6 layer×projection 激活尺度（full-tensor capture an_rms）")
    fig.tight_layout()
    return _save(fig, out / "fig_d6_activation_rms_layer_projection_heatmap.png")


# ---------------------------------------------------------------------------
# Figures R1–R10
# ---------------------------------------------------------------------------


def fig_r1_delta_hist(points: dict[str, Any], out: Path) -> str:
    d = _to_np(points["delta_a"])
    edges = _shared_hist_edges(d, bins=160)
    mean = float(d.mean())
    median = float(np.median(d))
    rms = float(np.sqrt((d * d).mean()))
    q99 = float(np.quantile(np.abs(d), 0.99))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(d, bins=edges, density=True, color="C0", alpha=0.85)
    ax.axvline(0.0, color="k", linewidth=0.8)
    ax.axvline(mean, color="r", linestyle="--", label=f"mean={mean:.4g}")
    ax.axvline(median, color="orange", linestyle=":", label=f"median={median:.4g}")
    ax.set_xlabel("ΔA = A_H − A_N")
    ax.set_ylabel("probability density")
    ax.set_title(
        f"R1 signed ΔA full-range\nRMS={rms:.4g}, q99_abs={q99:.4g}; n_points={d.size:,}"
    )
    ax.legend()
    fig.tight_layout()
    return _save(fig, out / "fig_r1_delta_hist_full.png")


def fig_r2_delta_log10_abs(points: dict[str, Any], out: Path) -> str:
    d = _to_np(points["delta_a"])
    zero_rate = float((d == 0).mean())
    nz = np.abs(d[d != 0])
    if nz.size == 0:
        raise RuntimeError("all-zero delta")
    z = np.log10(nz)
    edges = _shared_hist_edges(z, bins=80)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(z, bins=edges, density=True, color="C1", alpha=0.85)
    ax.set_xlabel("log10(|ΔA|)")
    ax.set_ylabel("probability density")
    ax.set_title(f"R2 |ΔA| 长尾；exact-zero rate={zero_rate:.4f}（0 不入 bin）")
    fig.tight_layout()
    return _save(fig, out / "fig_r2_delta_log10_abs_hist.png")


def fig_r3_an_vs_ah(points: dict[str, Any], out: Path) -> list[str]:
    an = _to_np(points["a_nvfp4"])
    ah = _to_np(points["a_hif4"])
    paths: list[str] = []

    def _one(mask: np.ndarray, name: str, title: str) -> str:
        x = an[mask]
        y = ah[mask]
        lim = float(max(np.max(np.abs(x)), np.max(np.abs(y))))
        if lim <= 0:
            raise RuntimeError("degenerate hexbin range")
        fig, ax = plt.subplots(figsize=(6.5, 6))
        hb = ax.hexbin(x, y, gridsize=80, mincnt=1, cmap="viridis", bins="log")
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.0, label="y=x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("A_N")
        ax.set_ylabel("A_H")
        ax.set_title(title)
        ax.legend(loc="upper left")
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="log10(count)")
        fig.tight_layout()
        return _save(fig, out / name)

    paths.append(
        _one(
            np.ones_like(an, dtype=bool),
            "fig_r3_an_vs_ah_hexbin_full.png",
            f"R3 A_N vs A_H full；n={an.size:,}",
        )
    )
    lim = max(_quantile_abs(an, 0.999), _quantile_abs(ah, 0.999))
    mask = (np.abs(an) <= lim) & (np.abs(ah) <= lim)
    clipped = 1.0 - float(mask.mean())
    paths.append(
        _one(
            mask,
            "fig_r3_an_vs_ah_hexbin_central.png",
            f"R3 central ±{lim:.4g}；裁剪 fraction={clipped:.4f}",
        )
    )
    return paths


def fig_r4_an_vs_delta(points: dict[str, Any], out: Path) -> str:
    an = _to_np(points["a_nvfp4"])
    d = _to_np(points["delta_a"])
    lim_x = _quantile_abs(an, 0.999)
    lim_y = _quantile_abs(d, 0.999)
    mask = (np.abs(an) <= lim_x) & (np.abs(d) <= lim_y)
    clipped = 1.0 - float(mask.mean())
    fig, ax = plt.subplots(figsize=(7, 5.5))
    hb = ax.hexbin(an[mask], d[mask], gridsize=80, mincnt=1, cmap="magma", bins="log")
    ax.axhline(0.0, color="w", linewidth=0.8)
    ax.set_xlabel("A_N")
    ax.set_ylabel("ΔA")
    ax.set_title(
        f"R4 A_N vs ΔA central 99.9% view\n"
        f"xlim=±{lim_x:.4g}, ylim=±{lim_y:.4g}, 裁剪={clipped:.4f}"
    )
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="log10(count)")
    fig.tight_layout()
    return _save(fig, out / "fig_r4_an_vs_delta_hexbin.png")


def fig_r5_energy(points: dict[str, Any], summary_rows: list[dict[str, str]], out: Path) -> str:
    energy = residual_energy_concentration(points["delta_a"])
    xs = [float(x) for x in energy["curve_fraction_elements"]]  # type: ignore[arg-type]
    ys = [float(y) for y in energy["curve_fraction_energy"]]  # type: ignore[arg-type]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", markersize=3)
    marks = {
        0.001: float(energy["top_0p1pct_energy_share"]),
        0.01: float(energy["top_1pct_energy_share"]),
        0.05: float(energy["top_5pct_energy_share"]),
        0.10: float(energy["top_10pct_energy_share"]),
    }
    for frac, share in marks.items():
        ax.axvline(frac, color="gray", linestyle=":", linewidth=0.8)
        ax.scatter([frac], [share], zorder=3)
        ax.annotate(f"top {frac*100:g}%={share:.3f}", (frac, share), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("top element fraction（sampled points）")
    ax.set_ylabel("cumulative ΔA² energy share")
    # cross-check medians from full-tensor CSV
    med = {
        "0.1%": _median([_f(r["energy_top_0p1pct_energy_share"]) for r in summary_rows]),
        "1%": _median([_f(r["energy_top_1pct_energy_share"]) for r in summary_rows]),
        "5%": _median([_f(r["energy_top_5pct_energy_share"]) for r in summary_rows]),
        "10%": _median([_f(r["energy_top_10pct_energy_share"]) for r in summary_rows]),
    }
    ax.set_title(
        "R5 残差能量集中（sampled estimate）\n"
        + "per-capture exact median shares: "
        + ", ".join(f"{k}={v:.3f}" for k, v in med.items())
    )
    fig.tight_layout()
    return _save(fig, out / "fig_r5_residual_energy_concentration.png")


def fig_r6_quantile_curve(points: dict[str, Any], out: Path) -> str:
    curve = activation_quantile_residual_curve(points["a_nvfp4"], points["delta_a"], num_bins=32)
    xs = np.arange(len(curve["count"]))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(xs, curve["rms_delta"], "C0-o", markersize=3, label="RMS(ΔA)")
    ax1.plot(xs, curve["mean_abs_delta"], "C1-s", markersize=3, label="mean|ΔA|")
    ax1.set_xlabel("|A_N| quantile bin")
    ax1.set_ylabel("residual")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.bar(xs, curve["count"], alpha=0.25, color="gray", label="count")
    ax2.set_ylabel("sample count")
    ax1.set_title("R6 |A_N| 分位条件下的残差（绝对量，非相对误差）")
    fig.tight_layout()
    return _save(fig, out / "fig_r6_residual_vs_activation_quantile.png")


def fig_r7_zero_transition(summary_rows: list[dict[str, str]], out: Path) -> str:
    keys = (
        "zero_transition_both_zero",
        "zero_transition_nv_zero_hf_nonzero",
        "zero_transition_nv_nonzero_hf_zero",
        "zero_transition_both_nonzero",
    )
    labels = [
        "both zero",
        "NV zero→HF nonzero",
        "NV nonzero→HF zero",
        "both nonzero",
    ]
    global_means = [_mean([_f(r[k]) for r in summary_rows]) for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(labels, global_means, color=["C0", "C1", "C3", "C2"])
    axes[0].set_ylabel("rate（capture full-tensor mean）")
    axes[0].set_title("全局零点转换")
    axes[0].tick_params(axis="x", rotation=20)

    x = np.arange(len(LINEAR_PROJECTIONS))
    width = 0.2
    for i, (k, lab) in enumerate(zip(keys, labels)):
        ys = []
        for pj in LINEAR_PROJECTIONS:
            vals = [_f(r[k]) for r in summary_rows if r["projection"] == pj]
            if not vals:
                raise RuntimeError(f"missing zero transition for {pj}")
            ys.append(_mean(vals))
        axes[1].bar(x + (i - 1.5) * width, ys, width=width, label=lab)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LINEAR_PROJECTIONS, rotation=30, ha="right")
    axes[1].set_title("按 projection")
    axes[1].legend(fontsize=7)
    fig.suptitle("R7 零点转换（full-tensor capture stats）")
    fig.tight_layout()
    return _save(fig, out / "fig_r7_zero_transition.png")


def fig_r8_payload_transition(points: dict[str, Any], out: Path) -> list[str]:
    nv = np.abs(_to_np(points["nv_payload"]))
    hf = np.abs(_to_np(points["hf_payload"]))
    i_nv = _nearest_level_indices(nv, _NV_PAYLOAD_LEVELS)
    i_hf = _nearest_level_indices(hf, _HF_PAYLOAD_LEVELS)
    mat = np.zeros((len(_NV_PAYLOAD_LEVELS), len(_HF_PAYLOAD_LEVELS)), dtype=np.float64)
    for a, b in zip(i_nv, i_hf):
        mat[a, b] += 1.0
    if mat.sum() <= 0:
        raise RuntimeError("empty payload transition")

    paths: list[str] = []
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(_HF_PAYLOAD_LEVELS)))
    ax.set_xticklabels([str(x) for x in _HF_PAYLOAD_LEVELS])
    ax.set_yticks(range(len(_NV_PAYLOAD_LEVELS)))
    ax.set_yticklabels([str(x) for x in _NV_PAYLOAD_LEVELS])
    ax.set_xlabel("HiF4 S1P2 |payload|")
    ax.set_ylabel("NVFP4 E2M1 |payload|")
    ax.set_title("R8 payload transition count（sampled elements；非最终数值映射）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    paths.append(_save(fig, out / "fig_r8_payload_transition_heatmap_count.png"))

    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    mat_n = mat / row_sum
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(mat_n, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(len(_HF_PAYLOAD_LEVELS)))
    ax.set_xticklabels([str(x) for x in _HF_PAYLOAD_LEVELS])
    ax.set_yticks(range(len(_NV_PAYLOAD_LEVELS)))
    ax.set_yticklabels([str(x) for x in _NV_PAYLOAD_LEVELS])
    ax.set_xlabel("HiF4 S1P2 |payload|")
    ax.set_ylabel("NVFP4 E2M1 |payload|")
    ax.set_title("R8 payload transition row-normalized")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    paths.append(_save(fig, out / "fig_r8_payload_transition_heatmap_row_normalized.png"))
    return paths


def _heatmap_layer_proj(
    summary_rows: list[dict[str, str]],
    value_key: str,
    out_path: Path,
    title: str,
    cbar_label: str,
) -> str:
    layers = sorted({int(r["layer_idx"]) for r in summary_rows})
    projs = list(LINEAR_PROJECTIONS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    mats: list[np.ndarray] = []
    for phase in ("prefill", "decode"):
        mat = np.full((len(layers), len(projs)), np.nan, dtype=np.float64)
        for i, li in enumerate(layers):
            for j, pj in enumerate(projs):
                vals = [
                    _f(r[value_key])
                    for r in summary_rows
                    if int(r["layer_idx"]) == li
                    and r["projection"] == pj
                    and r["phase"] == phase
                ]
                if not vals:
                    raise RuntimeError(f"missing {value_key} for {phase}/L{li}/{pj}")
                mat[i, j] = _mean(vals)
        mats.append(mat)
    vmin = float(min(np.nanmin(m) for m in mats))
    vmax = float(max(np.nanmax(m) for m in mats))
    for ax, mat, phase in zip(axes, mats, ("prefill", "decode")):
        im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="inferno")
        ax.set_xticks(range(len(projs)))
        ax.set_xticklabels(projs, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(x) for x in layers])
        ax.set_title(phase)
        ax.set_ylabel("layer_idx")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, out_path)


def fig_r9_nmse_heatmaps(summary_rows: list[dict[str, str]], out: Path) -> list[str]:
    p1 = _heatmap_layer_proj(
        summary_rows,
        "residual_nmse_hif4_vs_nvfp4",
        out / "fig_r9_residual_nmse_layer_projection_heatmap.png",
        "R9 mean NMSE(A_H vs A_N)（full-tensor）",
        "mean NMSE",
    )
    p2 = _heatmap_layer_proj(
        summary_rows,
        "residual_rms",
        out / "fig_r9_residual_rms_layer_projection_heatmap.png",
        "R9 mean RMS(ΔA)（full-tensor）",
        "mean RMS(ΔA)",
    )
    return [p1, p2]


def fig_r10_boxplot(summary_rows: list[dict[str, str]], out: Path) -> str:
    metrics = (
        ("residual_rms", "RMS(ΔA)"),
        ("residual_nmse_hif4_vs_nvfp4", "NMSE"),
        ("residual_q99_abs", "q99_abs"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (key, title) in zip(axes, metrics):
        data = []
        for pj in LINEAR_PROJECTIONS:
            vals = [_f(r[key]) for r in summary_rows if r["projection"] == pj]
            if not vals:
                raise RuntimeError(f"missing {key} for {pj}")
            data.append(vals)
        ax.boxplot(data, tick_labels=list(LINEAR_PROJECTIONS))
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(title)
    fig.suptitle("R10 per-capture residual 分布（full-tensor stats，按 projection）")
    fig.tight_layout()
    return _save(fig, out / "fig_r10_residual_by_projection_boxplot.png")


# ---------------------------------------------------------------------------
# Figures R11–R13
# ---------------------------------------------------------------------------


def _select_worst_discovery_capture(
    summary_rows: list[dict[str, str]],
    layer_idx: int,
) -> dict[str, str]:
    cands = [
        r
        for r in summary_rows
        if r.get("split") == "discovery"
        and r.get("phase") == "prefill"
        and int(r["layer_idx"]) == int(layer_idx)
    ]
    if not cands:
        # allow undivided run without split tag only if all rows discovery-like
        cands = [
            r
            for r in summary_rows
            if r.get("phase") == "prefill" and int(r["layer_idx"]) == int(layer_idx)
        ]
    if not cands:
        raise RuntimeError(f"no discovery/prefill capture for layer {layer_idx}")
    return max(cands, key=lambda r: _f(r["residual_nmse_hif4_vs_nvfp4"]))


def _find_token_group_map(
    entries: list[dict[str, Any]],
    row: dict[str, str],
) -> torch.Tensor:
    want = (
        str(row["sample_id"]),
        int(row["layer_idx"]),
        str(row["projection"]),
        str(row["phase"]),
        int(row["decode_step"]),
    )
    for e in entries:
        key = (
            str(e["sample_id"]),
            int(e["layer_idx"]),
            str(e["projection"]),
            str(e["phase"]),
            int(e["decode_step"]),
        )
        if key == want:
            m = e["map"]
            if not torch.is_tensor(m):
                raise TypeError("token-group map must be tensor")
            return m.to(torch.float32)
    raise KeyError(f"token-group map not found for {want}")


def fig_r11_token_group(
    summary_rows: list[dict[str, str]],
    map_entries: list[dict[str, Any]],
    out: Path,
    layers: tuple[int, ...] = _R11_LAYERS,
) -> list[str]:
    paths: list[str] = []
    for li in layers:
        row = _select_worst_discovery_capture(summary_rows, li)
        m = _find_token_group_map(map_entries, row)
        arr = m.detach().cpu().numpy()
        if arr.ndim != 2:
            raise ValueError(f"map must be 2D, got {arr.shape}")
        t, g = arr.shape
        tt, gg = np.meshgrid(np.arange(t), np.arange(g), indexing="ij")

        fig = plt.figure(figsize=(8, 5.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(tt, gg, arr, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("token")
        ax.set_ylabel("64-group")
        ax.set_zlabel("RMS(ΔA)")
        ax.set_title(
            f"R11 L{li} 3D surface\n"
            f"sample={row['sample_id']} proj={row['projection']} "
            f"T={t} G={g} NMSE={_f(row['residual_nmse_hif4_vs_nvfp4']):.4g}"
        )
        fig.tight_layout()
        paths.append(
            _save(fig, out / f"fig_r11_3d_token_group_residual_surface_layer{li}.png")
        )

        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(arr.T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("token_position")
        ax.set_ylabel("64-group index")
        ax.set_title(
            f"R11 L{li} 2D heatmap | {row['sample_id']} / {row['projection']} / "
            f"NMSE={_f(row['residual_nmse_hif4_vs_nvfp4']):.4g}"
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMS(ΔA)")
        fig.tight_layout()
        paths.append(
            _save(fig, out / f"fig_r11_token_group_residual_heatmap_layer{li}.png")
        )
    return paths


def fig_r12_group_mechanism(group_rows: list[dict[str, str]], out: Path) -> tuple[list[str], dict[str, float]]:
    amax = np.array([_f(r["amax64_x_mean"]) for r in group_rows], dtype=np.float64)
    disp = np.array([_f(r["mean_sub16_dispersion"]) for r in group_rows], dtype=np.float64)
    rms = np.array([_f(r["rms_delta"]) for r in group_rows], dtype=np.float64)
    mask = np.isfinite(amax) & np.isfinite(disp) & np.isfinite(rms)
    if int(mask.sum()) < 2:
        raise RuntimeError("insufficient finite group rows for R12")
    amax = amax[mask]
    disp = disp[mask]
    rms = rms[mask]
    x_all = np.log2(amax + _TINY)
    y_all = disp
    z_all = np.log10(rms + _TINY)
    spearman_amax = _spearman(x_all, z_all)
    spearman_disp = _spearman(y_all, z_all)

    n = x_all.size
    if n > _MAX_GROUP_SCATTER:
        rng = np.random.default_rng(_GROUP_SCATTER_SEED)
        idx = rng.choice(n, size=_MAX_GROUP_SCATTER, replace=False)
        x, y, z = x_all[idx], y_all[idx], z_all[idx]
    else:
        x, y, z = x_all, y_all, z_all

    paths: list[str] = []
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=2, alpha=0.35, c=z, cmap="plasma")
    ax.set_xlabel("log2(amax64_x_mean)")
    ax.set_ylabel("mean_sub16_dispersion")
    ax.set_zlabel("log10(rms_delta)")
    ax.set_title(
        f"R12 group 机制 3D（plot n={x.size:,} / all={n:,}）\n"
        f"Spearman(amax,rms)={spearman_amax:.4f}; "
        f"Spearman(disp,rms)={spearman_disp:.4f}"
    )
    fig.tight_layout()
    paths.append(_save(fig, out / "fig_r12_3d_group_mechanism_scatter.png"))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(x_all, z_all, s=3, alpha=0.25)
    ax.set_xlabel("log2(amax64_x_mean)")
    ax.set_ylabel("log10(rms_delta)")
    ax.set_title(f"R12 2D amax vs rms；Spearman={spearman_amax:.4f}（全量 finite rows）")
    fig.tight_layout()
    paths.append(_save(fig, out / "fig_r12_amax64_vs_rms_delta_2d.png"))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(y_all, z_all, s=3, alpha=0.25)
    ax.set_xlabel("mean_sub16_dispersion")
    ax.set_ylabel("log10(rms_delta)")
    ax.set_title(f"R12 2D dispersion vs rms；Spearman={spearman_disp:.4f}")
    fig.tight_layout()
    paths.append(_save(fig, out / "fig_r12_dispersion_vs_rms_delta_2d.png"))

    stats = {
        "spearman_log2_amax64_vs_log10_rms_delta": spearman_amax,
        "spearman_mean_sub16_dispersion_vs_log10_rms_delta": spearman_disp,
        "num_group_rows_finite": float(n),
        "num_group_rows_plotted_3d": float(x.size),
        "mean_sub16_dispersion": float(y_all.mean()),
        "q90_sub16_dispersion": float(np.quantile(y_all, 0.9)),
    }
    return paths, stats


def fig_r13_landscape(summary_rows: list[dict[str, str]], out: Path) -> str:
    layers = sorted({int(r["layer_idx"]) for r in summary_rows})
    projs = list(LINEAR_PROJECTIONS)
    fig = plt.figure(figsize=(12, 5))
    for i, phase in enumerate(("prefill", "decode")):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        xs, ys, zs = [], [], []
        for li in layers:
            for j, pj in enumerate(projs):
                vals = [
                    _f(r["residual_nmse_hif4_vs_nvfp4"])
                    for r in summary_rows
                    if int(r["layer_idx"]) == li
                    and r["projection"] == pj
                    and r["phase"] == phase
                ]
                if not vals:
                    raise RuntimeError(f"missing NMSE for R13 {phase}/L{li}/{pj}")
                xs.append(float(li))
                ys.append(float(j))
                zs.append(_mean(vals))
        ax.scatter(xs, ys, zs, c=zs, cmap="inferno", s=40)
        ax.set_xlabel("layer_idx")
        ax.set_ylabel("projection index")
        ax.set_zlabel("mean NMSE")
        ax.set_yticks(range(len(projs)))
        ax.set_yticklabels(projs, fontsize=7)
        ax.set_title(phase)
    fig.suptitle("R13 layer×projection residual landscape（capture-summary 聚合）")
    fig.tight_layout()
    return _save(fig, out / "fig_r13_3d_layer_projection_residual_landscape.png")


# ---------------------------------------------------------------------------
# Stats + markdown
# ---------------------------------------------------------------------------


def _aggregate_stats(
    points: dict[str, Any],
    summary_rows: list[dict[str, str]],
    group_rows: list[dict[str, str]],
    group_stats: dict[str, float],
    viz_summary: dict[str, Any],
) -> dict[str, Any]:
    an = _to_np(points["a_nvfp4"])
    d = _to_np(points["delta_a"])
    ah = _to_np(points["a_hif4"])
    abs_an = np.abs(an)
    abs_d = np.abs(d)

    # projection RMS from full-tensor
    proj_rms: dict[str, float] = {}
    for pj in LINEAR_PROJECTIONS:
        vals = [_f(r["an_rms"]) for r in summary_rows if r["projection"] == pj]
        proj_rms[pj] = _mean(vals)

    phase_rms = {
        ph: _mean([_f(r["an_rms"]) for r in summary_rows if r["phase"] == ph])
        for ph in ("prefill", "decode")
    }

    energy_med = {
        "top_0p1pct": _median([_f(r["energy_top_0p1pct_energy_share"]) for r in summary_rows]),
        "top_1pct": _median([_f(r["energy_top_1pct_energy_share"]) for r in summary_rows]),
        "top_5pct": _median([_f(r["energy_top_5pct_energy_share"]) for r in summary_rows]),
        "top_10pct": _median([_f(r["energy_top_10pct_energy_share"]) for r in summary_rows]),
    }
    energy_q90 = {
        "top_0p1pct": _pct([_f(r["energy_top_0p1pct_energy_share"]) for r in summary_rows], 0.9),
        "top_1pct": _pct([_f(r["energy_top_1pct_energy_share"]) for r in summary_rows], 0.9),
        "top_5pct": _pct([_f(r["energy_top_5pct_energy_share"]) for r in summary_rows], 0.9),
        "top_10pct": _pct([_f(r["energy_top_10pct_energy_share"]) for r in summary_rows], 0.9),
    }
    sampled_energy = residual_energy_concentration(points["delta_a"])

    # hottest cells
    hot: list[tuple[float, str, int, str]] = []
    for r in summary_rows:
        hot.append(
            (
                _f(r["residual_nmse_hif4_vs_nvfp4"]),
                r["phase"],
                int(r["layer_idx"]),
                r["projection"],
            )
        )
    hot.sort(reverse=True)
    top_hot = hot[:10]

    both_nz = (an != 0) & (ah != 0)
    if int(both_nz.sum()) == 0:
        sign_flip = 0.0
    else:
        sign_flip = float(((an[both_nz] * ah[both_nz]) < 0).mean())

    return {
        "num_points": int(an.size),
        "num_captures": len(summary_rows),
        "num_group_rows": len(group_rows),
        "a_n": {
            "mean": float(an.mean()),
            "std": float(an.std()),
            "rms": float(np.sqrt((an * an).mean())),
            "zero_rate": float((an == 0).mean()),
            "q90_abs": float(np.quantile(abs_an, 0.90)),
            "q99_abs": float(np.quantile(abs_an, 0.99)),
            "q999_abs": float(np.quantile(abs_an, 0.999)),
            "projection_rms_mean": proj_rms,
            "projection_rms_max": max(proj_rms, key=proj_rms.get),  # type: ignore[arg-type]
            "projection_rms_min": min(proj_rms, key=proj_rms.get),  # type: ignore[arg-type]
            "phase_rms_mean": phase_rms,
        },
        "delta": {
            "mean_bias": float(d.mean()),
            "median": float(np.median(d)),
            "rms": float(np.sqrt((d * d).mean())),
            "nmse_mean_capture": _mean(
                [_f(r["residual_nmse_hif4_vs_nvfp4"]) for r in summary_rows]
            ),
            "q90_abs": float(np.quantile(abs_d, 0.90)),
            "q99_abs": float(np.quantile(abs_d, 0.99)),
            "q999_abs": float(np.quantile(abs_d, 0.999)),
            "sign_flip_rate_sampled": sign_flip,
            "nv_nonzero_hf_zero_mean": _mean(
                [_f(r["zero_transition_nv_nonzero_hf_zero"]) for r in summary_rows]
            ),
            "nv_zero_hf_nonzero_mean": _mean(
                [_f(r["zero_transition_nv_zero_hf_nonzero"]) for r in summary_rows]
            ),
            "hf_minus_nv_zero_rate_mean": _mean(
                [_f(r["zero_transition_hf_minus_nv_zero_rate"]) for r in summary_rows]
            ),
            "sampled_energy_shares": {
                "top_0p1pct": float(sampled_energy["top_0p1pct_energy_share"]),
                "top_1pct": float(sampled_energy["top_1pct_energy_share"]),
                "top_5pct": float(sampled_energy["top_5pct_energy_share"]),
                "top_10pct": float(sampled_energy["top_10pct_energy_share"]),
            },
            "capture_energy_share_median": energy_med,
            "capture_energy_share_q90": energy_q90,
        },
        "group": group_stats,
        "hottest_captures": [
            {"nmse": a, "phase": b, "layer_idx": c, "projection": d_}
            for a, b, c, d_ in top_hot
        ],
        "stability": viz_summary.get("stability"),
        "viz_summary_meta": {
            "run_id": viz_summary.get("run_id"),
            "analysis_seed": viz_summary.get("analysis_seed"),
            "max_prefill_stat_tokens": viz_summary.get("max_prefill_stat_tokens"),
            "theoretical_grids": viz_summary.get("theoretical_grids"),
        },
    }


def _pick_algorithm_direction(stats: dict[str, Any]) -> str:
    e = stats["delta"]["sampled_energy_shares"]
    z = stats["delta"]["nv_nonzero_hf_zero_mean"]
    sp_disp = stats["group"]["spearman_mean_sub16_dispersion_vs_log10_rms_delta"]
    sp_amax = stats["group"]["spearman_log2_amax64_vs_log10_rms_delta"]
    top1 = e["top_1pct"]
    if top1 >= 0.5:
        return (
            "A. residual energy 高度集中于少量元素（sampled top1% "
            f"={top1:.3f}）：优先 outlier-aware / selective scale / protected group。"
        )
    if z >= 0.05:
        return (
            "B. NV nonzero→HiF4 zero 偏高 "
            f"（mean={z:.4f}）：优先解决 0 附近网格/scale 对齐。"
        )
    if sp_amax >= 0.5:
        return (
            "C. residual 与 |激活/amax| 相关较强 "
            f"（Spearman amax={sp_amax:.3f}）：优先优化动态范围与高幅值 code 分配。"
        )
    if sp_disp >= 0.4:
        return (
            "D. group residual 与 sub16 dispersion 相关 "
            f"（Spearman={sp_disp:.3f}）：优先调整 group64 层级 scale 或更细 group。"
        )
    return (
        "E. 误差在幅值/group 上较广泛分布"
        f"（top1% energy={top1:.3f}, zero-collapse={z:.4f}）："
        "更像 payload/grid 全局表示能力问题，不应只做局部 outlier 修补。"
    )


def _render_markdown(
    run_dir: Path,
    stats: dict[str, Any],
    figures: list[str],
    theory_note: str,
) -> str:
    an = stats["a_n"]
    dlt = stats["delta"]
    grp = stats["group"]
    direction = _pick_algorithm_direction(stats)
    hot_lines = "\n".join(
        f"- NMSE={h['nmse']:.6g} | {h['phase']} L{h['layer_idx']} {h['projection']}"
        for h in stats["hottest_captures"][:5]
    )
    stab = stats.get("stability")
    if stab is None:
        stab_txt = "本 run 未同时包含 discovery/validation，或未计算 JS 稳定性。"
    else:
        stab_txt = (
            f"stable={stab.get('stable')}；max_global_js={stab.get('max_global_js')}；"
            f"projections_exceeding_0p05={stab.get('projections_exceeding_0p05')}"
        )

    fig_list = "\n".join(f"- `{Path(p).name}`" for p in figures)

    return f"""# 03 W4A4 激活分布与 NVFP4→HiF4 残差可视化

## 1. 实验目的

在 **NVFP4 W4A4 semantic inference** 下，刻画真正进入 Linear GEMM 的激活 `A_N` 分布，并量化同一 `X_in` 上 counterfactual `A_H=Q_HiF4(X_in)` 相对 `A_N` 的转换残差 `ΔA=A_H−A_N` 的结构，指导后续激活优化优先级。

## 2. W4A4 semantic inference 定义

本实验是 NVFP4 W4A4 **semantic inference**：weight 使用 NVFP4-QAT fake-dequant BF16 source value（不重新做 packed W4 fake quant），activation 在每个目标 Linear 前执行 NVFP4 A4 QDQ；`A_N` 回传进入 GEMM，`A_H` 仅旁路统计。不是 packed W4A4 kernel 性能实验。

主残差固定：`ΔA = A_H − A_N`（禁止改成相对 `X_in` 或绝对值差）。

## 3. 数据来源与采样方式

- run_dir: `{run_dir}`
- 分布形状图 = **stratified deterministic sample estimate**（`activation_viz_points.pt`）
- 每个 capture 的数值指标 = **full-tensor exact statistic**（`activation_capture_summary.csv` / `activation_group_residual.csv`）
- 抽样策略：每个代表层 capture 最多 `max_point_samples_per_capture=1024` 个元素级点；prefill 统计 token 上限 128（均匀子采样）；seed=`{stats['viz_summary_meta'].get('analysis_seed')}`
- 实际绘图点数：`{stats['num_points']:,}`；capture 数：`{stats['num_captures']}`；group 行数：`{stats['num_group_rows']}`

## 4. 理论 NVFP4 / HiF4 完整内部网格

- NVFP4：**E4M3FN × signed E2M1**（排除最外层 FP32 per-tensor scale）。标题与正文一律写 **E4M3FN**，不是 E8M0。
- HiF4：S0 × 2^(e8+e4) × signed S1P2
- {theory_note}

## 5. 真实 NVFP4 W4A4 激活分布（sampled points）

| 指标 | 值 |
|---|---|
| mean(A_N) | {an['mean']:.6g} |
| std(A_N) | {an['std']:.6g} |
| RMS(A_N) | {an['rms']:.6g} |
| zero rate | {an['zero_rate']:.6g} |
| q90/q99/q99.9 \\|A_N\\| | {an['q90_abs']:.6g} / {an['q99_abs']:.6g} / {an['q999_abs']:.6g} |
| projection RMS max/min | {an['projection_rms_max']} / {an['projection_rms_min']} |
| prefill vs decode mean RMS(A_N) | {an['phase_rms_mean']['prefill']:.6g} vs {an['phase_rms_mean']['decode']:.6g} |

## 6. NVFP4→HiF4 元素级残差

| 指标 | 值 |
|---|---|
| mean bias(ΔA) | {dlt['mean_bias']:.6g} |
| median(ΔA) | {dlt['median']:.6g} |
| RMS(ΔA) sampled | {dlt['rms']:.6g} |
| mean capture NMSE | {dlt['nmse_mean_capture']:.6g} |
| q90/q99/q99.9 \\|ΔA\\| | {dlt['q90_abs']:.6g} / {dlt['q99_abs']:.6g} / {dlt['q999_abs']:.6g} |
| sign flip (sampled, both nonzero) | {dlt['sign_flip_rate_sampled']:.6g} |
| NV≠0→HF=0 mean | {dlt['nv_nonzero_hf_zero_mean']:.6g} |
| NV=0→HF≠0 mean | {dlt['nv_zero_hf_nonzero_mean']:.6g} |
| HF−NV zero rate mean | {dlt['hf_minus_nv_zero_rate_mean']:.6g} |
| sampled energy top0.1/1/5/10% | {dlt['sampled_energy_shares']['top_0p1pct']:.4f} / {dlt['sampled_energy_shares']['top_1pct']:.4f} / {dlt['sampled_energy_shares']['top_5pct']:.4f} / {dlt['sampled_energy_shares']['top_10pct']:.4f} |
| capture median energy top0.1/1/5/10% | {dlt['capture_energy_share_median']['top_0p1pct']:.4f} / {dlt['capture_energy_share_median']['top_1pct']:.4f} / {dlt['capture_energy_share_median']['top_5pct']:.4f} / {dlt['capture_energy_share_median']['top_10pct']:.4f} |

## 7. 零点迁移与 payload 迁移

见 R7/R8。payload 热图只描述 codebook 使用迁移，因 scale 系统不同，不能解释为最终数值一一映射。

## 8. layer / projection / phase 热点

最高 NMSE capture（top5）：

{hot_lines}

## 9. token×group 空间结构

见 R11（discovery/prefill 各代表层 NMSE 最大 capture 的 token×64-group RMS(ΔA) 表面与 2D heatmap）。

## 10. group 动态范围/离散度与残差

- Spearman(log2(amax64_x_mean), log10(rms_delta)) = **{grp['spearman_log2_amax64_vs_log10_rms_delta']:.6f}**
- Spearman(mean_sub16_dispersion, log10(rms_delta)) = **{grp['spearman_mean_sub16_dispersion_vs_log10_rms_delta']:.6f}**
- mean / q90 sub16 dispersion = {grp['mean_sub16_dispersion']:.6g} / {grp['q90_sub16_dispersion']:.6g}

## 11. discovery vs validation 稳定性

{stab_txt}

## 12. 对激活优化的直接启示

{direction}

## 13. 限制与下一步

- 元素级图依赖确定性分层抽样，不能代替 full-tensor 指标。
- D4 三联图比较的是「内部可表示点密度」与「真实概率质量」，不是直接覆盖率。
- 下一步按第 12 节方向做针对性消融，并在 validation 上复验热点是否反转。

---

# §0 核心问题逐条回答

### Q1. 真正进入 GEMM 的 `A_N` 是什么分布？是否重尾、强零点、正负不对称？

`A_N` 抽样点：mean={an['mean']:.4g}，std={an['std']:.4g}，RMS={an['rms']:.4g}，zero_rate={an['zero_rate']:.4g}；\\|A_N\\| 的 q99/q99.9={an['q99_abs']:.4g}/{an['q999_abs']:.4g}。见 D1–D3：若 q99.9 ≫ RMS 则重尾明显；zero_rate 反映零点集中；mean 相对 0 的偏离反映正负不对称。本报告以抽样密度估计形状，full-tensor 矩见 capture CSV。

### Q2. 各 projection / 早中晚层 / prefill·decode 是否不同？

projection 平均 RMS(A_N) 最大={an['projection_rms_max']}、最小={an['projection_rms_min']}；prefill vs decode mean RMS={an['phase_rms_mean']['prefill']:.4g} vs {an['phase_rms_mean']['decode']:.4g}。见 D5（projection×phase log2\\|A_N\\|）、D6（layer×projection log10 RMS）。

### Q3. 理论可表示点与真实 `A_N` 概率质量分别集中在哪些数量级？

理论网格为 **E4M3FN×E2M1**（NV）与 HiF4 层级网格（均去掉最外层 FP32 scale）；真实 `A_N` 为 dequantized 概率密度。见 D4：A/B 纵轴为 unique count，C 为 density；x 不归一化。三者 y 含义不同，不可直接当覆盖率。

### Q4. `ΔA` 是否近似零均值？是否有系统偏置？

sampled mean bias={dlt['mean_bias']:.6g}，median={dlt['median']:.6g}，RMS={dlt['rms']:.6g}。若 \\|mean\\|/RMS 不可忽略，则存在系统偏置。见 R1。

### Q5. 大量小误差还是少量极大 outlier 主导？

sampled top0.1/1/5/10% 能量份额={dlt['sampled_energy_shares']['top_0p1pct']:.4f}/{dlt['sampled_energy_shares']['top_1pct']:.4f}/{dlt['sampled_energy_shares']['top_5pct']:.4f}/{dlt['sampled_energy_shares']['top_10pct']:.4f}；per-capture exact median 同序={dlt['capture_energy_share_median']['top_0p1pct']:.4f}/{dlt['capture_energy_share_median']['top_1pct']:.4f}/{dlt['capture_energy_share_median']['top_5pct']:.4f}/{dlt['capture_energy_share_median']['top_10pct']:.4f}。见 R5。

### Q6. 大残差主要在小/中/大激活区间？

见 R4（A_N vs ΔA）与 R6（\\|A_N\\| 分位 bin 的 RMS/mean\\|ΔA\\|）。不得用接近 0 时爆炸的相对误差替代。

### Q7. HiF4 是否更容易把非零压成 0？

capture 均值：NV≠0→HF=0 = {dlt['nv_nonzero_hf_zero_mean']:.6g}；反方向 NV=0→HF≠0 = {dlt['nv_zero_hf_nonzero_mean']:.6g}；HF−NV zero_rate = {dlt['hf_minus_nv_zero_rate_mean']:.6g}。见 R7。

### Q8. `ΔA` 在 token×64-group 空间均匀还是集中？

见 R11 各代表层最差 NMSE capture 的 3D/2D 图：若表面出现少数 token/group 尖峰则为空间集中，否则较弥散。

### Q9. amax64、sub16 离散度与 group residual RMS 的关系？AX2 能否视觉复现？

Spearman(amax, rms_delta)={grp['spearman_log2_amax64_vs_log10_rms_delta']:.4f}；Spearman(dispersion, rms_delta)={grp['spearman_mean_sub16_dispersion_vs_log10_rms_delta']:.4f}。见 R12。若 dispersion 相关强，则与 AX2「64-group 过大/子块不均」方向一致。

### Q10. 哪些 layer/projection/phase 残差最高，下一步优先打哪里？

热点见第 8 节与 R9/R10/R13。算法方向：{direction}

---

## 生成的 figures

{fig_list}
"""


def _load_map_entries(run_dir: Path, discovery_run_dir: Path | None) -> list[dict[str, Any]]:
    candidates = [run_dir / "activation_token_group_maps.pt"]
    if discovery_run_dir is not None:
        candidates.insert(0, discovery_run_dir / "activation_token_group_maps.pt")
    for p in candidates:
        if p.is_file():
            blob = torch.load(p, map_location="cpu", weights_only=False)
            entries = blob.get("entries")
            if not isinstance(entries, list) or not entries:
                raise RuntimeError(f"empty token-group map entries in {p}")
            return entries
    raise FileNotFoundError(
        "activation_token_group_maps.pt missing "
        f"(looked in {[str(c) for c in candidates]})"
    )


def _discovery_summary_rows(
    summary_rows: list[dict[str, str]],
    discovery_run_dir: Path | None,
) -> list[dict[str, str]]:
    if discovery_run_dir is not None:
        path = discovery_run_dir / "activation_capture_summary.csv"
        return _read_csv_rows(path)
    disc = [r for r in summary_rows if r.get("split") == "discovery"]
    if disc:
        return disc
    return summary_rows


def build_activation_viz_report(
    run_dir: Path,
    *,
    discovery_run_dir: Path | None = None,
) -> dict[str, Any]:
    """Build D1–D6 / R1–R13 figures and Chinese phase report from a merged viz run."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if discovery_run_dir is not None:
        discovery_run_dir = Path(discovery_run_dir)
        if not discovery_run_dir.is_dir():
            raise FileNotFoundError(discovery_run_dir)

    summary_path = _require_file(run_dir / "activation_capture_summary.csv")
    group_path = _require_file(run_dir / "activation_group_residual.csv")
    points_path = _require_file(run_dir / "activation_viz_points.pt")
    viz_summary_path = _require_file(run_dir / "activation_viz_summary.json")

    summary_rows = _read_csv_rows(summary_path)
    group_rows = _read_csv_rows(group_path)
    points = _load_points(points_path)
    viz_summary = json.loads(viz_summary_path.read_text(encoding="utf-8"))
    nv_grid, hf_grid, theory = _load_theory_grids(run_dir)

    stability: dict[str, Any] | None = None
    if discovery_run_dir is not None:
        from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_viz_pipeline import (
            compute_split_stability,
        )

        disc_pts_path = discovery_run_dir / "activation_viz_points.pt"
        if not disc_pts_path.is_file():
            raise FileNotFoundError(disc_pts_path)
        disc_points = _load_points(disc_pts_path)
        stability = compute_split_stability(disc_points, points)
        viz_summary["stability"] = stability
        atomic_write_json(viz_summary_path, viz_summary)

    fig_dir = ensure_dir(run_dir / "figures")
    figures: list[str] = []

    figures.append(fig_d1_activation_hist_full(points, fig_dir))
    figures.append(fig_d2_activation_hist_central(points, fig_dir))
    figures.append(fig_d3_activation_log2_abs(points, fig_dir))
    figures.append(fig_d4_theory_vs_real_triptych(points, nv_grid, hf_grid, fig_dir))
    figures.append(fig_d5_by_projection_phase(points, fig_dir))
    figures.append(fig_d6_rms_heatmap(summary_rows, fig_dir))

    figures.append(fig_r1_delta_hist(points, fig_dir))
    figures.append(fig_r2_delta_log10_abs(points, fig_dir))
    figures.extend(fig_r3_an_vs_ah(points, fig_dir))
    figures.append(fig_r4_an_vs_delta(points, fig_dir))
    figures.append(fig_r5_energy(points, summary_rows, fig_dir))
    figures.append(fig_r6_quantile_curve(points, fig_dir))
    figures.append(fig_r7_zero_transition(summary_rows, fig_dir))
    figures.extend(fig_r8_payload_transition(points, fig_dir))
    figures.extend(fig_r9_nmse_heatmaps(summary_rows, fig_dir))
    figures.append(fig_r10_boxplot(summary_rows, fig_dir))

    disc_rows = _discovery_summary_rows(summary_rows, discovery_run_dir)
    map_entries = _load_map_entries(run_dir, discovery_run_dir)
    # R11 layers: prefer canonical 4/18/34 if present, else all layers in data
    layers_in_data = sorted({int(r["layer_idx"]) for r in disc_rows if r.get("phase") == "prefill"})
    r11_layers = tuple(li for li in _R11_LAYERS if li in layers_in_data)
    if len(r11_layers) != len(_R11_LAYERS):
        raise RuntimeError(
            f"R11 requires layers {_R11_LAYERS} in discovery/prefill; found {layers_in_data}"
        )
    figures.extend(fig_r11_token_group(disc_rows, map_entries, fig_dir, layers=r11_layers))

    r12_paths, group_stats = fig_r12_group_mechanism(group_rows, fig_dir)
    figures.extend(r12_paths)
    figures.append(fig_r13_landscape(summary_rows, fig_dir))

    stats = _aggregate_stats(points, summary_rows, group_rows, group_stats, viz_summary)
    if stability is not None:
        stats["stability"] = stability
    theory_note = (
        f"nv_unique={int(nv_grid.numel())}, hf_unique={int(hf_grid.numel())}; "
        f"source={theory.get('note', viz_summary.get('theoretical_grids', {}))}"
    )
    md = _render_markdown(run_dir, stats, figures, theory_note)

    report_cn = run_dir / "activation_distribution_residual_report_cn.md"
    write_text(report_cn, md)
    ensure_dir(_PHASE_REPORT_PATH.parent)
    write_text(_PHASE_REPORT_PATH, md)

    out_summary = {
        "run_dir": str(run_dir),
        "discovery_run_dir": str(discovery_run_dir) if discovery_run_dir else None,
        "figures": figures,
        "figure_names": [Path(p).name for p in figures],
        "num_points": stats["num_points"],
        "stats": stats,
        "phase_report": str(_PHASE_REPORT_PATH),
        "report_cn": str(report_cn),
        "sampling": {
            "distribution_plots": "stratified_deterministic_sample_estimate",
            "capture_metrics": "full_tensor_exact_statistic",
            "max_point_samples_per_capture": 1024,
        },
        "stability": stability,
        "theory_grid": {
            "nvfp4": "E4M3FN x signed E2M1 (not E8M0)",
            "hif4": "S0 x 2^(e8+e4) x signed S1P2",
            "nv_unique": int(nv_grid.numel()),
            "hf_unique": int(hf_grid.numel()),
        },
        "algorithm_direction": _pick_algorithm_direction(stats),
    }
    atomic_write_json(run_dir / "activation_distribution_residual_summary.json", out_summary)
    return out_summary


__all__ = [
    "FIGURE_NAMES",
    "build_activation_viz_report",
    "fig_d1_activation_hist_full",
    "fig_d2_activation_hist_central",
    "fig_d3_activation_log2_abs",
    "fig_d4_theory_vs_real_triptych",
    "fig_d5_by_projection_phase",
    "fig_d6_rms_heatmap",
    "fig_r1_delta_hist",
    "fig_r2_delta_log10_abs",
    "fig_r3_an_vs_ah",
    "fig_r4_an_vs_delta",
    "fig_r5_energy",
    "fig_r6_quantile_curve",
    "fig_r7_zero_transition",
    "fig_r8_payload_transition",
    "fig_r9_nmse_heatmaps",
    "fig_r10_boxplot",
    "fig_r11_token_group",
    "fig_r12_group_mechanism",
    "fig_r13_landscape",
]
