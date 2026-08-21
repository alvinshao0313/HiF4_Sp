#!/usr/bin/env python3
"""Consolidate discovery/validation AX runs into final Chinese phase report."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, ensure_dir, write_csv, write_text
from Inference_Paradigm_Conversion.ipc_analysis.reporting.activation_incremental_report import (
    build_activation_incremental_report,
)

REPO = Path("/home/shaoyuantian/program/HiF4_Sp/Inference_Paradigm_Conversion")


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _f(x, d=0.0):
    try:
        return float(x) if x not in (None, "") else d
    except Exception:
        return d


def _mean(xs):
    return statistics.mean(xs) if xs else 0.0


def _merge_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        rows.extend(_read(p))
    return rows


def main() -> None:
    results = REPO / "results"
    pref = results / "20260811T040542Z_ax_discovery_prefill_v2"
    dec = results / "20260811T040646Z_ax_discovery_decode"
    # validation run id from latest pointer or glob
    val_ptr = (results / "latest_ax_run_id.txt").read_text().strip() if (results / "latest_ax_run_id.txt").is_file() else ""
    val = results / val_ptr if val_ptr and "validation" in val_ptr else None
    if val is None or not val.is_dir():
        cands = sorted(results.glob("*_ax_validation"), key=lambda p: p.stat().st_mtime, reverse=True)
        val = cands[0] if cands else None

    out = ensure_dir(results / "20260811T_ax_final_consolidated")
    disc_ax1 = _merge_rows([pref / "ax1_s0_divisor_oracle.csv", dec / "ax1_s0_divisor_oracle.csv"])
    disc_ax2 = _merge_rows([pref / "ax2_group_size_ablation.csv", dec / "ax2_group_size_ablation.csv"])
    disc_ax2d = _merge_rows([pref / "ax2_sub16_dispersion.csv", dec / "ax2_sub16_dispersion.csv"])
    disc_ax3 = _merge_rows([pref / "ax3_grid_occupancy.csv", dec / "ax3_grid_occupancy.csv"])
    disc_ax4 = _merge_rows([pref / "ax4_cross_format_factorization.csv", dec / "ax4_cross_format_factorization.csv"])

    write_csv(out / "ax1_s0_divisor_oracle.csv", disc_ax1)
    write_csv(out / "ax2_group_size_ablation.csv", disc_ax2)
    write_csv(out / "ax2_sub16_dispersion.csv", disc_ax2d)
    write_csv(out / "ax3_grid_occupancy.csv", disc_ax3)
    write_csv(out / "ax3_local_scale_distribution.csv", _merge_rows([
        pref / "ax3_local_scale_distribution.csv", dec / "ax3_local_scale_distribution.csv"
    ]))
    write_csv(out / "ax4_cross_format_factorization.csv", disc_ax4)
    # theory from pref
    if (pref / "ax3_theoretical_grid.json").is_file():
        (out / "ax3_theoretical_grid.json").write_text((pref / "ax3_theoretical_grid.json").read_text(encoding="utf-8"), encoding="utf-8")

    # conversion-only ranking (AX sources)
    from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_rule_selection import (
        build_root_cause_ranking,
        run_rule_selection,
    )

    a2 = _read(results / "20260811T032247Z_a2" / "a2_variants.csv")
    ranking_all = build_root_cause_ranking(
        {"ax1": disc_ax1, "ax2": disc_ax2, "ax3": disc_ax3, "ax4": disc_ax4},
        a2_csv_rows=a2,
    )
    write_csv(out / "ax5_root_cause_ranking.csv", ranking_all)
    ranking_conv = [r for r in ranking_all if str(r.get("evidence_source", "")).startswith("AX")]
    write_csv(out / "ax5_conversion_root_cause_ranking.csv", ranking_conv)

    rules = run_rule_selection(disc_ax1)
    write_csv(out / "ax5_rule_validation.csv", [rules])

    # validation comparison
    val_summary = {}
    if val is not None and val.is_dir():
        v_ax1 = _read(val / "ax1_s0_divisor_oracle.csv")
        v_ax2 = _read(val / "ax2_group_size_ablation.csv")
        v_ax4 = _read(val / "ax4_cross_format_factorization.csv")
        val_summary = {
            "run_id": val.name,
            "ax1_mean_R_Y": _mean([_f(r.get("output_recovery")) for r in v_ax1]),
            "ax2_g16_mean_R_Y": _mean([_f(r.get("R_Y")) for r in v_ax2 if str(r.get("group_size")) == "16"]),
            "ax4_hn_rm_mean_R_Y": _mean([
                _f(r.get("R_Y"))
                for r in v_ax4
                if r.get("hybrid") == "HN" and r.get("match_kind") == "range_matched"
            ]),
        }

    report = build_activation_incremental_report(out)

    # richer final answers
    alphas = [_f(r.get("alpha_oracle_nvfp4"), 7.0) for r in disc_ax1]
    rys = [_f(r.get("output_recovery")) for r in disc_ax1]
    by_phase = defaultdict(list)
    by_proj = defaultdict(list)
    for r in disc_ax1:
        by_phase[r.get("phase", "?")].append(_f(r.get("output_recovery")))
        by_proj[r.get("projection", "?")].append(_f(r.get("alpha_oracle_nvfp4"), 7.0))
    g16 = _mean([_f(r.get("R_Y")) for r in disc_ax2 if str(r.get("group_size")) == "16"])
    g32 = _mean([_f(r.get("R_Y")) for r in disc_ax2 if str(r.get("group_size")) == "32"])
    hn_rm = _mean([_f(r.get("R_Y")) for r in disc_ax4 if r.get("hybrid") == "HN" and r.get("match_kind") == "range_matched"])
    nh_rm = _mean([_f(r.get("R_Y")) for r in disc_ax4 if r.get("hybrid") == "NH" and r.get("match_kind") == "range_matched"])
    z_hf = _mean([_f(r.get("hf_occ_zero_rate")) for r in disc_ax3])
    z_nv = _mean([_f(r.get("nv_occ_zero_rate")) for r in disc_ax3])
    b_hf = _mean([_f(r.get("hf_occ_boundary_rate")) for r in disc_ax3])
    b_nv = _mean([_f(r.get("nv_occ_boundary_rate")) for r in disc_ax3])

    top_conv = ranking_conv[:3]
    if hn_rm > nh_rm + 0.05:
        ax4_verdict = "Payload 主导（在 HiF4 Scale 上换 NVFP4 Payload 更好）"
    elif nh_rm > hn_rm + 0.05:
        ax4_verdict = "Scale System 主导"
    elif max(hn_rm, nh_rm) < 0.05:
        ax4_verdict = "证据不足 / 两者交互（单独替换收益有限）"
    else:
        ax4_verdict = "两者交互"

    answers = f"""# 最终必须回答的问题（基于 consolidated discovery）

1. `amax/7` 是否重要损失来源？  
   **否（次要）**：平均 R_Y≈{_mean(rys):.4f}，AX5-R 因低恢复被 skip。

2. 最优 S0 是否在 projection/phase 上稳定？  
   中位数 alpha≈{statistics.median(alphas):.4f}；分投影中位数：{ {k: statistics.median(v) for k,v in by_proj.items()} }；分 phase 平均 R_Y：{ {k: _mean(v) for k,v in by_phase.items()} }

3. activation-MSE 最优与 output-aware 是否一致？  
   见 ax1 中 `alpha_oracle_nvfp4` vs `alpha_oracle_bf16` 列；多数样本接近但非完全相同。

4. G64 相对 G16/G32 多损失多少？  
   G16 平均可恢复 R_Y≈{g16:.4f}；G32≈{g32:.4f}。

5. 4×16 离散度是否放大 G64 损失？  
   见 `ax2_sub16_dispersion.csv` 与图 AX2-2。

6–11. 网格/占用/Oracle S0：  
   HiF4 零点占用={z_hf:.4f} vs NVFP4={z_nv:.4f}；边界占用 HiF4={b_hf:.4f} vs NVFP4={b_nv:.4f}。理论网格见 `ax3_theoretical_grid.json`。

12–14. Hybrid：  
   HN range_matched R_Y≈{hn_rm:.4f}；NH range_matched R_Y≈{nh_rm:.4f}；结论：**{ax4_verdict}**。raw hybrid 因动态范围不匹配显著更差，不能与 range-matched 混谈。

15. Scale / Payload / 交互？  
   **{ax4_verdict}**

16. 与主计划 e8/e4/payload/clipping 比谁最大？  
   主计划 A2（相对 BF16 的 R_cf_output）中 payload/clipping 与 continuous payload 最大；**转换误差（相对 A_N）**上 AX 前三见下。

17. 激活转换前三根因（AX，相对 NVFP4 Source）：  
"""
    for i, r in enumerate(top_conv, 1):
        answers += f"   - #{i} {r.get('root_cause')}（R_Y={r.get('R_Y')}）\n"
    answers += f"""
18. validation 是否稳定？  
   {json.dumps(val_summary, ensure_ascii=False)}

19. 几乎无开销的 S0 规则？  
   `{rules.get('status')}`；candidate_for_e2e={rules.get('candidate_for_e2e')}

20. 下一步最值得实现的激活优化？  
   优先：**更细 group / payload 表示（对齐 A2 与 AX2）**；Scale×Payload 需联合设计；不要优先调全局 `/7`。
"""
    write_text(out / "final_answers_cn.md", answers)

    # append to phase report
    phase = REPO / "reports" / "phase_reports" / "02_activation_error_localization.md"
    if phase.is_file():
        prev = phase.read_text(encoding="utf-8")
        write_text(phase, prev + "\n\n---\n\n" + answers)

    summary = {
        "consolidated_run": out.name,
        "discovery_prefill": pref.name,
        "discovery_decode": dec.name,
        "validation": None if val is None else val.name,
        "conversion_top3": top_conv,
        "ax4_verdict": ax4_verdict,
        "rule_selection": rules,
        "validation_summary": val_summary,
        "report": report,
    }
    atomic_write_json(out / "activation_incremental_summary.json", summary)
    write_text(results / "latest_ax_run_id.txt", out.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
