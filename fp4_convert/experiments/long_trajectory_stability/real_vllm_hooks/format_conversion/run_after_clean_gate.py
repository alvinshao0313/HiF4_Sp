#!/usr/bin/env python3
"""After clean E0/E1 Gate: stop on C, or unlock reduced RQ1/RQ2 per 2026-09-08 plan."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_json(path: Path):
    return json.loads(path.read_text())


def _mmlu_section(run_root: Path) -> tuple[list[str], bool]:
    mmlu_json = run_root / "20_mmlu_pro_300" / "CLEAN_MMLU_PRO_E0_E7_REPORT.json"
    mmlu_md = run_root / "20_mmlu_pro_300" / "CLEAN_MMLU_PRO_E0_E7_REPORT.md"
    if not mmlu_json.exists():
        return (
            [
                "## 统一 MMLU-Pro300 E0–E7 基线",
                "",
                "- **PENDING**：尚未完成；该项为无条件必做，不受 LCB Gate 取消。",
                f"- 完成后见 `{mmlu_md}`。",
                "",
            ],
            False,
        )
    summary = load_json(mmlu_json)
    scores = summary["scores"]
    delta = summary["delta_pp_vs_E0"]
    lines = [
        "## 统一 MMLU-Pro300 E0–E7 基线",
        "",
        "- 状态：**COMPLETE**（统一 runner / prompt / TP2 / KV BF16 / generation；E3/E4=adopted）。",
        f"- 详情：`{mmlu_md}`",
        "",
        "| Variant | score | delta vs E0 (pp) |",
        "|---|---:|---:|",
    ]
    for k in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"):
        lines.append(f"| {k} | {scores[k]*100:.2f}% | {delta[k]:+.2f} |")
    lines.append("")
    if summary.get("E0_E1_bootstrap"):
        boot = summary["E0_E1_bootstrap"]
        flips = summary.get("paired_flips", {}).get("E0_E1", {})
        lines.extend(
            [
                f"- E0 vs E1 flips n10/n01 = {flips.get('n10_E0pass_E1fail')}/{flips.get('n01_E0fail_E1pass')}",
                f"- delta = {boot['delta_pp']:.3f} pp; 95% CI [{boot['ci95_low_pp']:.3f}, {boot['ci95_high_pp']:.3f}]",
                f"- exact McNemar p = {summary.get('E0_E1_mcnemar_p')}",
                "",
            ]
        )
    return lines, True


def write_report(run_root: Path, old_run: Path, clean_summary: dict, matched_audit: dict | None) -> Path:
    gate = clean_summary["gate"]
    out = run_root / "analysis" / "FORMAT_CONVERSION_ERROR_LOCALIZATION_REPLAN_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    pass_e0 = clean_summary["pass_counts"]["E0"]
    pass_e1 = clean_summary["pass_counts"]["E1"]
    n = clean_summary["n_tasks"]
    pairs = clean_summary["pair_counts"]
    boot = clean_summary["paired_bootstrap_95ci"]
    mmlu_lines, mmlu_done = _mmlu_section(run_root)
    cancelled = [
        "decode-length monotonic accumulation scan",
        "LCB low-margin enrichment 大扫描",
        "router top-k enrichment 大扫描",
        "重复 token exact-rejoin rescue/poison",
        "旧 formal 61 semantic-frontier",
        "全 token × 全 layer quantizer feature scan",
        "全 48 层 QKV/O same-input puncture",
        "全 48 层 state-reset matrix",
        "大 alpha dose injection curve",
        "E2/E3 无条件完整 mechanism rerun",
        "adopted E3 作为 DIAG 本身机制解释",
        "E4 / E6 / E7 **机制深钻**（但其 MMLU-Pro300 benchmark 必须重跑）",
        "旧 run_go heavy chain after matched greedy audit",
    ]
    if gate["gate"] == "C":
        cancelled.extend(
            [
                "task-causal MoE puncture / e=q+p cohort",
                "state reset/injection",
                "E2 / DIAG diagnostic probe",
                "clean mechanism cohort bridge",
            ]
        )
    elif gate["gate"] == "B":
        cancelled.extend(
            [
                "broad state intervention",
                "E2/E3 task recovery",
                "cohort-level logistic / large RQ2",
            ]
        )

    matched_block = "未完成 / 不可用"
    if matched_audit:
        matched_block = (
            f"n={matched_audit['n_tasks']}; "
            f"regression={matched_audit['group_counts'].get('mechanism_regression', 0)}; "
            f"robust={matched_audit['group_counts'].get('mechanism_robust', 0)}; "
            f"e0_fail={matched_audit['group_counts'].get('mechanism_e0_fail', 0)} "
            f"（有偏 interest cohort，不可外推）"
        )

    status = "COMPLETE" if mmlu_done else "PENDING_MMLU"
    lines = [
        "# FORMAT_CONVERSION_ERROR_LOCALIZATION_REPLAN_REPORT",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).isoformat()}",
        "- 计划：`2026-09-08-qwen3-30b-hif4-error-localization-replan-after-lcb-prompt-audit-cn.md`",
        f"- 报告状态：**{status}**",
        f"- Gate：**{gate['gate']}** (`{gate['CLEAN_FORMAT_LOSS']}`)",
        "",
        "## 第一页对照",
        "",
        "| 来源 | E0 | E1 | 说明 |",
        "|---|---:|---:|---|",
        "| 旧 Phase-A formal observed | 95/175 | 34/175 | prompt mismatch，**不可归因**为纯格式转换 |",
        f"| 新 clean controlled sampled | {pass_e0}/{n} | {pass_e1}/{n} | exact same input_ids + {clean_summary.get('execution_shape', 'max_num_seqs=128')} + 官方 judge |",
        "",
        f"- delta (E1-E0) = **{clean_summary['delta_accuracy_pp']:.3f} pp**",
        f"- paired bootstrap 95% CI = **[{boot['ci95_low_pp']:.3f}, {boot['ci95_high_pp']:.3f}] pp**",
        f"- exact McNemar p = **{clean_summary['exact_mcnemar_p']:.6g}**",
        f"- n10 / n01 / both_pass / both_fail = "
        f"**{pairs['n10']} / {pairs['n01']} / {pairs['both_pass']} / {pairs['both_fail']}**",
        "",
        *mmlu_lines,
        "## 已确认事实",
        "",
        "1. 旧 formal E0 用 Qwen chat wrapper，E1/E2/E3 用裸题面：`FORMAL_PROMPT_PROTOCOL_ALIGNED=false`。",
        "2. matched-prompt greedy 55 题 audit 已完成：" + matched_block + "。",
        "3. clean controlled sampled E0/E1 全 175 已完成，并给出 paired 统计与 Gate。",
        "4. MMLU-Pro300 E0–E7 统一重跑为无条件必做项（不受 LCB Gate 取消）。"
        + (" 已完成。" if mmlu_done else " **进行中**。"),
        "",
        "## 统计关联",
        "",
        "- clean LCB paired delta / CI / McNemar 见上；这是同 prompt 协议下的长生成任务级关联证据。",
        "- matched 55 的 formal→greedy 转移矩阵只说明旧 formal regression 在对齐后多数不再是 E0-pass/E1-fail。",
        "- MMLU-Pro300 新横向表用于短程推理 / 算法对照；旧 MMLU 横向表不再作为最终排名。",
        "",
        "## 因果证据",
        "",
    ]
    if gate["gate"] == "C":
        lines.extend(
            [
                "- **本轮未建立** LCB 精度损失的 task-causal 机制证据。",
                "- Gate C：没有支持足够大的 clean format loss，因此按计划主动停止 task-causal chain。",
                "",
            ]
        )
    elif gate["gate"] == "B":
        lines.extend(
            [
                "- clean loss 存在但 SMALL_OR_UNCERTAIN；仅允许缩减版 RQ1（<=4 case）。",
                "- 大规模 state / E2E3 recovery **未解锁**。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- Gate A：允许 MoE-first `e=q+p`、少量 state、条件式 E2/DIAG。",
                "- 具体因果层证据见后续 RQ1/RQ2 产物（若本文件生成时尚未跑完，以 `analysis/` 增量更新为准）。",
                "",
            ]
        )

    lines.extend(
        [
            "## 已被否定或显著削弱的旧假设（本轮不再验证）",
            "",
        ]
    )
    for item in [
        "固定 history 下误差随 decode 长度单调累积",
        "first divergence 必然低 margin（不作 LCB 主假设）",
        "router top-k change 在 first divergence 富集",
        "token-level exact rejoin rescue/poison 作为任务正确性主指标",
        "旧 formal 61 = 已证实的纯格式转换 regression",
    ]:
        lines.append(f"- {item}")

    lines.extend(["", "## 因 Gate / 计划被主动取消的实验", ""])
    for item in cancelled:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 计划成功标准对照（§15）",
            "",
            f"1. 旧 formal prompt mismatch 已剔除：`{(old_run / '00_audit/FORMAL_PROTOCOL_MISMATCH.md').exists()}`",
            f"2. matched-prompt 55 greedy audit：`{matched_audit is not None}`"
            + (f"（regression={matched_audit['group_counts'].get('mechanism_regression')}）" if matched_audit else ""),
            f"3. LCB E0 复用 + E1×175 @128：`{clean_summary.get('e0_policy')}` / `{clean_summary.get('execution_shape')}`",
            "4. LCB paired delta + CI + McNemar：见上文第一页对照",
            f"5. MMLU-Pro300 E0 复用 + E1–E7 统一重跑：`{'COMPLETE' if mmlu_done else 'PENDING'}`",
            f"6. MMLU 横向表 + paired flips：`{'COMPLETE' if mmlu_done else 'PENDING'}`",
            f"7. Gate A/B/C：`{gate['gate']}`",
            f"8. 按 Gate 停止不必要 LCB 机制实验：`{'task-causal STOPPED' if gate['gate'] == 'C' else '见因果证据节'}`",
            "",
            "## 产物路径",
            "",
            f"- clean report: `{run_root / '10_clean_lcb/CLEAN_LCB_E0_E1_REPORT.json'}`",
            f"- matched audit: `{old_run / '04_greedy_judge/MATCHED_PROMPT_GREEDY_AUDIT.json'}`",
            f"- formal mismatch: `{old_run / '00_audit/FORMAL_PROTOCOL_MISMATCH.md'}`",
            f"- mmlu report: `{run_root / '20_mmlu_pro_300/CLEAN_MMLU_PRO_E0_E7_REPORT.json'}`",
            f"- requirement audit: `{run_root / 'analysis/REQUIREMENT_AUDIT.json'}`",
            "",
        ]
    )
    out.write_text("\n".join(lines))
    (run_root / "analysis" / "REPORT_STATUS.json").write_text(
        json.dumps(
            {
                "status": status,
                "mmlu_complete": mmlu_done,
                "gate": gate["gate"],
                "report": str(out.resolve()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--old_run", type=Path, required=True)
    args = parser.parse_args()
    clean_summary = load_json(args.run_root / "10_clean_lcb/CLEAN_LCB_E0_E1_REPORT.json")
    matched_path = args.old_run / "04_greedy_judge/MATCHED_PROMPT_GREEDY_AUDIT.json"
    matched = load_json(matched_path) if matched_path.exists() else None
    gate = clean_summary["gate"]["gate"]
    decision = {
        "gate": gate,
        "CLEAN_FORMAT_LOSS": clean_summary["gate"]["CLEAN_FORMAT_LOSS"],
        "MATERIAL": clean_summary["gate"]["MATERIAL"],
        "STATISTICALLY_SUPPORTED": clean_summary["gate"]["STATISTICALLY_SUPPORTED"],
        "action": {
            "A": "unlock_reduced_RQ1_RQ2_and_conditional_E2_DIAG",
            "B": "unlock_reduced_RQ1_max_4_cases_only",
            "C": "stop_task_causal_chain_write_final_report",
        }[gate],
        "max_pairs": {"A": 8, "B": 4, "C": 0}[gate],
        "launch_reduced_rq1": gate in ("A", "B"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (args.run_root / "10_clean_lcb/GATE_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")
    (args.run_root / "10_clean_lcb/GATE.txt").write_text(gate + "\n")
    report = write_report(args.run_root, args.old_run, clean_summary, matched)
    print(json.dumps({"gate": gate, "report": str(report), "decision": decision["action"]}, ensure_ascii=False))
    if gate == "C":
        print("GATE_C_STOP: task-causal mechanism cancelled by plan", flush=True)
        return
    # Unlock marker for detached RQ1 driver.
    unlock = {
        "gate": gate,
        "max_pairs": decision["max_pairs"],
        "script": str((args.run_root / "run_reduced_rq1.sh").resolve()),
        "status": "READY",
    }
    (args.run_root / "10_clean_lcb/RQ1_UNLOCK.json").write_text(json.dumps(unlock, indent=2) + "\n")
    if gate == "B":
        print("GATE_B: reduced RQ1 (<=4) allowed; broad state/E2E3 cancelled", flush=True)
        return
    print("GATE_A: full reduced RQ1/RQ2 unlocked", flush=True)


if __name__ == "__main__":
    main()
