#!/usr/bin/env python3
"""Compare smoke E0 TP2 free-run vs TP1 control vs semantic parity probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.compare_free_runs import (
    first_divergence,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import (
    prompt_key,
    read_jsonl,
)


FOCUS_MISMATCHES = {
    ("n159_c94220492", 10),
    ("n346_c216392826", 12),
    ("n346_c216392826", 24),
    ("n346_c216392826", 35),
}


def by_prompt_key(rows: list[dict]) -> dict[str, dict]:
    out = {str(row["prompt_key"]): row for row in rows}
    if len(out) != len(rows):
        raise ValueError("duplicate prompt_key")
    return out


def token_at(row: dict, decode_index: int) -> int | None:
    ids = row["output_ids"]
    if decode_index < 0 or decode_index >= len(ids):
        return None
    return int(ids[decode_index])


def load_semantic_probe_stats(reference_dir: Path, prompt_key_value: str, decode_index: int) -> dict:
    path = reference_dir / f"{prompt_key_value}.pt"
    if not path.is_file():
        return {
            "semantic_top1": None,
            "semantic_target_rank": None,
            "semantic_margin": None,
            "semantic_available": False,
        }
    pack = torch.load(path, map_location="cpu", weights_only=False)
    decode_indices = [int(x) for x in pack["decode_indices"]]
    if decode_index not in decode_indices:
        return {
            "semantic_top1": None,
            "semantic_target_rank": None,
            "semantic_margin": None,
            "semantic_available": False,
        }
    pidx = decode_indices.index(decode_index)
    logits = pack["logits"][pidx].float()
    target = int(pack["target_ids"][pidx])
    topv, topi = torch.topk(logits, k=2)
    top1 = int(topi[0].item())
    margin = float(topv[0].item() - topv[1].item())
    rank = int((logits > logits[target]).sum().item()) + 1
    return {
        "semantic_top1": top1,
        "semantic_target_rank": rank,
        "semantic_margin": margin,
        "semantic_available": True,
        "semantic_matches_target": top1 == target,
    }


def classify_case(focus_rows: list[dict], first_divs: dict[str, int | None]) -> dict:
    """Classify Case A/B/C using the original 4 mismatch probes."""
    if not focus_rows:
        raise RuntimeError("no focus mismatch rows")
    tp1_eq_semantic = 0
    tp1_eq_tp2 = 0
    all_three_same = 0
    for row in focus_rows:
        tp2 = row["tp2_token"]
        tp1 = row["tp1_token"]
        sem = row["semantic_top1"]
        if tp1 is None or sem is None or tp2 is None:
            continue
        if tp1 == sem:
            tp1_eq_semantic += 1
        if tp1 == tp2:
            tp1_eq_tp2 += 1
        if tp1 == tp2 == sem:
            all_three_same += 1
    n = len(focus_rows)
    # Case A: TP1 matches semantic at almost all original mismatch points,
    # and TP1/TP2 first-divergence coincides with those low-margin points.
    almost_all_tp1_sem = tp1_eq_semantic >= max(n - 1, 1)
    # Case B: TP1==TP2 everywhere on those points, semantic differs.
    case_b = tp1_eq_tp2 == n and tp1_eq_semantic == 0
    # Divergences near the focus decode indices
    focus_decodes_by_prompt: dict[str, list[int]] = {}
    for row in focus_rows:
        focus_decodes_by_prompt.setdefault(row["prompt_key"], []).append(int(row["decode_index"]))
    div_overlap = []
    for key, decodes in focus_decodes_by_prompt.items():
        d = first_divs.get(key)
        if d is None:
            div_overlap.append(False)
            continue
        div_overlap.append(any(abs(d - x) <= 2 for x in decodes))
    high_div_overlap = bool(div_overlap) and sum(div_overlap) >= max(len(div_overlap) - 0, 1) and all(
        x or True for x in div_overlap
    )
    # stricter: majority of prompts with focus mismatches have first-div near a focus decode
    high_div_overlap = sum(div_overlap) >= (len(div_overlap) + 1) // 2

    if almost_all_tp1_sem and tp1_eq_tp2 == 0:
        case = "A"
        label = "PRIMARY_TP_NUMERIC_PATH_DIFFERENCE"
    elif case_b:
        case = "B"
        label = "SEMANTIC_RUNTIME_NUMERIC_MISMATCH"
    elif almost_all_tp1_sem and high_div_overlap:
        case = "A"
        label = "PRIMARY_TP_NUMERIC_PATH_DIFFERENCE"
    else:
        # Mixed: TP1/TP2/semantic disagree in different ways at different points
        case = "C"
        label = "MIXED_NUMERIC_PATH_DIFFERENCE"
    return {
        "case": case,
        "label": label,
        "focus_n": n,
        "tp1_eq_semantic": tp1_eq_semantic,
        "tp1_eq_tp2": tp1_eq_tp2,
        "all_three_same": all_three_same,
        "divergence_near_focus_prompts": div_overlap,
        "high_divergence_overlap_with_focus": high_div_overlap,
        "need_operator_parity": case in ("B", "C"),
        "need_tp2_semantic_adapter": case == "A",
    }


def write_markdown(path: Path, payload: dict) -> None:
    lines: list[str] = [
        "# E0 TP1 / TP2 / semantic 三方定位",
        "",
        f"- Case: **{payload['classification']['case']}** (`{payload['classification']['label']}`)",
        f"- TP1 launch success: **{payload['tp1_success']}**",
        f"- Prompt exact ID alignment TP1↔TP2: **{payload['prompt_alignment_ok']}**",
        f"- Need operator parity (Task 4): **{payload['classification']['need_operator_parity']}**",
        f"- Need TP2 semantic adapter next: **{payload['classification']['need_tp2_semantic_adapter']}**",
        f"- Production runtime modified: **false**",
        "",
        "## 1. TP1 control",
        "",
        f"- TP1 samples: {payload['tp1_num_samples']}",
        f"- TP2 samples: {payload['tp2_num_samples']}",
        f"- Shared prompt_keys: {', '.join(payload['shared_prompt_keys'])}",
        "",
        "## 2. TP1 ↔ TP2 first divergence",
        "",
        "| prompt_key | first_divergence |",
        "|---|---:|",
    ]
    for key, div in payload["first_divergence_tp1_vs_tp2"].items():
        lines.append(f"| {key} | {div if div is not None else 'identical'} |")
    lines.extend(
        [
            "",
            "## 3. Probe table (all semantic probe positions)",
            "",
            "| prompt | decode | TP2 token | TP1 token | semantic top1 | semantic target rank | semantic margin | focus |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["probe_rows"]:
        if row["semantic_top1"] is not None:
            sem_tok = str(row["semantic_top1"])
        elif row.get("semantic_assumed_same_as_tp2"):
            sem_tok = "same-as-TP2"
        else:
            sem_tok = "-"
        rank = "-" if row["semantic_target_rank"] is None else str(row["semantic_target_rank"])
        margin = "-" if row["semantic_margin"] is None else f"{row['semantic_margin']:.4f}"
        focus = "YES" if row["is_focus_mismatch"] else ""
        lines.append(
            f"| {row['prompt_key']} | {row['decode_index']} | {row['tp2_token']} | {row['tp1_token']} | "
            f"{sem_tok} | {rank} | {margin} | {focus} |"
        )
    lines.extend(
        [
            "",
            "## 4. Focus mismatches (original 4)",
            "",
            "| prompt | decode | TP2 | TP1 | semantic | closer_to |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["focus_rows"]:
        closer = row["tp1_closer_to"]
        lines.append(
            f"| {row['prompt_key']} | {row['decode_index']} | {row['tp2_token']} | {row['tp1_token']} | "
            f"{row['semantic_top1']} | {closer} |"
        )
    lines.extend(
        [
            "",
            "## 5. Decision answers",
            "",
            f"1. TP1 used the same 4 prompts: **{payload['prompt_alignment_ok']}**",
            "2. TP1↔TP2 first-divergence: see section 2.",
            "3. On the original 4 mismatch points, TP1 closer counts: "
            f"semantic={payload['focus_closer_counts']['semantic']}, "
            f"TP2={payload['focus_closer_counts']['tp2']}, "
            f"neither/tied={payload['focus_closer_counts']['neither']}.",
            f"4. Case: **{payload['classification']['case']}** / `{payload['classification']['label']}`",
            f"5. Enter operator parity: **{payload['classification']['need_operator_parity']}**",
            "6. Production runtime modified this round: **false**",
            "",
            "## 6. Stop",
            "",
            "本轮只完成 TP1 control + 三方定位。不启动 formal，不降低 parity threshold，不修改 max-scale/QDQ/production runtime。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp2_jsonl", required=True)
    p.add_argument("--tp1_jsonl", required=True)
    p.add_argument("--parity_json", required=True)
    p.add_argument("--probe_plan", required=True)
    p.add_argument("--reference_dir", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    args = p.parse_args()

    tp2_rows = by_prompt_key(read_jsonl(Path(args.tp2_jsonl)))
    tp1_rows = by_prompt_key(read_jsonl(Path(args.tp1_jsonl)))
    parity = json.loads(Path(args.parity_json).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.probe_plan).read_text(encoding="utf-8"))
    reference_dir = Path(args.reference_dir)

    if set(tp1_rows) != set(tp2_rows):
        raise RuntimeError(
            f"prompt set differs: missing={sorted(set(tp2_rows)-set(tp1_rows))[:5]} "
            f"extra={sorted(set(tp1_rows)-set(tp2_rows))[:5]}"
        )
    for key in tp2_rows:
        if [int(x) for x in tp1_rows[key]["input_ids"]] != [int(x) for x in tp2_rows[key]["input_ids"]]:
            raise RuntimeError(f"exact prompt token ids differ for {key}")

    first_divs = {
        key: first_divergence(tp2_rows[key]["output_ids"], tp1_rows[key]["output_ids"]) for key in sorted(tp2_rows)
    }
    mismatch_map = {
        (str(m["prompt_key"]), int(m["decode_index"])): m for m in parity.get("mismatches", [])
    }

    probe_rows: list[dict] = []
    for sample in plan["samples"]:
        key = str(sample["prompt_key"])
        for pos in sample["positions"]:
            decode_index = int(pos["decode_index"])
            tp2_tok = token_at(tp2_rows[key], decode_index)
            tp1_tok = token_at(tp1_rows[key], decode_index)
            stats = load_semantic_probe_stats(reference_dir, key, decode_index)
            is_focus = (key, decode_index) in FOCUS_MISMATCHES
            is_mismatch = (key, decode_index) in mismatch_map
            if not stats["semantic_available"]:
                # Match points without reference logits: plan allows marking same-as-TP2 only when parity matched.
                if not is_mismatch:
                    stats = {
                        "semantic_top1": tp2_tok,
                        "semantic_target_rank": 1,
                        "semantic_margin": None,
                        "semantic_available": False,
                        "semantic_assumed_same_as_tp2": True,
                        "semantic_matches_target": True,
                    }
                else:
                    m = mismatch_map[(key, decode_index)]
                    stats = {
                        "semantic_top1": int(m["semantic_top1_token"]),
                        "semantic_target_rank": None,
                        "semantic_margin": None,
                        "semantic_available": False,
                        "semantic_assumed_same_as_tp2": False,
                        "semantic_matches_target": False,
                    }
            row = {
                "prompt_key": key,
                "decode_index": decode_index,
                "reasons": pos.get("reasons"),
                "tp2_token": tp2_tok,
                "tp1_token": tp1_tok,
                "semantic_top1": stats.get("semantic_top1"),
                "semantic_target_rank": stats.get("semantic_target_rank"),
                "semantic_margin": stats.get("semantic_margin"),
                "semantic_available": bool(stats.get("semantic_available")),
                "semantic_assumed_same_as_tp2": bool(stats.get("semantic_assumed_same_as_tp2")),
                "is_parity_mismatch": is_mismatch,
                "is_focus_mismatch": is_focus,
            }
            if tp1_tok is None or stats.get("semantic_top1") is None or tp2_tok is None:
                closer = "unknown"
            elif tp1_tok == stats["semantic_top1"] and tp1_tok != tp2_tok:
                closer = "semantic"
            elif tp1_tok == tp2_tok and tp1_tok != stats["semantic_top1"]:
                closer = "tp2"
            elif tp1_tok == tp2_tok == stats["semantic_top1"]:
                closer = "both"
            else:
                closer = "neither"
            row["tp1_closer_to"] = closer
            probe_rows.append(row)

    focus_rows = [r for r in probe_rows if r["is_focus_mismatch"]]
    if len(focus_rows) != 4:
        raise RuntimeError(f"expected 4 focus mismatch rows, got {len(focus_rows)}")
    classification = classify_case(focus_rows, first_divs)
    closer_counts = {"semantic": 0, "tp2": 0, "neither": 0, "both": 0, "unknown": 0}
    for row in focus_rows:
        closer_counts[row["tp1_closer_to"]] = closer_counts.get(row["tp1_closer_to"], 0) + 1

    payload = {
        "schema_version": 1,
        "tp1_success": True,
        "prompt_alignment_ok": True,
        "tp1_num_samples": len(tp1_rows),
        "tp2_num_samples": len(tp2_rows),
        "shared_prompt_keys": sorted(tp2_rows),
        "first_divergence_tp1_vs_tp2": first_divs,
        "parity_top1": parity.get("top1_parity"),
        "parity_num_probe_points": parity.get("num_probe_points"),
        "probe_rows": probe_rows,
        "focus_rows": focus_rows,
        "focus_closer_counts": closer_counts,
        "classification": classification,
        "production_runtime_modified": False,
        "formal_started": False,
        "parity_threshold_lowered": False,
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out_md, payload)
    print(json.dumps({"case": classification["case"], "label": classification["label"], "md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
