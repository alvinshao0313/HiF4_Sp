"""Predictor probes on one isolated E0 token history, shared by every runtime."""
from __future__ import annotations

import hashlib
import json


def first_subsequence(ids: list[int], needle: list[int], start: int = 0) -> int | None:
    if not needle:
        raise ValueError("empty token subsequence")
    return next((j for j in range(start, len(ids) - len(needle) + 1)
                 if ids[j:j + len(needle)] == needle), None)


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for j, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return j
    return min(len(a), len(b)) if len(a) != len(b) else None


def transitions(row: dict, tokenizer) -> dict:
    ids = row["output_ids"]
    end = first_subsequence(ids, tokenizer.encode("</think>", add_special_tokens=False))
    start = first_subsequence(ids, tokenizer.encode("<think>", add_special_tokens=False))
    # Code transition is defined as the first Markdown code fence after thinking.
    # Locate its predictor using exact prefix decode lengths, not retokenization.
    text = tokenizer.decode(ids, skip_special_tokens=False)
    char_start = text.find("</think>")
    code_char = text.find("```", char_start + len("</think>") if end is not None else 0)
    code = None
    if code_char >= 0:
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(tokenizer.decode(ids[:mid + 1], skip_special_tokens=False)) > code_char:
                hi = mid
            else:
                lo = mid + 1
        code = lo
    return {"thinking_start": start, "thinking_end": end, "code_start": code,
            "output_end": len(ids), "finished_thinking": end is not None}


def _positions(length: int, event: int | None, transition: dict) -> list[dict]:
    chosen: dict[int, set[str]] = {}
    def add(j, reason):
        if j is not None and 0 <= j < length:
            chosen.setdefault(int(j), set()).add(reason)
    if event is not None:
        for delta in (-8, -4, -2, -1, 0):
            add(event + delta, f"divergence_offset_{delta}")
    for j in (8, 16, 32, 64, 128):
        if event is None or abs(j - event) >= 8:
            add(j, "stable_control")
    for field in ("thinking_end", "code_start"):
        if transition[field] is not None:
            add(transition[field] - 1, field + "_previous_predictor")
    return [{"decode_index": j, "reasons": sorted(reasons), "bin": "mechanism"}
            for j, reasons in sorted(chosen.items())]


def build_plans(cohort: list[dict], e0: list[dict], e1: list[dict], tokenizer) -> tuple[dict, dict, list[dict], list[dict]]:
    index0 = {r["prompt_key"]: r for r in e0}
    index1 = {r["prompt_key"]: r for r in e1}
    heavy, feature, events, transition_rows = [], [], [], []
    cohort_by_key = {r["prompt_key"]: r for r in cohort}
    for c in cohort:
        key = c["prompt_key"]
        a, b = index0[key], index1[key]
        if a["input_ids"] != b["input_ids"]:
            raise RuntimeError(f"mechanism prompt mismatch: {key}")
        t = first_divergence(a["output_ids"], b["output_ids"])
        ta, tb = transitions(a, tokenizer), transitions(b, tokenizer)
        group = c["mechanism_group"]
        positions = _positions(len(a["output_ids"]), t if group == "mechanism_regression" else None, ta)
        if group == "mechanism_robust":
            paired = c["matched_regression_key"]
            if paired not in cohort_by_key:
                raise RuntimeError(f"unmatched mechanism control: {key}")
            ar, br = index0[paired], index1[paired]
            tr = first_divergence(ar["output_ids"], br["output_ids"])
            reg_positions = _positions(len(ar["output_ids"]), tr, transitions(ar, tokenizer))
            existing = {p["decode_index"]: p for p in positions}
            for p in reg_positions:
                j = round(p["decode_index"] * (len(a["output_ids"]) - 1) / max(len(ar["output_ids"]) - 1, 1))
                if j not in existing:
                    existing[j] = {"decode_index": j, "bin": "mechanism", "reasons": ["matched_normalized_position"]}
            positions = [existing[j] for j in sorted(existing)]
        common = {**c, **a, "mechanism_group": group, "t_lex": t,
                  "canonical_variant": "E0", "canonical_output_sha256": hashlib.sha256(json.dumps(a["output_ids"]).encode()).hexdigest()}
        heavy.append({**common, "positions": positions, "max_required_decode_index": max(p["decode_index"] for p in positions)})
        limit = min(max(63, (t or 0) + 8), 255) if group == "mechanism_regression" else 63
        limit = min(limit, len(a["output_ids"]) - 1)
        feature.append({**common, "positions": [{"decode_index": j, "reasons": ["dense_feature_scan"]} for j in range(limit + 1)],
                        "max_required_decode_index": limit})
        events.append({"sample_key": key, "mechanism_group": group, "t_lex": t,
                       "predictor_abs_position": len(a["input_ids"]) + t - 1 if t is not None else None,
                       "inside_thinking": t < ta["thinking_end"] if t is not None and ta["thinking_end"] is not None else None,
                       "distance_to_e0_thinking_end": ta["thinking_end"] - t if t is not None and ta["thinking_end"] is not None else None})
        transition_rows.append({"sample_key": key, "mechanism_group": group, "E0": ta, "E1": tb})
    metadata = {"schema_version": 1, "history": "identical_isolated_E0_forced_tokens", "variants": ["E0", "E1", "E2", "E3"]}
    return ({**metadata, "samples": heavy}, {**metadata, "samples": feature}, events, transition_rows)
