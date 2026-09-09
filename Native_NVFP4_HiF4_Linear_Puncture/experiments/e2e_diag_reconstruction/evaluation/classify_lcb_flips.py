#!/usr/bin/env python3
"""Re-judge E0-pass / E1-fail LCB items with official extract_code + check_correctness."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
LIGHTEVAL_SRC = REPO_ROOT / "3rdparty" / "lighteval" / "src"
if str(LIGHTEVAL_SRC) not in sys.path:
    sys.path.insert(0, str(LIGHTEVAL_SRC))

from lighteval.tasks.tasks.lcb.codegen_metrics import check_correctness, extract_code  # noqa: E402

DEFAULT_MAX_NEW_TOKENS = 38912
OFFICIAL_TIMEOUT = 6


def _load_items(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} is not a lighteval details list")
    return data


def _doc_id(item: dict) -> str:
    return str(item["doc"]["id"])


def _passed(item: dict) -> bool:
    return float(item["metric"]["codegen_pass@1:16"]) == 1.0


def _generation(item: dict) -> str:
    texts = item["model_response"]["text_post_processed"]
    if not texts:
        return ""
    if len(texts) != 1:
        raise ValueError(f"expected 1 generation, got {len(texts)} for id={_doc_id(item)}")
    return texts[0]


def _sample_payload(item: dict) -> dict:
    spec = item["doc"]["specific"]
    return {
        "input_output": json.dumps(
            {
                "inputs": spec["inputs"],
                "outputs": spec["outputs"],
                "fn_name": spec.get("fn_name"),
            }
        )
    }


def _first_non_pass(results: list) -> object:
    for value in results:
        if value is True:
            continue
        return value
    return True


def _syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _count_fence_lines(text: str) -> int:
    return sum(1 for line in text.split("\n") if "```" in line)


def classify_item(
    *,
    item: dict,
    n_tokens: int,
    max_new_tokens: int,
    timeout: int,
) -> dict:
    text = _generation(item)
    n_fences = _count_fence_lines(text)
    extracted = extract_code(text)
    truncated = n_tokens >= max_new_tokens
    record = {
        "id": _doc_id(item),
        "n_tokens": n_tokens,
        "truncated": truncated,
        "n_fence_lines": n_fences,
        "extracted_chars": len(extracted),
        "extracted_empty": extracted.strip() == "",
        "syntax_ok": None,
        "judge_results": None,
        "first_non_pass": None,
        "cls": None,
    }

    if extracted.strip() == "":
        record["cls"] = "generation_loop" if truncated else "extract_empty"
        return record

    record["syntax_ok"] = _syntax_ok(extracted)
    results = check_correctness(_sample_payload(item), extracted, timeout=timeout)
    fixed = []
    for value in results:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool):
            value = bool(value)
        fixed.append(value)
    record["judge_results"] = fixed
    first = _first_non_pass(fixed)
    record["first_non_pass"] = first if not isinstance(first, bool) else first

    if first is True:
        record["cls"] = "unexpected_pass"
    elif first in (-3, -1):
        record["cls"] = "tle"
    elif first is False or first == -2:
        record["cls"] = "wa"
    elif first == -4:
        fn_name = item["doc"]["specific"].get("fn_name")
        missing_fn = bool(fn_name) and fn_name not in extracted
        record["cls"] = "compile" if (not record["syntax_ok"] or missing_fn) else "runtime"
    else:
        record["cls"] = f"unmapped:{first!r}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0_details", type=Path, required=True)
    parser.add_argument("--e1_details", type=Path, required=True)
    parser.add_argument("--tokenizer_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--timeout", type=int, default=OFFICIAL_TIMEOUT)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print("loading details...", flush=True)
    e0_items = _load_items(args.e0_details)
    e1_items = _load_items(args.e1_details)
    if len(e0_items) != len(e1_items):
        raise ValueError(f"length mismatch: E0={len(e0_items)} E1={len(e1_items)}")
    for left, right in zip(e0_items, e1_items):
        if _doc_id(left) != _doc_id(right):
            raise ValueError(f"id order mismatch: {_doc_id(left)} vs {_doc_id(right)}")

    flips = []
    both_pass = both_fail = e1_only = 0
    for e0, e1 in zip(e0_items, e1_items):
        p0, p1 = _passed(e0), _passed(e1)
        if p0 and not p1:
            flips.append(e1)
        elif p0 and p1:
            both_pass += 1
        elif (not p0) and (not p1):
            both_fail += 1
        else:
            e1_only += 1

    print(
        f"aligned={len(e0_items)} e0_pass_e1_fail={len(flips)} "
        f"both_pass={both_pass} both_fail={both_fail} e1_only={e1_only}",
        flush=True,
    )

    print(f"loading tokenizer {args.tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), trust_remote_code=True)

    records = []
    for i, item in enumerate(flips, start=1):
        text = _generation(item)
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        n_tests = len(item["doc"]["specific"]["inputs"])
        print(
            f"[{i}/{len(flips)}] id={_doc_id(item)} tokens={n_tokens} "
            f"fences={_count_fence_lines(text)} n_tests={n_tests}",
            flush=True,
        )
        record = classify_item(
            item=item,
            n_tokens=n_tokens,
            max_new_tokens=args.max_new_tokens,
            timeout=args.timeout,
        )
        print(f"    -> {record['cls']} extracted_empty={record['extracted_empty']}", flush=True)
        records.append(record)

    counts = Counter(r["cls"] for r in records)
    truncated_n = sum(1 for r in records if r["truncated"])
    payload = {
        "n_e0_pass_e1_fail": len(records),
        "both_pass": both_pass,
        "both_fail": both_fail,
        "e1_only": e1_only,
        "max_new_tokens": args.max_new_tokens,
        "timeout": args.timeout,
        "counts": dict(counts),
        "n_truncated": truncated_n,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print("counts", dict(counts), flush=True)
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
