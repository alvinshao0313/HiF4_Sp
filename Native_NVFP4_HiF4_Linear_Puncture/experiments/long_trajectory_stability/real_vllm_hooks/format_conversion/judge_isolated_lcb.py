"""Official LiveCodeBench checker for isolated greedy and intervention completions."""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
LIGHTEVAL_SRC = REPO_ROOT / "3rdparty/lighteval/src"
for path in (REPO_ROOT, LIGHTEVAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lighteval.tasks.tasks.lcb.codegen_metrics import check_correctness, extract_code
from lighteval.utils.utils import remove_reasoning_tags
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import read_jsonl, write_jsonl

OFFICIAL_TIMEOUT = 6


def contains_subsequence(ids: list[int], subsequence: list[int]) -> bool:
    if not subsequence:
        raise ValueError("empty token subsequence")
    return any(ids[index:index + len(subsequence)] == subsequence for index in range(len(ids) - len(subsequence) + 1))


def official_checker_manifest() -> dict:
    path = Path(inspect.getsourcefile(check_correctness)).resolve()
    expected = (LIGHTEVAL_SRC / "lighteval/tasks/tasks/lcb/codegen_metrics.py").resolve()
    if path != expected or Path(inspect.getsourcefile(extract_code)).resolve() != expected:
        raise RuntimeError(f"unexpected official checker import: {path}")
    return {"checker": str(path), "checker_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "extractor": "lighteval.tasks.tasks.lcb.codegen_metrics.extract_code",
            "checker_function": "lighteval.tasks.tasks.lcb.codegen_metrics.check_correctness",
            "timeout": OFFICIAL_TIMEOUT, "reasoning_preprocessing": "official remove_reasoning_tags([('<think>', '</think>')])"}


def _judge_text(doc_row: dict, text: str, timeout: int = OFFICIAL_TIMEOUT) -> dict:
    if timeout != OFFICIAL_TIMEOUT:
        raise ValueError("official LCB timeout is fixed at 6 seconds")
    spec = doc_row["doc"]["specific"] if "doc" in doc_row else doc_row["specific"]
    if not spec["inputs"] or len(spec["inputs"]) != len(spec["outputs"]):
        raise ValueError("missing or misaligned official LCB test cases")
    sample = {"input_output": json.dumps({key: spec[key] for key in ("inputs", "outputs", "fn_name")})}
    code = extract_code(text)
    try:
        ast.parse(code)
        syntax_ok = True
    except SyntaxError:
        syntax_ok = False
    # Even an empty extraction is sent to the same production official checker.
    results = check_correctness(sample, code, timeout=timeout)
    normalized = [value.item() if hasattr(value, "item") else value for value in results]
    if not normalized:
        raise RuntimeError("official LCB checker returned no test results")
    passed = all(value > 0 for value in normalized)
    if passed:
        failure = "pass"
    elif not code.strip():
        failure = "extract_empty"
    elif not syntax_ok:
        failure = "syntax_error"
    elif any(value == -3 or value == -1 for value in normalized):
        failure = "timeout"
    elif any(value == -4 for value in normalized):
        failure = "runtime_or_missing_function"
    else:
        failure = "wrong_answer"
    return {"pass": passed, "extracted_empty": not code.strip(), "syntax_ok": syntax_ok,
            "judge_results": normalized, "failure_class": failure,
            "extracted_code_sha256": hashlib.sha256(code.encode()).hexdigest()}


def judge_completion(cohort_row: dict, raw_text: str, output_ids: list[int], *, think_end_ids: list[int],
                     max_new_tokens: int = 38912, timeout: int = OFFICIAL_TIMEOUT) -> dict:
    official_checker_manifest()
    if not isinstance(raw_text, str) or not output_ids:
        raise ValueError("isolated judge requires raw text and nonempty exact output IDs")
    result = _judge_text(cohort_row, remove_reasoning_tags(raw_text, [("<think>", "</think>")]), timeout)
    finished = contains_subsequence(output_ids, think_end_ids)
    result.update({"finished_thinking": finished, "finished_thinking_source": "raw_output_token_ids",
                   "generation_len": len(output_ids), "budget_exhausted": len(output_ids) >= max_new_tokens,
                   "raw_text_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
                   "output_ids_sha256": hashlib.sha256(json.dumps(output_ids, separators=(",", ":")).encode()).hexdigest()})
    if not result["pass"] and not finished and result["budget_exhausted"]:
        result["failure_class"] = "thinking_unfinished_budget_exhausted"
    return result


def judge_formal_sentinels(details_by_variant: dict, doc_ids=("58", "44")) -> tuple[list[dict], dict]:
    """Rejudge preserved official text and reject any discrepancy with formal labels."""
    records = []
    for variant in ("E0", "E1", "E2", "E3"):
        index = {str(row["doc"]["id"]): row for row in details_by_variant[variant]}
        for doc_id in doc_ids:
            item = index[str(doc_id)]
            texts = item["model_response"]["text_post_processed"]
            if len(texts) != 1:
                raise ValueError(f"formal sentinel has nonunit generation count: {variant}/{doc_id}")
            result = _judge_text(item, texts[0])
            expected = float(item["metric"]["codegen_pass@1:16"]) == 1.0
            records.append({"variant": variant, "doc_id": str(doc_id), "formal_pass": expected,
                            "label_reproduced": result["pass"] == expected, **result})
    summary = {"schema_version": 1, "status": "PASS" if all(row["label_reproduced"] for row in records) else "OFFICIAL_JUDGE_BLOCKED",
               "n_checked": len(records), "checker": official_checker_manifest(),
               "mismatches": [row for row in records if not row["label_reproduced"]]}
    return records, summary


def build_greedy_matrix(cohort: list[dict], isolated_by_variant: dict[str, list[dict]], *, think_end_ids: list[int],
                        existing_judgements: dict[tuple[str, str], dict] | None = None) -> tuple[list[dict], dict]:
    if not {"E0", "E1"}.issubset(isolated_by_variant) or set(isolated_by_variant) - {"E0", "E1", "E2", "E3"}:
        raise ValueError("greedy bridge requires E0/E1 and permits only E2/E3 as controls")
    indices = {variant: {str(row["prompt_key"]): row for row in rows} for variant, rows in isolated_by_variant.items()}
    matrix = []
    for source in cohort:
        key = str(source["prompt_key"])
        row = {"doc_id": str(source["doc_id"]), "prompt_key": key, "formal_group": source["formal_group"],
               "label_protocol": "isolated_greedy_official_LCB_judge"}
        for variant, index in indices.items():
            trajectory = index[key]
            if trajectory["input_ids"] != source["input_ids"] or str(trajectory["doc_id"]) != str(source["doc_id"]):
                raise ValueError(f"isolated prompt/doc mismatch: {variant}/{key}")
            judged = judge_completion(source, trajectory["raw_text"], trajectory["output_ids"], think_end_ids=think_end_ids)
            row.update({f"{variant}_{field}": value for field, value in judged.items()})
        row["mechanism_group"] = ("mechanism_regression" if not row["E1_pass"] else "mechanism_robust") if row["E0_pass"] else "mechanism_e0_fail"
        matrix.append(row)
    counts = dict(Counter(row["mechanism_group"] for row in matrix))
    return matrix, {"schema_version": 1, "n_tasks": len(matrix), "group_counts": counts,
                    "MECHANISM_LABEL_BRIDGE_INSUFFICIENT": counts.get("mechanism_regression", 0) < 4,
                    "checker": official_checker_manifest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--phasea_root", type=Path)
    parser.add_argument("--formal_sentinels", action="store_true")
    parser.add_argument("--model_path")
    parser.add_argument("--variants", nargs="+", default=["E0", "E1"], choices=["E0", "E1", "E2", "E3"])
    args = parser.parse_args()
    output = args.run_root / "04_greedy_judge"
    if args.formal_sentinels:
        from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.build_cohort import load_formal_artifacts
        details, _ = load_formal_artifacts(args.phasea_root)
        rows, summary = judge_formal_sentinels(details)
        filename = "formal_judge_identity"
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        cohort = read_jsonl(args.run_root / "02_cohort/benchmark_cohort.jsonl")
        isolated = {variant: read_jsonl(args.run_root / f"03_isolated/{variant}.jsonl") for variant in args.variants}
        rows, summary = build_greedy_matrix(cohort, isolated, think_end_ids=tokenizer.encode("</think>", add_special_tokens=False))
        filename = "greedy_task"
    write_jsonl(output / f"{filename}_matrix.jsonl", rows)
    (output / f"{filename}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if summary.get("status") == "OFFICIAL_JUDGE_BLOCKED":
        raise RuntimeError("OFFICIAL_JUDGE_BLOCKED: formal labels did not reproduce")


if __name__ == "__main__":
    main()
