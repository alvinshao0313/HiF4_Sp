from __future__ import annotations

from copy import deepcopy

import pytest

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion import build_cohort as cohort
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion import judge_isolated_lcb as judge


def _formal_item(doc_id, passed, prompt="question"):
    return {"doc": {"id": doc_id, "specific": {"inputs": ["2\n"], "outputs": ["2\n"], "fn_name": None}},
            "metric": {"codegen_pass@1:16": int(passed)},
            "model_response": {"input": prompt, "text_post_processed": ["```python\nprint(input())\n```"]}}


def test_formal_alignment_is_by_doc_id_and_rejects_missing_duplicates():
    details = {variant: [_formal_item(2, False), _formal_item(1, True)] for variant in cohort.VARIANT_DIRS}
    details["E2"].reverse()
    rows, summary = cohort.build_formal_matrix(details)
    assert [row["doc_id"] for row in rows] == ["1", "2"]
    assert summary["FORMAL_PROMPT_PROTOCOL_ALIGNED"]
    assert rows[0]["E0_generation_len"] is None
    assert rows[0]["E0_finished_thinking"] == "unknown"
    bad = deepcopy(details)
    bad["E2"].pop()
    with pytest.raises(ValueError, match="ID sets differ"):
        cohort.build_formal_matrix(bad)
    bad = deepcopy(details)
    bad["E2"].append(bad["E2"][0])
    with pytest.raises(ValueError, match="duplicate"):
        cohort.build_formal_matrix(bad)


def test_prompt_mismatch_is_preserved_as_fact_and_blocks_mechanism_cohort():
    details = {variant: [_formal_item(1, True)] for variant in cohort.VARIANT_DIRS}
    details["E1"][0]["model_response"]["input"] = "plain question instead of chat wrapper"
    rows, summary = cohort.build_formal_matrix(details)
    assert not summary["FORMAL_PROMPT_PROTOCOL_ALIGNED"]
    assert rows[0]["E1_pass"]
    with pytest.raises(ValueError, match="FORMAL_PROMPT_PROTOCOL_MISMATCH"):
        cohort.build_benchmark_cohort(rows, details["E0"], None)


def test_interest_cohort_requires_exact_e0_wrap_and_marks_noncausal():
    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 97 + 1 for c in text[:48]] or [1]

    def wrap(question: str) -> str:
        return "<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n"

    q_reg, q_rob = "solve regression case", "solve robust case"
    details = {}
    for variant in cohort.VARIANT_DIRS:
        reg_pass = variant == "E0"
        rob_pass = True
        details[variant] = [
            _formal_item(58, reg_pass, prompt=q_reg if variant != "E0" else wrap(q_reg)),
            _formal_item(2, rob_pass, prompt=q_rob if variant != "E0" else wrap(q_rob)),
        ]
        details[variant][0]["model_response"]["output_tokens"] = [[1] * 100]
        details[variant][1]["model_response"]["output_tokens"] = [[1] * 90]
    matrix, summary = cohort.build_formal_matrix(details, Tok())
    assert not summary["FORMAL_PROMPT_PROTOCOL_ALIGNED"]
    rows, manifest = cohort.build_interest_cohort_under_prompt_mismatch(matrix, details, Tok(), count=1)
    assert manifest["FORMAL_PROMPT_PROTOCOL_ALIGNED"] is False
    assert manifest["cohort_role"] == "interest_seed_under_formal_prompt_mismatch"
    assert all(not row["formal_format_only_regression"] for row in rows)
    assert all(row["prompt_source"] == "phaseA_formal_e0_text_retokens_once" for row in rows)
    assert len({row["prompt_key"] for row in rows}) == len(rows)


def test_deterministic_length_coverage_and_one_to_one_matching():
    regressions = [{"doc_id": str(i), "E0_generation_len": i * 10, "formal_group": "formal_regression"} for i in range(1, 62)]
    left = cohort.select_regressions(regressions, count=16)
    right = cohort.select_regressions(list(reversed(regressions)), count=16)
    assert left == right
    assert left[0]["doc_id"] == "58"
    assert len({row["doc_id"] for row in left}) == 16
    assert any(row["doc_id"] == "1" for row in left)
    assert any(row["doc_id"] == "61" for row in left)
    regressions = [{"doc_id": "1", "E0_generation_len": 100}, {"doc_id": "2", "E0_generation_len": 100}]
    robust = [{"doc_id": "3", "E0_generation_len": 101}, {"doc_id": "4", "E0_generation_len": 99}]
    pairs = cohort.match_robust(regressions, robust, {"1": 50, "2": 60, "3": 60, "4": 50})
    assert [(a["doc_id"], b["doc_id"]) for a, b in pairs] == [("1", "4"), ("2", "3")]
    with pytest.raises(ValueError, match="GENERATION_LENGTH_UNAVAILABLE"):
        cohort.select_regressions([{"doc_id": "1", "formal_group": "formal_regression", "E0_generation_len": None}])


def test_official_checker_import_and_real_completion():
    from lighteval.tasks.tasks.lcb.codegen_metrics import check_correctness, extract_code
    assert judge.check_correctness is check_correctness
    assert judge.extract_code is extract_code
    result = judge.judge_completion(_formal_item(1, True), "<think>reason</think>\n```python\nprint(input())\n```",
                                    [10, 20, 30], think_end_ids=[20, 30])
    assert result["pass"] is True
    assert result["finished_thinking"] is True
    assert result["judge_results"] == [True]
    assert result["generation_len"] == 3


def test_think_subsequence_requires_complete_match():
    assert judge.contains_subsequence([1, 2, 3, 4], [2, 3])
    assert not judge.contains_subsequence([1, 2, 3, 4], [3, 2])
    assert not judge.contains_subsequence([1, 2], [2, 3])
    with pytest.raises(ValueError, match="empty"):
        judge.contains_subsequence([1], [])
