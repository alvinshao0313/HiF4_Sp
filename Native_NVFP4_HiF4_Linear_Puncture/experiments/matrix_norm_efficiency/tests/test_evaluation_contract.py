"""Protocol and final-answer regressions without model inference."""
from Native_NVFP4_HiF4_Linear_Puncture.experiments.matrix_norm_efficiency.evaluate import task_protocol
from Native_NVFP4_HiF4_Linear_Puncture.experiments.reasoning_trace_calibration.scoring import aime_submission


def test_full_task_counts_and_actual_multiplicity():
    counts = {"mmlu_pro": 12032, "lcb": 175, "gsm8k": 1319, "aime24": 30, "aime25": 30}
    for task, count in counts.items():
        protocol = task_protocol(task)
        assert protocol["count"] == count
        assert protocol["max_samples"] is None
        assert protocol["generations_per_doc"] == (5 if task.startswith("aime") else 1)
        smoke = task_protocol(task, smoke=True)
        assert smoke["count"] == 1
        for key in ("max_new_tokens", "max_model_length", "tensor_parallel_size",
                    "max_num_batched_tokens", "batch_size_max_num_seqs"):
            assert smoke[key] == protocol[key]


def test_both_aime_years_use_avg5_and_distinct_task_names():
    for year in (24, 25):
        protocol = task_protocol(f"aime{year}")
        assert protocol["task_name"] == f"aime{year}_avg5|0"
        assert protocol["metric"] == "avg@n:n=5"


def test_final_submission_excludes_thinking_candidate():
    response = r"Maybe \boxed{123}. </think>Therefore, \boxed{456}."
    result = aime_submission(response, "123")
    assert result["score"] == 0
    assert result["answer"] == "456"
    assert result["review"] is False


def test_incomplete_thinking_is_not_a_submission():
    result = aime_submission(r"I think it is \boxed{123} but must check", "123")
    assert result["score"] == 0
    assert result["decision"] == "no_final_submission"


def test_conflicting_final_answers_require_review():
    result = aime_submission(r"</think>First \boxed{123}, then \boxed{456}.", "456")
    assert result["review"] is True
    assert result["boxes"] == ["123", "456"]


def test_unboxed_final_answer_requires_review():
    result = aime_submission("</think>The final answer is 123.", "123")
    assert result["review"] is True
    assert result["score"] is None
