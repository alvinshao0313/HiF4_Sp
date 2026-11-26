"""Scoring regressions: reasoning must not become a submitted answer."""
import pytest

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.requests import Doc

from .task import GPQAAvg10, TASKS_TABLE, extract_final_answer


@pytest.mark.parametrize(("raw", "expected"), [
    ('Consider {"answer": "A"}. </think>\n{"answer": "C"}', ("C", "ok")),
    ('</think>\n```json\n{"answer": "B"}\n```', ("B", "ok")),
    ('<think>Still thinking: {"answer": "C"}', (None, "unfinished_thinking")),
    ('{"answer": "C"}', ("C", "ok")),
    ('</think>\nThe answer is C.', (None, "missing_answer_field")),
    ('</think> {"answer":"A"} {"answer":"C"}', (None, "conflicting_answer_fields")),
])
def test_final_submission(raw, expected):
    assert extract_final_answer(raw) == expected


def test_mean_and_per_generation_evidence():
    doc = Doc(query="", choices=list("ABCD"), gold_index=2)
    raw = ['</think> {"answer": "C"}'] * 7 + ['</think> {"answer": "B"}'] * 3
    # Ignore a contradictory post-processing field; the raw response is the evidence.
    response = ModelResponse(text=raw, text_post_processed=['{"answer": "A"}'] * 10)
    assert GPQAAvg10().compute(doc, response) == 0.7
    assert doc.specific["gpqa_scoring"]["scores"] == [1] * 7 + [0] * 3
    assert max(LightevalTask(TASKS_TABLE[0]).num_samples) == 10


def test_incomplete_coverage_fails():
    doc = Doc(query="", choices=list("ABCD"), gold_index=0)
    with pytest.raises(ValueError, match="Expected 10 generations"):
        GPQAAvg10().compute(doc, ModelResponse(text=['</think> {"answer":"A"}']))
