"""Ten-generation coverage and the existing final-submission parser contract."""
import pytest
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.requests import Doc

from .task import GPQAAccuracy, GPQAAvg10, TASKS_TABLE


def test_ten_samples_mean_and_raw_scoring():
    doc = Doc(query="", choices=list("ABCD"), gold_index=2)
    response = ModelResponse(
        text=['<think>answer: A</think>**Answer**: C'] * 6 + ['</think>\\boxed{C}']
             + ['</think>answer: B'] * 2 + ['<think>answer: C'],
        text_post_processed=['answer: A'] * 10,
    )
    assert GPQAAvg10().compute(doc, response) == 0.7
    assert doc.specific["gpqa_scoring"]["scores"] == [1] * 7 + [0] * 3
    assert max(LightevalTask(TASKS_TABLE[0]).num_samples) == 10


def test_incomplete_sampling_rejected():
    with pytest.raises(ValueError, match="Expected 10 generations"):
        GPQAAvg10().compute(Doc(query="", choices=list("ABCD"), gold_index=0), ModelResponse(text=['A']))


def test_single_sample_uses_same_final_submission_rules():
    scorer = GPQAAccuracy(1)
    task = next(config for config in TASKS_TABLE if config.name.endswith("single"))
    assert max(LightevalTask(task).num_samples) == 1
    for raw, expected in [("<think>answer: A</think>**Answer**: C", 1),
                          ("</think>\\boxed{B}", 0), ("<think>answer: C", 0)]:
        doc = Doc(query="", choices=list("ABCD"), gold_index=2)
        response = ModelResponse(text=[raw], text_post_processed=["answer: A"])
        assert scorer.compute(doc, response) == expected
        assert doc.specific["gpqa_scoring"]["scores"] == [expected]
    with pytest.raises(ValueError, match="Expected 1 generations"):
        scorer.compute(doc, ModelResponse(text=["A", "B"]))
