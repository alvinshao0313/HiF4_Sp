"""GPQA Diamond with Qwen3's JSON answer prompt and ten-sample mean."""
from __future__ import annotations

import hashlib
import os
import random
import re
from statistics import mean

from lighteval.metrics.metrics_sample import AvgAtN
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


DATASET_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"
DATASET_PATH = os.environ.get("GPQA_DATASET_PATH")
ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"([A-D])"')
INSTRUCTION = (
    "Answer the following multiple choice question. Think step by step before answering. "
    'Please show your choice in the `answer` field with only the choice letter, '
    'e.g., `"answer": "C"`.'
)


def gpqa_prompt(line: dict, task_name: str | None = None) -> Doc:
    # Match the built-in GPQA option placement; main.py seeds Python with 1234.
    gold_index = random.randint(0, 3)
    choices = [line[f"Incorrect Answer {i}"].strip() for i in range(1, 4)]
    choices.insert(gold_index, line["Correct Answer"].strip())
    question = line["Question"].strip()
    options = "\n".join(f"{letter}) {answer}" for letter, answer in zip("ABCD", choices))
    return Doc(
        task_name=task_name,
        query=f"{INSTRUCTION}\n\n{question}\n\n{options}",
        choices=list("ABCD"),
        gold_index=gold_index,
        specific={"question_sha256": hashlib.sha256(question.encode()).hexdigest()},
    )


def extract_final_answer(raw: str) -> tuple[str | None, str]:
    """Score only the completed final response to the thinking-enabled prompt.

    The checkpoint template leaves the assistant prefix open; the model emits
    its own thinking tags. An opened but unclosed block has no final submission.
    Conflicting JSON fields are retained for review, never resolved using the gold.
    """
    if "<think>" in raw and "</think>" not in raw:
        return None, "unfinished_thinking"
    final = raw.rsplit("</think>", 1)[-1]
    answers = set(ANSWER_FIELD.findall(final))
    if not answers:
        return None, "missing_answer_field"
    if len(answers) != 1:
        return None, "conflicting_answer_fields"
    return answers.pop(), "ok"


class GPQAAvg10(AvgAtN):
    def __init__(self):
        super().__init__(n=10)

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        if len(model_response.text) != self.n:
            raise ValueError(f"Expected {self.n} generations, got {len(model_response.text)}")
        parsed = [extract_final_answer(raw) for raw in model_response.text]
        gold = doc.get_golds()[0]
        scores = [int(answer == gold) for answer, _ in parsed]
        if doc.specific is None:
            doc.specific = {}
        # main.py retains doc.specific in details; raw generations remain in SampleCache.
        doc.specific["gpqa_scoring"] = {
            "answers": [answer for answer, _ in parsed],
            "parse_status": [status for _, status in parsed],
            "scores": scores,
        }
        return mean(scores)


TASKS_TABLE = [
    LightevalTaskConfig(
        name="gpqa_diamond_qwen_avg10",
        prompt_function=gpqa_prompt,
        hf_repo=DATASET_PATH or "Idavidrein/gpqa",
        hf_subset="gpqa_diamond",
        hf_revision=None if DATASET_PATH else DATASET_REVISION,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=32768,
        stop_sequence=[],
        metrics=[SampleLevelMetric(
            metric_name="gpqa_accuracy_avg10",
            sample_level_fn=GPQAAvg10(),
            category=SamplingMethod.GENERATIVE,
            corpus_level_fn=mean,
            higher_is_better=True,
        )],
        version=0,
    )
]
