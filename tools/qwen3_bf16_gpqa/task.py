"""GPQA Diamond, Qwen JSON prompt and shared final-submission scoring."""
from statistics import mean

from lighteval.metrics.metrics_sample import AvgAtN
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import SamplingMethod

from Native_NVFP4_HiF4_Linear_Puncture.experiments.nvfp4_gpqa.task import (
    DATASET_PATH, DATASET_REVISION, extract_final_answer, gpqa_prompt,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4.review_answers import parse_final


class GPQAAccuracy(AvgAtN):
    def __init__(self, n):
        super().__init__(n=n)

    def compute(self, doc, model_response, **kwargs):
        if len(model_response.text) != self.n:
            raise ValueError(f"Expected {self.n} generations, got {len(model_response.text)}")
        parsed = [parse_final(raw) for raw in model_response.text]
        original = [extract_final_answer(raw) for raw in model_response.text]
        gold = doc.get_golds()[0]
        scores = [int(item["answer"] == gold) for item in parsed]
        if doc.specific is None:
            doc.specific = {}
        doc.specific["gpqa_scoring"] = {
            "answers": [item["answer"] for item in parsed],
            "parse_status": [item["status"] for item in parsed],
            "scores": scores,
            "strict_json_answers": [answer for answer, _ in original],
            "strict_json_scores": [int(answer == gold) for answer, _ in original],
        }
        return mean(scores)


class GPQAAvg10(GPQAAccuracy):
    def __init__(self):
        super().__init__(10)


def task_config(n):
    suffix = "single" if n == 1 else "avg10"
    return LightevalTaskConfig(
        name=f"gpqa_diamond_qwen_bf16_{suffix}",
        prompt_function=gpqa_prompt,
        hf_repo=DATASET_PATH or "Idavidrein/gpqa",
        hf_subset="gpqa_diamond",
        hf_revision=None if DATASET_PATH else DATASET_REVISION,
        hf_avail_splits=["train"], evaluation_splits=["train"],
        few_shots_split=None, few_shots_select=None,
        generation_size=32768, stop_sequence=[],
        metrics=[SampleLevelMetric(
            metric_name=f"gpqa_accuracy_{suffix}", sample_level_fn=GPQAAccuracy(n),
            category=SamplingMethod.GENERATIVE, corpus_level_fn=mean, higher_is_better=True,
        )], version=0,
    )


TASKS_TABLE = [task_config(10), task_config(1)]
