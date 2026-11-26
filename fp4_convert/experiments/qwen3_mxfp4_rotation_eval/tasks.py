"""Ten responses per question, preserving the existing prompts and scorers."""
from copy import deepcopy
from dataclasses import replace
import json
from statistics import mean

import numpy as np

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import AvgAtN
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.requests import SamplingMethod
from lighteval.tasks.tasks.mmlu_pro import mmlu_pro
from lighteval.tasks.tasks.lcb.main import TASKS_TABLE as LCB_TASKS
from lighteval.tasks.tasks.lcb.codegen_metrics import codegen_metrics, extract_code


class RecordedAvg10(AvgAtN):
    def __init__(self, task):
        super().__init__(n=10)
        self.task = task

    def compute(self, doc, model_response, **kwargs):
        if len(model_response.text) != 10 or len(model_response.final_text) != 10:
            raise ValueError("This experiment requires exactly ten responses per question")
        if self.task == "mmlu_pro":
            scorer = Metrics.gpqa_instruct_metric.value.sample_level_fn
            scores = []
            for index in range(10):
                single = model_response[index]
                # ModelResponse.__getitem__ drops processed text. Restore it so
                # each response follows the original single-response scorer.
                if model_response.text_post_processed is not None:
                    single.text_post_processed = [model_response.text_post_processed[index]]
                scores.append(float(scorer.compute(doc=doc, model_response=single)))
        else:
            sample = {key: doc.specific[key] for key in ("inputs", "outputs", "fn_name")}
            metrics, results = codegen_metrics(
                [{"input_output": json.dumps(sample)}],
                [[extract_code(text) for text in model_response.final_text]],
                k_list=[1], num_process_evaluate=8,
            )
            if set(results) != {0} or len(results[0]) != 10:
                raise ValueError("LCB checker did not return ten code-test results")
            scores = [int(np.all(np.asarray(checks) > 0)) for checks in results[0]]
            if abs(float(metrics["pass@1"]) - mean(scores)) > 1e-12:
                raise ValueError("LCB pass@1 is inconsistent with individual code-test scores")
        if any(score not in (0, 1) for score in scores):
            raise ValueError("Individual response accuracy must be binary")
        if doc.specific is None:
            doc.specific = {}
        doc.specific["rotation_eval_scoring"] = {"scores": scores, "task": self.task,
                                                "basis": "unchanged official single-response scorer"}
        return mean(scores)


def repeated(config, name, task):
    return replace(deepcopy(config), name=name, metrics=[SampleLevelMetric(
        metric_name="accuracy_avg10", sample_level_fn=RecordedAvg10(task),
        category=SamplingMethod.GENERATIVE, corpus_level_fn=mean, higher_is_better=True,
    )])


TASKS_TABLE = [
    repeated(mmlu_pro, "rotation_mmlu_pro_avg10", "mmlu_pro"),
    repeated(next(task for task in LCB_TASKS if task.name == "lcb:codegeneration_v6"),
             "rotation_lcb_v6_avg10", "lcb"),
]
