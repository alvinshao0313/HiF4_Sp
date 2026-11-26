"""Pinned AISBench zero-shot prompts, with five actual responses and existing scorers."""
from copy import deepcopy
from dataclasses import replace
import json
from statistics import mean
from string import ascii_uppercase

import numpy as np

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import AvgAtN
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.mmlu_pro import mmlu_pro
from lighteval.tasks.tasks.lcb.main import TASKS_TABLE as LCB_TASKS, lcb_codegeneration_prompt_fn
from lighteval.tasks.tasks.lcb.codegen_metrics import codegen_metrics, extract_code

MMLU_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.

Question:

{question}

Options:

{options_str}"""
LCB_TEMPLATE = "### Question:\n{question_content}\n\n{format_prompt}### Answer: (use the provided format with backticks)\n\n"
LCB_WITH_STARTER = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
LCB_WITHOUT_STARTER = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."


def mmlu_prompt(line, task_name=None):
    options = "\n".join(f"{ascii_uppercase[i]}. {text}" for i, text in enumerate(line["options"]) if text != "N/A").strip()
    query = MMLU_TEMPLATE.format(question=line["question"], options_str=options)
    return Doc(task_name=task_name, query=query, instruction=query,
               choices=list(ascii_uppercase[:len(line["options"])]), gold_index=line["answer_index"],
               specific={"category": line["category"], "question_id": line["question_id"]})


def lcb_prompt(line, task_name=None):
    doc = lcb_codegeneration_prompt_fn(line, task_name)
    starter = line.get("starter_code")
    instruction = LCB_WITH_STARTER if starter else LCB_WITHOUT_STARTER
    formatting = f"### Format: {instruction}\n```python\n{starter or '# YOUR CODE HERE'}\n```\n\n"
    # AISBench declares a system-message constant but does not use it in this config.
    doc.query = LCB_TEMPLATE.format(question_content=line["question_content"], format_prompt=formatting)
    return doc


class RecordedAvg5(AvgAtN):
    def __init__(self, task):
        super().__init__(n=5)
        self.task = task

    def compute(self, doc, model_response, **kwargs):
        if len(model_response.text) != 5 or len(model_response.final_text) != 5:
            raise ValueError("Exactly five actual responses per question are required")
        if self.task == "mmlu_pro":
            scorer = Metrics.gpqa_instruct_metric.value.sample_level_fn
            scores = []
            for index in range(5):
                single = model_response[index]
                # __getitem__ omits processed text; preserve the original scoring input.
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
            if set(results) != {0} or len(results[0]) != 5:
                raise ValueError("LCB checker must return five code-test results")
            scores = [int(np.all(np.asarray(checks) > 0)) for checks in results[0]]
            if abs(float(metrics["pass@1"]) - mean(scores)) > 1e-12:
                raise ValueError("LCB aggregate differs from the five individual code-test results")
        if any(score not in (0, 1) for score in scores):
            raise ValueError("Individual response scores must be binary")
        if doc.specific is None:
            doc.specific = {}
        doc.specific["combination_eval_scoring"] = {"scores": scores, "task": self.task,
                                                    "basis": "existing lighteval single-response scorer"}
        return mean(scores)


def repeated(config, name, task, prompt):
    return replace(deepcopy(config), name=name, prompt_function=prompt, metrics=[SampleLevelMetric(
        metric_name="accuracy_avg5", sample_level_fn=RecordedAvg5(task),
        category=SamplingMethod.GENERATIVE, corpus_level_fn=mean, higher_is_better=True,
    )])


TASKS_TABLE = [
    repeated(mmlu_pro, "combination_mmlu_pro_aisbench_avg5", "mmlu_pro", mmlu_prompt),
    repeated(next(task for task in LCB_TASKS if task.name == "lcb:codegeneration_v6"),
             "combination_lcb_v6_aisbench_avg5", "lcb", lcb_prompt),
]
