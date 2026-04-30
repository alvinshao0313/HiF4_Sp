"""AIME 2024 / 2025 的自定义任务（avg@N）。

- ``aime24_avg5`` / ``aime25_avg5``：avg@5（更省时间的小规模对比）

Prompt 与 ``lighteval/tasks/tasks/aime.py`` 内置版本一致，便于与 lighteval
内置的 ``aime24``/``aime25``/``aime24_avg``/``aime25_avg`` 直接对齐。

注：lighteval v0.13.0 已经自带了 ``aime24_avg`` / ``aime25_avg``（均为 avg@64，
定义与我们此前的实现等价），所以这里不再重复定义，直接复用上游内置任务即可；
只有 ``avg5`` 变体是我们额外添加的。
"""
from textwrap import dedent

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# 与 lighteval/tasks/tasks/aime.py 内置 prompt 保持一致
MATH_PROMPT_TEMPLATE = dedent("""
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{prompt}
""").strip()


def record_to_sample(record):
    from inspect_ai.dataset import Sample

    return Sample(input=record["problem"], target=record["answer"])


def aime_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_PROMPT_TEMPLATE.format(prompt=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def _make_aime(name: str, hf_repo: str, n: int) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=name,
        prompt_function=aime_prompt,
        sample_fields=record_to_sample,
        hf_repo=hf_repo,
        hf_subset="default",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=None,
        metrics=[Metrics.avg_at_n_math(sample_params={"n": n})],
        version=2,
    )


aime24_avg5 = _make_aime("aime24_avg5", "HuggingFaceH4/aime_2024", n=5)
aime25_avg5 = _make_aime("aime25_avg5", "yentinglin/aime_2025", n=5)


TASKS_TABLE = [aime24_avg5, aime25_avg5]
