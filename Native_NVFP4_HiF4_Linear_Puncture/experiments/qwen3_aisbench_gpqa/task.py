"""AISBench zero-shot CoT wording; preserve data placement and both old scorers."""
from copy import deepcopy

from Native_NVFP4_HiF4_Linear_Puncture.experiments.nvfp4_gpqa import task as original

UPSTREAM_REVISION = "aff1554354fbf19dccbdf2199cc3c22119f815a2"
UPSTREAM_PATH = "ais_bench/benchmark/configs/datasets/gpqa/gpqa_gen_0_shot_cot_chat_prompt.py"
INSTRUCTION = (
    "Answer the following multiple choice question. The last line of your response should be "
    "of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
    "Think step by step before answering."
)
TEMPLATE = INSTRUCTION + "\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"


def gpqa_prompt(line, task_name=None):
    doc = original.gpqa_prompt(line, task_name)
    if not doc.query.startswith(original.INSTRUCTION + "\n\n"):
        raise ValueError("Original prompt structure changed")
    doc.query = INSTRUCTION + doc.query[len(original.INSTRUCTION):]
    return doc


TASKS_TABLE = deepcopy(original.TASKS_TABLE)
TASKS_TABLE[0].name = "gpqa_diamond_aisbench_avg10"
TASKS_TABLE[0].prompt_function = gpqa_prompt
