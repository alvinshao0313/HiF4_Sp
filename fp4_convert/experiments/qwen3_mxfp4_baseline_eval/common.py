"""Fixed references and three finite TP2 queues for the requested baselines."""
from pathlib import Path

from ..qwen3_mxfp4_combination_eval.common import (
    REPO, TASKS, PROBE_QUESTIONS, digest, file_hash, model_view, now, read, save,
)

HERE = Path(__file__).resolve().parent
PACKAGE = "fp4_convert.experiments.qwen3_mxfp4_baseline_eval"
BASE = REPO / "Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4"
REFERENCE = BASE / "combination_downstream_20261114T053205Z_v2"
GPQA_REFERENCE = BASE / "aisbench_gpqa_20261113T023500Z"
TASK_FILE = HERE.parent / "qwen3_mxfp4_combination_eval/tasks.py"
VARIANTS = ("native", "direct")
LANES = (
    ("2,3", (("native", "mmlu_pro"),)),
    ("4,5", (("direct", "mmlu_pro"),)),
    ("6,7", (("native", "lcb"), ("direct", "lcb"))),
)
