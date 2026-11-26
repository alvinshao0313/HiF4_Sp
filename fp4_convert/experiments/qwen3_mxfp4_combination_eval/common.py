"""Fixed task definitions and references to completed TP2 qualification."""
from pathlib import Path

from ..qwen3_mxfp4_combinations.format import VARIANTS
from ..qwen3_mxfp4_rotation_eval.common import (
    REPO, digest, file_hash, install_datasets, model_view, now, read, save,
)

HERE = Path(__file__).resolve().parent
PACKAGE = "fp4_convert.experiments.qwen3_mxfp4_combination_eval"
REFERENCE = REPO / "Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/combinations_tp2_20261113T154617Z"
DATA_REFERENCE = REPO / "Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/rotation_downstream_20261111T021111Z/datasets.json"
TASKS = {
    "lcb": {"tag": "combination_lcb_v6_aisbench_avg5|0", "questions": 175, "max_new_tokens": 38912},
    "mmlu_pro": {"tag": "combination_mmlu_pro_aisbench_avg5|0", "questions": 12032, "max_new_tokens": 32768},
}
DEVICES = ("0,1", "2,3", "4,5", "6,7")
PROBE_QUESTIONS = 52  # Five responses per question fill the formal 256-sequence batch.
UPSTREAM_REVISION = "aff1554354fbf19dccbdf2199cc3c22119f815a2"
UPSTREAM_FILES = {
    "mmlu_config.py": "ais_bench/benchmark/configs/datasets/mmlu_pro/mmlu_pro_gen_0_shot_str.py",
    "mmlu_dataset.py": "ais_bench/benchmark/datasets/mmlu_pro.py",
    "lcb_config.py": "ais_bench/benchmark/configs/datasets/livecodebench/livecodebench_0_shot_chat_v6.py",
    "lcb_prompts.py": "ais_bench/benchmark/datasets/livecodebench/prompts.py",
}
