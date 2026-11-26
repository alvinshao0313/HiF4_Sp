"""Fixed, AIME-free roles and protocols; historical experiments remain untouched."""
from pathlib import Path

from ..reasoning_trace_calibration.config import NATIVE, REPO, ROOT, COHORT
from ..matrix_norm_efficiency.config import SOURCE_RUN
from ..downstream_accuracy_recovery.protocol import protocol_for

MODULE = "fp4_convert.experiments.thesis_format_conversion_supplement"
EFFICIENCY = ROOT / "results/matrix_norm_efficiency/run_20261029T023839Z"
PLAN = ROOT / "plans/qwen3_30b_a3b/2026-11-16-thesis-format-conversion-supplement-plan-cn.md"
TASKS = ("mmlu_pro", "lcb", "gsm8k", "arc_easy", "arc_challenge")
REASONING = TASKS[:3]
COUNTS = dict(zip(TASKS, (12032, 175, 1319, 2376, 1172)))
MAIN = ("native", "direct", "r64", "r64_scale", "r64_gptq", "ours_a20")
EFFICIENCY_ROLES = ("b20", "full_e5", "random50_e5", "coverage50_e5")
HISTORY = {"native": (COHORT, "native_nvfp4"), "direct": (COHORT, "direct_hif4"),
           "r64": (COHORT, "fixed_r64"), "ours_a20": (SOURCE_RUN, "matrix_norm_s1k"),
           "b20": (SOURCE_RUN, "matrix_norm_native_traces"),
           **{k: (EFFICIENCY, k) for k in EFFICIENCY_ROLES[1:]}}
PROTOCOL = {
    "tasks": {k: protocol_for(k if not k.startswith("arc_") else "arc") for k in TASKS},
    "repetitions": {"count": 5, "aggregation": "arithmetic mean of five complete runs",
                    "seeds": "retain task protocol seed in each run", "independent_caches": True,
                    "scope": "all five tasks, every reported method; authorized 2026-11-16"},
    "counts": COUNTS, "native_revision": NATIVE.name, "main": MAIN,
    "efficiency": EFFICIENCY_ROLES, "efficiency_primary_tasks": ["lcb", "mmlu_pro"],
    "calibration": {"source": str(SOURCE_RUN / "calibration/s1k_clean"),
                    "train": 128, "val": 32, "seed": 42, "full_sequences": True},
    "scale": {"budget": "full", "s0_mode": "hardware", "offsets": [-2, -1, 0, 1, 2],
              "group_size": 64, "axis": -1, "data_free": True, "enumerate_e8_e4": True},
    "gptq": {"percdamp": .01, "blocksize": 128, "groupsize": 64, "column_order": "natural", "stream_experts": 32,
             "quantizer": "WeightHiFxQuantizer(hifx4); standard cached hierarchy, no scale search",
             "hessian_input": "actual routed input after R64 before activation QDQ",
             "unvisited_expert": "fail with coverage evidence; no implicit RTN or zeroing"},
    "orthogonal": {"epochs": 20, "batch_size": 4, "lr": .0003, "weight_decay": 0,
                   "scheduler": "cosine", "activation": "fixed R64", "weight": "Cayley @ R64",
                   "norm_trainable": True, "router_trainable": False, "router_kl": 0,
                   "selection": "validation block_delta_nmse", "group_size": 64},
    "mechanism": {"layers": [0, 15, 31, 47], "validation_sequences": 32},
    "bootstrap": {"repeats": 10000, "seed": 42, "unit": "question within each task"},
    "lcb_repeat": {"conditional": False, "count": 5, "seed": 1234, "questions_per_run": 175,
                   "supersedes": "plan section 9.2 conditional three rounds"},
    "storage": "BF16 saved HiF4 QDQ weights, logical W4A4; no packed storage or speed claim",
}
