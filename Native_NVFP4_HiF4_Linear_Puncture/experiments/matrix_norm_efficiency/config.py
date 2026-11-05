"""First-stage candidates and the user-confirmed accuracy/time objective."""
from pathlib import Path

from ..reasoning_trace_calibration.config import ENGINE, GENERATION, MMLU_DATA, NATIVE

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "Native_NVFP4_HiF4_Linear_Puncture"
MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.matrix_norm_efficiency"
SOURCE_RUN = ROOT / "results/reasoning_trace_calibration/continue_20261016T135141Z"
LAYERS, BATCH_SIZE, SEED, LR = 48, 4, 42, 3e-4
TOKEN_FRACTION = .5
CANDIDATES = {
    "full_e5": {"selection": "full", "epochs": 5},
    "random50_e5": {"selection": "random50", "epochs": 5},
    "coverage50_e5": {"selection": "coverage50", "epochs": 5},
}
TASKS = ("mmlu_pro", "lcb", "gsm8k", "aime24", "aime25")
FORMAL_COUNTS = dict(zip(TASKS, (12032, 175, 1319, 30, 30)))
BASELINE_SECONDS = 72146.30592771294
PROTOCOL = {
    "native_snapshot": str(NATIVE), "target": "HiF4 W4A4", "source_run": str(SOURCE_RUN),
    "stage": "data_efficiency_independent_short_training", "candidates": CANDIDATES,
    "calibration": {"token_fraction": TOKEN_FRACTION, "seed": SEED,
        "strata": "domain_then_three_equal_count_length_bins",
        "allocation": "largest_remainder_by_bin_tokens_within_domain",
        "sequences": "complete_no_truncation_no_cross_bin_borrowing",
        "coverage": "prompt_word_jaccard_and_character5gram_jaccard_facility_gain_per_token",
        "validation": "same_32_native_traces", "development": "same_64_math_plus_64_code"},
    "training": {"layers": LAYERS, "batch_size": BATCH_SIZE, "lr": LR,
        "optimizer": "AdamW", "weight_decay": 0., "scheduler": "independent_cosine",
        "variant": "matrix_norm", "loss": "block_delta_nmse_on_student_prefix",
        "checkpoint_selection": "validation_nmse", "group_size": 64,
        "trainable": ["weight_side_matrix", "input_norm", "moe_norm"],
        "router_trainable": False, "router_kl_weight": 0.},
    "evaluation": {"tasks": TASKS, "counts": FORMAL_COUNTS, "aime_generations": 5,
        "engine": ENGINE, "generation": GENERATION,
        "accuracy_loss": "mean(max(0,100*(native-candidate)))", "maximum_loss_pp": .4},
    "efficiency": {"baseline_seconds": BASELINE_SECONDS, "ideal_seconds": 10800,
        "stretch_seconds": 7200, "fallback_seconds": BASELINE_SECONDS / 2,
        "boundary": "initial_embedding_through_48_layer_training_validation_save_propagation_and_source_check",
        "resource_comparison": "same_gpu_model_and_count", "other_costs": "reported_separately"},
}
