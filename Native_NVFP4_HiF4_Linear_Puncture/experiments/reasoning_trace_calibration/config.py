"""The confirmed protocol; calibration data is the only A/B training variable."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "Native_NVFP4_HiF4_Linear_Puncture"
COHORT = ROOT / "results/module_design_ablation/formal_20261008T074812Z"
NATIVE = Path("/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3")
S1K = ROOT / "results/e2e_diag_reconstruction/shared_calibration/s1k_original_n128_v32_seed42"
S1K_REVISION = "f375c33cb9f540d89b5fa39bb18f6903e2b97e33"
S1K_ARROW = Path("/home/shaoyuantian/.cache/huggingface/datasets/simplescaling___s1_k-1.1_tokenized/default/0.0.0") / S1K_REVISION / "s1_k-1.1_tokenized-train.arrow"
OLD_TRAINING = COHORT / "training/matrix_norm"
MMLU_DATA = COHORT / "mmlu_revision_recovery_20261014"
MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.reasoning_trace_calibration"
SOURCES = {
    "math": {"repo": "DigitalLearningGmbH/MATH-lighteval", "revision": "0530c78699ea5e8eb5530600900e1f328b48acad"},
    "code": {"repo": "deepmind/code_contests", "revision": "802411c3010cb00d1b05bad57ca77365a3c699d6"},
}
MODELS = ("matrix_norm_s1k", "matrix_norm_native_traces", "winner_router_aligned")
COUNTS = {"train": 64, "val": 16, "dev": 64}  # Per domain.
SEED = 42
GENERATION = {"temperature": .6, "top_p": .95, "top_k": 20, "min_p": 0., "max_tokens": 38912, "seed": 1234}
ENGINE = {"tensor_parallel_size": 2, "max_model_len": 40960, "max_num_batched_tokens": 4096,
          "max_num_seqs": 128, "gpu_memory_utilization": .9, "enforce_eager": True,
          "kv_cache_dtype": "bfloat16"}
BASELINE = {"MMLU-Pro": .7537400265957447, "LCB": 95 / 175, "GSM8K": 1259 / 1319,
            "AIME25 avg5": 106 / 150}
PRIMARY = tuple(BASELINE)
PROTOCOL = {"models": MODELS, "sources": SOURCES, "calibration": {"train": 128, "val": 32, "seed": SEED,
            "full_sequences": True, "correctness_filter": False,
            "s1k_policy": "retain_clean_original_slots_fill_unused_seed42_order",
            "s1k_revision": S1K_REVISION, "s1k_retrain_required": True}, "generation": GENERATION, "engine": ENGINE,
            "matrix_norm": {"layers": 48, "epochs": 20, "batch_size": 4, "lr": .0003,
                            "optimizer": "AdamW", "weight_decay": 0., "scheduler": "cosine",
                            "checkpoint_selection": "validation_nmse", "group_size": 64,
                            "loss": "block_delta_nmse", "router_trainable": False, "router_kl_weight": 0.},
            "router": {"layers": 48, "epochs": 5, "batch_size": 4, "learning_rates": [1e-5, 3e-5, 1e-4],
                       "screen_epochs": 3, "loss": "top_mass", "top_k": 8, "temperature": 1.,
                       "optimizer": "AdamW", "weight_decay": 0., "master_dtype": "float32", "forward_export_dtype": "bfloat16",
                       "include_initial_epoch": True}, "baseline": BASELINE, "primary_tasks": PRIMARY,
            "native_snapshot": str(NATIVE), "development": {"math": 64, "code": 64, "tie": MODELS[0]},
            "isolation": {"character_ngram": 5, "jaccard_minimum": .8, "numeric_masking": True},
            "formal_counts": {"mmlu_pro": 12032, "lcb": 175, "gsm8k": 1319, "aime25_questions": 30,
                              "aime25_generations_per_question": 5, "arc_easy": 2376, "arc_challenge": 1172}}
