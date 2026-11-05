"""The user-approved protocol; no downstream selection of training settings."""
from ..reasoning_trace_calibration.config import BASELINE, ENGINE, GENERATION, NATIVE, REPO

MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.reasoning_trace_last_head"
ORIGIN = "matrix_norm_native_traces"
CANDIDATE = "B_last_head_vocab64"
PROTOCOL = {
    "candidate": CANDIDATE, "origin": ORIGIN, "teacher": str(NATIVE),
    "teacher_prefix": "independent_native", "student_prefix": "independent_B",
    "layer": 47, "vocab": 151936, "hidden": 2048, "group_size": 64,
    "train": 128, "val": 32, "batch_size": 4, "seed": 42,
    "epochs": 20, "screen_epochs": 3, "learning_rates": [1e-5, 3e-5, 1e-4],
    "optimizer": "AdamW", "betas": [.9, .999], "eps": 1e-8,
    "weight_decay": 0., "scheduler": "constant", "epoch_zero_allowed": True,
    "parameter_dtype": "float32", "gradient_dtype": "float32", "moment_dtype": "float32",
    "forward_dtype": "bfloat16", "loss_dtype": "float32", "trainable_parameters": 40638464,
    "loss": "vocabulary_top_mass_KL_teacher_to_student", "top_k": 64, "temperature": 1.,
    "token_chunk": 128, "full_trajectories": True, "head_quantized": False,
    "head_matrix_initialization": "identity", "baseline": BASELINE,
    "engine": ENGINE, "generation": GENERATION,
}
