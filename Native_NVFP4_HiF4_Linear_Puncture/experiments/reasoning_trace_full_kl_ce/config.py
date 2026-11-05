"""Shared training and inference conditions; only the objective differs."""
from ..reasoning_trace_last_head.config import BASELINE, ENGINE, GENERATION, NATIVE, ORIGIN, REPO

MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.reasoning_trace_full_kl_ce"
CANDIDATES = {"full_kl": "B_last_head_full_kl", "hard_ce": "B_last_head_hard_ce"}
PROTOCOL = {
    "origin": ORIGIN, "teacher": str(NATIVE), "teacher_prefix": "independent_native",
    "student_prefix": "independent_B", "layer": 47, "vocab": 151936,
    "hidden": 2048, "trainable_parameters": 40638464, "train": 128, "val": 32,
    "batch_size": 4, "seed": 42, "epochs": 20, "screen_epochs": 3,
    "learning_rates": [1e-5, 3e-5, 1e-4], "optimizer": "AdamW", "betas": [.9, .999],
    "eps": 1e-8, "weight_decay": 0., "scheduler": "constant",
    "parameter_gradient_moment_dtype": "float32", "forward_export_dtype": "bfloat16",
    "loss_dtype": "float32", "token_chunk": 128, "temperature": 1.,
    "full_trajectories": True, "head_quantized": False,
    "full_kl": "KL(native_teacher || B_student), full vocabulary, all original effective positions",
    "hard_ce": "original native trajectory token[t+1] from student position[t], original full-sequence loss mask at t+1; final position excluded",
    "development": "original fixed isolated 64 math + 64 code; no formal test selection",
    "baseline": BASELINE, "engine": ENGINE, "generation": GENERATION,
}
