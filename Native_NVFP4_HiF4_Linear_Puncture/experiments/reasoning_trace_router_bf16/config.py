"""Fixed protocol for the additional B experiment."""
from ..reasoning_trace_calibration.config import BASELINE, ENGINE, GENERATION, NATIVE, REPO

MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.reasoning_trace_router_bf16"
ORIGIN = "matrix_norm_native_traces"
CANDIDATE = "B_router_bf16"
PROTOCOL = {
    "candidate": CANDIDATE, "origin": ORIGIN, "teacher": str(NATIVE),
    "teacher_prefix": "independent_native", "student_prefix": "independent_student",
    "layers": 48, "train": 128, "val": 32, "batch_size": 4, "seed": 42,
    "epochs": 5, "screen_epochs": 3, "learning_rates": [1e-5, 3e-5, 1e-4],
    "optimizer": "AdamW", "betas": [.9, .999], "eps": 1e-8,
    "weight_decay": 0., "scheduler": "constant", "epoch_zero_allowed": True,
    "parameter_dtype": "bfloat16", "gradient_dtype": "bfloat16",
    "moment_dtype": "bfloat16", "fp32_master": False, "loss_dtype": "float32",
    "loss": "top_mass", "top_k": 8, "temperature": 1.,
    "teacher_experts": 128, "full_trajectories": True,
    "baseline": BASELINE, "engine": ENGINE, "generation": GENERATION,
}
