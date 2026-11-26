"""Shared evaluation constants and GPU checks."""

from __future__ import annotations

import torch

REASONING_EVAL_NUM_GPUS = 2
REASONING_EVAL_GROUPS = ("mmlu_pro_300", "aime25_avg5")
# lm_eval + vLLM TP=2 groups (same GPU count as reasoning eval).
VLLM_TP2_EVAL_GROUPS = ("arc", "mmlu", "mmlu_pro_300", "aime25_avg5")


def require_visible_cuda_count(min_gpus: int) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this eval group")
    n = int(torch.cuda.device_count())
    if n < min_gpus:
        raise RuntimeError(f"need at least {min_gpus} visible CUDA devices, got {n}")
    return n
