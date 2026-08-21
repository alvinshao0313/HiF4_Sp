"""Eager HF model loader for numerical attribution (not vLLM fused path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_source_model_for_capture(
    checkpoint: Path | str,
    device: str | torch.device = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, Any]:
    """Load Qwen3 source checkpoint in eager mode for hook-based capture.

    Hadamard must not be re-applied; checkpoint already has folded Hadamard.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = Path(checkpoint)
    tok = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    # Ensure no gradient / no cache side effects for capture forwards.
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok
