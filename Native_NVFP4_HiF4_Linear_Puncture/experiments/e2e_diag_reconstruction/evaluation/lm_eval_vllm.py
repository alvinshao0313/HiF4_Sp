"""lm_eval ARC adapter backed by vLLM kwargs for Qwen3-MoE."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_lm_eval_vllm_kwargs(
    *,
    model_path: str,
    hif4_runtime_spec_path: str | None = None,
    native_nvfp4: bool = False,
    max_model_len: int = 4096,
    max_num_batched_tokens: int | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "pretrained": model_path,
        "trust_remote_code": True,
        "dtype": "auto",
        "tensor_parallel_size": 2,
        "kv_cache_dtype": "bfloat16",
        "enforce_eager": True,
        "max_model_len": int(max_model_len),
    }
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
    additional: dict[str, Any] = {}
    if native_nvfp4:
        kwargs["linear_backend"] = "emulation"
        kwargs["moe_backend"] = "emulation"
    if hif4_runtime_spec_path:
        if not Path(hif4_runtime_spec_path).is_file():
            raise FileNotFoundError(hif4_runtime_spec_path)
        additional["hif4_runtime_spec_path"] = hif4_runtime_spec_path
    if additional:
        kwargs["additional_config"] = additional
    return kwargs
