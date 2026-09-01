"""lm_eval ARC adapter backed by vLLM kwargs for Qwen3-MoE.

Keeps lm_eval's existing vLLM request protocol; only adapts engine kwargs and
the legacy ``LLM.generate(prompt_token_ids=...)`` call shape that lm_eval 0.4.x
still uses against newer vLLM entrypoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


_GENERATE_COMPAT_ATTR = "_hif4_lm_eval_prompt_token_ids_compat"


def patch_lm_eval_vllm_generate_compat() -> None:
    """Map lm_eval's legacy ``prompt_token_ids=`` onto current ``prompts=``."""
    from vllm import LLM

    current = LLM.generate
    if getattr(current, _GENERATE_COMPAT_ATTR, False):
        return

    def generate(self, prompts=None, sampling_params=None, *args, prompt_token_ids=None, **kwargs):  # noqa: ANN001
        if prompt_token_ids is not None:
            if prompts is not None:
                raise TypeError("pass only one of prompts or prompt_token_ids")
            # list[int] is a valid PromptType in current vLLM.
            prompts = prompt_token_ids
        return current(self, prompts, sampling_params, *args, **kwargs)

    setattr(generate, _GENERATE_COMPAT_ATTR, True)
    LLM.generate = generate  # type: ignore[method-assign]


def build_lm_eval_vllm_kwargs(
    *,
    model_path: str,
    hif4_runtime_spec_path: str | None = None,
    native_nvfp4: bool = False,
    max_model_len: int = 4096,
    max_num_batched_tokens: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "pretrained": model_path,
        "trust_remote_code": True,
        "dtype": "auto",
        "tensor_parallel_size": 2,
        "kv_cache_dtype": "bfloat16",
        "enforce_eager": True,
        "max_model_len": int(max_model_len),
        "seed": int(seed),
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
        kwargs["moe_backend"] = "triton"
        additional["hif4_runtime_spec_path"] = hif4_runtime_spec_path
    if additional:
        kwargs["additional_config"] = additional
    return kwargs
