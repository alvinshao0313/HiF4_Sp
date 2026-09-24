"""QAD 并行：FSDP FULL_SHARD（ZeRO-3 等价）与按层 device_map 备选。"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger("qad.parallel")

__all__ = [
    "FSDP_TRANSFORMER_LAYER_CLS",
    "build_fsdp_training_kwargs",
    "load_model_for_qad",
    "is_oom_error",
]

# Qwen3.5 text decoder layer；FSDP auto_wrap 用
FSDP_TRANSFORMER_LAYER_CLS = ("Qwen3_5DecoderLayer",)


def is_oom_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        key in text
        for key in (
            "out of memory",
            "cuda oom",
            "cuda error: out of memory",
            "cudacudaoutofmemoryerror",
        )
    )


def build_fsdp_training_kwargs(
    *,
    transformer_layer_cls: tuple[str, ...] = FSDP_TRANSFORMER_LAYER_CLS,
) -> dict[str, Any]:
    """HF TrainingArguments 的 FSDP FULL_SHARD 配置。"""
    return {
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": list(transformer_layer_cls),
            "backward_prefetch": "backward_pre",
            "forward_prefetch": False,
            "use_orig_params": True,
            "sync_module_states": True,
            # 多 rank 只在 rank0 完整加载，避免 8×54GB CPU 内存爆炸
            "cpu_ram_efficient_loading": True,
            "limit_all_gathers": True,
            # 与 Trainer gradient_checkpointing 二选一；FSDP 路径用这个
            "activation_checkpointing": True,
        },
    }


def load_model_for_qad(
    model_path: str,
    *,
    parallel_mode: str,
    bf16: bool = True,
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = True,
):
    """按并行模式加载模型。

    - fsdp / ddp: CPU 加载，交给 Trainer/accelerate 分片或搬到各卡
    - layer: device_map=auto 按层切到可见 GPU
    """
    from transformers import AutoModelForCausalLM

    mode = str(parallel_mode).strip().lower()
    dtype = torch.bfloat16 if bf16 else torch.float32
    common: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,
        "dtype": dtype,
    }

    if mode in {"fsdp", "ddp", "none"}:
        logger.info("Loading model on CPU for parallel_mode=%s", mode)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=None,
            low_cpu_mem_usage=True,
            **common,
        )
        return model

    if mode == "layer":
        if not torch.cuda.is_available():
            raise RuntimeError("parallel_mode=layer requires CUDA")
        n = torch.cuda.device_count()
        # 双权重（teacher BF16 + frozen B ≈ 2×线性层）；卡少时加载阶段多留余量给 wrap/激活
        per_gib = "40GiB" if n <= 4 else "65GiB"
        max_memory = {i: per_gib for i in range(n)}
        logger.info(
            "Loading model with device_map=auto on %d GPUs (max_memory=%s)",
            n,
            max_memory,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            **common,
        )
        return model

    raise ValueError(f"Unknown parallel_mode={parallel_mode!r}; expected fsdp|layer|ddp|none")
