from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from block_pruning.config import GradientBlockPruningConfig


def resolve_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def resolve_model_input_device(model: torch.nn.Module) -> torch.device:
    """Device that should receive input_ids for (possibly sharded) HF models."""
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict) and hf_map:
        devices = []
        for dev in hf_map.values():
            if isinstance(dev, int):
                devices.append(torch.device(f"cuda:{dev}"))
            else:
                devices.append(torch.device(dev))
        return sorted(devices, key=str)[0]
    return next(model.parameters()).device


def _summarize_device_map(model: torch.nn.Module) -> str:
    hf_map = getattr(model, "hf_device_map", None)
    if not isinstance(hf_map, dict) or not hf_map:
        try:
            dev = next(model.parameters()).device
            return f"single_device={dev}"
        except StopIteration:
            return "empty_model"
    counts: dict[str, int] = {}
    for dev in hf_map.values():
        key = str(dev)
        counts[key] = counts.get(key, 0) + 1
    parts = [f"{dev}:{n}" for dev, n in sorted(counts.items())]
    return "device_map={" + ", ".join(parts) + "}"


def load_model_and_tokenizer(config: GradientBlockPruningConfig):
    """Load Qwen3.5-27B (or compatible CausalLM) for MLP block pruning.

    Hub id ``Qwen/Qwen3.5-27B`` ships a multimodal ``Qwen3_5Config`` wrapper.
    Causal LM scoring/export must use ``text_config`` + ``Qwen3_5ForCausalLM`` so
    the saved checkpoint matches vLLM's text model loader (same as Qmodel/).

    CUDA placement follows ``CUDA_VISIBLE_DEVICES``:
    - 1 visible GPU: load onto that device
    - 2+ visible GPUs: ``device_map='auto'`` shards across all visible GPUs
    """
    torch_dtype = resolve_torch_dtype(config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    auto_cfg = AutoConfig.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
    )

    common_kwargs: dict = dict(
        torch_dtype=torch_dtype,
        trust_remote_code=config.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    use_cuda = config.device != "cpu" and torch.cuda.is_available()
    if config.device == "cpu":
        common_kwargs["device_map"] = {"": "cpu"}
    elif use_cuda:
        n_visible = torch.cuda.device_count()
        if n_visible < 1:
            raise RuntimeError("device=cuda but torch.cuda.device_count() == 0")
        if n_visible == 1:
            common_kwargs["device_map"] = {"": 0}
        else:
            # Shard across every GPU made visible by CUDA_VISIBLE_DEVICES.
            common_kwargs["device_map"] = "auto"
        print(
            f"[prune] visible_gpus={n_visible} "
            f"device_map={common_kwargs['device_map']!r}",
            flush=True,
        )
    else:
        raise RuntimeError(
            f"Unsupported device={config.device!r} (cuda unavailable)"
        )

    if getattr(auto_cfg, "model_type", None) == "qwen3_5" and hasattr(auto_cfg, "text_config"):
        from transformers import Qwen3_5ForCausalLM

        model = Qwen3_5ForCausalLM.from_pretrained(
            config.model_path,
            config=auto_cfg.text_config,
            **common_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            **common_kwargs,
        )

    # Ensure export metadata is text CausalLM for vLLM.
    if getattr(model.config, "model_type", None) == "qwen3_5_text":
        model.config.architectures = ["Qwen3_5ForCausalLM"]

    print(f"[prune] placement: {_summarize_device_map(model)}", flush=True)

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    if config.requires_gradient_checkpointing() and config.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model, tokenizer

