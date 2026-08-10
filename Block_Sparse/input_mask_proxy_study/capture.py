from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer


@dataclass(frozen=True)
class CapturedLinearData:
    activations_bf16: torch.Tensor
    weight_bf16: torch.Tensor
    module_name: str
    layer_index: int
    projection: str
    sample_indices: tuple[int, ...]
    sample_block_counts: tuple[int, ...]


def _load_qwen35_causal_lm(model_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    auto_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(auto_cfg, "model_type", None) == "qwen3_5" and hasattr(
        auto_cfg, "text_config"
    ):
        from transformers import Qwen3_5ForCausalLM

        model = Qwen3_5ForCausalLM.from_pretrained(
            model_path,
            config=auto_cfg.text_config,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def _find_unique_module(model: torch.nn.Module, suffix: str) -> tuple[str, torch.nn.Module]:
    matches = [(n, m) for n, m in model.named_modules() if n.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one module ending with {suffix!r}, found {len(matches)}: "
            f"{[n for n, _ in matches]}"
        )
    return matches[0]


def capture_or_load(
    *,
    model_path: str,
    dataset_hf_id: str,
    seed: int,
    num_samples: int,
    max_seq_len: int,
    max_activation_blocks: int,
    layer_index: int,
    projection: str,
    activation_block_rows: int,
    cache_path: str | Path,
    expected_meta: dict[str, Any] | None = None,
) -> CapturedLinearData:
    """Capture activations.

    Semantics:
    - Each accepted sample must have length >= max_seq_len and contributes exactly
      ``blocks_per_sample = max_seq_len // activation_block_rows`` row blocks
      (e.g. 1024 tokens -> 32 blocks).
    - Collect exactly ``num_samples`` such samples.
    - ``max_activation_blocks`` must equal ``num_samples * blocks_per_sample``.
    """
    cache_path = Path(cache_path)
    if cache_path.is_file():
        return load_capture_cache(cache_path, expected_meta=expected_meta)

    if max_seq_len % activation_block_rows != 0:
        raise ValueError(
            f"max_seq_len={max_seq_len} not divisible by "
            f"activation_block_rows={activation_block_rows}"
        )
    blocks_per_sample = max_seq_len // activation_block_rows
    expected_total = num_samples * blocks_per_sample
    if max_activation_blocks != expected_total:
        raise ValueError(
            f"max_activation_blocks={max_activation_blocks} != "
            f"num_samples*blocks_per_sample={expected_total}"
        )

    ds = load_dataset(dataset_hf_id, split="train")
    if num_samples > len(ds):
        raise ValueError(
            f"num_samples={num_samples} exceeds dataset size {len(ds)}"
        )
    # Shuffle a full candidate pool so short samples can be skipped.
    candidate_indices = list(range(len(ds)))
    random.Random(seed).shuffle(candidate_indices)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = _load_qwen35_causal_lm(model_path)
    suffix = f"layers.{layer_index}.mlp.{projection}"
    module_name, module = _find_unique_module(model, suffix)
    weight = module.weight.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()

    captured_rows: list[torch.Tensor] = []
    used_indices: list[int] = []
    sample_block_counts: list[int] = []
    pending_capture: dict[str, Any] = {"rows": None}

    def pre_hook(_mod, args):
        if not args:
            raise RuntimeError("forward_pre_hook received empty args")
        inp = args[0]
        if not torch.is_tensor(inp):
            raise RuntimeError(f"expected tensor input, got {type(inp)}")
        if inp.ndim != 3 or int(inp.shape[0]) != 1:
            raise RuntimeError(
                f"hook input must be [1,T,K], got shape {tuple(inp.shape)}"
            )
        act = inp.detach()[0].to(dtype=torch.bfloat16, device="cpu").contiguous()
        t = int(act.shape[0])
        if t != max_seq_len:
            raise RuntimeError(
                f"hook input length must equal max_seq_len={max_seq_len}, got {t}"
            )
        n_full = t // activation_block_rows
        if n_full != blocks_per_sample:
            raise RuntimeError(
                f"expected {blocks_per_sample} blocks from one sample, got {n_full}"
            )
        pending_capture["rows"] = act[: blocks_per_sample * activation_block_rows]

    hook_handle = module.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            for idx in candidate_indices:
                if len(used_indices) >= num_samples:
                    break
                text = ds[idx]["text"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"sample {idx} has empty text")
                encoded = tokenizer(
                    text,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"]
                # Require a full-length sample: one sample == 32 row blocks when
                # max_seq_len=1024 and activation_block_rows=32.
                if int(input_ids.shape[1]) < max_seq_len:
                    continue
                input_ids = input_ids[:, :max_seq_len].to(device="cuda:0")
                attention_mask = torch.ones_like(input_ids)
                pending_capture["rows"] = None
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                rows = pending_capture["rows"]
                if rows is None:
                    raise RuntimeError(f"hook did not fire for sample {idx}")
                captured_rows.append(rows)
                used_indices.append(int(idx))
                sample_block_counts.append(blocks_per_sample)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if len(used_indices) != num_samples:
        raise RuntimeError(
            f"captured only {len(used_indices)} full-length samples "
            f"(need {num_samples} samples with length>={max_seq_len})"
        )
    total_blocks = sum(sample_block_counts)
    if total_blocks != max_activation_blocks:
        raise RuntimeError(
            f"captured {total_blocks} blocks, need {max_activation_blocks}"
        )

    activations = torch.cat(captured_rows, dim=0).contiguous()
    need_rows = max_activation_blocks * activation_block_rows
    if int(activations.shape[0]) != need_rows:
        raise RuntimeError(
            f"activation rows {activations.shape[0]} != expected {need_rows}"
        )

    data = CapturedLinearData(
        activations_bf16=activations,
        weight_bf16=weight,
        module_name=module_name,
        layer_index=layer_index,
        projection=projection,
        sample_indices=tuple(used_indices),
        sample_block_counts=tuple(sample_block_counts),
    )
    save_capture_cache(cache_path, data, extra_meta=expected_meta)
    return data


def save_capture_cache(
    path: Path,
    data: CapturedLinearData,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "activations_bf16": data.activations_bf16,
        "weight_bf16": data.weight_bf16,
        "module_name": data.module_name,
        "layer_index": data.layer_index,
        "projection": data.projection,
        "sample_indices": list(data.sample_indices),
        "sample_block_counts": list(data.sample_block_counts),
        "meta": dict(extra_meta or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_capture_cache(
    path: Path,
    expected_meta: dict[str, Any] | None = None,
) -> CapturedLinearData:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    data = CapturedLinearData(
        activations_bf16=payload["activations_bf16"],
        weight_bf16=payload["weight_bf16"],
        module_name=str(payload["module_name"]),
        layer_index=int(payload["layer_index"]),
        projection=str(payload["projection"]),
        sample_indices=tuple(int(x) for x in payload["sample_indices"]),
        sample_block_counts=tuple(int(x) for x in payload["sample_block_counts"]),
    )
    if expected_meta is not None:
        meta = payload.get("meta", {})
        for key, value in expected_meta.items():
            if key not in meta or meta[key] != value:
                raise ValueError(
                    f"capture cache metadata mismatch for {key!r}: "
                    f"cache={meta.get(key)!r} expected={value!r}"
                )
    return data


def capture_manifest_fields(data: CapturedLinearData) -> dict[str, Any]:
    return {
        "module_name": data.module_name,
        "layer_index": data.layer_index,
        "projection": data.projection,
        "activation_shape": list(data.activations_bf16.shape),
        "weight_shape": list(data.weight_bf16.shape),
        "sample_indices": list(data.sample_indices),
        "sample_block_counts": list(data.sample_block_counts),
        "K": int(data.activations_bf16.shape[-1]),
        "N": int(data.weight_bf16.shape[0]),
    }
