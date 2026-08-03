"""HF-side PPL and lm_eval helpers for threshold experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# lm_eval compatibility with transformers>=4.50 where AutoModelForVision2Seq was renamed.
_orig_tf_getattr = transformers.__class__.__getattr__


def _tf_getattr_patched(self, name):  # noqa: ANN001
    if name == "AutoModelForVision2Seq":
        return transformers.AutoModelForImageTextToText
    return _orig_tf_getattr(self, name)


transformers.__class__.__getattr__ = _tf_getattr_patched
setattr(
    transformers,
    "AutoModelForVision2Seq",
    transformers.AutoModelForImageTextToText,
)

from .fixed_thresholds import get_baseline_config
from .model_hooks import replace_linears_with_act_quant
from .quantizer import HiF4QuantConfig, quantize_hif4
from .weight_search import search_weight_groups


def load_model_tokenizer(
    model_name: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[nn.Module, Any]:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map={"": 0} if device.startswith("cuda") else device,
        )
    model.eval()
    return model, tok


@torch.no_grad()
def apply_weight_quantization(
    model: nn.Module,
    *,
    mode: str,
    device: str = "cuda",
    budget: str = "fast",
    fixed_config: HiF4QuantConfig | None = None,
    layer_whitelist: list[str] | None = None,
    precomputed_updates: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """In-place replace Linear weights with reconstructed quantized weights.

    mode:
      - standard / scalar_mse / no_clip / fixed: fixed-threshold RTN
      - search_fast / search_full: per-group search
      - none: leave weights unchanged

    If ``precomputed_updates`` is provided for search modes, apply those tensors
    directly (keys are module names) and skip live search.
    """
    from .metrics import nmse
    from .model_hooks import iter_target_linears

    stats: dict[str, float] = {}
    cfg = fixed_config
    if mode in ("standard", "scalar_mse", "no_clip"):
        cfg = get_baseline_config(mode)
    elif mode == "fixed":
        if cfg is None:
            raise ValueError("fixed mode requires fixed_config")
    elif mode in ("search_fast", "search_full"):
        budget = "fast" if mode == "search_fast" else "full"
    elif mode == "none":
        return stats
    else:
        raise ValueError(f"unknown weight mode {mode}")

    for name, mod in iter_target_linears(model):
        if layer_whitelist is not None and name not in layer_whitelist:
            continue
        w = mod.weight.data

        if (
            mode in ("search_fast", "search_full")
            and precomputed_updates is not None
            and name in precomputed_updates
        ):
            wq = precomputed_updates[name].to(dtype=w.dtype, device=w.device)
            stats[name] = nmse(w.detach().float(), wq.detach().float())
            mod.weight.data.copy_(wq)
            continue

        # Prefer quantizing along in_features (last dim) if divisible by 64.
        if w.shape[-1] % 64 != 0:
            if w.shape[0] % 64 != 0:
                continue
            wt = w.detach().float().T.contiguous()
            if mode in ("search_fast", "search_full"):
                out = search_weight_groups(wt, budget=budget, device=device)  # type: ignore[arg-type]
                wq = out.reconstruction.T.to(dtype=w.dtype, device=w.device)
                stats[name] = out.nmse
            else:
                assert cfg is not None
                qcfg = HiF4QuantConfig(
                    s0_divisor=cfg.s0_divisor,
                    e8_threshold=cfg.e8_threshold,
                    e4_threshold=cfg.e4_threshold,
                    s0_mode=cfg.s0_mode,
                    group_dim=-1,
                )
                recon = quantize_hif4(wt, config=qcfg).reconstruction
                stats[name] = nmse(wt, recon)
                wq = recon.T.to(dtype=w.dtype, device=w.device)
        else:
            wf = w.detach().float().contiguous()
            if mode in ("search_fast", "search_full"):
                out = search_weight_groups(wf, budget=budget, device=device)  # type: ignore[arg-type]
                wq = out.reconstruction.to(dtype=w.dtype, device=w.device)
                stats[name] = out.nmse
            else:
                assert cfg is not None
                recon = quantize_hif4(wf, config=cfg).reconstruction
                stats[name] = nmse(wf, recon)
                wq = recon.to(dtype=w.dtype, device=w.device)
        mod.weight.data.copy_(wq)
    return stats


def apply_activation_params(
    model: nn.Module,
    param_map: dict[str, HiF4QuantConfig] | None,
    *,
    default: HiF4QuantConfig | None = None,
) -> list[str]:
    if param_map is None and default is None:
        return []
    return replace_linears_with_act_quant(
        model, param_map or {}, default=default
    )


@torch.no_grad()
def evaluate_ppl_wikitext2(
    model: nn.Module,
    tokenizer,
    *,
    max_length: int = 2048,
    stride: int = 2048,
    device: str = "cuda",
) -> float:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t is not None])
    # Do not truncate to model_max_length; we stride manually.
    enc = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
    input_ids = enc.input_ids.to(device)
    nlls = []
    seq_len = input_ids.size(1)
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        input_chunk = input_ids[:, begin:end]
        target = input_chunk.clone()
        target[:, :-trg_len] = -100
        outputs = model(input_chunk, labels=target)
        neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        prev_end = end
        if end == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / max(prev_end, 1))
    return float(ppl.item())


def evaluate_lm_eval_tasks(
    model: nn.Module,
    tokenizer,
    tasks: list[str],
    *,
    num_fewshot: int = 0,
    batch_size: int = 8,
    limit: int | None = None,
) -> dict[str, Any]:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    hf = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=hf,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
    )
    return results


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
