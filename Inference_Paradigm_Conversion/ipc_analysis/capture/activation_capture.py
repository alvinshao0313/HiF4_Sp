"""Hook-based linear input capture with prefill/decode separation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Union

import torch
import torch.nn as nn


@dataclass
class CapturedTensor:
    sample_id: str
    phase: str
    token_positions: list[int]
    layer_idx: int
    module_name: str
    stage: str
    tensor: torch.Tensor
    extras: dict[str, Any] = field(default_factory=dict)


class _TokenBudget:
    def __init__(self, max_tokens: int, max_raw: int) -> None:
        self.max_tokens = max_tokens
        self.max_raw = max_raw
        self.n_stats = 0
        self.n_raw = 0

    def accept_stats(self, n: int) -> int:
        left = self.max_tokens - self.n_stats
        take = min(n, left)
        self.n_stats += take
        return take

    def accept_raw(self, n: int) -> int:
        left = self.max_raw - self.n_raw
        take = min(n, left)
        self.n_raw += take
        return take


def _parse_layer_idx(name: str) -> int:
    # model.layers.12.mlp.down_proj
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return -1


@torch.no_grad()
def capture_linear_inputs(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    phase: str,
    module_filter: Callable[[str, nn.Module], bool] | None = None,
    *,
    sample_id: str,
    max_tokens_per_module: int = 4096,
    max_raw_tokens_per_module: int = 256,
    keep_raw: bool = True,
    return_outputs: bool = False,
) -> list[CapturedTensor] | tuple[list[CapturedTensor], Any]:
    """Capture Linear inputs without modifying forward numerics.

    phase must be 'prefill' or 'decode'.
    Padding tokens (attention_mask==0) are excluded when mask matches activation length.
    """
    if phase not in {"prefill", "decode"}:
        raise ValueError(f"phase must be prefill|decode, got {phase}")

    budgets: dict[str, _TokenBudget] = {}
    captured: list[CapturedTensor] = []
    handles = []

    def _want(name: str, mod: nn.Module) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if name.endswith("lm_head") or ".lm_head" in name:
            return False
        if module_filter is not None:
            return module_filter(name, mod)
        return True

    attn_mask = batch.get("attention_mask")

    def make_hook(name: str):
        def hook(_module, inputs):
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            # x: [B, T, K] or [B*T, K]
            xt = x.detach()
            budget = budgets.setdefault(
                name, _TokenBudget(max_tokens_per_module, max_raw_tokens_per_module)
            )
            if xt.ndim == 3:
                bsz, tlen, hidden = xt.shape
                flat = xt.reshape(bsz * tlen, hidden)
                positions = torch.arange(tlen, device=xt.device).repeat(bsz)
                # Prefill: mask aligns with sequence. Decode+KV-cache: activation
                # is only the new tokens while attention_mask is full length.
                if attn_mask is not None and attn_mask.numel() == bsz * tlen:
                    valid = attn_mask.reshape(-1).bool()
                    flat = flat[valid]
                    positions = positions[valid]
            elif xt.ndim == 2:
                flat = xt
                positions = torch.arange(flat.shape[0], device=xt.device)
            else:
                raise ValueError(f"unexpected activation rank {xt.ndim} at {name}")

            n = flat.shape[0]
            take = budget.accept_stats(n)
            if take <= 0:
                return
            # Deterministic head slice of valid tokens for this forward
            flat = flat[:take].to(dtype=torch.bfloat16)
            pos_list = positions[:take].tolist()
            raw_take = budget.accept_raw(take) if keep_raw else 0
            tensor_keep = flat[:raw_take].cpu() if raw_take > 0 else flat[:0].cpu()
            captured.append(
                CapturedTensor(
                    sample_id=sample_id,
                    phase=phase,
                    token_positions=[int(p) for p in pos_list[: max(raw_take, 0)]],
                    layer_idx=_parse_layer_idx(name),
                    module_name=name,
                    stage="linear_input",
                    tensor=tensor_keep,
                    extras={
                        "num_tokens_stats": take,
                        "num_tokens_raw": raw_take,
                        "hidden_size": int(flat.shape[-1]),
                        # keep a small CPU BF16 sample always for A analysis even if raw capped
                        "stat_sample": flat[: min(256, flat.shape[0])].cpu(),
                    },
                )
            )

        return hook

    for name, mod in model.named_modules():
        if _want(name, mod):
            handles.append(mod.register_forward_pre_hook(make_hook(name)))

    try:
        outputs = model(**batch)
    finally:
        for h in handles:
            h.remove()

    if return_outputs:
        return captured, outputs
    return captured


def capture_linear_inputs_iter(
    model: nn.Module,
    batches: Iterator[tuple[str, str, dict[str, torch.Tensor]]],
    **kwargs,
) -> Iterator[CapturedTensor]:
    """batches yield (sample_id, phase, batch_dict)."""
    for sample_id, phase, batch in batches:
        for item in capture_linear_inputs(
            model, batch, phase=phase, sample_id=sample_id, **kwargs
        ):
            yield item
