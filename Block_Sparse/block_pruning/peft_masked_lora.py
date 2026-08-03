"""peft LoRA + block-mask adaptation for pruned MLP linears.

Subclass peft ``lora.Linear`` so both forward and ``merge_and_unload`` apply
``ΔW = scale·(B@A) ⊙ M``. Stock peft forward does not go through
``get_delta_weight``; both paths must be overridden.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.tuners.lora import Linear as PeftLoraLinear

from block_pruning.block_utils import expand_block_mask
from block_pruning.mask_apply import verify_masks_and_weights
from block_pruning.mlp_registry import MLPLinearTarget, collect_mlp_linears

logger = logging.getLogger(__name__)

MLP_LORA_TARGET_MODULES = ("gate_proj", "up_proj", "down_proj")

__all__ = [
    "MLP_LORA_TARGET_MODULES",
    "MaskedLoraLinear",
    "build_mlp_lora_config",
    "wrap_pruned_mlp_with_peft_lora",
    "attach_block_masks_to_peft_model",
    "assert_only_lora_trainable",
    "merge_and_verify",
    "resolve_mask_key",
]


class MaskedLoraLinear(PeftLoraLinear):
    """peft LoRA linear whose delta is element-masked by ``element_mask``."""

    def get_delta_weight(self, adapter) -> torch.Tensor:
        delta = super().get_delta_weight(adapter)
        mask = getattr(self, "element_mask", None)
        if mask is None:
            raise RuntimeError(
                "MaskedLoraLinear.element_mask is missing; call "
                "attach_block_masks_to_peft_model before training or merge"
            )
        if mask.shape != delta.shape:
            raise RuntimeError(
                f"element_mask shape {tuple(mask.shape)} != delta shape {tuple(delta.shape)}"
            )
        return delta * mask.to(device=delta.device, dtype=delta.dtype)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is not None:
            raise NotImplementedError(
                "MaskedLoraLinear does not support mixed adapter_names batches"
            )

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            if self.use_dora.get(active_adapter, False):
                raise NotImplementedError(
                    "MaskedLoraLinear does not support DoRA; set use_dora=False"
                )
            lora_A = self.lora_A[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            x_cast = x.to(lora_A.weight.dtype)
            # get_delta_weight already includes scaling and element_mask
            delta = self.get_delta_weight(active_adapter)
            result = result + F.linear(dropout(x_cast), delta)
        return result.to(torch_result_dtype)


def build_mlp_lora_config(
    *,
    r: int,
    lora_alpha: int,
    lora_dropout: float = 0.0,
) -> LoraConfig:
    if int(r) <= 0:
        raise ValueError(f"lora r must be > 0, got {r}")
    if int(lora_alpha) <= 0:
        raise ValueError(f"lora_alpha must be > 0, got {lora_alpha}")
    return LoraConfig(
        r=int(r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=list(MLP_LORA_TARGET_MODULES),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=False,
    )


def resolve_mask_key(module_name: str, mask_keys: set[str]) -> str:
    """Map peft / HF module names onto pruning mask keys like ``model.layers.N.mlp.gate_proj``."""
    if module_name in mask_keys:
        return module_name
    parts = module_name.split(".")
    for i in range(len(parts)):
        candidate = ".".join(parts[i:])
        if candidate in mask_keys:
            return candidate
    raise KeyError(
        f"No pruning mask key for module {module_name!r}. "
        f"Example mask keys: {sorted(mask_keys)[:3]}"
    )


def _iter_peft_lora_linears(model: nn.Module) -> list[tuple[str, PeftLoraLinear]]:
    found: list[tuple[str, PeftLoraLinear]] = []
    for name, module in model.named_modules():
        if isinstance(module, PeftLoraLinear):
            found.append((name, module))
    return found


def attach_block_masks_to_peft_model(
    model: nn.Module,
    masks: dict[str, torch.Tensor],
    block_height: int,
    block_width: int,
) -> int:
    """Promote peft LoRA linears to ``MaskedLoraLinear`` and attach element masks."""
    if not masks:
        raise ValueError("masks is empty")
    mask_keys = set(masks.keys())
    lora_layers = _iter_peft_lora_linears(model)
    if not lora_layers:
        raise RuntimeError("No peft lora.Linear modules found to attach masks")

    attached = 0
    used_keys: set[str] = set()
    for name, module in lora_layers:
        if any(module.use_dora.get(a, False) for a in module.use_dora):
            raise NotImplementedError(
                f"DoRA is not supported for masked LoRA ({name})"
            )
        mask_key = resolve_mask_key(name, mask_keys)
        block_mask = masks[mask_key]
        if block_mask.dtype != torch.bool:
            raise TypeError(
                f"Mask for {mask_key} must be bool, got {block_mask.dtype}"
            )
        base = module.get_base_layer()
        if not isinstance(base, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear base for {name}, got {type(base).__name__}"
            )
        weight = base.weight
        expected = (
            weight.shape[0] // block_height,
            weight.shape[1] // block_width,
        )
        if tuple(block_mask.shape) != expected:
            raise ValueError(
                f"Mask shape {tuple(block_mask.shape)} != expected {expected} "
                f"for {mask_key} (module={name}, weight={tuple(weight.shape)}, "
                f"block={block_height}x{block_width})"
            )
        element_mask = expand_block_mask(block_mask, block_height, block_width)
        element_mask = element_mask.to(device=weight.device, dtype=weight.dtype)

        module.__class__ = MaskedLoraLinear
        module.register_buffer("element_mask", element_mask, persistent=True)
        used_keys.add(mask_key)
        attached += 1

    missing = sorted(mask_keys - used_keys)
    if missing:
        raise RuntimeError(
            f"Pruning masks not attached to any LoRA layer ({len(missing)} keys). "
            f"Examples: {missing[:5]}"
        )
    logger.info(
        "Attached element masks to %d peft LoRA layers (block=%dx%d)",
        attached,
        block_height,
        block_width,
    )
    return attached


def wrap_pruned_mlp_with_peft_lora(
    model: nn.Module,
    masks: dict[str, torch.Tensor],
    *,
    block_height: int,
    block_width: int,
    r: int,
    lora_alpha: int,
    lora_dropout: float = 0.0,
) -> PeftModel:
    """Apply peft LoRA on MLP projections, then attach block masks."""
    # Validate targets exist before peft wrap
    targets = collect_mlp_linears(model, block_height, block_width)
    target_names = {t.module_name for t in targets}
    mask_keys = set(masks.keys())
    if target_names != mask_keys:
        only_targets = sorted(target_names - mask_keys)
        only_masks = sorted(mask_keys - target_names)
        raise RuntimeError(
            "MLP target / mask key mismatch: "
            f"only_in_targets={only_targets[:5]} only_in_masks={only_masks[:5]}"
        )

    config = build_mlp_lora_config(
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )
    peft_model = get_peft_model(model, config)
    attach_block_masks_to_peft_model(
        peft_model, masks, block_height=block_height, block_width=block_width
    )
    return peft_model


def assert_only_lora_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    bad: list[str] = []
    for name, param in model.named_parameters():
        n = int(param.numel())
        total += n
        if not param.requires_grad:
            continue
        trainable += n
        if "lora_A" not in name and "lora_B" not in name:
            bad.append(name)
    if bad:
        raise RuntimeError(
            "Non-LoRA trainable parameters found: " + ", ".join(bad[:20])
        )
    if trainable == 0:
        raise RuntimeError("No trainable LoRA parameters")
    return trainable, total


def merge_and_verify(
    peft_model: PeftModel,
    masks: dict[str, torch.Tensor],
    *,
    block_height: int,
    block_width: int,
) -> nn.Module:
    """Merge masked LoRA into base weights, unload peft, verify pruned blocks stay zero."""
    masked_count = sum(
        1 for _, m in peft_model.named_modules() if isinstance(m, MaskedLoraLinear)
    )
    if masked_count == 0:
        raise RuntimeError(
            "merge_and_verify expected MaskedLoraLinear modules; masks were not attached"
        )

    merged = peft_model.merge_and_unload()
    targets = collect_mlp_linears(merged, block_height, block_width)
    verify_masks_and_weights(
        masks=masks,
        targets=targets,
        block_height=block_height,
        block_width=block_width,
    )
    logger.info(
        "Merged masked LoRA into base; verified %d MLP matrices against block masks",
        len(targets),
    )
    return merged
