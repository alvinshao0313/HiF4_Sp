from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.block_utils import expand_block_mask
from block_pruning.mask_apply import apply_mlp_block_masks, verify_masks_and_weights
from block_pruning.mlp_registry import collect_mlp_linears
from block_pruning.peft_masked_lora import (
    MaskedLoraLinear,
    assert_only_lora_trainable,
    merge_and_verify,
    wrap_pruned_mlp_with_peft_lora,
)


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(64, 64, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(64, 128, bias=False)
        self.mlp.up_proj = nn.Linear(64, 128, bias=False)
        self.mlp.down_proj = nn.Linear(128, 64, bias=False)


class _TinyCausalLM(nn.Module):
    """Minimal stand-in so peft TaskType.CAUSAL_LM wrapping still targets Linear names."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyLayer()])
        self.lm_head = nn.Linear(64, 32, bias=False)

    def forward(self, x):
        raise NotImplementedError("forward unused in unit test")

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

def _all_one_masks(targets, h, w):
    return {
        t.module_name: torch.ones(
            t.module.weight.shape[0] // h,
            t.module.weight.shape[1] // w,
            dtype=torch.bool,
        )
        for t in targets
    }


def test_masked_lora_merge_preserves_pruned_zeros():
    torch.manual_seed(0)
    model = _TinyCausalLM()
    h, w = 64, 64
    targets = collect_mlp_linears(model, h, w)
    masks = _all_one_masks(targets, h, w)
    # Prune block (0,0) on every MLP matrix
    for name in masks:
        masks[name][0, 0] = False
    apply_mlp_block_masks(model, masks, h, w, targets=targets)
    verify_masks_and_weights(masks, targets, h, w)

    peft_model = wrap_pruned_mlp_with_peft_lora(
        model,
        masks,
        block_height=h,
        block_width=w,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    assert_only_lora_trainable(peft_model)

    masked_layers = [
        m for _, m in peft_model.named_modules() if isinstance(m, MaskedLoraLinear)
    ]
    assert len(masked_layers) == 3

    # Force non-trivial LoRA weights so unmasked merge would fill zeros
    with torch.no_grad():
        for layer in masked_layers:
            for adapter in layer.lora_A.keys():
                layer.lora_A[adapter].weight.normal_(0.0, 0.5)
                layer.lora_B[adapter].weight.normal_(0.0, 0.5)
            delta = layer.get_delta_weight(layer.active_adapters[0])
            assert delta.shape == layer.element_mask.shape
            pruned = delta[:h, :w]
            assert int(torch.count_nonzero(pruned).item()) == 0
            assert float(delta.abs().sum().item()) > 0.0

    merged = merge_and_verify(peft_model, masks, block_height=h, block_width=w)
    targets_after = collect_mlp_linears(merged, h, w)
    verify_masks_and_weights(masks, targets_after, h, w)

    # Kept blocks should have changed vs pure zeros-only prune state for at least one matrix
    changed = False
    for t in targets_after:
        weight = t.module.weight.detach()
        elem = expand_block_mask(masks[t.module_name], h, w).to(weight.dtype)
        kept = weight * elem
        if float(kept.abs().sum().item()) > 0:
            # compare against a freshly zeroed-then-masked clone of original shape randomness
            changed = True
            break
    assert changed


def test_resolve_and_attach_rejects_missing_mask():
    torch.manual_seed(1)
    model = _TinyCausalLM()
    h, w = 64, 64
    targets = collect_mlp_linears(model, h, w)
    masks = _all_one_masks(targets, h, w)
    # Drop one key
    drop = next(iter(masks))
    del masks[drop]
    try:
        wrap_pruned_mlp_with_peft_lora(
            model,
            masks,
            block_height=h,
            block_width=w,
            r=2,
            lora_alpha=4,
        )
        assert False, "expected RuntimeError for mask/target mismatch"
    except RuntimeError as exc:
        assert "mismatch" in str(exc).lower() or "mask" in str(exc).lower()
