from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from Block_Sparse.dynamic_input_sparse.common import flatten_tokens, ratio_to_keep_count
from Block_Sparse.dynamic_input_sparse.config import (
    DynamicInputMaskMethod,
    DynamicInputSparseConfig,
)
from Block_Sparse.dynamic_input_sparse.m1_oracle import (
    predict_m1_full_output_mask,
    predict_m1_full_output_mask_multiweight,
)
from Block_Sparse.dynamic_input_sparse.m8_energy import (
    compute_all_output_weight_energy,
    predict_m8_input_mask,
)
from Block_Sparse.dynamic_input_sparse.masked_linear import (
    apply_input_block_mask,
    masked_linear_reference,
)


class DynamicInputSparseMLPReference(nn.Module):
    """HF MLP wrapper: shared gate/up MX, down MX from sparse intermediate."""

    def __init__(
        self,
        mlp: nn.Module,
        config: DynamicInputSparseConfig,
        *,
        capture_masks: bool = False,
    ) -> None:
        super().__init__()
        if not (
            hasattr(mlp, "gate_proj")
            and hasattr(mlp, "up_proj")
            and hasattr(mlp, "down_proj")
        ):
            raise TypeError("mlp must expose gate_proj/up_proj/down_proj")
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn = getattr(mlp, "act_fn", None)
        if self.act_fn is None:
            raise TypeError("mlp must expose act_fn")
        self.config = config
        self.capture_masks = bool(capture_masks)
        self.last_mx_gate_up: torch.Tensor | None = None
        self.last_mx_down: torch.Tensor | None = None
        self.last_down_input: torch.Tensor | None = None

        self._g_gate_up: torch.Tensor | None = None
        self._g_down: torch.Tensor | None = None
        if config.method == DynamicInputMaskMethod.M8_ENERGY:
            g_gate = compute_all_output_weight_energy(
                self.gate_proj.weight,
                k_block_size=config.k_block_size,
                output_block_size=config.output_energy_block_size,
            )
            g_up = compute_all_output_weight_energy(
                self.up_proj.weight,
                k_block_size=config.k_block_size,
                output_block_size=config.output_energy_block_size,
            )
            self._g_gate_up = (g_gate + g_up).contiguous()
            self._g_down = compute_all_output_weight_energy(
                self.down_proj.weight,
                k_block_size=config.k_block_size,
                output_block_size=config.output_energy_block_size,
            )

    def _predict_gate_up(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if cfg.method == DynamicInputMaskMethod.NONE or float(cfg.keep_ratio) == 1.0:
            x_flat, _ = flatten_tokens(x)
            kb = int(x_flat.shape[-1]) // cfg.k_block_size
            return torch.ones(
                x_flat.shape[0], kb, dtype=torch.bool, device=x_flat.device
            )
        if cfg.method == DynamicInputMaskMethod.M8_ENERGY:
            assert self._g_gate_up is not None
            return predict_m8_input_mask(
                x,
                self._g_gate_up.to(device=x.device),
                cfg.keep_ratio,
                k_block_size=cfg.k_block_size,
            )
        if cfg.method == DynamicInputMaskMethod.M1_ORACLE:
            return predict_m1_full_output_mask_multiweight(
                x,
                [self.gate_proj.weight, self.up_proj.weight],
                cfg.keep_ratio,
                token_chunk_size=cfg.m1_token_chunk_size,
                k_block_size=cfg.k_block_size,
            )
        raise ValueError(f"unknown method {cfg.method}")

    def _predict_down(self, h: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if cfg.method == DynamicInputMaskMethod.NONE or float(cfg.keep_ratio) == 1.0:
            h_flat, _ = flatten_tokens(h)
            kb = int(h_flat.shape[-1]) // cfg.k_block_size
            return torch.ones(
                h_flat.shape[0], kb, dtype=torch.bool, device=h_flat.device
            )
        if cfg.method == DynamicInputMaskMethod.M8_ENERGY:
            assert self._g_down is not None
            return predict_m8_input_mask(
                h,
                self._g_down.to(device=h.device),
                cfg.keep_ratio,
                k_block_size=cfg.k_block_size,
            )
        if cfg.method == DynamicInputMaskMethod.M1_ORACLE:
            return predict_m1_full_output_mask(
                h,
                self.down_proj.weight,
                cfg.keep_ratio,
                token_chunk_size=cfg.m1_token_chunk_size,
                k_block_size=cfg.k_block_size,
            )
        raise ValueError(f"unknown method {cfg.method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mx_gate_up = self._predict_gate_up(x)
        x_masked = apply_input_block_mask(
            x, mx_gate_up, block_size=self.config.k_block_size
        )
        y_gate = self.gate_proj(x_masked)
        y_up = self.up_proj(x_masked)
        h = self.act_fn(y_gate) * y_up
        mx_down = self._predict_down(h)
        if self.capture_masks:
            self.last_mx_gate_up = mx_gate_up.detach()
            self.last_mx_down = mx_down.detach()
            self.last_down_input = h.detach()
        return masked_linear_reference(
            h,
            self.down_proj.weight,
            self.down_proj.bias,
            mx_down,
            block_size=self.config.k_block_size,
        )


def install_dynamic_input_sparse_on_hf_model(
    model: nn.Module,
    config: DynamicInputSparseConfig,
    *,
    capture_masks: bool = False,
) -> list[str]:
    """Replace each text MLP with DynamicInputSparseMLPReference. Returns prefixes."""
    if config.method == DynamicInputMaskMethod.NONE:
        return []

    replaced: list[str] = []

    def _walk(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name
            if (
                hasattr(child, "gate_proj")
                and hasattr(child, "up_proj")
                and hasattr(child, "down_proj")
                and hasattr(child, "act_fn")
                and "mlp" in name
            ):
                wrapped = DynamicInputSparseMLPReference(
                    child, config, capture_masks=capture_masks
                )
                setattr(module, name, wrapped)
                replaced.append(child_prefix)
            else:
                _walk(child, child_prefix)

    _walk(model)
    if not replaced:
        raise RuntimeError("no MLP modules were wrapped for dynamic input sparsity")
    return replaced


def expected_keep_counts(config: DynamicInputSparseConfig) -> dict[str, int]:
    """Qwen3.5-4B geometry helper for smoke checks."""
    return {
        "gate_up_kb": 2560 // config.k_block_size,
        "down_kb": 9216 // config.k_block_size,
        "gate_up_keep": ratio_to_keep_count(config.keep_ratio, 2560 // config.k_block_size),
        "down_keep": ratio_to_keep_count(config.keep_ratio, 9216 // config.k_block_size),
    }


def assert_weights_unchanged(
    before: dict[str, torch.Tensor], modules: list[nn.Module]
) -> None:
    after: dict[str, torch.Tensor] = {}
    for i, m in enumerate(modules):
        for n, p in m.named_parameters():
            after[f"{i}.{n}"] = p.detach().float().cpu()
    for k, v in before.items():
        if k not in after:
            raise AssertionError(f"missing weight {k} after forward")
        if not torch.equal(v, after[k]):
            raise AssertionError(f"weight changed: {k}")
