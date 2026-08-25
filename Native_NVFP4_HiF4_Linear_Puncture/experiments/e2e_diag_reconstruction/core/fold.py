"""Fusable fold and online freeze. Transform math comes only from transforms.py."""

from __future__ import annotations

import torch
import torch.nn as nn

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.semantic_hif4 import (
    ALL_PROJS,
    ATTN_PROJS,
    SwitchableNVHiF4Linear,
    enable_eval_weight_cache,
    quant_weight,
    set_layer_runtime_mode,
)


def _linear(layer: nn.Module, proj: str) -> SwitchableNVHiF4Linear:
    parent = layer.self_attn if proj in ATTN_PROJS else layer.mlp
    mod = getattr(parent, proj)
    if not isinstance(mod, SwitchableNVHiF4Linear):
        raise TypeError(f"{proj} is {type(mod)}")
    return mod


class FusedDiagRMSNorm(nn.Module):
    """Bake DIAG into an FP32 fused weight, apply original RMSNorm then FP32 D.

    Forward matches unfolded Linear input: ``float(RMSNorm(x)) * D``.
    ``self.weight`` is the fused ``w * D`` used for audit / later export.
    """

    def __init__(self, base: nn.Module, scale: torch.Tensor) -> None:
        super().__init__()
        if not hasattr(base, "weight"):
            raise TypeError(f"{type(base)} has no weight to fold DIAG into")
        d = scale.detach().to(device=base.weight.device, dtype=torch.float32).reshape(-1)
        if int(d.numel()) != int(base.weight.numel()):
            raise ValueError(
                f"DIAG length {int(d.numel())} != RMSNorm weight {int(base.weight.numel())}"
            )
        self.base = base
        self.register_buffer("diag", d.contiguous())
        self.weight = nn.Parameter((base.weight.detach().float() * d).contiguous(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x).float() * self.diag


def _fuse_norm(parent: nn.Module, name: str, scale: torch.Tensor) -> None:
    orig = getattr(parent, name)
    setattr(parent, name, FusedDiagRMSNorm(orig, scale))


def fold_fusable_layer_inplace(layer: nn.Module, diag_state, use_r64: bool) -> None:
    q = _linear(layer, "q_proj")
    if bool(use_r64) != bool(q.use_r64):
        raise ValueError("fold use_r64 does not match the layer wrappers")
    _fuse_norm(layer, "input_layernorm", diag_state.d_qkv())
    _fuse_norm(layer, "post_attention_layernorm", diag_state.d_gu())

    for proj in ALL_PROJS:
        linear = _linear(layer, proj)
        w_final = linear.transformed_master_weight()
        linear.replace_master_weight_(w_final)
        bias_fp32 = None if linear.bias is None else linear.bias.float()
        if proj == "v_proj" and bias_fp32 is not None:
            bias_fp32 = bias_fp32 * diag_state.d_vo().to(bias_fp32.device)
            with torch.no_grad():
                linear.bias.copy_(bias_fp32.to(dtype=linear.bias.dtype))
        if proj == "up_proj" and bias_fp32 is not None:
            bias_fp32 = bias_fp32 * diag_state.d_ud().to(bias_fp32.device)
            with torch.no_grad():
                linear.bias.copy_(bias_fp32.to(dtype=linear.bias.dtype))
        linear.set_folded_bias_fp32(bias_fp32)
        linear.set_mode("folded")
        linear.enable_weight_cache()


def freeze_online_layer_for_eval(
    layer: nn.Module,
    diag_state,
    use_r64: bool,
    rot_order: str,
) -> None:
    del diag_state, use_r64, rot_order
    set_layer_runtime_mode(layer, "hif4_eval")
    enable_eval_weight_cache(layer)
    for proj in ALL_PROJS:
        linear = _linear(layer, proj)
        with torch.no_grad():
            w_t = linear.transformed_master_weight()
            linear._weight_qdq_calls += 1
            linear._cached_w_h = quant_weight(w_t, use_ste=False).detach()
