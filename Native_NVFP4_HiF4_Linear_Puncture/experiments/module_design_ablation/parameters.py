"""Native-initialized parameters and differentiable folds for each ablation."""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from ..diag_gradient.r64_transform import r64_matrix
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import build_moe_diag_state
from ..e2e_diag_reconstruction.core.moe_transforms import (
    apply_r64_no_cross_head,
    fusable_weight_transform_no_h,
    transform_router_weight,
)
from ..non_equivalent_reconstruction.transforms import block_right, projection_weights
from .rotation import CayleyRotation, apply_group_matrix
from .variants import get_variant


class LayerParameters(nn.Module):
    """Only the selected degrees of freedom belong to the optimizer.

    ``base`` must come directly from the native checkpoint loader. In particular,
    this module has no E4 artifact argument or previously learned initializer.
    """

    def __init__(self, base, variant_name: str):
        super().__init__()
        self.variant = get_variant(variant_name)
        if self.variant.is_reference:
            raise ValueError(f"{variant_name} uses its reference model, not a new student")
        self.register_buffer("native_input_norm", base.input_layernorm_weight.detach().float().clone())
        self.register_buffer("native_moe_norm", base.post_attention_layernorm_weight.detach().float().clone())
        self.norms = nn.ParameterDict()
        if self.variant.train_norm:
            self.norms["input"] = nn.Parameter(self.native_input_norm.clone())
            self.norms["moe"] = nn.Parameter(self.native_moe_norm.clone())
        self.diag = (build_moe_diag_state(base.spec, "fusable").to(base.router_weight.device)
                     if self.variant.transform == "diag" else None)
        self.matrices = nn.ParameterDict()
        self.rotations = nn.ModuleDict()
        for name, weight in projection_weights(base):
            if weight.requires_grad:
                raise ValueError(f"native base weight {name} must be frozen")
            if weight.shape[-1] % 64:
                raise ValueError(f"{name} input width must be divisible by 64")
            groups = weight.shape[-1] // 64
            if name == "router":
                if self.variant.train_router:
                    initial = torch.eye(64, device=weight.device).repeat(groups, 1, 1)
                    self.matrices[name] = nn.Parameter(initial)
            elif self.variant.transform == "matrix":
                initial = r64_matrix(device=weight.device).repeat(groups, 1, 1)
                self.matrices[name] = nn.Parameter(initial)
            elif self.variant.transform == "rotation":
                self.rotations[name] = CayleyRotation(groups, device=weight.device)

    @property
    def input_norm(self):
        if self.diag is not None:
            return self.native_input_norm * self.diag.d_qkv()
        return self.norms["input"] if self.variant.train_norm else self.native_input_norm

    @property
    def moe_norm(self):
        if self.diag is not None:
            return self.native_moe_norm * self.diag.d_gu()
        return self.norms["moe"] if self.variant.train_norm else self.native_moe_norm

    def diag_weight(self, name, weight, *, head_dim=None):
        if self.diag is None:
            return weight.float()
        diag = self.diag
        out_scale = torch.ones(weight.shape[0], device=weight.device, dtype=torch.float32)
        if name in {"q_proj", "k_proj", "v_proj"}:
            in_scale = diag.d_qkv()
            if name == "v_proj":
                out_scale = diag.d_vo()
        elif name == "o_proj":
            in_scale = diag.d_vo_expanded()
        else:
            _, index, projection = name.split("_", 2)
            if projection == "down_proj":
                in_scale = diag.d_ud(int(index))
            elif projection in {"gate_proj", "up_proj"}:
                in_scale = diag.d_gu()
                if projection == "up_proj":
                    out_scale = diag.d_ud(int(index))
            else:
                raise ValueError(f"unknown expert projection {name!r}")
        return fusable_weight_transform_no_h(
            weight, in_scale, out_scale, use_r64=False, head_dim=head_dim,
        )

    def projection(self, x, name, weight, *, head_dim=None):
        """Transform both operands before HiF4 QDQ, preserving tied gradients."""
        family = self.variant.transform
        if family == "direct":
            return x.float(), weight.float()
        if family == "rotation":
            q = self.rotations[name]()
            return (apply_group_matrix(x, q, head_dim=head_dim),
                    apply_group_matrix(weight, q, head_dim=head_dim))
        activation = apply_r64_no_cross_head(x.float(), head_dim=head_dim)
        return activation, self.weight(name, weight, head_dim=head_dim)

    def weight(self, name, weight, *, head_dim=None):
        """The exact pre-QDQ weight also used by materialization."""
        family = self.variant.transform
        if name == "router":
            if self.diag is not None:
                return transform_router_weight(weight, self.diag.d_gu())
            if self.variant.train_router:
                return block_right(weight, self.matrices[name])
            return weight.float()
        if family == "direct":
            return weight.float()
        if family == "rotation":
            return apply_group_matrix(weight, self.rotations[name](), head_dim=head_dim)
        if family == "matrix":
            return block_right(weight, self.matrices[name])
        folded = self.diag_weight(name, weight, head_dim=head_dim)
        return apply_r64_no_cross_head(folded, head_dim=head_dim)

    def snapshot(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}


@torch.no_grad()
def transformed_state(base, learned: LayerParameters):
    """Fold learned parameters without mutating native weights or applying QDQ.

    Rotation variants ALSO require their learned activation matrices at runtime;
    this folded weight state alone is not a deployable rotation checkpoint.
    """
    attention = {
        name: learned.weight(name, weight, head_dim=base.spec.head_dim if name == "o_proj" else None).detach()
        for name, weight in base.attention.items()
    }
    experts = [replace(expert, **{
        name: learned.weight(f"expert_{index}_{name}", getattr(expert, name)).detach()
        for name in ("gate_proj", "up_proj", "down_proj")
    }) for index, expert in enumerate(base.experts)]
    return replace(
        base, attention=attention, experts=experts,
        input_layernorm_weight=learned.input_norm.detach().to(torch.bfloat16),
        post_attention_layernorm_weight=learned.moe_norm.detach().to(torch.bfloat16),
        router_weight=learned.weight("router", base.router_weight).detach().to(torch.bfloat16),
    )
