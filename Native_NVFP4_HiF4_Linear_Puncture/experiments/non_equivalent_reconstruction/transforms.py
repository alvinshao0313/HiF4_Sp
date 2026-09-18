"""Right-multiply input G64 blocks; the online activation R64 stays fixed."""
from __future__ import annotations

from dataclasses import replace
import torch
from torch import nn

from Native_NVFP4_HiF4_Linear_Puncture.experiments.diag_gradient.r64_transform import r64_matrix
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_fold import fold_fusable_moe_layer_state
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import build_moe_diag_state


def block_right(weight: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or weight.shape[-1] % 64:
        raise ValueError("weight must be [out,in] with in divisible by 64")
    groups = weight.shape[-1] // 64
    if tuple(matrix.shape) not in {(64, 64), (groups, 64, 64)}:
        raise ValueError(f"invalid block matrix shape {tuple(matrix.shape)} for {groups} groups")
    w = weight.float().reshape(weight.shape[0], groups, 64)
    if matrix.ndim == 2:
        result = w @ matrix.float()
    else:
        result = torch.einsum("ogk,gkj->ogj", w, matrix.float())
    return result.reshape_as(weight)


def fold_initial_diag(state, z: dict[str, torch.Tensor]):
    diag = build_moe_diag_state(state.spec, "fusable").to(state.router_weight.device)
    diag.load_snapshot(z)
    # This folds D_out W D_in^-1, not R64 and not HiF4 quantization.
    with torch.no_grad():
        return fold_fusable_moe_layer_state(state, diag, use_r64=False)


def projection_weights(state):
    yield from state.attention.items()
    for index, expert in enumerate(state.experts):
        for name in ("gate_proj", "up_proj", "down_proj"):
            yield f"expert_{index}_{name}", getattr(expert, name)
    yield "router", state.router_weight


class LayerParameters(nn.Module):
    def __init__(self, base, sharing: str):
        super().__init__()
        if sharing not in {"linear", "group"}:
            raise ValueError(sharing)
        self.input_norm = nn.Parameter(base.input_layernorm_weight.detach().float().clone())
        self.moe_norm = nn.Parameter(base.post_attention_layernorm_weight.detach().float().clone())
        matrices = {}
        for name, weight in projection_weights(base):
            if weight.shape[-1] % 64:
                raise ValueError(f"{name} input width must be divisible by 64")
            initial = (torch.eye(64, device=weight.device) if name == "router"
                       else r64_matrix(device=weight.device))
            if sharing == "group":
                initial = initial.unsqueeze(0).repeat(weight.shape[-1] // 64, 1, 1)
            matrices[name] = nn.Parameter(initial.clone())
        self.matrices = nn.ParameterDict(matrices)

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {name: p.detach().cpu().clone() for name, p in self.state_dict().items()}


def transformed_state(base, params: LayerParameters):
    """Float master state shared by numerical verification and materialization."""
    attention = {name: block_right(w, params.matrices[name]).detach()
                 for name, w in base.attention.items()}
    experts = [replace(expert, **{
        name: block_right(getattr(expert, name), params.matrices[f"expert_{i}_{name}"]).detach()
        for name in ("gate_proj", "up_proj", "down_proj")
    }) for i, expert in enumerate(base.experts)]
    return replace(base, attention=attention, experts=experts,
                   input_layernorm_weight=params.input_norm.detach().to(torch.bfloat16),
                   post_attention_layernorm_weight=params.moe_norm.detach().to(torch.bfloat16),
                   router_weight=block_right(base.router_weight, params.matrices["router"]).detach().to(torch.bfloat16))
