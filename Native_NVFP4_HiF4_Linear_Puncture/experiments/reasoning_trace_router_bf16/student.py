"""Run B's exported QDQ values directly; only its real BF16 gate is trainable."""
from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file

from ..e2e_diag_reconstruction.core.moe_model_spec import Qwen3MoeModelSpec
from ..e2e_diag_reconstruction.core.moe_transforms import apply_r64_no_cross_head
from ..non_equivalent_reconstruction.student import Student, quantize, weighted_linear
from .artifacts import read_json, state_digest
from .config import NATIVE


def load_layer(model, layer, device):
    model = Path(model)
    prefix = f"model.layers.{layer}."
    index = read_json(model / "model.safetensors.index.json")["weight_map"]
    wanted = {k: v for k, v in index.items() if k.startswith(prefix)}
    tensors = {}
    for shard in sorted(set(wanted.values())):
        values = load_file(str(model / shard), device=str(device))
        tensors.update({k: v for k, v in values.items() if k in wanted})
    if set(tensors) != set(wanted) or any(t.dtype != torch.bfloat16 or t.requires_grad for t in tensors.values()):
        raise ValueError("B layer must contain frozen BF16 exported tensors")
    get = lambda suffix: tensors[prefix + suffix]
    spec = Qwen3MoeModelSpec.from_snapshot(NATIVE)
    return SimpleNamespace(
        layer_idx=layer, spec=spec, tensors=tensors,
        input_layernorm_weight=get("input_layernorm.weight"),
        post_attention_layernorm_weight=get("post_attention_layernorm.weight"),
        q_norm_weight=get("self_attn.q_norm.weight"), k_norm_weight=get("self_attn.k_norm.weight"),
        router_weight=get("mlp.gate.weight"),
        attention={n: get(f"self_attn.{n}.weight") for n in ("q_proj", "k_proj", "v_proj", "o_proj")},
        experts=[SimpleNamespace(**{n: get(f"mlp.experts.{i}.{n}.weight")
                                   for n in ("gate_proj", "up_proj", "down_proj")})
                 for i in range(spec.num_experts)],
    )


class FrozenNorms(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.register_buffer("input_norm", base.input_layernorm_weight)
        self.register_buffer("moe_norm", base.post_attention_layernorm_weight)


class RouterStudent(Student):
    def __init__(self, base):
        nn.Module.__init__(self)
        self.base, self.eps = base, 1e-6
        self.learned = FrozenNorms(base)
        self.attention_view = SimpleNamespace(
            num_key_value_groups=base.spec.num_attention_heads // base.spec.num_key_value_heads, training=False)
        self.router_weight = nn.Parameter(base.router_weight.detach().clone())
        self.check_trainable()

    def check_trainable(self):
        if ([n for n, p in self.named_parameters() if p.requires_grad] != ["router_weight"]
                or self.router_weight.dtype != torch.bfloat16):
            raise RuntimeError("only the real BF16 router weight may be optimized")

    def linear(self, x, name, weight, *, use_ste, head_dim=None, routing_weight=None):
        # Weight-side R64/matrices and HiF4 QDQ are already in the export.
        activation = quantize(apply_r64_no_cross_head(x.float(), head_dim=head_dim), False)
        return (weighted_linear(activation, weight, routing_weight) if routing_weight is not None
                else F.linear(activation, weight))

    def router(self, normed):
        return F.linear(normed.to(torch.bfloat16), self.router_weight)

    def frozen_digest(self):
        return state_digest({**self.base.tensors, **dict(self.learned.named_buffers())})
