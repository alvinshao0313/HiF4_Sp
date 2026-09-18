"""Train the actual folded norm and weight matrices, with fixed activation R64."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.bias import causal_upper_left
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    _rms_norm, _sdpa_attention_forward, qdq_hif4_ste_bf16,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_transforms import apply_r64_no_cross_head
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct
from .transforms import LayerParameters, block_right


def causal_mask(hidden):
    # Batches are right-padded. No valid query can see a padded future key.
    return causal_upper_left(hidden.shape[1], hidden.shape[1])


def quantize(x, use_ste):
    return qdq_hif4_ste_bf16(x) if use_ste else qdq_hif4_direct(x, output_dtype=torch.bfloat16)


class _FloatAccumulatorMM(torch.autograd.Function):
    """BF16 operands, FP32 accumulator output, with the ordinary matmul derivative.

    torch.mm(out_dtype=float32) has no autograd implementation in torch 2.10.
    Retain the accumulator so router weights multiply BEFORE BF16 rounding,
    as in vLLM's fused down projection.
    """
    @staticmethod
    def forward(ctx, a, weight):
        ctx.save_for_backward(a, weight)
        return torch.mm(a, weight.T, out_dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad):
        a, weight = ctx.saved_tensors
        return (grad.float() @ weight.float()).to(a.dtype), (grad.float().T @ a.float()).to(weight.dtype)


def weighted_linear(a, weight, routing_weight):
    if a.is_cuda:
        accumulator = _FloatAccumulatorMM.apply(a, weight)
    else:
        accumulator = F.linear(a.float(), weight.float())
    return (accumulator * routing_weight[:, None]).to(torch.bfloat16)


@dataclass
class Output:
    output: torch.Tensor
    router_logits: torch.Tensor
    pre_moe_norm: torch.Tensor
    expert_counts: torch.Tensor


class Student(nn.Module):
    def __init__(self, base, sharing, *, rms_norm_eps=1e-6):
        super().__init__()
        self.base = base
        self.learned = LayerParameters(base, sharing)
        self.eps = rms_norm_eps
        self.attention_view = SimpleNamespace(
            num_key_value_groups=base.spec.num_attention_heads // base.spec.num_key_value_heads,
            training=False,
        )

    def linear(self, x, name, weight, *, use_ste, head_dim=None, routing_weight=None):
        x_rot = apply_r64_no_cross_head(x.float(), head_dim=head_dim)
        w = block_right(weight, self.learned.matrices[name])
        a_h, w_h = quantize(x_rot, use_ste), quantize(w, use_ste)
        if routing_weight is not None:
            return weighted_linear(a_h, w_h, routing_weight)
        return F.linear(a_h, w_h)

    def attention(self, x, position_embeddings, attention_mask, *, use_ste):
        base = self.base
        shape = (*x.shape[:-1], -1, base.spec.head_dim)
        q, k, v = [self.linear(x, name, base.attention[name], use_ste=use_ste)
                   for name in ("q_proj", "k_proj", "v_proj")]
        q = _rms_norm(q.view(shape), base.q_norm_weight, self.eps).transpose(1, 2)
        k = _rms_norm(k.view(shape), base.k_norm_weight, self.eps).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        out = _sdpa_attention_forward(
            self.attention_view, q, k, v, attention_mask,
            scaling=1.0 / math.sqrt(base.spec.head_dim), dropout=0.0,
        ).reshape(*x.shape[:-1], -1)
        return self.linear(out, "o_proj", base.attention["o_proj"],
                           use_ste=use_ste, head_dim=base.spec.head_dim)

    def router(self, normed):
        weight = block_right(self.base.router_weight, self.learned.matrices["router"])
        return F.linear(normed.to(torch.bfloat16), weight.to(torch.bfloat16))

    def router_aux_logits(self, pre_moe_norm):
        # Stop upstream gradients BEFORE the learnable norm, not after it.
        normed = _rms_norm(pre_moe_norm.detach(), self.learned.moe_norm, self.eps)
        return self.router(normed)

    def forward(self, hidden, *, position_embeddings, attention_mask=None, use_ste=True):
        if attention_mask is None:
            attention_mask = causal_mask(hidden)
        normed = _rms_norm(hidden, self.learned.input_norm, self.eps)
        residual = hidden + self.attention(normed, position_embeddings, attention_mask, use_ste=use_ste)
        normed = _rms_norm(residual, self.learned.moe_norm, self.eps)
        result, logits, counts = self.moe(normed, use_ste=use_ste)
        return Output(residual + result, logits, residual, counts)

    def moe(self, normed, *, use_ste):
        logits = self.router(normed)
        flat = normed.reshape(-1, normed.shape[-1])
        probs = logits.reshape(-1, logits.shape[-1]).float().softmax(-1)
        weights, ids = probs.topk(self.base.spec.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        routed = torch.zeros(flat.shape[0], self.base.spec.top_k, flat.shape[-1],
                             device=flat.device, dtype=flat.dtype)
        counts = torch.bincount(ids.reshape(-1), minlength=self.base.spec.num_experts)
        for index in ids.unique(sorted=True).tolist():
            rows, positions = torch.where(ids == index)
            x = flat[rows]
            expert = self.base.experts[index]
            gate = self.linear(x, f"expert_{index}_gate_proj", expert.gate_proj, use_ste=use_ste)
            up = self.linear(x, f"expert_{index}_up_proj", expert.up_proj, use_ste=use_ste)
            # vLLM silu_and_mul casts SiLU back to BF16 before packed_mul.
            intermediate = F.silu(gate) * up
            down = self.linear(intermediate, f"expert_{index}_down_proj", expert.down_proj,
                               use_ste=use_ste, routing_weight=weights[rows, positions])
            routed[rows, positions] = down
        result = routed.float().sum(1).to(flat.dtype)
        return result.reshape_as(normed), logits, counts
