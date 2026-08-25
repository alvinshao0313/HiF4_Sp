"""Qwen3-MoE layer semantic runtime for native NVFP4 and HiF4 reconstruction.

This module intentionally has no H16 path.  Routing and RoPE use the installed
Transformers Qwen3-MoE implementations; only the replaceable quantized linear
operations are supplied by the experiment runtime.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeTopKRouter,
    apply_rotary_pos_emb,
    repeat_kv,
)

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    MoELayerMasterState,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_transforms import (
    apply_r64_no_cross_head,
    expand_vo_scale_qwen3_moe,
    fusable_weight_transform_no_h,
    online_weight_transform_no_h,
    transform_router_weight,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct


def qdq_native_nvfp4(x: torch.Tensor, input_global_scale_inv: torch.Tensor) -> torch.Tensor:
    """Native ModelOpt activation QDQ using the vLLM emulation definition."""
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        ref_nvfp4_quant_dequant,
    )

    original_shape = x.shape
    x2d = x.reshape(-1, original_shape[-1])
    qdq = ref_nvfp4_quant_dequant(
        x2d, input_global_scale_inv.to(device=x.device, dtype=torch.float32), block_size=16
    )
    return qdq.reshape(original_shape)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = x.to(torch.float32)
    normed = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return normed.to(dtype) * weight.to(device=x.device, dtype=dtype)


def native_nvfp4_linear(x: torch.Tensor, weight_fp32: torch.Tensor, input_global_scale_inv: torch.Tensor) -> torch.Tensor:
    x_qdq = qdq_native_nvfp4(x, input_global_scale_inv)
    return F.linear(x_qdq, weight_fp32.to(device=x.device, dtype=x_qdq.dtype))


def qdq_hif4_ste_bf16(x: torch.Tensor) -> torch.Tensor:
    xf = x.to(torch.float32)
    with torch.no_grad():
        recon = qdq_hif4_direct(xf.detach(), output_dtype=torch.bfloat16).to(torch.float32)
    y = xf + (recon - xf).detach()
    return y.to(torch.bfloat16)


def _scale_from_z(z: torch.Tensor) -> torch.Tensor:
    return torch.exp2(z.to(torch.float32))


class MoEFusableDiagState(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 2048,
        num_experts: int = 128,
        moe_intermediate_size: int = 768,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        self.diag_mode = "fusable"
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.z_qkv = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.z_vo = nn.Parameter(torch.zeros(num_key_value_heads * head_dim, dtype=torch.float32))
        self.z_gu = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.z_ud = nn.Parameter(torch.zeros(num_experts, moe_intermediate_size, dtype=torch.float32))

    def d_qkv(self) -> torch.Tensor:
        return _scale_from_z(self.z_qkv)

    def d_vo(self) -> torch.Tensor:
        return _scale_from_z(self.z_vo)

    def d_vo_expanded(self) -> torch.Tensor:
        return expand_vo_scale_qwen3_moe(self.d_vo())

    def d_gu(self) -> torch.Tensor:
        return _scale_from_z(self.z_gu)

    def d_ud(self, expert_idx: int) -> torch.Tensor:
        return _scale_from_z(self.z_ud[int(expert_idx)])

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters()}

    def load_snapshot(self, snapshot: dict[str, torch.Tensor]) -> None:
        params = dict(self.named_parameters())
        if set(snapshot) != set(params):
            raise ValueError(f"snapshot keys {sorted(snapshot)} != {sorted(params)}")
        with torch.no_grad():
            for name, value in snapshot.items():
                params[name].copy_(value.to(device=params[name].device, dtype=torch.float32))

    def clamp_log2_(self, bounds: tuple[float, float] = (-4.0, 4.0)) -> None:
        lo, hi = bounds
        with torch.no_grad():
            for p in self.parameters():
                p.clamp_(lo, hi)


class MoEOnlineDiagState(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 2048,
        num_experts: int = 128,
        moe_intermediate_size: int = 768,
        o_input_size: int = 4096,
    ) -> None:
        super().__init__()
        self.diag_mode = "online"
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.o_input_size = int(o_input_size)
        self.z_q = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.z_k = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.z_v = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.z_o = nn.Parameter(torch.zeros(o_input_size, dtype=torch.float32))
        self.z_gate = nn.Parameter(torch.zeros(num_experts, hidden_size, dtype=torch.float32))
        self.z_up = nn.Parameter(torch.zeros(num_experts, hidden_size, dtype=torch.float32))
        self.z_down = nn.Parameter(torch.zeros(num_experts, moe_intermediate_size, dtype=torch.float32))

    def d_for(self, proj: str, expert_idx: int | None = None) -> torch.Tensor:
        if proj in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            return _scale_from_z(getattr(self, {"q_proj": "z_q", "k_proj": "z_k", "v_proj": "z_v", "o_proj": "z_o"}[proj]))
        if expert_idx is None:
            raise ValueError(f"{proj} requires expert_idx")
        name = {"gate_proj": "z_gate", "up_proj": "z_up", "down_proj": "z_down"}[proj]
        return _scale_from_z(getattr(self, name)[int(expert_idx)])

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters()}

    def load_snapshot(self, snapshot: dict[str, torch.Tensor]) -> None:
        params = dict(self.named_parameters())
        if set(snapshot) != set(params):
            raise ValueError(f"snapshot keys {sorted(snapshot)} != {sorted(params)}")
        with torch.no_grad():
            for name, value in snapshot.items():
                params[name].copy_(value.to(device=params[name].device, dtype=torch.float32))

    def clamp_log2_(self, bounds: tuple[float, float] = (-4.0, 4.0)) -> None:
        lo, hi = bounds
        with torch.no_grad():
            for p in self.parameters():
                p.clamp_(lo, hi)


def build_moe_diag_state(spec, diag_mode: str) -> nn.Module:
    if diag_mode == "fusable":
        return MoEFusableDiagState(
            hidden_size=spec.hidden_size,
            num_experts=spec.num_experts,
            moe_intermediate_size=spec.moe_intermediate_size,
            num_key_value_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim,
        )
    if diag_mode == "online":
        return MoEOnlineDiagState(
            hidden_size=spec.hidden_size,
            num_experts=spec.num_experts,
            moe_intermediate_size=spec.moe_intermediate_size,
            o_input_size=spec.num_attention_heads * spec.head_dim,
        )
    raise ValueError(f"invalid diag_mode={diag_mode!r}")


@dataclass
class StudentStepCache:
    transformed_weight_qdq: dict[tuple[str, int | None], torch.Tensor]
    weight_qdq_calls_by_proj: dict[str, int]

    @classmethod
    def new(cls) -> "StudentStepCache":
        return cls(transformed_weight_qdq={}, weight_qdq_calls_by_proj={})

    def clear(self) -> None:
        self.transformed_weight_qdq.clear()

    def cached_weight(self, key: tuple[str, int | None], build) -> torch.Tensor:
        if key not in self.transformed_weight_qdq:
            self.transformed_weight_qdq[key] = build()
            proj = key[0]
            self.weight_qdq_calls_by_proj[proj] = self.weight_qdq_calls_by_proj.get(proj, 0) + 1
        return self.transformed_weight_qdq[key]


def _student_activation(
    x: torch.Tensor,
    d: torch.Tensor,
    *,
    use_r64: bool,
    rot_order: str,
    head_dim: int | None,
) -> torch.Tensor:
    xf = x.to(torch.float32)
    if rot_order == "diag_then_rot":
        y = xf * d.to(device=xf.device)
        return apply_r64_no_cross_head(y, head_dim=head_dim) if use_r64 else y
    if rot_order == "rot_then_diag":
        y = apply_r64_no_cross_head(xf, head_dim=head_dim) if use_r64 else xf
        return y * d.to(device=xf.device)
    raise ValueError(f"invalid rot_order={rot_order!r}")


def _attention_d_in_out(proj: str, diag_state: nn.Module, out_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(diag_state, MoEFusableDiagState):
        if proj in {"q_proj", "k_proj"}:
            return diag_state.d_qkv(), torch.ones(out_features, dtype=torch.float32, device=diag_state.z_qkv.device)
        if proj == "v_proj":
            return diag_state.d_qkv(), diag_state.d_vo()
        if proj == "o_proj":
            return diag_state.d_vo_expanded(), torch.ones(out_features, dtype=torch.float32, device=diag_state.z_qkv.device)
    raise ValueError(f"invalid fusable attention proj={proj!r}")


def forward_student_attention_proj(
    proj: str,
    x: torch.Tensor,
    master_weight: torch.Tensor,
    diag_state: nn.Module,
    *,
    use_r64: bool,
    rot_order: str,
    step_cache: StudentStepCache,
    use_ste: bool = True,
    head_dim: int | None = None,
) -> torch.Tensor:
    if isinstance(diag_state, MoEFusableDiagState):
        d_in, d_out = _attention_d_in_out(proj, diag_state, int(master_weight.shape[0]))
        act_d = torch.ones_like(d_in) if proj == "o_proj" else d_in
        x_t = _student_activation(x, act_d, use_r64=use_r64, rot_order="diag_then_rot", head_dim=head_dim)
        w_t = fusable_weight_transform_no_h(master_weight, d_in, d_out, use_r64=use_r64, head_dim=head_dim)
    elif isinstance(diag_state, MoEOnlineDiagState):
        d = diag_state.d_for(proj)
        x_t = _student_activation(x, d, use_r64=use_r64, rot_order=rot_order, head_dim=head_dim)
        w_t = online_weight_transform_no_h(master_weight, d, use_r64=use_r64, rot_order=rot_order, head_dim=head_dim)
    else:
        raise TypeError(f"unsupported diag_state={type(diag_state)!r}")
    a_h = qdq_hif4_ste_bf16(x_t) if use_ste else qdq_hif4_direct(x_t, output_dtype=torch.bfloat16)
    w_h = step_cache.cached_weight((proj, None), lambda: qdq_hif4_ste_bf16(w_t) if use_ste else qdq_hif4_direct(w_t, output_dtype=torch.bfloat16))
    return F.linear(a_h.to(dtype=w_h.dtype), w_h)


def _expert_d_in_out(
    proj: str,
    expert_idx: int,
    diag_state: MoEFusableDiagState,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones(out_features, dtype=torch.float32, device=diag_state.z_qkv.device)
    if proj == "gate_proj":
        return diag_state.d_gu(), ones
    if proj == "up_proj":
        return diag_state.d_gu(), diag_state.d_ud(expert_idx)
    if proj == "down_proj":
        return diag_state.d_ud(expert_idx), ones
    raise ValueError(proj)


def forward_student_expert_proj(
    proj: str,
    expert_idx: int,
    x: torch.Tensor,
    master_weight: torch.Tensor,
    diag_state: nn.Module,
    *,
    use_r64: bool,
    rot_order: str,
    step_cache: StudentStepCache,
    use_ste: bool = True,
) -> torch.Tensor:
    if isinstance(diag_state, MoEFusableDiagState):
        d_in, d_out = _expert_d_in_out(proj, expert_idx, diag_state, int(master_weight.shape[0]))
        act_d = torch.ones_like(d_in) if proj == "down_proj" else d_in
        x_t = _student_activation(x, act_d, use_r64=use_r64, rot_order="diag_then_rot", head_dim=None)
        w_t = fusable_weight_transform_no_h(master_weight, d_in, d_out, use_r64=use_r64)
    elif isinstance(diag_state, MoEOnlineDiagState):
        d = diag_state.d_for(proj, expert_idx)
        x_t = _student_activation(x, d, use_r64=use_r64, rot_order=rot_order, head_dim=None)
        w_t = online_weight_transform_no_h(master_weight, d, use_r64=use_r64, rot_order=rot_order)
    else:
        raise TypeError(f"unsupported diag_state={type(diag_state)!r}")
    a_h = qdq_hif4_ste_bf16(x_t) if use_ste else qdq_hif4_direct(x_t, output_dtype=torch.bfloat16)
    w_h = step_cache.cached_weight((proj, int(expert_idx)), lambda: qdq_hif4_ste_bf16(w_t) if use_ste else qdq_hif4_direct(w_t, output_dtype=torch.bfloat16))
    return F.linear(a_h.to(dtype=w_h.dtype), w_h)


@dataclass
class StudentMoEOutput:
    output: torch.Tensor
    router_logits: torch.Tensor
    routing_weights: torch.Tensor
    selected_experts: torch.Tensor
    per_expert_routed_token_count: torch.Tensor
    router_input: torch.Tensor | None = None


def forward_student_routed_moe(
    x: torch.Tensor,
    state: MoELayerMasterState,
    diag_state: nn.Module,
    *,
    use_r64: bool,
    rot_order: str,
    step_cache: StudentStepCache,
    use_ste: bool = True,
) -> StudentMoEOutput:
    flat = x.reshape(-1, x.shape[-1])
    router_weight = state.router_weight
    if isinstance(diag_state, MoEFusableDiagState):
        router_weight = transform_router_weight(router_weight, diag_state.d_gu()).to(device=flat.device, dtype=router_weight.dtype)
        router_input = flat * diag_state.d_gu().to(device=flat.device, dtype=flat.dtype)
    else:
        router_input = flat
    router_weight = router_weight.to(device=flat.device)
    router_logits = F.linear(router_input.to(dtype=router_weight.dtype), router_weight)
    probs = torch.softmax(router_logits, dtype=torch.float32, dim=-1)
    routing_weights, selected_experts = torch.topk(probs, state.spec.top_k, dim=-1)
    if state.spec.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(router_logits.dtype)
    output = torch.zeros_like(flat)
    counts = torch.zeros(state.spec.num_experts, dtype=torch.long, device=flat.device)
    for expert_idx in selected_experts.unique(sorted=True).tolist():
        expert = state.experts[int(expert_idx)]
        token_idx, topk_pos = torch.where(selected_experts == expert_idx)
        counts[int(expert_idx)] = token_idx.numel()
        current = flat[token_idx]
        gate = forward_student_expert_proj(
            "gate_proj", int(expert_idx), current, expert.gate_proj, diag_state,
            use_r64=use_r64, rot_order=rot_order, step_cache=step_cache, use_ste=use_ste,
        )
        up = forward_student_expert_proj(
            "up_proj", int(expert_idx), current, expert.up_proj, diag_state,
            use_r64=use_r64, rot_order=rot_order, step_cache=step_cache, use_ste=use_ste,
        )
        hidden = F.silu(gate) * up
        down = forward_student_expert_proj(
            "down_proj", int(expert_idx), hidden, expert.down_proj, diag_state,
            use_r64=use_r64, rot_order=rot_order, step_cache=step_cache, use_ste=use_ste,
        )
        output.index_add_(0, token_idx, down.to(output.dtype) * routing_weights[token_idx, topk_pos, None])
    return StudentMoEOutput(
        output.reshape_as(x),
        router_logits,
        routing_weights,
        selected_experts,
        counts,
        router_input=x,
    )


@dataclass
class NativeMoEForward:
    output: torch.Tensor
    router_logits: torch.Tensor
    routing_weights: torch.Tensor
    selected_experts: torch.Tensor


@dataclass
class NativeAttentionProjections:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o_input: torch.Tensor
    o: torch.Tensor


def _sdpa_attention_forward(
    module: SimpleNamespace,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    dropout: float = 0.0,
) -> torch.Tensor:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_output = F.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=dropout if module.training else 0.0,
        is_causal=False,
        scale=scaling,
    )
    return attn_output.transpose(1, 2).contiguous()


class NativeQwen3MoELayerRuntime(nn.Module):
    """One lazy-materialized native layer with vLLM-aligned NVFP4 QDQ/GEMM."""

    def __init__(self, state: MoELayerMasterState, *, rms_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.state = state
        self.rms_norm_eps = float(rms_norm_eps)
        config = Qwen3MoeConfig(
            hidden_size=state.spec.hidden_size,
            num_hidden_layers=state.spec.num_layers,
            num_attention_heads=state.spec.num_attention_heads,
            num_key_value_heads=state.spec.num_key_value_heads,
            head_dim=state.spec.head_dim,
            num_experts=state.spec.num_experts,
            num_experts_per_tok=state.spec.top_k,
            moe_intermediate_size=state.spec.moe_intermediate_size,
            norm_topk_prob=state.spec.norm_topk_prob,
        )
        self.router = Qwen3MoeTopKRouter(config)
        self.router.weight = nn.Parameter(state.router_weight, requires_grad=False)
        self._attention_view = SimpleNamespace(
            num_key_value_groups=state.spec.num_attention_heads // state.spec.num_key_value_heads,
            training=False,
        )

    def attention_projections(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> NativeAttentionProjections:
        state = self.state
        shape = (*x.shape[:-1], -1, state.spec.head_dim)
        q = native_nvfp4_linear(x, state.attention["q_proj"], state.attention_metadata["q_proj"].input_global_scale_inv)
        k = native_nvfp4_linear(x, state.attention["k_proj"], state.attention_metadata["k_proj"].input_global_scale_inv)
        v = native_nvfp4_linear(x, state.attention["v_proj"], state.attention_metadata["v_proj"].input_global_scale_inv)
        q_attn = _rms_norm(q.view(shape), state.q_norm_weight, self.rms_norm_eps).transpose(1, 2)
        k_attn = _rms_norm(k.view(shape), state.k_norm_weight, self.rms_norm_eps).transpose(1, 2)
        v_attn = v.view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        q_attn, k_attn = apply_rotary_pos_emb(q_attn, k_attn, cos, sin)
        out = _sdpa_attention_forward(
            self._attention_view,
            q_attn,
            k_attn,
            v_attn,
            attention_mask,
            scaling=1.0 / math.sqrt(state.spec.head_dim),
            dropout=0.0,
        )
        o_input = out.reshape(*x.shape[:-1], -1).contiguous()
        o = native_nvfp4_linear(o_input, state.attention["o_proj"], state.attention_metadata["o_proj"].input_global_scale_inv)
        return NativeAttentionProjections(q=q, k=k, v=v, o_input=o_input, o=o)

    def routed_moe(self, x: torch.Tensor) -> NativeMoEForward:
        state = self.state
        flat = x.reshape(-1, x.shape[-1])
        router_logits, routing_weights, selected_experts = self.router(flat)
        output = torch.zeros_like(flat)
        # This mirrors Transformers Qwen3MoeExperts.forward, but only
        # materializes/QDQs experts that were actually routed in this batch.
        for expert_idx in selected_experts.unique(sorted=True).tolist():
            expert = state.experts[int(expert_idx)]
            token_idx, topk_pos = torch.where(selected_experts == expert_idx)
            current = flat[token_idx]
            gate = native_nvfp4_linear(current, expert.gate_proj, expert.gate_metadata.input_global_scale_inv)
            up = native_nvfp4_linear(current, expert.up_proj, expert.up_metadata.input_global_scale_inv)
            hidden = F.silu(gate) * up
            down = native_nvfp4_linear(hidden, expert.down_proj, expert.down_metadata.input_global_scale_inv)
            down = down * routing_weights[token_idx, topk_pos, None]
            output.index_add_(0, token_idx, down.to(output.dtype))
        return NativeMoEForward(output.reshape_as(x), router_logits, routing_weights, selected_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> NativeMoEForward:
        residual = hidden_states
        normed = _rms_norm(hidden_states, self.state.input_layernorm_weight, self.rms_norm_eps)
        hidden_states = residual + self.attention_projections(normed, attention_mask, position_embeddings).o
        residual = hidden_states
        normed = _rms_norm(hidden_states, self.state.post_attention_layernorm_weight, self.rms_norm_eps)
        moe = self.routed_moe(normed)
        return NativeMoEForward(residual + moe.output, moe.router_logits, moe.routing_weights, moe.selected_experts)


def qwen3_moe_config_from_snapshot(snapshot: str | Path) -> Qwen3MoeConfig:
    config = json.loads((Path(snapshot) / "config.json").read_text(encoding="utf-8"))
    return Qwen3MoeConfig.from_dict(config)


@dataclass
class ExpertCoverageStats:
    active_experts_train: set[int]
    active_experts_val: set[int]
    per_expert_routed_token_count: torch.Tensor
    min_routed_tokens: int
    median_routed_tokens: float
    max_routed_tokens: int
    never_routed_experts: list[int]
    weight_qdq_calls_by_proj: dict[str, int]


class StudentQwen3MoELayerRuntime(nn.Module):
    def __init__(
        self,
        state: MoELayerMasterState,
        diag_state: nn.Module,
        *,
        use_r64: bool,
        rot_order: str,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.state = state
        self.diag_state = diag_state
        self.use_r64 = bool(use_r64)
        self.rot_order = rot_order
        self.rms_norm_eps = float(rms_norm_eps)
        self._attention_view = SimpleNamespace(
            num_key_value_groups=state.spec.num_attention_heads // state.spec.num_key_value_heads,
            training=False,
        )

    def attention(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        step_cache: StudentStepCache,
        *,
        use_ste: bool = True,
    ) -> torch.Tensor:
        state = self.state
        shape = (*x.shape[:-1], -1, state.spec.head_dim)
        q = forward_student_attention_proj(
            "q_proj", x, state.attention["q_proj"], self.diag_state,
            use_r64=self.use_r64, rot_order=self.rot_order, step_cache=step_cache, use_ste=use_ste,
        )
        k = forward_student_attention_proj(
            "k_proj", x, state.attention["k_proj"], self.diag_state,
            use_r64=self.use_r64, rot_order=self.rot_order, step_cache=step_cache, use_ste=use_ste,
        )
        v = forward_student_attention_proj(
            "v_proj", x, state.attention["v_proj"], self.diag_state,
            use_r64=self.use_r64, rot_order=self.rot_order, step_cache=step_cache, use_ste=use_ste,
        )
        q_attn = _rms_norm(q.view(shape), state.q_norm_weight, self.rms_norm_eps).transpose(1, 2)
        k_attn = _rms_norm(k.view(shape), state.k_norm_weight, self.rms_norm_eps).transpose(1, 2)
        v_attn = v.view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        q_attn, k_attn = apply_rotary_pos_emb(q_attn, k_attn, cos, sin)
        out = _sdpa_attention_forward(
            self._attention_view,
            q_attn,
            k_attn,
            v_attn,
            attention_mask,
            scaling=1.0 / math.sqrt(state.spec.head_dim),
            dropout=0.0,
        )
        o_input = out.reshape(*x.shape[:-1], -1).contiguous()
        return forward_student_attention_proj(
            "o_proj",
            o_input,
            state.attention["o_proj"],
            self.diag_state,
            use_r64=self.use_r64,
            rot_order=self.rot_order,
            step_cache=step_cache,
            use_ste=use_ste,
            head_dim=state.spec.head_dim,
        )

    def forward_to_router_input(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        step_cache: StudentStepCache | None = None,
        use_ste: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return post-attention residual and the true pre-D_GU router input."""
        cache = step_cache if step_cache is not None else StudentStepCache.new()
        residual = hidden_states
        normed = _rms_norm(hidden_states, self.state.input_layernorm_weight, self.rms_norm_eps)
        hidden_after_attention = residual + self.attention(
            normed,
            attention_mask,
            position_embeddings,
            cache,
            use_ste=use_ste,
        )
        router_input = _rms_norm(
            hidden_after_attention,
            self.state.post_attention_layernorm_weight,
            self.rms_norm_eps,
        )
        return hidden_after_attention, router_input

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        step_cache: StudentStepCache | None = None,
        use_ste: bool = True,
    ) -> StudentMoEOutput:
        cache = step_cache if step_cache is not None else StudentStepCache.new()
        residual, normed = self.forward_to_router_input(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            step_cache=cache,
            use_ste=use_ste,
        )
        moe = forward_student_routed_moe(
            normed,
            self.state,
            self.diag_state,
            use_r64=self.use_r64,
            rot_order=self.rot_order,
            step_cache=cache,
            use_ste=use_ste,
        )
        return StudentMoEOutput(
            residual + moe.output,
            moe.router_logits,
            moe.routing_weights,
            moe.selected_experts,
            moe.per_expert_routed_token_count,
            router_input=moe.router_input,
        )


def summarize_expert_coverage(
    *,
    train_counts: torch.Tensor,
    val_counts: torch.Tensor,
    step_cache: StudentStepCache,
) -> ExpertCoverageStats:
    total = train_counts.to(torch.long).cpu() + val_counts.to(torch.long).cpu()
    nonzero = total[total > 0]
    if nonzero.numel() == 0:
        min_count = 0
        median = 0.0
        max_count = 0
    else:
        min_count = int(nonzero.min().item())
        median = float(nonzero.to(torch.float32).median().item())
        max_count = int(nonzero.max().item())
    return ExpertCoverageStats(
        active_experts_train=set(torch.nonzero(train_counts.cpu(), as_tuple=False).flatten().tolist()),
        active_experts_val=set(torch.nonzero(val_counts.cpu(), as_tuple=False).flatten().tolist()),
        per_expert_routed_token_count=total,
        min_routed_tokens=min_count,
        median_routed_tokens=median,
        max_routed_tokens=max_count,
        never_routed_experts=torch.nonzero(total == 0, as_tuple=False).flatten().tolist(),
        weight_qdq_calls_by_proj=dict(step_cache.weight_qdq_calls_by_proj),
    )
