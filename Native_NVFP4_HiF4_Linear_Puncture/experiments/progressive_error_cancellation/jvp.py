"""STE-surrogate JVP with fixed native routing branches."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    NativeQwen3MoELayerRuntime,
    NativeMoEForward,
    NativeAttentionProjections,
    _sdpa_attention_forward,
    _rms_norm,
    qdq_native_nvfp4,
    _row_parallel_linear,
    repeat_kv,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb


def _chunked_math_attention(module, query, key, value, attention_mask, *, scaling, chunk_size=256):
    """Memory-bounded math SDPA used only by the forward-mode JVP probe."""
    output_dtype = query.dtype
    query = query.float()
    key = key.float()
    value = value.float()
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    if getattr(module, "is_causal", False) and attention_mask is None:
        raise RuntimeError("chunked JVP attention requires an explicit causal mask")
    outputs = []
    for start in range(0, query.shape[2], chunk_size):
        stop = min(start + chunk_size, query.shape[2])
        mask = attention_mask
        if mask is not None and mask.ndim >= 2:
            mask = mask[..., start:stop, :]
        with sdpa_kernel(SDPBackend.MATH):
            output = torch.nn.functional.scaled_dot_product_attention(
                query[:, :, start:stop], key, value, attn_mask=mask,
                dropout_p=0.0, is_causal=False, scale=scaling,
            )
            outputs.append(output.to(output_dtype))
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous()


def route_ids(runtime, x):
    normed = _rms_norm(x, runtime.state.input_layernorm_weight, runtime.rms_norm_eps)
    return runtime.router(normed.reshape(-1, normed.shape[-1]))[2].detach()


class FixedRouteNativeRuntime(NativeQwen3MoELayerRuntime):
    """Native forward with the reference top-k IDs frozen for a JVP."""

    def __init__(self, state, selected, *, rms_norm_eps, o_proj_tp_size=1):
        super().__init__(state, rms_norm_eps=rms_norm_eps, o_proj_tp_size=o_proj_tp_size)
        selected = selected.detach().cpu()
        self.selected_shape = tuple(selected.shape)
        self.selected_values = selected.reshape(-1).tolist()
        self.expert_ids = sorted(set(self.selected_values))

    @staticmethod
    def _ste_qdq(x, scale):
        xf = x.float()
        with torch.no_grad():
            quantized = qdq_native_nvfp4(xf.detach(), scale)
        # Keep the native forward dtype while retaining an identity STE for
        # the derivative used by the surrogate JVP.
        return (xf + (quantized.float() - xf).detach()).to(x.dtype)

    def _linear(self, x, weight, metadata, *, tp_size=1):
        activation = self._ste_qdq(x, metadata.input_global_scale_inv)
        return _row_parallel_linear(
            x=activation,
            weight=weight.to(device=activation.device, dtype=activation.dtype),
            tp_size=tp_size,
        )

    def attention_projections(self, x, attention_mask, position_embeddings):
        state = self.state
        shape = (*x.shape[:-1], -1, state.spec.head_dim)
        q = self._linear(x, state.attention["q_proj"], state.attention_metadata["q_proj"])
        k = self._linear(x, state.attention["k_proj"], state.attention_metadata["k_proj"])
        v = self._linear(x, state.attention["v_proj"], state.attention_metadata["v_proj"])
        q_attn = _rms_norm(q.view(shape), state.q_norm_weight, self.rms_norm_eps).transpose(1, 2)
        k_attn = _rms_norm(k.view(shape), state.k_norm_weight, self.rms_norm_eps).transpose(1, 2)
        v_attn = v.view(shape).transpose(1, 2)
        q_attn, k_attn = apply_rotary_pos_emb(q_attn, k_attn, *position_embeddings)
        out = _chunked_math_attention(
            self._attention_view, q_attn, k_attn, v_attn, attention_mask,
            scaling=1.0 / (state.spec.head_dim ** .5),
        )
        o_input = out.reshape(*x.shape[:-1], -1).contiguous()
        o = self._linear(o_input, state.attention["o_proj"], state.attention_metadata["o_proj"],
                         tp_size=self.o_proj_tp_size)
        return NativeAttentionProjections(q=q, k=k, v=v, o_input=o_input, o=o)

    def routed_moe(self, x):
        state = self.state
        flat = x.reshape(-1, x.shape[-1])
        logits = F.linear(flat, self.router.weight)
        probs = torch.softmax(logits.float(), dim=-1)
        # Keep route control flow outside forward-mode tensors.  IDs are
        # fixed at the native point and reconstructed as an ordinary index
        # tensor for gather/where operations.
        selected = torch.tensor(self.selected_values, device=flat.device,
                                dtype=torch.long).reshape(self.selected_shape)
        if selected.shape != (flat.shape[0], state.spec.top_k):
            raise ValueError("fixed route shape differs from JVP input")
        weights = probs.gather(-1, selected)
        if state.spec.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.to(logits.dtype)
        output = torch.zeros_like(flat)
        for expert_idx in self.expert_ids:
            expert = state.experts[int(expert_idx)]
            token_idx, topk_pos = torch.where(selected == expert_idx)
            current = flat[token_idx]
            gate = self._linear(current, expert.gate_proj, expert.gate_metadata)
            up = self._linear(current, expert.up_proj, expert.up_metadata)
            hidden = F.silu(gate) * up
            down = self._linear(hidden, expert.down_proj, expert.down_metadata)
            output.index_add_(0, token_idx, down.to(output.dtype) * weights[token_idx, topk_pos, None])
        return NativeMoEForward(output.reshape_as(x), logits, weights, selected)


def fixed_route_native_jvp(runtime, x, tangent, call, selected):
    if x.shape != tangent.shape:
        raise ValueError("JVP primal and tangent shapes differ")
    fixed = FixedRouteNativeRuntime(runtime.state, selected,
                                    rms_norm_eps=runtime.rms_norm_eps,
                                    o_proj_tp_size=runtime.o_proj_tp_size).to(x.device)

    def function(value):
        return fixed(value, attention_mask=call.attention_mask,
                     position_embeddings=call.position_embeddings).output

    try:
        # The math SDPA surrogate has forward-mode support in the installed
        # PyTorch build.  torch.func.jvp avoids the double-backward path used
        # by autograd.functional.jvp, while unsupported operators still raise
        # here and are never replaced by finite differences.
        value, jvp = torch.func.jvp(
            function, (x.detach(),), (tangent.detach(),), strict=True,
        )
    except (NotImplementedError, RuntimeError) as exc:
        raise RuntimeError(
            "native STE JVP is unavailable for an active operator; refusing a fallback"
        ) from exc
    if not torch.isfinite(jvp).all():
        raise RuntimeError("native STE JVP produced nonfinite values")
    return jvp.detach(), value.detach()


def toy_jvp(linear_weight, x, tangent):
    """Small deterministic probe used by unit tests and smoke gating."""
    fn = lambda z: z @ linear_weight.T
    return torch.autograd.functional.jvp(fn, x, tangent, strict=True)[1]
