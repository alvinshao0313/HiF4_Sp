"""One GPU, TP2 arithmetic; frozen suffix layers retain input gradients."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

from ..non_equivalent_reconstruction.student import Student, quantize, weighted_linear
from ..non_equivalent_reconstruction.transforms import block_right
from ..e2e_diag_reconstruction.core.moe_transforms import apply_r64_no_cross_head
from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from .losses import output_kl


class TP2Student(Student):
    def weight(self, name, weight, use_ste):
        return quantize(block_right(weight, self.learned.matrices[name]), use_ste)

    def project(self, x, name, weight, *, use_ste, rows=None, columns=None, head_dim=None, routing_weight=None):
        w = self.weight(name, weight, use_ste)
        if rows is not None:
            w = w[rows]
        if columns is not None:
            w = w[:, columns]
        a = quantize(apply_r64_no_cross_head(x.float(), head_dim=head_dim), use_ste)
        return F.linear(a, w) if routing_weight is None else weighted_linear(a, w, routing_weight)

    def linear(self, x, name, weight, *, use_ste, head_dim=None, routing_weight=None):
        if name == "o_proj":
            middle = x.shape[-1] // 2
            if middle % 64:
                raise ValueError("TP2 split must preserve G64 boundaries")
            return sum(self.project(x[..., s], name, weight, use_ste=use_ste,
                                    columns=s, head_dim=head_dim)
                       for s in (slice(0, middle), slice(middle, None)))
        if name in ("q_proj", "k_proj", "v_proj"):
            middle = weight.shape[0] // 2
            return torch.cat([self.project(x, name, weight, use_ste=use_ste, rows=s)
                              for s in (slice(0, middle), slice(middle, None))], -1)
        return self.project(x, name, weight, use_ste=use_ste, head_dim=head_dim,
                            routing_weight=routing_weight)

    def router_aux_logits(self, pre_moe_norm):
        # Unlike the legacy experiment, the auxiliary objective reaches attention.
        from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import _rms_norm
        return self.router(_rms_norm(pre_moe_norm, self.learned.moe_norm, self.eps))

    def moe(self, normed, *, use_ste):
        logits = self.router(normed)
        flat = normed.reshape(-1, normed.shape[-1])
        probs = logits.reshape(-1, logits.shape[-1]).float().softmax(-1)
        weights, ids = probs.topk(self.base.spec.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        counts = torch.bincount(ids.reshape(-1), minlength=self.base.spec.num_experts)
        rank_outputs = []
        for rank in range(2):
            routed = torch.zeros(len(flat), self.base.spec.top_k, flat.shape[-1],
                                 device=flat.device, dtype=flat.dtype)
            for index in ids.unique(sorted=True).tolist():
                rows, positions = torch.where(ids == index)
                x = flat[rows]
                expert = self.base.experts[index]
                middle = expert.gate_proj.shape[0] // 2
                if middle % 64:
                    raise ValueError("expert TP2 split must preserve G64 groups")
                span = slice(rank * middle, (rank + 1) * middle)
                gate = self.project(x, f"expert_{index}_gate_proj", expert.gate_proj,
                                    rows=span, use_ste=use_ste)
                up = self.project(x, f"expert_{index}_up_proj", expert.up_proj,
                                  rows=span, use_ste=use_ste)
                down = self.project(F.silu(gate) * up, f"expert_{index}_down_proj", expert.down_proj,
                                    columns=span, use_ste=use_ste, routing_weight=weights[rows, positions])
                routed[rows, positions] = down
            # Production sums the selected experts in FP32, casts each TP rank,
            # then adds ranks. Do not cast routing weights or down accumulators early.
            rank_outputs.append(routed.float().sum(1).to(flat.dtype))
        return (rank_outputs[0] + rank_outputs[1]).reshape_as(normed), logits, counts


class FrozenTP2Student(TP2Student):
    """Weights are already R64-transformed and HiF4 QDQ; never transform twice."""
    def __init__(self, base, eps):
        nn.Module.__init__(self)
        self.base, self.eps = base, eps
        self.learned = nn.Module()
        self.learned.register_buffer("input_norm", base.input_layernorm_weight)
        self.learned.register_buffer("moe_norm", base.post_attention_layernorm_weight)
        self.attention_view = SimpleNamespace(num_key_value_groups=base.spec.num_attention_heads // base.spec.num_key_value_heads,
                                             training=False)

    def weight(self, name, weight, use_ste):
        return weight

    def router(self, normed):
        return F.linear(normed.to(torch.bfloat16), self.base.router_weight)


class ProductionTP2Student(TP2Student):
    """Match TP2 prefill chunking and the production CUDA kernels on one GPU."""
    def forward(self, hidden, *, position_embeddings, attention_mask=None, use_ste=True):
        from .config import PROTOCOL
        from .production import rms_norm, RotateQuantize, Rotary, PagedAttention, Routing, moe_rank
        from ..non_equivalent_reconstruction.student import Output
        if not hidden.is_cuda or hidden.ndim != 3 or hidden.shape[0] != 1 or attention_mask is not None:
            raise ValueError("production training requires one complete unpadded CUDA sequence")
        x = hidden[0]
        chunks = [slice(i, min(i+PROTOCOL.prefill_chunk, len(x))) for i in range(0, len(x), PROTOCOL.prefill_chunk)]
        spec = self.base.spec
        width = spec.head_dim
        cos, sin = (p[0] for p in position_embeddings)
        normalized = [rms_norm(x[c], self.learned.input_norm, self.eps) for c in chunks]
        aq = [RotateQuantize.apply(h) for h in normalized]
        attn_weights = {p: self.weight(p, w, use_ste) for p, w in self.base.attention.items()}
        attention_parts = []
        for rank in range(2):
            packed_w = torch.cat([attn_weights[p].chunk(2, 0)[rank] for p in ('q_proj', 'k_proj', 'v_proj')]).contiguous()
            qkv = torch.cat([F.linear(h, packed_w) for h in aq])
            qs, ks = spec.num_attention_heads*width//2, spec.num_key_value_heads*width//2
            q, k, v = qkv.split((qs, ks, ks), -1)
            q = torch.cat([rms_norm(q[c].reshape(-1, qs//width, width), self.base.q_norm_weight, self.eps, fused=False) for c in chunks])
            k = torch.cat([rms_norm(k[c].reshape(-1, ks//width, width), self.base.k_norm_weight, self.eps, fused=False) for c in chunks])
            v = v.reshape(-1, ks//width, width)
            q, k = Rotary.apply(q, k, cos, sin)
            attention = torch.cat([PagedAttention.apply(q[c], k[:c.stop], v[:c.stop]).reshape(c.stop-c.start, -1) for c in chunks])
            w = attn_weights['o_proj'].chunk(2, -1)[rank].contiguous()
            attention_parts.append(torch.cat([F.linear(RotateQuantize.apply(attention[c]), w) for c in chunks]))
        residual = x + (attention_parts[0] + attention_parts[1])
        normed = torch.cat([rms_norm(residual[c], self.learned.moe_norm, self.eps) for c in chunks])
        logits = torch.cat([self.router(normed[c]) for c in chunks])
        weights, ids = Routing.apply(logits, spec.top_k)
        counts = torch.bincount(ids.long().reshape(-1), minlength=spec.num_experts)
        rank_outputs = []
        for rank in range(2):
            w13, w2 = [], []
            for i, e in enumerate(self.base.experts):
                w13.append(torch.cat([self.weight(f'expert_{i}_{p}', getattr(e, p), use_ste).chunk(2, 0)[rank] for p in ('gate_proj', 'up_proj')]))
                w2.append(self.weight(f'expert_{i}_down_proj', e.down_proj, use_ste).chunk(2, -1)[rank].contiguous())
            w13, w2 = torch.stack(w13), torch.stack(w2)
            rank_outputs.append(torch.cat([moe_rank(normed[c].contiguous(), w13, w2, weights[c].contiguous(), ids[c].contiguous()) for c in chunks]))
        output = residual + (rank_outputs[0] + rank_outputs[1])
        return Output(output.unsqueeze(0), logits.unsqueeze(0), residual.unsqueeze(0), counts)

    def router_aux_logits(self, pre_moe_norm):
        from .production import rms_norm
        return self.router(rms_norm(pre_moe_norm, self.learned.moe_norm, self.eps))


class FrozenProductionStudent(ProductionTP2Student):
    __init__ = FrozenTP2Student.__init__
    weight = FrozenTP2Student.weight
    router = FrozenTP2Student.router


def frozen_state(model_dir, layer, device, spec):
    model_dir = Path(model_dir)
    tensors = load_file(str(model_dir / f"model-layer-{layer:05d}-of-00048.safetensors"), device=str(device))
    prefix = f"model.layers.{layer}"
    return SimpleNamespace(layer_idx=layer, spec=spec,
        input_layernorm_weight=tensors[f"{prefix}.input_layernorm.weight"],
        post_attention_layernorm_weight=tensors[f"{prefix}.post_attention_layernorm.weight"],
        q_norm_weight=tensors[f"{prefix}.self_attn.q_norm.weight"],
        k_norm_weight=tensors[f"{prefix}.self_attn.k_norm.weight"],
        router_weight=tensors[f"{prefix}.mlp.gate.weight"],
        attention={p: tensors[f"{prefix}.self_attn.{p}.weight"] for p in ("q_proj", "k_proj", "v_proj", "o_proj")},
        experts=[SimpleNamespace(**{p: tensors[f"{prefix}.mlp.experts.{i}.{p}.weight"]
                                    for p in ("gate_proj", "up_proj", "down_proj")}) for i in range(spec.num_experts)])


class Runtime:
    def __init__(self, snapshot, baseline, layer, device="cuda:0"):
        self.snapshot, self.baseline = Path(snapshot), Path(baseline)
        self.layer, self.device = layer, torch.device(device)
        cfg = json.loads((self.snapshot / "config.json").read_text())
        self.eps = cfg["rms_norm_eps"]
        self.num_layers = cfg["num_hidden_layers"]
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.rotary_embedding import get_rope
        # vLLM constructs the frequency table on its default CUDA device. The
        # HF CPU inverse-frequency table differs at BF16 rounding boundaries.
        with torch.device(self.device), set_current_vllm_config(VllmConfig()):
            rope = get_rope(cfg['head_dim'], max_position=cfg['max_position_embeddings'],
                rope_parameters={'rope_type': 'default', 'rope_theta': cfg['rope_theta']}, dtype=torch.bfloat16)
        self.rope_cache = rope.cos_sin_cache
        self.student = ProductionTP2Student(load_qwen3_moe_layer_state(snapshot, layer, device), "group",
                                  rms_norm_eps=self.eps).to(device)
        shared = load_file(str(self.baseline / "model-non-layer.safetensors"), device=str(device))
        self.norm = shared["model.norm.weight"]
        self.head = shared["lm_head.weight"]
        self.student.train()

    def local(self, x, *, use_ste=True):
        return self.student(x, position_embeddings=self.positions(len(x[0])), use_ste=use_ste)

    def positions(self, length):
        cos, sin = self.rope_cache[:length].chunk(2, -1)
        return torch.cat((cos, cos), -1).unsqueeze(0), torch.cat((sin, sin), -1).unsqueeze(0)

    def suffix(self, hidden):
        y = hidden
        for layer in range(self.layer + 1, self.num_layers):
            def one(h, index=layer):
                model = FrozenProductionStudent(frozen_state(self.baseline, index, self.device, self.student.base.spec), self.eps)
                return model(h, position_embeddings=self.positions(len(h[0])), use_ste=True).output
            y = checkpoint(one, y, use_reentrant=False) if torch.is_grad_enabled() and y.requires_grad else one(y)
        return y

    def kl(self, hidden, teacher, denominator):
        from .production import final_norm
        return output_kl(self.suffix(hidden), self.norm, self.head, teacher, denominator, eps=self.eps,
                         norm_forward=final_norm)

    def snapshot_parameters(self):
        return self.student.learned.snapshot()

    def restore(self, params):
        self.student.learned.load_state_dict(params, strict=True)
