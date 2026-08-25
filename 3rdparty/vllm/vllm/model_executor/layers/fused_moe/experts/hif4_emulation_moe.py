"""HiF4 fusable/direct MoE correctness runtime."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization import hif4_fake


def apply_hif4_fused_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden = hidden_states.shape
    out = torch.zeros_like(hidden_states)
    x_q = hif4_fake.hif4_fake_quantize_hifx4(hidden_states)
    for token in range(num_tokens):
        acc = torch.zeros(hidden, device=hidden_states.device, dtype=torch.float32)
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot].item())
            gu = F.linear(x_q[token], w13[expert])
            gate, up = gu.chunk(2, dim=-1)
            act = F.silu(gate) * up
            act_q = hif4_fake.hif4_fake_quantize_hifx4(act.unsqueeze(0)).squeeze(0)
            down = F.linear(act_q, w2[expert])
            acc += down.float() * float(topk_weights[token, slot].item())
        out[token] = acc.to(out.dtype)
    return out
