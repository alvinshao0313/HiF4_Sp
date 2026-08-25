"""HiF4 online expert-specific MoE correctness runtime."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization import hif4_fake


def _scale(z: torch.Tensor) -> torch.Tensor:
    return torch.exp2(z.to(torch.float32))


def apply_hif4_online_reference_moe(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    z_gate: torch.Tensor,
    z_up: torch.Tensor,
    z_down: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden = hidden_states.shape
    out = torch.zeros_like(hidden_states)
    for token in range(num_tokens):
        acc = torch.zeros(hidden, device=hidden_states.device, dtype=torch.float32)
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot].item())
            x_gate = hif4_fake.hif4_fake_quantize_hifx4(
                hidden_states[token].float().mul(_scale(z_gate[expert]).to(hidden_states.device)).unsqueeze(0)
            ).squeeze(0)
            x_up = hif4_fake.hif4_fake_quantize_hifx4(
                hidden_states[token].float().mul(_scale(z_up[expert]).to(hidden_states.device)).unsqueeze(0)
            ).squeeze(0)
            gate = F.linear(x_gate.to(gate_weight.dtype), gate_weight[expert])
            up = F.linear(x_up.to(up_weight.dtype), up_weight[expert])
            act = F.silu(gate) * up
            act_q = hif4_fake.hif4_fake_quantize_hifx4(
                act.float().mul(_scale(z_down[expert]).to(act.device)).unsqueeze(0)
            ).squeeze(0)
            down = F.linear(act_q.to(down_weight.dtype), down_weight[expert])
            acc += down.float() * float(topk_weights[token, slot].item())
        out[token] = acc.to(out.dtype)
    return out
