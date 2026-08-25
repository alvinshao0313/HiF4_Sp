"""Call boundary for one lazy Qwen3-MoE decoder layer.

This module owns the non-weight inputs needed by the Task-3 native semantic
teacher: position ids, RoPE embeddings and the single-layer runtime call.  It
does not materialize the full model and does not implement any H16 path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    NativeMoEForward,
    NativeQwen3MoELayerRuntime,
    qwen3_moe_config_from_snapshot,
)


@dataclass(frozen=True)
class Qwen3MoeLayerCall:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    position_ids: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]


def build_position_ids(
    batch_size: int,
    seq_len: int,
    *,
    device: torch.device | str,
    start: int = 0,
) -> torch.Tensor:
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    return torch.arange(start, start + seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)


def build_qwen3_moe_layer_call(
    snapshot: str,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> Qwen3MoeLayerCall:
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be [batch, seq, hidden], got {tuple(hidden_states.shape)}")
    if position_ids is None:
        position_ids = build_position_ids(
            int(hidden_states.shape[0]),
            int(hidden_states.shape[1]),
            device=hidden_states.device,
        )
    config = qwen3_moe_config_from_snapshot(snapshot)
    rope = Qwen3MoeRotaryEmbedding(config).to(hidden_states.device)
    return Qwen3MoeLayerCall(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=rope(hidden_states, position_ids),
    )


def call_native_qwen3_moe_layer(
    runtime: NativeQwen3MoELayerRuntime,
    call: Qwen3MoeLayerCall,
) -> NativeMoEForward:
    return runtime(
        call.hidden_states,
        attention_mask=call.attention_mask,
        position_embeddings=call.position_embeddings,
    )
