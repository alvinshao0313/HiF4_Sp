"""Minimal checkpoint IO helpers retained for frozen-input operator parity."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core import (
    moe_semantic_hif4 as sem,
)


def install_causal_sdpa() -> None:
    def causal_sdpa(module, query, key, value, attention_mask, *, scaling: float, dropout: float = 0.0):
        key_states = sem.repeat_kv(key, module.num_key_value_groups)
        value_states = sem.repeat_kv(value, module.num_key_value_groups)
        output = F.scaled_dot_product_attention(
            query,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=attention_mask is None,
            scale=scaling,
        )
        return output.transpose(1, 2).contiguous()

    sem._sdpa_attention_forward = causal_sdpa


def load_index(snapshot: Path) -> dict[str, str]:
    return json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))[
        "weight_map"
    ]


def load_tensor(snapshot: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard = snapshot / weight_map[key]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)
