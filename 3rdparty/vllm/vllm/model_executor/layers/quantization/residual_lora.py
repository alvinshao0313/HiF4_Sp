"""Replicated FP32 low-rank corrections to the unnormalized residual stream."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .hif4_runtime import current_hif4_runtime_spec


class ResidualLoRA(nn.Module):
    def __init__(self, a, b, scale):
        super().__init__()
        # Buffers come from the sidecar, not from the HF weight loader. Each TP
        # rank has the full residual and applies the correction exactly once.
        self.register_buffer("a", a.to(dtype=torch.float32).contiguous())
        self.register_buffer("b", b.to(dtype=torch.float32).contiguous())
        self.scale = scale

    def forward(self, x):
        # Sidecar tensors are loaded on CPU before the vLLM worker device is
        # known. Move them once on the first actual forward, then retain them
        # there so token generation does not repeat a host-to-device copy.
        if self.a.device != x.device:
            self.a = self.a.to(x.device)
            self.b = self.b.to(x.device)
        return (F.linear(F.linear(x.float(), self.a), self.b) * self.scale).to(x.dtype)


def load_residual_loras(path, layer_index, hidden_size):
    spec = current_hif4_runtime_spec(path)
    payload = None if spec is None else spec.get("residual_lora")
    if payload is None:
        return None, None
    if payload.get("schema_version") != 1 or payload.get("compute_dtype") != "float32":
        raise ValueError("unsupported residual LoRA sidecar")
    rank, alpha = payload["rank"], payload["alpha"]
    if rank <= 0 or rank >= hidden_size or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("invalid residual LoRA rank/alpha")
    mode = payload["mode"]
    if mode not in {"attention", "moe", "both"}:
        raise ValueError(f"invalid residual LoRA mode: {mode}")
    if set(payload["layers"]) != {str(i) for i in range(spec["num_layers"])}:
        raise ValueError("incomplete residual LoRA layers")
    layer = payload["layers"][str(layer_index)]
    result = []
    for branch in ("attention", "moe"):
        if mode not in {branch, "both"}:
            result.append(None)
            continue
        a, b = layer[f"{branch}_lora_A"], layer[f"{branch}_lora_B"]
        if a.shape != (rank, hidden_size) or b.shape != (hidden_size, rank):
            raise ValueError(f"invalid residual LoRA shape in layer {layer_index}")
        if not a.isfinite().all() or not b.isfinite().all():
            raise ValueError("nonfinite residual LoRA weights")
        result.append(ResidualLoRA(a, b, alpha / rank))
    return tuple(result)
