"""按层 hook：流式 LAFD 打分 / 只捕获 selected 层 hidden。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "get_decoder_layers",
    "StreamingTeacherSelector",
    "SelectedHiddenCapture",
]


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Qwen3.5ForCausalLM: model.model.layers。"""
    root = model
    if hasattr(root, "module"):
        root = root.module
    if hasattr(root, "model") and hasattr(root.model, "layers"):
        layers = root.model.layers
    elif hasattr(root, "layers"):
        layers = root.layers
    else:
        raise RuntimeError(f"Cannot find decoder layers on {type(root).__name__}")
    if not isinstance(layers, nn.ModuleList):
        raise RuntimeError(f"layers is not ModuleList: {type(layers)}")
    return layers


def _masked_mean_cosine(
    a: torch.Tensor,
    b: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> float:
    a = a.float()
    b = b.float()
    if a.device != b.device:
        b = b.to(device=a.device)
    cos = F.cosine_similarity(a, b, dim=-1)
    if attention_mask is None:
        return float(cos.mean().item())
    mask = attention_mask.to(device=cos.device, dtype=cos.dtype)
    while mask.ndim < cos.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(cos)
    return float((cos * mask).sum().item() / mask.sum().clamp_min(1.0).item())


@dataclass
class StreamingTeacherSelector:
    """Teacher 前向时只保留上一层 hidden，流式算 cosine，不存齐 65 层。"""

    attention_mask: Optional[torch.Tensor]
    scores: list[float] = field(default_factory=list)
    _prev: Optional[torch.Tensor] = None
    _handles: list = field(default_factory=list)
    last_hidden: Optional[torch.Tensor] = None
    emb_hidden: Optional[torch.Tensor] = None

    def attach(self, model: nn.Module) -> None:
        layers = get_decoder_layers(model)
        self._handles = []

        # embedding 后：hook 第一层输入或 embed
        root = model.module if hasattr(model, "module") else model
        backbone = root.model if hasattr(root, "model") else root

        def emb_hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.emb_hidden = h.detach()
            self._prev = h.detach()

        if hasattr(backbone, "embed_tokens"):
            self._handles.append(backbone.embed_tokens.register_forward_hook(emb_hook))

        for layer in layers:
            def make_hook():
                def hook(_module, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h_det = h.detach()
                    if self._prev is None:
                        raise RuntimeError("Teacher selector missing previous hidden")
                    score = _masked_mean_cosine(h_det, self._prev, self.attention_mask)
                    self.scores.append(score)
                    self._prev = h_det
                    self.last_hidden = h_det

                return hook

            self._handles.append(layer.register_forward_hook(make_hook()))

    def selected_indices(self, topk: int) -> list[int]:
        if not self.scores:
            raise RuntimeError("No layer scores collected")
        topk = min(max(1, int(topk)), len(self.scores))
        order = sorted(range(len(self.scores)), key=lambda i: self.scores[i])[:topk]
        return order

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._prev = None


@dataclass
class SelectedHiddenCapture:
    """Student（或二次 teacher）只捕获 selected 层输出 + last_hidden。"""

    selected: Sequence[int]
    attention_mask: Optional[torch.Tensor] = None
    hiddens: dict[int, torch.Tensor] = field(default_factory=dict)
    last_hidden: Optional[torch.Tensor] = None
    _handles: list = field(default_factory=list)
    keep_grad: bool = True

    def attach(self, model: nn.Module) -> None:
        layers = get_decoder_layers(model)
        selected_set = set(int(i) for i in self.selected)
        self._handles = []
        n = len(layers)

        for idx, layer in enumerate(layers):
            if idx not in selected_set and idx != n - 1:
                continue

            def make_hook(layer_idx: int, is_last: bool):
                def hook(_module, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if is_last:
                        self.last_hidden = h if self.keep_grad else h.detach()
                    if layer_idx in selected_set:
                        self.hiddens[layer_idx] = h if self.keep_grad else h.detach()

                return hook

            self._handles.append(
                layer.register_forward_hook(make_hook(idx, is_last=(idx == n - 1)))
            )

    def ordered_hiddens(self) -> list[torch.Tensor]:
        return [self.hiddens[int(i)] for i in self.selected]

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
