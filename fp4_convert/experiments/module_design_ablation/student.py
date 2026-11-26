"""Ablation transforms on the existing causal HiF4 layer reconstruction path."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from ..non_equivalent_reconstruction.student import Student as FixedR64Student
from ..non_equivalent_reconstruction.student import quantize, weighted_linear
from .parameters import LayerParameters


class Student(FixedR64Student):
    def __init__(self, native_base, variant_name, *, rms_norm_eps=1e-6):
        # Do not construct the old all-matrix/E4 parameterization first.
        nn.Module.__init__(self)
        self.base = native_base
        self.learned = LayerParameters(native_base, variant_name)
        self.eps = rms_norm_eps
        self.attention_view = SimpleNamespace(
            num_key_value_groups=native_base.spec.num_attention_heads // native_base.spec.num_key_value_heads,
            training=False,
        )

    def linear(self, x, name, weight, *, use_ste, head_dim=None, routing_weight=None):
        activation, transformed = self.learned.projection(x, name, weight, head_dim=head_dim)
        a_h, w_h = quantize(activation, use_ste), quantize(transformed, use_ste)
        if routing_weight is not None:
            return weighted_linear(a_h, w_h, routing_weight)
        return F.linear(a_h, w_h)

    def router(self, normed):
        weight = self.learned.weight("router", self.base.router_weight)
        return F.linear(normed.to(torch.bfloat16), weight.to(torch.bfloat16))
