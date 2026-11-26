"""Local calibration on decoded ModelOpt operands, without an additional rotation."""
from dataclasses import dataclass
from types import SimpleNamespace
from collections.abc import Callable

import torch

from HiFloat4.hif4gptq.gptq.gptq_utils import GPTQ, WeightHiFxQuantizer
from ..qwen3_mxfp4.format import MIDPOINTS, VALUES, quantize, decode_scales
from .operands import operand


@dataclass
class CalibratedWeights:
    weights: list[torch.Tensor]
    input_scale: torch.Tensor | None
    ratio: float | None = None


class MXGroupQuantizer:
    """Freeze one E8M0 scale per row while GPTQ updates the 32 columns."""

    def ready(self):
        return True

    def find_params(self, weights):
        if weights.shape[-1] != 32:
            raise ValueError("MXFP4 GPTQ requires complete groups of 32")
        _, encoded = quantize(weights)
        self.scale = decode_scales(encoded).flatten()
        self.midpoints = weights.new_tensor(MIDPOINTS)
        self.values = weights.new_tensor(VALUES)

    def __call__(self, column):
        x = column.flatten() / self.scale
        magnitude = x.abs()
        code = torch.bucketize(magnitude, self.midpoints)
        ties = magnitude == self.midpoints[code.clamp_max(6)]
        code += (ties & code.bitwise_and(1).bool()).long()
        return (torch.copysign(self.values[code], x) * self.scale).reshape_as(column)


@torch.no_grad()
def gptq(weight, inputs, target):
    """Each item is one sequence of valid, pre-QDQ input rows for this projection."""
    matrix = torch.nn.Parameter(weight.float(), requires_grad=False)
    block = GPTQ(SimpleNamespace(weight=matrix))
    width = {"hif4": 64, "mxfp4": 32}[target]
    if matrix.shape[-1] % width:
        raise ValueError("GPTQ requires full input groups")
    for sequence in inputs:
        if sequence.ndim != 2 or not len(sequence) or sequence.shape[-1] != matrix.shape[-1]:
            raise ValueError("Invalid calibration sequence")
        block.add_batch(sequence.to(weight.device), None)
    if not block.nsamples or not torch.isfinite(block.H).all() or not torch.diagonal(block.H).sum() > 0:
        raise ValueError("GPTQ requires finite, nonzero activation coverage")
    block.quantizer = WeightHiFxQuantizer("hifx4") if target == "hif4" else MXGroupQuantizer()
    block.fasterquant(blocksize=128, groupsize=width, percdamp=.01)
    return CalibratedWeights([matrix.detach().bfloat16()], None)


@torch.no_grad()
def scaled_weight(weight, scale, target):
    # The published QAD output head is also quantized. Limit scratch space.
    result = torch.empty_like(weight, dtype=torch.bfloat16)
    for start in range(0, len(weight), 256):
        rows = slice(start, start + 256)
        result[rows] = operand(weight[rows].float() * scale, target)
    return result


@torch.no_grad()
def smoothquant(weights, activation_max, target, alpha=0.5):
    """One channel scale for same-input projections, with inverse scaling on x."""
    if not 0 <= alpha <= 1 or not weights or not bool(activation_max.max() > 0):
        raise ValueError("SmoothQuant requires observed activations and alpha in [0,1]")
    weight_max = torch.stack([w.float().abs().amax(0) for w in weights]).amax(0).clamp_min(1e-5)
    scale = (activation_max.float().pow(alpha) / weight_max.pow(1 - alpha)).clamp_min(1e-5)
    return CalibratedWeights([scaled_weight(w, scale, target) for w in weights], scale)


@torch.no_grad()
def awq(weights, activation_mean, target, output_loss: Callable[[list[torch.Tensor]], float]):
    """Search AWQ's activation-only 20-point grid, with W-only output error.

    The caller evaluates the original parent (or projection) on unchanged
    calibration inputs, using the supplied effective quantized weights.
    At inference use the stored Q(W*s) and target QDQ on x/s.
    """
    if not weights or not torch.isfinite(activation_mean).all() or not activation_mean.max() > 0:
        raise ValueError("AWQ requires finite observed input channels")
    best_loss = float("inf")
    best = None
    for index in range(20):
        ratio = index / 20
        scale = activation_mean.float().pow(ratio).clamp_min(1e-4)
        scale = scale / (scale.max() * scale.min()).sqrt()
        q = [scaled_weight(w, scale, target) for w in weights]
        loss = output_loss([w.float() / scale for w in q])
        if loss < best_loss:
            best_loss = loss
            best = CalibratedWeights(q, scale, ratio)
    if best is None:
        raise ValueError("No finite AWQ candidate on calibration inputs")
    return best
