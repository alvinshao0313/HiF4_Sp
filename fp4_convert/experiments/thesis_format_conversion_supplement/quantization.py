"""R64 scale search and streamed, actual-input MoE GPTQ adapters."""
from types import SimpleNamespace
import time

import torch
from torch import nn
import torch.nn.functional as F

from HiFloat4.hif4_scale_threshold_optimization.src.weight_search import search_weight_groups
from ..e2e_diag_reconstruction.core.moe_transforms import apply_r64_no_cross_head
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import _rms_norm
from ..e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from ..module_design_ablation.student import Student
from ..non_equivalent_reconstruction.student import quantize, weighted_linear
from ..non_equivalent_reconstruction.transforms import projection_weights
from ...src.formats import qdq_hif4_direct
from .artifacts import write_json
from .config import NATIVE


def rotated(name, weight, spec):
    return apply_r64_no_cross_head(weight.float(), head_dim=spec.head_dim if name == "o_proj" else None)


@torch.no_grad()
def search_projection(weight):
    result = search_weight_groups(weight, budget="full", s0_mode="hardware", group_chunk_size=4096,
                                  enumerate_e8_e4=True, device=weight.device)
    standard = qdq_hif4_direct(weight, output_dtype=torch.float32)
    standard_error = (weight.float()-standard).square().sum().item()
    error = (weight.float()-result.reconstruction.float()).square().sum().item()
    # Both outputs are legal HiF4 levels, exactly representable by BF16 at these scales.
    if error > standard_error * (1+2e-6) + 1e-10:
        raise ValueError(f"full search regressed from its included standard candidate: {error} > {standard_error}")
    q = result.reconstruction.to(torch.bfloat16)
    if not torch.equal(q.float(), result.reconstruction.float()):
        raise ValueError("BF16 storage altered searched HiF4 values")
    metadata = {"s0": result.s0.cpu(), "e8": result.e8.to(torch.uint8).cpu(),
                "e4": result.e4.to(torch.uint8).cpu(), "s0_index": result.s0_index.to(torch.uint8).cpu()}
    metrics = {"nmse": result.nmse, "standard_nmse": standard_error / weight.square().sum().item(),
               "search_seconds": result.elapsed_s, "peak_allocated_bytes": result.peak_memory_bytes,
               "offset_counts": torch.bincount(result.s0_index.flatten(), minlength=5).cpu().tolist(),
               "budget": "full", "group_chunk_size": 4096, "bf16_exact": True}
    return q, metadata, metrics


class ConvertedStudent(Student):
    """Already-quantized weights never pass through weight QDQ a second time.

    Before a projection is converted its rotated effective weight is used. Input
    activation QDQ is always enabled, matching the deployed W4A4 layer. Hessians
    observe its actual input immediately before that QDQ, with valid tokens only.
    """
    def __init__(self, base):
        super().__init__(base, "fixed_r64")
        self.quantized = {}
        self.collectors = {}
        self.valid = None
        self.routed_valid = None
        self.coverage_counts = None

    def linear(self, x, name, weight, *, use_ste, head_dim=None, routing_weight=None):
        a = apply_r64_no_cross_head(x.float(), head_dim=head_dim)
        if name in self.collectors:
            mask = self.routed_valid if name.startswith("expert_") else self.valid
            valid = a.reshape(-1, a.shape[-1])[mask]
            if valid.numel():
                block = self.collectors[name]
                block.add_batch(valid, None)
                block.effective_tokens += valid.shape[0]
        q = self.quantized.get(name)
        w = q if q is not None else rotated(name, weight, self.base.spec).to(torch.bfloat16)
        ah = quantize(a, False)
        return weighted_linear(ah, w, routing_weight) if routing_weight is not None else F.linear(ah, w)

    def moe(self, normed, *, use_ste):
        logits = self.router(normed)
        flat = normed.reshape(-1, normed.shape[-1])
        probs = logits.reshape(-1, logits.shape[-1]).float().softmax(-1)
        weights, ids = probs.topk(self.base.spec.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        routed = torch.zeros(flat.shape[0], self.base.spec.top_k, flat.shape[-1], device=flat.device, dtype=flat.dtype)
        counts = torch.bincount(ids.reshape(-1), minlength=self.base.spec.num_experts)
        if self.coverage_counts is not None:
            valid_ids = ids if self.valid is None else ids[self.valid]
            self.coverage_counts += torch.bincount(valid_ids.reshape(-1), minlength=self.base.spec.num_experts)
        for index in ids.unique(sorted=True).tolist():
            rows, positions = torch.where(ids == index)
            self.routed_valid = self.valid[rows] if self.valid is not None else None
            expert = self.base.experts[index]
            x = flat[rows]
            gate = self.linear(x, f"expert_{index}_gate_proj", expert.gate_proj, use_ste=False)
            up = self.linear(x, f"expert_{index}_up_proj", expert.up_proj, use_ste=False)
            down = self.linear(F.silu(gate)*up, f"expert_{index}_down_proj", expert.down_proj,
                               use_ste=False, routing_weight=weights[rows, positions])
            routed[rows, positions] = down
        self.routed_valid = None
        return routed.float().sum(1).to(flat.dtype).reshape_as(normed), logits, counts


@torch.no_grad()
def replay(student, samples, hidden, *, propagate=False):
    outputs = {}
    for index, sample in enumerate(samples):
        x = hidden[sample.sample_id].to(student.base.router_weight.device).unsqueeze(0)
        student.valid = sample.loss_mask.bool().to(x.device).reshape(-1)
        call = build_qwen3_moe_layer_call(str(NATIVE), x)
        out = student(x, position_embeddings=call.position_embeddings, use_ste=False)
        if not torch.isfinite(out.output).all():
            raise ValueError("nonfinite GPTQ layer output")
        if propagate:
            outputs[sample.sample_id] = out.output[0].cpu().contiguous()
        if student.collectors and ((index + 1) % 128 == 0 or index + 1 == len(samples)):
            name, collector = next(iter(student.collectors.items()))
            if getattr(collector, "H", None) is not None:
                trace = float(torch.diagonal(collector.H).sum())
                print(f"GPTQ layer {student.base.layer_idx} collecting {name}: "
                      f"{index+1}/{len(samples)} complete sequences, "
                      f"input_tokens={collector.effective_tokens}, hessian_trace={trace:.6g}", flush=True)
    student.valid = None
    return outputs


@torch.no_grad()
def gptq_layer(base, samples, hidden, output):
    from HiFloat4.hif4gptq.gptq.gptq_utils import GPTQ, WeightHiFxQuantizer
    student = ConvertedStudent(base)
    weights = {n:w for n,w in projection_weights(base) if n != "router"}
    # Bound Hessian residency to one attention group or 32 experts. Gate and up
    # share actual inputs, but retain the core's independent update semantics.
    groups = [["q_proj", "k_proj", "v_proj"], ["o_proj"]]
    for start in range(0, base.spec.num_experts, 32):
        indices = range(start, min(start+32, base.spec.num_experts))
        groups += [[f"expert_{i}_{p}" for i in indices for p in ("gate_proj", "up_proj")],
                   [f"expert_{i}_down_proj" for i in indices]]
    stats = {}
    expert_counts = None
    for group_index, names in enumerate(groups):
        t0 = time.monotonic()
        for name in names:
            # SimpleNamespace carries the real weight; no dummy layer allocation.
            weight = rotated(name, weights[name], base.spec)
            block = GPTQ(SimpleNamespace(weight=nn.Parameter(weight, requires_grad=False)))
            block.quantizer = WeightHiFxQuantizer("hifx4")
            block.effective_tokens = 0
            student.collectors[name] = block
        if group_index == 2:
            student.coverage_counts = torch.zeros(base.spec.num_experts, dtype=torch.long, device=base.router_weight.device)
        replay(student, samples, hidden)
        collect_seconds = time.monotonic()-t0
        if group_index == 2:
            expert_counts = student.coverage_counts.cpu().tolist()
            student.coverage_counts = None
            missing = [i for i, count in enumerate(expert_counts) if not count]
            write_json({"complete": not missing, "layer": base.layer_idx,
                        "calibration_sequences": len(samples),
                        "valid_tokens": sum(int(s.loss_mask.bool().sum()) for s in samples),
                        "expert_counts": expert_counts, "unvisited": missing,
                        "scope": "actual propagated GPTQ inputs after attention conversion, before expert conversion"},
                       output / "expert_coverage.json")
            if missing:
                raise ValueError(f"layer {base.layer_idx}: unvisited experts {missing}; see {output / 'expert_coverage.json'}")
        for name, block in student.collectors.items():
            diagonal = torch.diag(block.H)
            entry = {"tokens": block.effective_tokens, "batches": block.nsamples,
                     "zero_diagonal": int((diagonal == 0).sum()), "columns": block.columns,
                     "group_hessian_seconds": collect_seconds, "hessian_group": group_index}
            stats[name] = entry
            write_json(stats, output / "coverage.json")
            if not block.effective_tokens:
                raise ValueError(f"unvisited projection {name}; GPTQ stopped before dead-column zeroing; see {output}")
            if name.startswith("expert_") and block.effective_tokens != expert_counts[int(name.split("_")[1])]:
                raise ValueError(f"actual routed tokens and GPTQ input capture disagree: {name}")
            started = time.monotonic()
            block.fasterquant(blocksize=128, groupsize=64, percdamp=.01)
            student.quantized[name] = block.layer.weight.detach().to(torch.bfloat16)
            entry["quantization_seconds"] = time.monotonic()-started
            block.free()
        student.collectors.clear()
        write_json(stats, output / "coverage.json")
        print(f"GPTQ layer {base.layer_idx}: {len(student.quantized)}/{len(weights)} projections", flush=True)
    if set(student.quantized) != set(weights):
        raise ValueError("incomplete GPTQ projection coverage")
    return student, stats
