"""Calibrate every expert on all valid tokens, preserving the original top-k sum.

The calibration/output distinction follows LLM Compressor's Qwen3-MoE adapter.
Only the current Hessian group runs densely; over the full pass every projection
sees the same complete corpus. Other experts keep their sparse forward path.
"""
import time
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from HiFloat4.hif4gptq.gptq.gptq_utils import GPTQ, WeightHiFxQuantizer
from ..non_equivalent_reconstruction.transforms import projection_weights
from ..thesis_format_conversion_supplement.artifacts import write_json
from ..thesis_format_conversion_supplement.quantization import ConvertedStudent, replay, rotated


class AllExpertStudent(ConvertedStudent):
    def __init__(self, base):
        super().__init__(base)
        self.calibration_experts = set()

    def moe(self, normed, *, use_ste):
        if not self.calibration_experts:
            return super().moe(normed, use_ste=use_ste)
        logits = self.router(normed)
        flat = normed.reshape(-1, normed.shape[-1])
        probs = logits.reshape(-1, logits.shape[-1]).float().softmax(-1)
        weights, ids = probs.topk(self.base.spec.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        routed = torch.zeros(flat.shape[0], self.base.spec.top_k, flat.shape[-1],
                             device=flat.device, dtype=flat.dtype)
        counts = torch.bincount(ids.flatten(), minlength=self.base.spec.num_experts)
        if self.coverage_counts is not None:
            valid_ids = ids if self.valid is None else ids[self.valid]
            self.coverage_counts += torch.bincount(valid_ids.flatten(), minlength=self.base.spec.num_experts)
        for index in sorted(set(ids.unique().tolist()) | self.calibration_experts):
            rows, positions = torch.where(ids == index)
            dense = index in self.calibration_experts
            x = flat if dense else flat[rows]
            self.routed_valid = self.valid if dense else (self.valid[rows] if self.valid is not None else None)
            expert = self.base.experts[index]
            gate = self.linear(x, f"expert_{index}_gate_proj", expert.gate_proj, use_ste=False)
            up = self.linear(x, f"expert_{index}_up_proj", expert.up_proj, use_ste=False)
            routing_weight = weights[rows, positions]
            if dense:
                routing_weight = torch.zeros(flat.shape[0], device=flat.device, dtype=weights.dtype)
                routing_weight[rows] = weights[rows, positions]
            # Capture the REAL expert-specific SiLU(gate)*up input before the
            # router weight is applied. Nonselected tokens still inform GPTQ.
            down = self.linear(F.silu(gate) * up, f"expert_{index}_down_proj", expert.down_proj,
                               use_ste=False, routing_weight=routing_weight)
            routed[rows, positions] = down[rows] if dense else down
        self.routed_valid = None
        return routed.float().sum(1).to(flat.dtype).reshape_as(normed), logits, counts


@torch.no_grad()
def gptq_layer(base, samples, hidden, output):
    student = AllExpertStudent(base)
    weights = {name: weight for name, weight in projection_weights(base) if name != "router"}
    groups = [["q_proj", "k_proj", "v_proj"], ["o_proj"]]
    for start in range(0, base.spec.num_experts, 32):
        indices = range(start, min(start + 32, base.spec.num_experts))
        groups += [[f"expert_{i}_{p}" for i in indices for p in ("gate_proj", "up_proj")],
                   [f"expert_{i}_down_proj" for i in indices]]
    valid_tokens = sum(int(s.loss_mask.bool().sum()) for s in samples)
    if valid_tokens <= 0:
        raise ValueError("GPTQ requires real valid calibration tokens")
    stats, routed_counts = {}, None
    for group_index, names in enumerate(groups):
        started = time.monotonic()
        student.calibration_experts = {int(n.split("_")[1]) for n in names if n.startswith("expert_")}
        for name in names:
            weight = rotated(name, weights[name], base.spec)
            block = GPTQ(SimpleNamespace(weight=nn.Parameter(weight, requires_grad=False)))
            block.quantizer = WeightHiFxQuantizer("hifx4")
            block.effective_tokens = 0
            student.collectors[name] = block
        if group_index == 2:
            student.coverage_counts = torch.zeros(base.spec.num_experts, dtype=torch.long,
                                                   device=base.router_weight.device)
        replay(student, samples, hidden)
        collect_seconds = time.monotonic() - started
        if group_index == 2:
            routed_counts = student.coverage_counts.cpu().tolist()
            student.coverage_counts = None
            if sum(routed_counts) != valid_tokens * base.spec.top_k:
                raise ValueError("original top-k routing count changed")
        for name, block in student.collectors.items():
            diagonal = torch.diagonal(block.H)
            if block.effective_tokens != valid_tokens or block.nsamples != len(samples):
                raise ValueError(f"all-token calibration count mismatch: {name}")
            if not torch.isfinite(block.H).all() or not diagonal.sum() > 0:
                raise ValueError(f"invalid all-token Hessian: {name}")
            entry = {"tokens": block.effective_tokens, "batches": block.nsamples,
                     "zero_diagonal": int((diagonal == 0).sum()), "columns": block.columns,
                     "hessian_trace": float(diagonal.sum()), "hessian_group": group_index,
                     "group_hessian_seconds": collect_seconds,
                     "input_scope": "all valid tokens, R64 before activation QDQ"}
            stats[name] = entry
            write_json(stats, output / "coverage.json")
            started = time.monotonic()
            block.fasterquant(blocksize=128, groupsize=64, percdamp=.01)
            q = block.layer.weight.detach()
            if not torch.isfinite(q).all() or not torch.equal(q, q.to(torch.bfloat16).float()):
                raise ValueError(f"GPTQ output is nonfinite or not BF16 exact: {name}")
            student.quantized[name] = q.to(torch.bfloat16)
            entry["quantization_seconds"] = time.monotonic() - started
            block.free()
        student.collectors.clear()
        write_json(stats, output / "coverage.json")
        print(f"All-expert GPTQ layer {base.layer_idx}: {len(student.quantized)}/{len(weights)} projections; "
              f"every captured projection saw {valid_tokens} tokens", flush=True)
    student.calibration_experts.clear()
    if set(student.quantized) != set(weights):
        raise ValueError("incomplete GPTQ projections")
    write_json({"complete": True, "layer": base.layer_idx, "calibration_sequences": len(samples),
                "valid_tokens": valid_tokens, "calibration_tokens_per_expert": [valid_tokens] * base.spec.num_experts,
                "original_routed_tokens_per_expert": routed_counts,
                "naturally_unvisited": [i for i, n in enumerate(routed_counts) if not n],
                "unvisited_calibration": [], "gate_up_down_counts_checked": True,
                "scope": "all-token calibration; output remains original top-k weighted sum"},
               output / "expert_coverage.json")
    return student, stats
