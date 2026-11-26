"""Real complete-sequence checks for dense statistics and unchanged sparse routing."""
import torch
import torch.nn.functional as F

from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import _rms_norm
from ..e2e_diag_reconstruction.core.moe_transforms import apply_r64_no_cross_head
from ..e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from ..thesis_format_conversion_supplement.config import NATIVE


class Capture:
    def __init__(self, retain=False):
        self.effective_tokens = 0
        self.values = None
        self.retain = retain

    def add_batch(self, values, output):
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite expert calibration input")
        if self.retain:
            self.values = values.clone()


@torch.no_grad()
def routing_check(student, samples, hidden):
    records = []
    for sample in samples:
        x = hidden[sample.sample_id].to(student.base.router_weight.device).unsqueeze(0)
        student.valid = sample.loss_mask.bool().to(x.device).reshape(-1)
        call = build_qwen3_moe_layer_call(str(NATIVE), x)
        student.calibration_experts.clear()
        sparse = student(x, position_embeddings=call.position_embeddings, use_ste=False)
        student.collectors = {f"expert_{i}_{p}": Capture(retain=i in (0, 127))
                              for i in range(128) for p in ("gate_proj", "up_proj", "down_proj")}
        student.calibration_experts = set(range(128))
        dense = student(x, position_embeddings=call.position_embeddings, use_ste=False)
        captures = student.collectors
        student.collectors = {}
        if not torch.equal(sparse.router_logits, dense.router_logits) or not torch.equal(sparse.expert_counts, dense.expert_counts):
            raise ValueError("calibration changed original routing")
        nmse = float((sparse.output.float() - dense.output.float()).square().sum() /
                     sparse.output.float().square().sum())
        # Dense/sparse GEMMs can choose different reduction tilings. This bound
        # is below BF16 operand error and does not relax routing/count equality.
        if not torch.isfinite(dense.output).all() or nmse > 1e-5:
            raise ValueError(f"dense calibration altered layer output: NMSE={nmse}")
        count = int(student.valid.sum())
        if any(c.effective_tokens != count for c in captures.values()):
            raise ValueError("some experts did not observe every valid token")
        normed = _rms_norm(dense.pre_moe_norm, student.learned.moe_norm, student.eps).reshape(-1, x.shape[-1])
        expected_gate = apply_r64_no_cross_head(normed.float())[student.valid]
        for index in (0, 127):
            for p in ("gate_proj", "up_proj"):
                if not torch.equal(captures[f"expert_{index}_{p}"].values, expected_gate):
                    raise ValueError("gate/up Hessian does not observe complete real normalized inputs")
            expert = student.base.experts[index]
            gate = student.linear(normed, f"expert_{index}_gate_proj", expert.gate_proj, use_ste=False)
            up = student.linear(normed, f"expert_{index}_up_proj", expert.up_proj, use_ste=False)
            expected_down = apply_r64_no_cross_head((F.silu(gate) * up).float())[student.valid]
            if not torch.equal(captures[f"expert_{index}_down_proj"].values, expected_down):
                raise ValueError("down Hessian does not observe the real expert-specific intermediate")
        records.append({"id": sample.sample_id, "tokens": count, "all_384_projection_counts_equal": True,
                        "gate_up_actual_input_equal": True, "down_actual_input_equal": True,
                        "router_logits_and_counts_exact": True, "dense_sparse_output_nmse": nmse,
                        "naturally_unvisited": torch.where(sparse.expert_counts == 0)[0].cpu().tolist(),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        student.calibration_experts.clear()
        del captures, dense, sparse, x, normed, expected_gate, expected_down, gate, up
    student.valid = None
    return records
