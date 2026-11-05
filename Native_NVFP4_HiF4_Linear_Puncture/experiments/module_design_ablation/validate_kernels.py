"""Compare the deployed rotation/MoE kernels with the real trained layer."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.training.moe_trainer import _load_tensor
from ..non_equivalent_reconstruction.artifact import load, write_json
from ..non_equivalent_reconstruction.student import quantize
from .config import NATIVE_SNAPSHOT, CALIB_CACHE
from .rotation import apply_group_matrix
from .rotation_kernel import rotate, rotate_qdq
from .rotation_moe import apply_moe
from .student import Student


@torch.no_grad()
def validate(run_dir, output):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = load_qwen3_moe_layer_state(NATIVE_SNAPSHOT, 0, "cuda:0")
    student = Student(base, "rotation_norm_kl")
    student.learned.load_state_dict(load(Path(run_dir) / "layers/000/selected.pt")["parameters"], strict=True)
    samples = torch.load(CALIB_CACHE / "calibration/train.pt", weights_only=False)
    ids = torch.cat([sample.input_ids[:128] for sample in samples[:4]])
    embedding = _load_tensor(NATIVE_SNAPSHOT, "model.embed_tokens.weight")
    x = embedding[ids].to(device="cuda:0", dtype=torch.bfloat16).contiguous()
    q = student.learned.rotations["q_proj"]()
    reference = apply_group_matrix(x, q)
    actual = rotate(x, q)
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=2e-6)
    expected_qdq = quantize(reference, False)
    kernel_qdq = rotate_qdq(x, q)
    qdq_rel_l2 = float((kernel_qdq.float() - expected_qdq.float()).norm() / expected_qdq.float().norm())
    if qdq_rel_l2 > 1e-3:
        raise AssertionError(f"rotation QDQ discrepancy exceeds arithmetic tolerance: {qdq_rel_l2}")
    expected, logits, _ = student.moe(x, use_ste=False)
    probs = logits.float().softmax(-1)
    weights, ids = probs.topk(base.spec.top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    w13 = torch.stack([torch.cat([
        quantize(student.learned.weight(f"expert_{i}_{name}", getattr(expert, name)), False)
        for name in ("gate_proj", "up_proj")]) for i, expert in enumerate(base.experts)])
    w2 = torch.stack([quantize(student.learned.weight(f"expert_{i}_down_proj", expert.down_proj), False)
                      for i, expert in enumerate(base.experts)])
    rotations = {name: torch.stack([student.learned.rotations[f"expert_{i}_{name}"]()
                                    for i in range(base.spec.num_experts)])
                 for name in ("gate_proj", "up_proj", "down_proj")}
    actual = apply_moe(x, w13, w2, weights, ids, rotations)
    difference = actual.float() - expected.float()
    relative = float(difference.norm() / expected.float().norm())
    if relative > 3e-3:
        raise AssertionError(f"real MoE operator discrepancy exceeds BF16 arithmetic tolerance: {relative}")
    result = {"passed": True, "qdq_relative_l2": qdq_rel_l2, "moe_relative_l2": relative,
              "moe_max_abs": float(difference.abs().max()), "tokens": x.shape[0],
              "real_experts": base.spec.num_experts, "runtime_tp2_verified": False}
    write_json(result, Path(output))
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output", required=True)
    validate(**vars(parser.parse_args()))
