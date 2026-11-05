"""Numerical contracts for sparse targets, chunking, and real BF16 execution."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import unittest

from .artifacts import atomic_save, load, state_digest, write_json
from .losses import kl_sum, output_loss, target


def math_checks():
    torch.manual_seed(42)
    q_logits = torch.randn(259, 193)
    h0, w0 = torch.randn(259, 17), torch.randn(193, 17) / 4
    sparse = target(q_logits)
    h, w = h0.clone().requires_grad_(), w0.clone().requires_grad_()
    p = F.linear(h, w).softmax(-1)
    q = q_logits.softmax(-1)
    ids = sparse["ids"].long()
    outside = torch.ones_like(q).scatter(-1, ids, 0.)
    qb = torch.cat((q.gather(-1, ids), (q * outside).sum(-1, keepdim=True)), -1)
    pb = torch.cat((p.gather(-1, ids), (p * outside).sum(-1, keepdim=True)), -1)
    reference = (qb * (qb.log() - pb.log())).sum(-1).mean()
    reference_grad = torch.autograd.grad(reference, (h, w))
    results = []
    for chunk, recompute in ((128, True), (259, False)):
        a, b = h0.clone().requires_grad_(), w0.clone().requires_grad_()
        value = output_loss(a, b, sparse, chunk=chunk, recompute=recompute)
        gradients = torch.autograd.grad(value, (a, b))
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5)
        for grad, ref in zip(gradients, reference_grad):
            torch.testing.assert_close(grad, ref, atol=1e-6, rtol=1e-5)
        results.append({"chunk": chunk, "recompute": recompute, "KL": float(value.detach())})
    torch.testing.assert_close(sparse["top"].exp(), qb[:, :64], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(sparse["tail"].exp(), qb[:, 64], atol=1e-6, rtol=1e-5)
    return {"passed": True, "FP32_atol": 1e-6, "FP32_rtol": 1e-5, "executions": results}


def aime_checks():
    from ..reasoning_trace_calibration.tests import ProtocolTests
    names = ("test_scoring_uses_final_submission", "test_replay_reviewed_aime_600_answers")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(ProtocolTests(name) for name in names))
    if not result.wasSuccessful():
        raise RuntimeError("confirmed AIME scoring regression")


def real_checks(model, cache, collator, training, validation, directory):
    device = model.head_base.device
    # This diagnostic isolates chunking; the subsequent trial uses all longest tokens.
    x, targets = cache.batch(training, device)
    base = x[0, :256].detach().clone()
    sparse = {k: v[:256] for k, v in targets.items()}
    outputs = []
    for chunk, recompute in ((128, True), (256, False)):
        model.zero_grad(set_to_none=True)
        h = base.clone().requires_grad_()
        value = output_loss(h, model.fused_head(), sparse, chunk=chunk, recompute=recompute)
        hg, mg = torch.autograd.grad(value, (h, model.head_matrix))
        outputs.append((float(value.detach()), hg.detach().float(), mg.detach().clone()))
    errors = {}
    for index, key in ((1, "hidden"), (2, "head_matrix")):
        a, b = outputs[0][index], outputs[1][index]
        relative = float((a - b).norm() / b.norm().clamp_min(1e-20))
        cosine = float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
        # BF16 GEMM/gradient accumulation may round differently across chunks.
        if relative > .02 or cosine < .999:
            raise RuntimeError(f"BF16 chunked gradient discrepancy: {key}: {relative}, {cosine}")
        errors[key] = {"relative_L2": relative, "cosine": cosine}
    if abs(outputs[0][0] - outputs[1][0]) > 1e-5:
        raise RuntimeError("BF16 chunked forward KL differs")
    initial = model.snapshot()
    path = directory / "epoch_zero.pt"
    atomic_save({"epoch": 0, "parameters": initial}, path)
    model.restore(load(path)["parameters"])
    if state_digest(model.snapshot()) != state_digest(initial):
        raise RuntimeError("epoch-zero checkpoint changed on reload")
    write_json({"passed": True, "chunked_BF16": errors, "KL": outputs[0][0],
                "relative_L2_tolerance": .02, "cosine_minimum": .999,
                "epoch_zero_exact_reload": True, "trainable_names": model.check_trainable(),
                "prefixes": "separate processes: teacher native 48 layers; student frozen B 47 layers"}, directory / "contracts.json")
    model.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = math_checks()
    aime_checks()
    result["AIME_600_reviewed_answers"] = "passed"
    write_json(result, args.run / "qualification/math.json")
