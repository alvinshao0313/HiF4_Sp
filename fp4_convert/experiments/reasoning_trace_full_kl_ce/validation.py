"""FP32 reference objectives and actual BF16 full-vocabulary chunk gradients."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from ..reasoning_trace_last_head.validation import aime_checks
from .artifacts import write_json
from .losses import ce_positions, full_kl_sum, output_loss


def math_checks():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    h0, w0 = torch.randn(9, 17), torch.randn(151936, 17) / 4
    th, tw = torch.randn_like(h0), torch.randn_like(w0) / 4
    labels = torch.randint(0, 151936, (9,))
    results = {}
    for objective in ("full_kl", "hard_ce"):
        h, w = h0.clone().requires_grad_(), w0.clone().requires_grad_()
        logits, teacher = F.linear(h, w), F.linear(th, tw)
        reference = (F.kl_div(logits.log_softmax(-1), teacher.softmax(-1), reduction="sum") / len(h)
                     if objective == "full_kl" else F.cross_entropy(logits, labels))
        gradients = torch.autograd.grad(reference, (h, w), retain_graph=True)
        if objective == "full_kl":
            soft_ce = -(teacher.softmax(-1) * logits.log_softmax(-1)).sum() / len(h)
            soft_grad = torch.autograd.grad(soft_ce, (h, w))
            for a, b in zip(gradients, soft_grad):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        for chunk in (4, 9):
            a, b = h0.clone().requires_grad_(), w0.clone().requires_grad_()
            value = output_loss(a, b, objective=objective, teacher_hidden=th, teacher_head=tw,
                                labels=labels, chunk=chunk, recompute=True)
            actual = torch.autograd.grad(value, (a, b))
            torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5)
            for x, y in zip(actual, gradients):
                torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
        results[objective] = float(reference.detach())
    packed = {"input_ids": torch.tensor([[11, 12, 13, 14], [21, 22, 0, 0]]),
              "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
              "loss_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])}
    valid, actual = ce_positions(packed)
    assert actual.tolist() == [12, 13, 14, 22] and valid.sum() == 4
    from .development import choose
    original = {"tag": "B", "equal_mean": .7, "is_original_B": True, "epoch": 0, "lr": 0.}
    early = {"tag": "early", "equal_mean": .7, "is_original_B": False, "epoch": 1, "lr": 1e-5}
    late = {"tag": "late", "equal_mean": .7, "is_original_B": False, "epoch": 2, "lr": 1e-5}
    assert choose([late, early, original]) == original
    assert choose([late, early]) == early
    assert choose([original, {**late, "equal_mean": .71}])["tag"] == "late"
    return {"passed": True, "vocabulary": 151936, "FP32_atol": 1e-6, "FP32_rtol": 1e-5,
            "losses": results, "soft_CE_full_KL_gradients_equivalent": True,
            "next_token_shift_and_padding": True, "development_selection_and_ties": True}


def real_checks(model, cache, collator, objective, samples, directory):
    device = model.head_base.device
    values = cache.batch(samples, device)
    # Isolate vocabulary chunking; the trial below covers the complete longest batch.
    hidden = values["student"][0, :256].detach()
    teacher = values.get("teacher_hidden")
    teacher = teacher[0, :256].detach() if teacher is not None else None
    labels = samples[0].input_ids[1:257].to(device)
    outputs = []
    for chunk, recompute in ((128, True), (256, False)):
        h = hidden.clone().requires_grad_()
        loss = output_loss(h, model.fused_head(), objective=objective,
                           teacher_hidden=teacher, teacher_head=model.head_base, labels=labels,
                           chunk=chunk, recompute=recompute)
        gradients = torch.autograd.grad(loss, (h, model.head_matrix))
        outputs.append((float(loss.detach()), *[g.detach().float() for g in gradients]))
    errors = {}
    for i, name in ((1, "hidden"), (2, "head_matrix")):
        a, b = outputs[0][i], outputs[1][i]
        relative = float((a - b).norm() / b.norm().clamp_min(1e-20))
        cosine = float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
        if relative > .02 or cosine < .999:
            raise RuntimeError(f"BF16 chunk gradient mismatch: {name}: {relative}, {cosine}")
        errors[name] = {"relative_L2": relative, "cosine": cosine}
    if abs(outputs[0][0] - outputs[1][0]) > 1e-5:
        raise RuntimeError("BF16 chunk loss mismatch")
    write_json({"passed": True, "objective": objective, "BF16_chunk_gradients": errors,
                "relative_L2_tolerance": .02, "cosine_minimum": .999,
                "loss": outputs[0][0], "trainable_names": model.check_trainable()}, directory / "contracts.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = math_checks()
    aime_checks()
    result["AIME_600_reviewed_answers"] = "passed"
    write_json(result, args.run / "qualification/math.json")
