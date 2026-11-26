"""Both objectives retain all vocabulary entries and one effective-token denominator."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def full_kl_sum(student_logits, teacher_logits):
    lp = F.log_softmax(student_logits.float(), -1)
    lq = F.log_softmax(teacher_logits.detach().float(), -1)
    return (lq.exp() * (lq - lp)).sum()


def output_loss(hidden, fused_head, *, objective, teacher_hidden=None, teacher_head=None,
                labels=None, chunk=128, recompute=True):
    n = len(hidden)
    if n == 0 or fused_head.shape[0] != 151936:
        raise ValueError("complete nonempty token coverage and full 151936 vocabulary required")
    if objective == "full_kl":
        if teacher_hidden.shape != hidden.shape or teacher_hidden.requires_grad or teacher_head.requires_grad:
            raise ValueError("teacher targets must be detached independent-prefix hidden states")

        def block(h, w, th, tw):
            with torch.no_grad():
                teacher_logits = F.linear(th, tw)
            return full_kl_sum(F.linear(h, w), teacher_logits)
    elif objective == "hard_ce":
        if labels.shape != (n,) or labels.dtype != torch.long:
            raise ValueError("CE requires one shifted integer label per effective position")

        def block(h, w, y):
            return F.cross_entropy(F.linear(h, w).float(), y, reduction="sum")
    else:
        raise ValueError(objective)
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, n, chunk):
        args = ((hidden[start:start + chunk], fused_head, teacher_hidden[start:start + chunk], teacher_head)
                if objective == "full_kl" else (hidden[start:start + chunk], fused_head, labels[start:start + chunk]))
        total = total + (checkpoint(block, *args, use_reentrant=False)
                         if recompute and torch.is_grad_enabled() else block(*args))
    return total / n


def ce_positions(packed):
    mask = packed["attention_mask"].bool()
    valid = mask[:, :-1] & mask[:, 1:] & packed["loss_mask"][:, 1:].bool()
    return valid, packed["input_ids"][:, 1:][valid]
