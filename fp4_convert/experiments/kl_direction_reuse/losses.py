"""Signed adjoints and full-vocabulary KL, with one fixed token denominator."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..non_equivalent_reconstruction.losses import router_loss_per_token


def mse_term(output, target, batch_tokens):
    if output.shape != target.shape or batch_tokens <= 0:
        raise ValueError("invalid MSE boundary or normalization")
    return (output.float() - target.detach().float()).square().sum() / (batch_tokens * output.shape[-1])


def router_term(logits, target, batch_tokens, top_k=8):
    if batch_tokens <= 0:
        raise ValueError("invalid router denominator")
    return router_loss_per_token(logits, target, kind="top_partial", top_k=top_k,
                                 temperature=1.).sum() / batch_tokens


def signed_term(output, anchor, gradient):
    if not (output.shape == anchor.shape == gradient.shape):
        raise ValueError("cached boundary shapes differ")
    return (gradient.detach().float() * (output.float() - anchor.detach().float())).sum()


def kl_sum(student_logits, teacher_logits):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("KL logits must have identical token/vocabulary coverage")
    lp = F.log_softmax(student_logits.float(), -1)
    lq = F.log_softmax(teacher_logits.detach().float(), -1)
    return (lq.exp() * (lq - lp)).sum()


def head_logits(hidden, weight):
    # This model's vocabulary is a multiple of vLLM's padding size (64), so
    # its two vocabulary shards need no padding. Keep the rank-local GEMMs.
    if len(weight) % 64:
        raise ValueError("the TP2 LM head requires an unpadded vocabulary divisible by 64")
    return torch.cat([F.linear(hidden, w) for w in weight.chunk(2, 0)], dim=-1)


def output_kl(hidden, norm_weight, head_weight, teacher, batch_tokens, *, eps, token_chunk=256, norm_forward=None):
    from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import _rms_norm
    h = hidden.reshape(-1, hidden.shape[-1])
    if teacher.shape != (len(h), len(head_weight)) or batch_tokens <= 0:
        raise ValueError("teacher logits or batch normalization mismatch")
    h = (norm_forward or _rms_norm)(h, norm_weight, eps)
    total = h.new_zeros((), dtype=torch.float32)
    for start in range(0, len(h), token_chunk):
        stop = min(start + token_chunk, len(h))
        def one(block, a=start, b=stop):
            q = teacher[a:b].to(block.device)
            logits = head_logits(block, head_weight)
            return kl_sum(logits, q) / batch_tokens
        if torch.is_grad_enabled() and h.requires_grad:
            value = checkpoint(one, h[start:stop], use_reentrant=False)
        else:
            value = one(h[start:stop])
        total = total + value
    return total


@torch.no_grad()
def distribution_metrics(logits, teacher, ids, start):
    """KL on every predictor; NLL only where a next text token exists."""
    z = logits.float()
    if not torch.isfinite(z).all() or not torch.isfinite(teacher).all():
        raise RuntimeError("nonfinite evaluation logits")
    n = min(len(z), len(ids) - 1 - start)
    nll = F.cross_entropy(z[:n], ids[start+1:start+1+n].to(z.device), reduction="sum") if n > 0 else z.new_zeros(())
    return {"kl_sum": float(kl_sum(z, teacher)), "kl_tokens": len(z),
            "nll_sum": float(nll), "nll_tokens": max(0, n)}
