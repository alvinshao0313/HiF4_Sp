"""Full-vocabulary normalization followed by 64 singleton bins and one tail."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def log_bins(logits, ids):
    z = logits.float()
    normalizer = z.logsumexp(-1)
    top = z.gather(-1, ids.long()) - normalizer[:, None]
    # logsumexp of the complement avoids cancellation in 1 - sum(top64).
    outside = z.scatter(-1, ids.long(), float("-inf"))
    tail = outside.logsumexp(-1) - normalizer
    return top, tail


@torch.no_grad()
def target(logits, top_k=64):
    ids = logits.float().topk(top_k, dim=-1).indices
    top, tail = log_bins(logits, ids)
    return {"ids": ids.to(torch.int32), "top": top, "tail": tail}


def kl_sum(logits, ids, q_top, q_tail):
    p_top, p_tail = log_bins(logits, ids)
    return ((q_top.exp() * (q_top - p_top)).sum(-1)
            + q_tail.exp() * (q_tail - p_tail)).sum()


def output_loss(hidden, fused_head, targets, *, chunk=128, recompute=True):
    """Each chunk contributes to one graph and one batch-level backward/update."""
    n = hidden.shape[0]
    if n == 0 or targets["ids"].shape != (n, 64):
        raise ValueError("invalid complete effective-token target coverage")
    total = torch.zeros((), device=hidden.device, dtype=torch.float32)

    def block(h, w, ids, top, tail):
        return kl_sum(F.linear(h, w), ids, top, tail)

    for start in range(0, n, chunk):
        end = start + chunk
        args = (hidden[start:end], fused_head, targets["ids"][start:end],
                targets["top"][start:end], targets["tail"][start:end])
        value = (checkpoint(block, *args, use_reentrant=False) if recompute and torch.is_grad_enabled()
                 else block(*args))
        total = total + value
    return total / n
