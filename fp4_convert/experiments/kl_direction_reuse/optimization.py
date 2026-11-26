"""Shared math used by cached and direct training, and by gradient verification."""
import torch

from .losses import mse_term, router_term, signed_term


def collect_direction(runtime, x, teacher_logits, batch_tokens):
    with torch.no_grad():
        anchor = runtime.local(x).output.detach()
    boundary = anchor.clone().requires_grad_(True)
    loss = runtime.kl(boundary, teacher_logits, batch_tokens)
    gradient, = torch.autograd.grad(loss, boundary)
    if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
        raise RuntimeError("nonfinite final-KL direction")
    return dict(anchor=anchor.detach().cpu(), gradient=gradient.detach().float().cpu(),
                kl_sum=float(loss.detach()) * batch_tokens)


def backward_sample(runtime, x, target, teacher_logits, batch_tokens, objective, cached=None):
    out = runtime.local(x)
    aux = router_term(out.router_logits, target["router_logits"], batch_tokens,
                      top_k=runtime.student.base.spec.top_k)
    if objective == "mse":
        main = mse_term(out.output, target["output"], batch_tokens)
    elif objective == "direct_kl":
        main = runtime.kl(out.output, teacher_logits, batch_tokens)
    elif objective == "cached_kl":
        if cached is None:
            raise ValueError("cached direction missing")
        main = signed_term(out.output, cached["anchor"].to(x.device), cached["gradient"].to(x.device))
    else:
        raise ValueError(objective)
    total = main + aux
    if not torch.isfinite(total):
        raise RuntimeError("nonfinite training loss")
    total.backward()
    return dict(main=float(main.detach()), router=float(aux.detach()))


def finite_gradients(parameters):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads or any(not torch.isfinite(g).all() for g in grads):
        raise RuntimeError("missing or nonfinite trainable gradients")
    return sum(float(g.float().square().sum()) for g in grads) ** .5
