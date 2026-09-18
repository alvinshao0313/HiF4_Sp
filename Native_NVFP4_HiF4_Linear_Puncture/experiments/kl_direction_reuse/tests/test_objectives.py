from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.losses import (
    distribution_metrics, head_logits, kl_sum, mse_term, output_kl, router_term, signed_term,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.optimization import (
    backward_sample, collect_direction,
)


class ToyRuntime:
    """Small differentiable network; exercises real objective/adjoint code on CPU."""
    def __init__(self):
        self.student = nn.Module()
        self.student.learned = nn.Linear(3, 3, bias=False)
        self.student.base = SimpleNamespace(spec=SimpleNamespace(top_k=2))
        self.router = torch.randn(5, 3)
        self.head = torch.randn(7, 3)
        self.device = torch.device("cpu")
        self.layer = 8

    def local(self, x):
        h = self.student.learned(x).tanh()
        return SimpleNamespace(output=h, router_logits=F.linear(h, self.router))

    def kl(self, h, teacher, denominator):
        return kl_sum(F.linear(h.sin(), self.head), teacher) / denominator

    def snapshot_parameters(self):
        return {k: p.detach().clone() for k, p in self.student.learned.state_dict().items()}

    def restore(self, state):
        self.student.learned.load_state_dict(state)


def test_fresh_cached_and_direct_gradients_with_variable_lengths_and_router():
    torch.manual_seed(42)
    runtime = ToyRuntime()
    samples = []
    lengths = [2, 3, 5, 8]
    for n in lengths:
        x = torch.randn(1, n, 3)
        q = torch.randn(1, n, 7)
        target = {"output": torch.randn_like(x), "router_logits": torch.randn(1, n, 5)}
        samples.append((x, q, target, collect_direction(runtime, x, q, sum(lengths))))
    initial = runtime.snapshot_parameters()
    grads, steps = {}, {}
    for objective in ("direct_kl", "cached_kl"):
        runtime.restore(initial)
        opt = torch.optim.AdamW(runtime.student.learned.parameters(), lr=1e-4, weight_decay=0.)
        opt.zero_grad(set_to_none=True)
        for x, q, target, cache in samples:
            backward_sample(runtime, x, target, q, sum(lengths), objective, cache)
        grads[objective] = runtime.student.learned.weight.grad.clone()
        opt.step()
        steps[objective] = runtime.snapshot_parameters()["weight"]
    torch.testing.assert_close(grads["direct_kl"], grads["cached_kl"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(steps["direct_kl"], steps["cached_kl"], rtol=1e-6, atol=1e-7)
    assert grads["direct_kl"].norm() > 0
    assert not torch.equal(steps["direct_kl"], initial["weight"])


def test_signed_objective_keeps_sign_and_does_not_divide_again():
    h = torch.tensor([1., 2., 3.], requires_grad=True)
    g = torch.tensor([-2., 0., 4.], requires_grad=True)
    anchor = h.detach().clone().requires_grad_(True)
    loss = signed_term(h, anchor, g)
    assert loss == 0
    loss.backward()
    assert torch.equal(h.grad, g.detach())
    assert anchor.grad is None and g.grad is None
    assert signed_term(h.detach()-g.detach(), anchor, g) < 0


def test_standard_mse_token_channel_normalization_and_target_detach():
    target = torch.tensor([[[10., 20.], [30., 40.]]], requires_grad=True)
    h = (target.detach()+2).requires_grad_()
    loss = mse_term(h, target, 2)
    assert loss == 4
    loss.backward()
    assert torch.equal(h.grad, torch.ones_like(h))
    assert target.grad is None
    assert mse_term(h, target, 4) == 2


def test_partial_router_uses_full_softmax_and_detaches_teacher():
    s = torch.tensor([[.2, -.3, .7, .1]], requires_grad=True)
    t = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
    ids = t.topk(2, -1).indices
    p, q = s.log_softmax(-1), t.log_softmax(-1)
    expected = (q.exp() * (q-p)).gather(-1, ids).sum()/7
    actual = router_term(s, t, 7, top_k=2)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert t.grad is None
    # Tail experts still receive gradients through the full-softmax denominator.
    assert s.grad[0, :2].abs().sum() > 0


def test_chunked_final_kl_matches_dense_loss_and_gradient():
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import _rms_norm
    torch.manual_seed(4)
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    norm, head, q = torch.ones(4), torch.randn(64, 4), torch.randn(7, 64)
    dense = kl_sum(head_logits(_rms_norm(hidden[0], norm, 1e-6), head), q)/13
    grad, = torch.autograd.grad(dense, hidden)
    chunked = output_kl(hidden, norm, head, q, 13, eps=1e-6, token_chunk=3)
    chunk_grad, = torch.autograd.grad(chunked, hidden)
    torch.testing.assert_close(chunked, dense)
    torch.testing.assert_close(chunk_grad, grad)


def test_full_text_nll_and_kl_chunk_boundaries():
    z = torch.tensor([[1., 2., 0.], [3., 1., 0.], [0., 2., 3.], [1., 0., 2.]])
    q, ids = z+.1, torch.tensor([0, 1, 0, 2])
    a, b = distribution_metrics(z[:3], q[:3], ids, 0), distribution_metrics(z[3:], q[3:], ids, 3)
    assert a["nll_tokens"] == 3 and b["nll_tokens"] == 0
    assert a["kl_tokens"] + b["kl_tokens"] == 4
    assert a["nll_sum"] == pytest.approx(F.cross_entropy(z[:-1], ids[1:], reduction="sum").item())


def test_stale_direction_is_reused_but_no_longer_exact():
    torch.manual_seed(7)
    runtime = ToyRuntime()
    x, q = torch.randn(1, 3, 3), torch.randn(1, 3, 7)
    cached = collect_direction(runtime, x, q, 3)
    with torch.no_grad():
        runtime.student.learned.weight.add_(.3)
    fresh = collect_direction(runtime, x, q, 3)
    assert not torch.allclose(cached["gradient"], fresh["gradient"])
    assert not cached["gradient"].requires_grad

