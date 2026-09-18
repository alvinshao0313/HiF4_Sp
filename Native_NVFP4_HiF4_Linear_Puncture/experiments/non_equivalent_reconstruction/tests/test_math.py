import math

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.losses import (
    router_loss_per_token, masked_router_loss, router_metrics,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.transforms import block_right


@pytest.mark.parametrize("sharing", ["linear", "group"])
def test_block_right_matches_explicit_block_diagonal_and_gradient(sharing):
    torch.manual_seed(3)
    w = torch.randn(7, 128)
    matrix = torch.randn((64, 64) if sharing == "linear" else (2, 64, 64), requires_grad=True)
    blocks = [matrix, matrix] if sharing == "linear" else list(matrix.unbind())
    reference = w @ torch.block_diag(*blocks)
    actual = block_right(w, matrix)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), matrix)[0],
                               torch.autograd.grad(reference.square().sum(), matrix)[0], atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("kind", ["top_partial", "top_mass"])
@pytest.mark.parametrize("k", [1, 3, 7, 100])
def test_router_formula_and_teacher_detach(kind, k):
    torch.manual_seed(4)
    s = torch.randn(2, 3, 7, requires_grad=True)
    t = torch.randn(2, 3, 7, requires_grad=True)
    temperature = 1.7
    q, p = (t.detach() / temperature).softmax(-1), (s / temperature).softmax(-1)
    ids = t.detach().topk(min(k, 7), dim=-1).indices
    qt, pt = q.gather(-1, ids), p.gather(-1, ids)
    reference = (qt * (qt.log() - pt.log())).sum(-1)
    if kind == "top_mass" and k < 7:
        qo, po = 1 - qt.sum(-1), 1 - pt.sum(-1)
        reference = reference + qo * (qo.log() - po.log())
    actual = router_loss_per_token(s, t, kind=kind, top_k=k, temperature=temperature)
    torch.testing.assert_close(actual, reference * temperature**2, atol=2e-6, rtol=2e-5)
    actual.sum().backward()
    assert t.grad is None
    assert torch.isfinite(s.grad).all()


def test_partial_negative_and_mass_zero_when_equal():
    t = torch.tensor([[math.log(0.6), math.log(0.4)]])
    s = torch.tensor([[math.log(0.9), math.log(0.1)]])
    assert router_loss_per_token(s, t, kind="top_partial", top_k=1).item() < 0
    assert router_loss_per_token(s, t, kind="top_mass", top_k=1).item() > 0
    for kind in ("top_partial", "top_mass"):
        torch.testing.assert_close(router_loss_per_token(t, t, kind=kind, top_k=1), torch.zeros(1))


@pytest.mark.parametrize("kind", ["top_partial", "top_mass"])
def test_extreme_logits_and_padding(kind):
    s = torch.tensor([[[10000., -10000., 0.], [float("nan")] * 3]], requires_grad=True)
    t = torch.tensor([[[-10000., 10000., 0.], [float("nan")] * 3]])
    mask = torch.tensor([[True, False]])
    loss = masked_router_loss(s, t, mask, kind=kind, top_k=1)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(s.grad).all()
    assert s.grad[0, 1].count_nonzero() == 0


def test_mass_is_blind_to_redistribution_within_tail_but_metric_is_not():
    t = torch.tensor([[0.4, 0.3, 0.15, 0.15]]).log()
    s = torch.tensor([[0.4, 0.3, 0.299, 0.001]]).log()
    torch.testing.assert_close(router_loss_per_token(s, t, top_k=2), torch.zeros(1), atol=1e-6, rtol=0)
    metrics = router_metrics(s, t, torch.ones(1, dtype=torch.bool), top_k=2)
    assert metrics["topk_set_match"] == 1


def test_invalid_inputs_fail():
    with pytest.raises(ValueError):
        block_right(torch.ones(2, 65), torch.eye(64))
    with pytest.raises(ValueError):
        masked_router_loss(torch.ones(1, 3), torch.ones(1, 3), torch.zeros(1, dtype=torch.bool))
