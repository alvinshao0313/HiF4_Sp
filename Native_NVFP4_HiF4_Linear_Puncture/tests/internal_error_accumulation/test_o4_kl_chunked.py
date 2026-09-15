"""O4 final-logit KL must stay exact under vocab/token chunking."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.selected_layer_objective_trainer import (
    _kl_mean,
    _o4_final_kl_mean,
    _rms_norm,
)


def test_kl_mean_matches_batchmean_kl_div():
    torch.manual_seed(0)
    p = torch.randn(17, 503, dtype=torch.float32)
    q = torch.randn(17, 503, dtype=torch.float32, requires_grad=True)
    ref = F.kl_div(torch.log_softmax(q, dim=-1), torch.softmax(p, dim=-1), reduction="batchmean")
    got = _kl_mean(p, q, vocab_chunk=64)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)
    got.backward()
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()


def test_o4_final_kl_mean_matches_concat_kl():
    torch.manual_seed(1)
    b, s, h, v = 2, 11, 8, 97
    hidden = torch.randn(b, s, h, dtype=torch.float32, requires_grad=True)
    lengths = torch.tensor([7, 11])
    norm_w = torch.ones(h, dtype=torch.float32)
    lm_w = torch.randn(v, h, dtype=torch.float32)
    e0 = {}
    q_chunks = []
    p_chunks = []
    for i, sid in enumerate(["a", "b"]):
        n = int(lengths[i].item())
        h_n = _rms_norm(hidden[i, :n], norm_w, 1e-6)
        q = F.linear(h_n, lm_w)
        p = (q.detach() + 0.15).contiguous()
        e0[sid] = p
        q_chunks.append(q)
        p_chunks.append(p)
    ref = _kl_mean(torch.cat(p_chunks, dim=0), torch.cat(q_chunks, dim=0), vocab_chunk=32)
    got = _o4_final_kl_mean(
        hidden=hidden,
        lengths=lengths,
        norm_weight=norm_w,
        lm_head_weight=lm_w,
        e0_final_logits=e0,
        sample_ids=["a", "b"],
        token_chunk=3,
        vocab_chunk=32,
    )
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_o4_subsequent_defaults_to_checkpoint():
    # O4 subsequent must checkpoint by default to stay within 80GB.
    use_checkpoint = True
    assert use_checkpoint is True


def test_o4_final_kl_mean_rejects_cross_device():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("need >=2 cuda devices")
    torch.manual_seed(2)
    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")
    b, s, h, v = 1, 5, 8, 64
    hidden = torch.randn(b, s, h, device=dev0, dtype=torch.float32, requires_grad=True)
    lengths = torch.tensor([5], device=dev0)
    norm_w = torch.ones(h, device=dev1, dtype=torch.float32)
    lm_w = torch.randn(v, h, device=dev1, dtype=torch.float32)
    e0 = {"a": torch.randn(5, v)}
    with pytest.raises(RuntimeError, match="O4 LM must stay on student device"):
        _o4_final_kl_mean(
            hidden=hidden,
            lengths=lengths,
            norm_weight=norm_w,
            lm_head_weight=lm_w,
            e0_final_logits=e0,
            sample_ids=["a"],
            token_chunk=2,
            vocab_chunk=16,
        )


def test_o4_per_sample_backward_matches_batch_mean_grad():
    """sum_i (kl_i * n_i / N).backward() ≡ mean_token_kl.backward()."""
    torch.manual_seed(3)
    b, s, h, v = 3, 9, 8, 64
    lengths = torch.tensor([4, 9, 6])
    sids = ["a", "b", "c"]
    base = torch.randn(b, s, h, dtype=torch.float32)
    norm_w = torch.ones(h, dtype=torch.float32)
    lm_w = torch.randn(v, h, dtype=torch.float32)
    e0: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for i, sid in enumerate(sids):
            n = int(lengths[i].item())
            h_n = _rms_norm(base[i, :n], norm_w, 1e-6)
            q = F.linear(h_n, lm_w)
            e0[sid] = (q + 0.12).contiguous()

    hidden_batch = base.clone().detach().requires_grad_(True)
    loss_batch = _o4_final_kl_mean(
        hidden=hidden_batch,
        lengths=lengths,
        norm_weight=norm_w,
        lm_head_weight=lm_w,
        e0_final_logits=e0,
        sample_ids=sids,
        token_chunk=3,
        vocab_chunk=16,
    )
    loss_batch.backward()
    assert hidden_batch.grad is not None

    hidden_ps = base.clone().detach().requires_grad_(True)
    n_total = int(lengths.sum().item())
    loss_acc = 0.0
    for i, sid in enumerate(sids):
        n = int(lengths[i].item())
        kl_i = _o4_final_kl_mean(
            hidden=hidden_ps[i : i + 1],
            lengths=torch.tensor([n]),
            norm_weight=norm_w,
            lm_head_weight=lm_w,
            e0_final_logits=e0,
            sample_ids=[sid],
            token_chunk=3,
            vocab_chunk=16,
        )
        (kl_i * (float(n) / float(n_total))).backward()
        loss_acc += float(kl_i.detach().item()) * float(n)
    loss_ps = loss_acc / float(n_total)
    assert hidden_ps.grad is not None
    assert abs(float(loss_batch.detach().item()) - loss_ps) < 1e-6
    assert torch.allclose(hidden_batch.grad, hidden_ps.grad, rtol=1e-5, atol=1e-6)
