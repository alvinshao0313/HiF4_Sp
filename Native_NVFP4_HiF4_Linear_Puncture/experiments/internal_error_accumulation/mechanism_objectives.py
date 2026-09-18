"""Mechanism controls: change the loss residual, never the inference path."""
from __future__ import annotations
import torch


def cumulative_prediction(r0: torch.Tensor, r1: torch.Tensor,
                          branch: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha not in (0.0, 0.5, 1.0):
        raise ValueError('prespecified cumulative coefficients are 0, 0.5, 1')
    if r0.shape != r1.shape or r1.shape != branch.shape:
        raise ValueError('cumulative control boundary shapes differ')
    if any(x.dtype != torch.bfloat16 for x in (r0, r1, branch)):
        raise ValueError('cumulative control requires actual BF16 residual semantics')
    upstream = r0 if alpha == 0 else r1 if alpha == 1 else (
        r0.float() + alpha * (r1.float() - r0.float())).to(torch.bfloat16)
    return upstream + branch


def direction_controls(error: torch.Tensor, update: torch.Tensor,
                       seeds=(20260916, 20260917, 20260918, 20260919)) -> dict:
    """FP64 direction controls; degenerate vectors are explicit, never filled."""
    e, u = error.detach().cpu().double(), update.detach().cpu().double()
    if e.shape != u.shape or not torch.isfinite(e).all() or not torch.isfinite(u).all():
        raise ValueError('invalid intervention directions')
    norm = u.norm()
    if norm == 0:
        return {'status': 'ZERO_UPDATE', 'vectors': {}}
    vectors = {'original': u}
    degenerate = []
    if e.square().sum() == 0:
        degenerate += ['parallel', 'orthogonal']
    else:
        parallel = (u*e).sum()/e.square().sum()*e
        for name, v in [('parallel', parallel), ('orthogonal', u-parallel)]:
            if v.norm() == 0:
                degenerate.append(name)
            else:
                vectors[name] = v/v.norm()*norm
    for seed in seeds:
        v = torch.randn(u.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
        vectors[f'random_{seed}'] = v/v.norm()*norm
    return {'status': 'READY', 'vectors': vectors, 'degenerate': degenerate}
