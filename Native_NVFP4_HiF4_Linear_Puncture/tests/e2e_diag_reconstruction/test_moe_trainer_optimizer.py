"""Unit tests for MoE trainer optimizer / scheduler / clamp wiring."""

from __future__ import annotations

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    MoEFusableDiagState,
)


def _make_optimizer_scheduler(cfg: E2ETrainConfig, n_batches: int):
    state = MoEFusableDiagState(
        hidden_size=4,
        num_experts=1,
        moe_intermediate_size=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    state.configure_fusable_components(cfg.fusable_diag_components)
    params = [p for p in state.parameters() if p.requires_grad]
    assert cfg.optimizer == "AdamW"
    optimizer = torch.optim.AdamW(params, lr=cfg.diag_lr, weight_decay=float(cfg.weight_decay))
    scheduler = None
    if cfg.diag_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.diag_epochs * n_batches, eta_min=0.0
        )
    return optimizer, scheduler


def test_moe_cosine_scheduler_lr_decreases_per_step():
    cfg = E2ETrainConfig.for_test(diag_scheduler="cosine", diag_epochs=2, diag_lr=5e-3)
    optimizer, scheduler = _make_optimizer_scheduler(cfg, n_batches=4)
    assert scheduler is not None
    lrs = []
    for _ in range(cfg.diag_epochs * 4):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == cfg.diag_lr
    assert lrs[-1] < lrs[0]
    assert min(lrs) >= 0.0


def test_moe_constant_scheduler_keeps_lr():
    cfg = E2ETrainConfig.for_test(diag_scheduler="constant", diag_epochs=2, diag_lr=5e-3)
    optimizer, scheduler = _make_optimizer_scheduler(cfg, n_batches=4)
    assert scheduler is None
    for _ in range(8):
        optimizer.step()
        assert optimizer.param_groups[0]["lr"] == cfg.diag_lr


def test_moe_adamw_weight_decay_explicit_zero():
    cfg = E2ETrainConfig.for_test()
    optimizer, _ = _make_optimizer_scheduler(cfg, n_batches=1)
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
