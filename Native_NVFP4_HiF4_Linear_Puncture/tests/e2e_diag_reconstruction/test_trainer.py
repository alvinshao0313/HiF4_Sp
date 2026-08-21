from __future__ import annotations

import torch
import torch.nn as nn

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.semantic_hif4 import (
    LayerDiagState,
    ONLINE_Z_NAMES,
    SwitchableNVHiF4Linear,
    load_layer_diag_snapshot,
    snapshot_layer_diag,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    sample_from_ids_and_mask,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.layer_runtime import (
    ProgressiveHiddenCache,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.losses import (
    finalize_reconstruction_loss,
    masked_reconstruction_components,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training import trainer as trainer_mod
from Native_NVFP4_HiF4_Linear_Puncture.src.semantic_model import NativeNVFP4SemanticLinear


def test_block_delta_nmse_matches_spec():
    torch.manual_seed(0)
    delta_h = torch.randn(2, 4, 8)
    delta_n = torch.randn(2, 4, 8)
    loss_mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    mask3 = (loss_mask.bool() & attention_mask.bool()).to(torch.float32)[..., None]
    num = ((delta_h.float() - delta_n.float()).pow(2) * mask3).sum()
    den = (delta_n.float().pow(2) * mask3).sum().clamp_min(1e-30)
    got_num, got_den = masked_reconstruction_components(
        delta_h, delta_n, loss_mask, attention_mask, "block_delta_nmse"
    )
    loss = finalize_reconstruction_loss(got_num, got_den, "block_delta_nmse")
    assert torch.allclose(got_num, num)
    assert torch.allclose(got_den, den)
    assert torch.allclose(loss, num / den)


def test_validation_accumulates_global_num_den():
    nums = [torch.tensor(1.0), torch.tensor(2.0)]
    dens = [torch.tensor(1.0), torch.tensor(10.0)]
    loss = finalize_reconstruction_loss(sum(nums), sum(dens), "block_delta_nmse")
    assert abs(float(loss.item()) - (3.0 / 11.0)) < 1e-6
    mean_of_batch = ((1 / 1) + (2 / 10)) / 2
    assert abs(float(loss.item()) - mean_of_batch) > 1e-6


def _native(in_f, out_f, name):
    base = nn.Linear(in_f, out_f, bias=False)
    return NativeNVFP4SemanticLinear(
        base,
        module_name=name,
        input_global_scale=torch.tensor(1.0),
        rotation_matrix=torch.eye(16, dtype=torch.bfloat16),
        rotation_group_size=16,
    )


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        k_dims = {p: 64 for p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}
        k_dims["down_proj"] = 64
        self.diag_state = LayerDiagState(
            diag_mode="online",
            hidden_size=64,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            intermediate_size=64,
            k_dims=k_dims,
        )
        self.self_attn = nn.Module()
        self.self_attn.q_proj = SwitchableNVHiF4Linear(
            _native(64, 64, "q_proj"),
            diag_state=self.diag_state,
            proj="q_proj",
            diag_mode="online",
            use_r64=False,
            rot_order="diag_then_rot",
        )
        self.mlp = nn.Identity()

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([FakeLayer()])
        inner.embed_tokens = nn.Embedding(8, 64)
        self.model = inner

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return h


def test_rollback_zeros_all_z_when_val_worse(monkeypatch):
    model = FakeModel()
    layer = model.model.layers[0]
    calls = {"n": 0}

    def fake_eval(**kwargs):
        calls["n"] += 1
        zabs = float(layer.diag_state.z_q.abs().sum().item())
        if zabs == 0:
            return 0.4, 1.0, 2.0
        return 0.9, 1.0, 2.0

    def fake_train(**kwargs):
        with torch.no_grad():
            layer.diag_state.z_q.fill_(1.5)
        return 0.2, 1

    monkeypatch.setattr(trainer_mod, "evaluate_layer_real_qdq", fake_eval)
    monkeypatch.setattr(trainer_mod, "_train_one_epoch", fake_train)
    monkeypatch.setattr(
        trainer_mod,
        "build_teacher_targets",
        lambda **k: trainer_mod.TeacherTargetCache(),
    )
    monkeypatch.setattr(
        trainer_mod,
        "build_length_bucket_batches",
        lambda samples, batch_size, seed: [samples],
    )

    samples = [
        sample_from_ids_and_mask("a", 0, torch.tensor([1, 2, 3]), torch.ones(3, dtype=torch.long), {})
    ]
    cache = ProgressiveHiddenCache()
    cache.store("a", torch.zeros(3, 64, dtype=torch.bfloat16), 3)
    cfg = E2ETrainConfig.for_test(diag_mode="online", diag_epochs=1, diag_batch_size=1)
    result = trainer_mod.train_layer_joint(
        model=model,
        layer_idx=0,
        cfg=cfg,
        train_samples=samples,
        val_samples=samples,
        collator=DynamicCalibrationCollator(0),
        x_cache=cache,
        device=torch.device("cpu"),
    )
    assert result.rollback is True
    assert result.accepted is False
    assert result.metrics["would_rollback"] is True
    assert result.metrics["rollback_applied"] is True
    assert torch.equal(layer.diag_state.z_q, torch.zeros_like(layer.diag_state.z_q))


def test_rollback_off_keeps_worse_z(monkeypatch):
    model = FakeModel()
    layer = model.model.layers[0]

    def fake_eval(**kwargs):
        zabs = float(layer.diag_state.z_q.abs().sum().item())
        if zabs == 0:
            return 0.4, 1.0, 2.0
        return 0.9, 1.0, 2.0

    def fake_train(**kwargs):
        with torch.no_grad():
            layer.diag_state.z_q.fill_(1.5)
        return 0.2, 1

    monkeypatch.setattr(trainer_mod, "evaluate_layer_real_qdq", fake_eval)
    monkeypatch.setattr(trainer_mod, "_train_one_epoch", fake_train)
    monkeypatch.setattr(
        trainer_mod,
        "build_teacher_targets",
        lambda **k: trainer_mod.TeacherTargetCache(),
    )
    monkeypatch.setattr(
        trainer_mod,
        "build_length_bucket_batches",
        lambda samples, batch_size, seed: [samples],
    )
    samples = [
        sample_from_ids_and_mask("a", 0, torch.tensor([1, 2, 3]), torch.ones(3, dtype=torch.long), {})
    ]
    cache = ProgressiveHiddenCache()
    cache.store("a", torch.zeros(3, 64, dtype=torch.bfloat16), 3)
    cfg = E2ETrainConfig.for_test(
        diag_mode="online", diag_epochs=1, diag_batch_size=1, layer_rollback="off"
    )
    result = trainer_mod.train_layer_joint(
        model=model,
        layer_idx=0,
        cfg=cfg,
        train_samples=samples,
        val_samples=samples,
        collator=DynamicCalibrationCollator(0),
        x_cache=cache,
        device=torch.device("cpu"),
    )
    assert result.metrics["would_rollback"] is True
    assert result.metrics["rollback_applied"] is False
    assert result.rollback is False
    assert result.accepted is False
    assert torch.allclose(layer.diag_state.z_q, torch.full_like(layer.diag_state.z_q, 1.5))


def test_linear_independent_only_updates_current_z(monkeypatch):
    model = FakeModel()
    layer = model.model.layers[0]
    for name, p in layer.diag_state.named_parameters():
        if name != "z_q":
            # extra online params exist
            pass
    before = snapshot_layer_diag(layer)

    n_eval = {"n": 0}

    def fake_eval(**kwargs):
        n_eval["n"] += 1
        if n_eval["n"] == 1:
            return 0.5, 1.0, 10.0
        return 0.1, 1.0, 10.0

    monkeypatch.setattr(trainer_mod, "evaluate_layer_real_qdq", fake_eval)
    teacher = trainer_mod.TeacherTargetCache()
    sid = "a"
    teacher.linear_in[sid] = {p: torch.zeros(3, 64, dtype=torch.bfloat16) for p in ONLINE_Z_NAMES}
    teacher.linear_out[sid] = {p: torch.zeros(3, 64, dtype=torch.bfloat16) for p in ONLINE_Z_NAMES}
    teacher.output[sid] = torch.zeros(3, 64, dtype=torch.bfloat16)
    monkeypatch.setattr(trainer_mod, "build_teacher_targets", lambda **k: teacher)

    samples = [sample_from_ids_and_mask("a", 0, torch.tensor([1, 2, 3]), torch.ones(3, dtype=torch.long), {})]
    cache = ProgressiveHiddenCache()
    cache.store("a", torch.zeros(3, 64, dtype=torch.bfloat16), 3)
    cfg = E2ETrainConfig.for_test(
        diag_mode="online",
        diag_train_scope="linear_independent",
        diag_epochs=1,
        diag_batch_size=1,
        use_r64=False,
    )
    # Missing k/v/... wrappers would break _proj_module. Patch the loop to only q_proj.
    orig_order = trainer_mod.ONLINE_TRAIN_ORDER
    trainer_mod.ONLINE_TRAIN_ORDER = ("q_proj",)
    try:
        result = trainer_mod.train_layer_linear_independent(
            model=model,
            layer_idx=0,
            cfg=cfg,
            train_samples=samples,
            val_samples=samples,
            collator=DynamicCalibrationCollator(0),
            x_cache=cache,
            device=torch.device("cpu"),
        )
    finally:
        trainer_mod.ONLINE_TRAIN_ORDER = orig_order
    after = snapshot_layer_diag(layer)
    for name in after:
        if name == "z_q":
            continue
        assert torch.equal(after[name], before[name])
    assert result.accepted is True
