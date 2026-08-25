from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import (
    apply_conversion_state,
    load_conversion_state,
    save_conversion_artifact,
    save_layer_artifacts,
    select_layer_diag,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.fold import (
    fold_fusable_layer_inplace,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.semantic_hif4 import (
    LayerDiagState,
    SwitchableNVHiF4Linear,
    upgrade_semantic_model_inplace,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core import semantic_hif4 as sem_mod
from Native_NVFP4_HiF4_Linear_Puncture.src.metrics import relative_l2
from Native_NVFP4_HiF4_Linear_Puncture.src.semantic_model import NativeNVFP4SemanticLinear


class TinyRMS(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n, dtype=torch.float32))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.variance_epsilon) * self.weight).to(x.dtype)


def _native(in_f, out_f, name, bias=True):
    base = nn.Linear(in_f, out_f, bias=bias)
    with torch.no_grad():
        base.weight.copy_(torch.randn(out_f, in_f) * 0.05)
        if bias:
            base.bias.copy_(torch.randn(out_f) * 0.01)
    return NativeNVFP4SemanticLinear(
        base,
        module_name=name,
        input_global_scale=torch.tensor(1.0),
        rotation_matrix=torch.eye(16, dtype=torch.bfloat16),
        rotation_group_size=16,
    )


class TinyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _native(64, 64, "q_proj")
        self.k_proj = _native(64, 64, "k_proj")
        self.v_proj = _native(64, 64, "v_proj", bias=True)
        self.o_proj = _native(64, 64, "o_proj")

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(v)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = _native(64, 64, "gate_proj")
        self.up_proj = _native(64, 64, "up_proj", bias=True)
        self.down_proj = _native(64, 64, "down_proj")

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        return self.down_proj(u)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = TinyRMS(64)
        self.self_attn = TinyAttn()
        self.post_attention_layernorm = TinyRMS(64)
        self.mlp = TinyMLP()

    def forward(self, hidden_states, **kwargs):
        h = self.input_layernorm(hidden_states)
        h = hidden_states + self.self_attn(h)
        n = self.post_attention_layernorm(h)
        return h + self.mlp(n)


class TinyCfg:
    hidden_size = 64
    num_attention_heads = 1
    num_key_value_heads = 1
    head_dim = 64
    intermediate_size = 64


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TinyCfg()
        inner = nn.Module()
        inner.layers = nn.ModuleList([TinyLayer()])
        self.model = inner

    def forward(self, hidden_states):
        return self.model.layers[0](hidden_states)


def test_fusable_fold_unquantized_matches_unfolded(monkeypatch):
    monkeypatch.setattr(sem_mod, "quant_activation", lambda x, use_ste: x.to(torch.float32))
    monkeypatch.setattr(sem_mod, "quant_weight", lambda w, use_ste: w.to(torch.float32))
    model = TinyModel()
    cfg = E2ETrainConfig.for_test(diag_mode="fusable", use_r64=True)
    upgrade_semantic_model_inplace(model, cfg)
    layer = model.model.layers[0]
    with torch.no_grad():
        layer.diag_state.z_qkv.copy_(torch.linspace(-0.4, 0.4, 64))
        layer.diag_state.z_vo.copy_(torch.linspace(-0.2, 0.3, 64))
        layer.diag_state.z_gu.copy_(torch.linspace(0.1, -0.3, 64))
        layer.diag_state.z_ud.copy_(torch.linspace(-0.15, 0.25, 64))
    for m in layer.modules():
        if isinstance(m, SwitchableNVHiF4Linear):
            m.set_mode("hif4_eval")
    x = torch.randn(2, 4, 64, dtype=torch.float32)
    with torch.no_grad():
        y0 = layer(x).float()
        fold_fusable_layer_inplace(layer, layer.diag_state, use_r64=True)
        y1 = layer(x).float()
    assert relative_l2(y1, y0) < 1e-6


def _make_model(cfg: E2ETrainConfig):
    torch.manual_seed(0)
    model = TinyModel()
    upgrade_semantic_model_inplace(model, cfg)
    return model


def test_artifact_roundtrip_restores_z_and_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(sem_mod, "quant_activation", lambda x, use_ste: x.to(torch.bfloat16))
    monkeypatch.setattr(sem_mod, "quant_weight", lambda w, use_ste: w.to(torch.bfloat16))
    cfg = E2ETrainConfig.for_test(diag_mode="fusable", use_r64=False, output_dir=str(tmp_path))
    model = _make_model(cfg)
    layer = model.model.layers[0]
    with torch.no_grad():
        layer.diag_state.z_qkv.fill_(0.25)
    fold_fusable_layer_inplace(layer, layer.diag_state, use_r64=False)
    x = torch.randn(1, 3, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        y0 = layer(x).clone()
    path = save_conversion_artifact(
        cfg=cfg,
        layer_records={
            0: {
                "accepted": True,
                "rollback": False,
                "best_epoch": 3,
                "z": layer.diag_state.snapshot(),
            }
        },
        out_dir=tmp_path,
    )
    fresh = _make_model(cfg)
    apply_conversion_state(fresh, torch.load(path, map_location="cpu", weights_only=False))
    with torch.no_grad():
        y1 = fresh.model.layers[0](x)
    assert relative_l2(y1.float(), y0.float()) < 1e-6
    assert torch.allclose(fresh.model.layers[0].diag_state.z_qkv, torch.full((64,), 0.25))


def test_moe_artifact_schema_v3_preserves_candidate_and_adopted_replay(tmp_path):
    cfg = E2ETrainConfig.for_test(
        output_dir=str(tmp_path),
        use_r64=True,
        router_rollback="on",
        router_align_loss_weight=0.5,
    )
    candidate = {
        "z_qkv": torch.full((2048,), 0.1),
        "z_vo": torch.full((512,), 0.2),
        "z_gu": torch.full((2048,), 0.3),
        "z_ud": torch.full((128, 768), 0.4),
    }
    adopted = {name: value.clone() for name, value in candidate.items()}
    adopted["z_gu"].zero_()
    layer_records = {
        i: {
            "accepted": False,
            "rollback": True,
            "best_epoch": 2,
            "candidate_best_epoch": 2,
            "candidate_z": candidate,
            "adopted_z": adopted,
            "z": adopted,
            "loss_rollback_applied": False,
            "router_rollback_applied": True,
        }
        for i in range(48)
    }
    path = save_conversion_artifact(cfg=cfg, layer_records=layer_records, out_dir=tmp_path)
    state = load_conversion_state(path)
    assert state["schema_version"] == 3
    assert state["model_type"] == "qwen3_moe"
    assert state["num_layers"] == 48
    assert state["num_experts"] == 128
    assert state["kv_cache_dtype"] == "bfloat16"
    assert state["use_r64"] is True
    assert state["rot_order"] == "diag_then_rot"
    assert state["router_rollback"] == "on"
    assert state["router_align_type"] == "kl"
    assert state["router_align_temperature"] == 1.0
    assert state["router_align_loss_weight"] == 0.5
    assert state["artifact_diag_variants"] == ["adopted", "candidate"]
    assert set(state["layers"]) == {str(i) for i in range(48)}
    rec = state["layers"]["0"]
    torch.testing.assert_close(select_layer_diag(rec, "candidate")["z_gu"], candidate["z_gu"])
    torch.testing.assert_close(select_layer_diag(rec, "adopted")["z_gu"], adopted["z_gu"])
    assert rec["router_rollback_applied"] is True


def test_layer_artifact_preserves_pre_rollback_candidate(tmp_path):
    final_z = {"z_gu": torch.zeros(4)}
    candidate_z = {"z_gu": torch.tensor([0.25, -0.5, 0.75, -1.0])}
    d = save_layer_artifacts(
        tmp_path,
        0,
        z=final_z,
        metrics={"rollback": True},
        train_log=[],
        candidate_z=candidate_z,
        candidate_metrics={"candidate_best_val_loss": 0.1, "router_topk_mismatches": 3},
    )
    final_loaded = torch.load(d / "best_diag.pt", map_location="cpu", weights_only=False)
    candidate_loaded = torch.load(d / "candidate_best_diag.pt", map_location="cpu", weights_only=False)
    assert torch.equal(final_loaded["z_gu"], final_z["z_gu"])
    assert torch.equal(candidate_loaded["z_gu"], candidate_z["z_gu"])
    assert (d / "candidate_metrics.json").is_file()
