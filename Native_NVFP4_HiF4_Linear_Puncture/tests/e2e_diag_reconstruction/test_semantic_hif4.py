from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
    FUSABLE_COMPONENT_MAP,
    FUSABLE_DIAG_COMPONENTS,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core import semantic_hif4 as sem_mod
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.semantic_hif4 import (
    LayerDiagState,
    SwitchableNVHiF4Linear,
    qdq_hif4_ste_bf16,
    set_layer_runtime_mode,
    upgrade_semantic_model_inplace,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.semantic_model import NativeNVFP4SemanticLinear


def _native_linear(in_f: int, out_f: int, name: str, *, bias: bool = True) -> NativeNVFP4SemanticLinear:
    base = nn.Linear(in_f, out_f, bias=bias)
    with torch.no_grad():
        base.weight.copy_(torch.randn(out_f, in_f) * 0.05)
        if bias:
            base.bias.copy_(torch.randn(out_f) * 0.01)
    return NativeNVFP4SemanticLinear(
        base_linear=base,
        module_name=name,
        input_global_scale=torch.tensor(1.0, dtype=torch.float32),
        rotation_matrix=torch.eye(16, dtype=torch.bfloat16),
        rotation_group_size=16,
    )


def _fusable_state() -> LayerDiagState:
    return LayerDiagState(
        diag_mode="fusable",
        hidden_size=64,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=128,
        k_dims={
            "q_proj": 64,
            "k_proj": 64,
            "v_proj": 64,
            "o_proj": 64,
            "gate_proj": 64,
            "up_proj": 64,
            "down_proj": 128,
        },
    )


def _online_state() -> LayerDiagState:
    return LayerDiagState(
        diag_mode="online",
        hidden_size=64,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=128,
        k_dims={
            "q_proj": 64,
            "k_proj": 64,
            "v_proj": 64,
            "o_proj": 64,
            "gate_proj": 64,
            "up_proj": 64,
            "down_proj": 128,
        },
    )


def test_fusable_component_presets_trainable_and_inactive_zero():
    name_map = {"qkv": "z_qkv", "vo": "z_vo", "gu": "z_gu", "ud": "z_ud"}
    for preset in FUSABLE_DIAG_COMPONENTS:
        st = _fusable_state()
        with torch.no_grad():
            st.z_qkv.fill_(0.0)
        st.configure_fusable_components(preset)
        active = {name_map[k] for k in FUSABLE_COMPONENT_MAP[preset]}
        trained = {n for n, p in st.named_parameters() if p.requires_grad}
        assert trained == active
        for n, p in st.named_parameters():
            assert torch.equal(p, torch.zeros_like(p))
            if n not in active:
                assert p.requires_grad is False
    all_state = _fusable_state()
    all_state.configure_fusable_components("all")
    assert {n for n, p in all_state.named_parameters() if p.requires_grad} == {
        "z_qkv",
        "z_vo",
        "z_gu",
        "z_ud",
    }


def test_layer_diag_state_fusable_shapes_init_zero():
    st = _fusable_state()
    assert st.z_qkv.shape == (64,)
    assert st.z_vo.shape == (64,)
    assert st.z_gu.shape == (64,)
    assert st.z_ud.shape == (128,)
    for p in st.parameters():
        assert p.dtype == torch.float32
        assert torch.equal(p, torch.zeros_like(p))


def test_layer_diag_state_online_k_dims_init_zero():
    st = _online_state()
    assert st.z_q.shape == (64,)
    assert st.z_k.shape == (64,)
    assert st.z_v.shape == (64,)
    assert st.z_o.shape == (64,)
    assert st.z_gate.shape == (64,)
    assert st.z_up.shape == (64,)
    assert st.z_down.shape == (128,)
    for p in st.parameters():
        assert p.dtype == torch.float32
        assert torch.equal(p, torch.zeros_like(p))


def test_qdq_hif4_ste_matches_real_forward_and_has_finite_grad():
    torch.manual_seed(0)
    x = torch.randn(3, 64, dtype=torch.float32, requires_grad=True)
    y = qdq_hif4_ste_bf16(x)
    with torch.no_grad():
        y_ref = sem_mod.qdq_hif4_direct(x.detach(), output_dtype=torch.bfloat16)
    assert torch.equal(y.detach(), y_ref)
    (y.float().pow(2).sum()).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_switchable_reuses_master_weight_object():
    native = _native_linear(64, 32, "mock.q_proj")
    wrap = SwitchableNVHiF4Linear(
        native,
        diag_state=_fusable_state(),
        proj="q_proj",
        diag_mode="fusable",
        use_r64=False,
        rot_order="diag_then_rot",
    )
    assert wrap.weight.data_ptr() == native.weight.data_ptr()


def test_native_mode_matches_original_wrapper(monkeypatch):
    monkeypatch.setattr(sem_mod, "qdq_nvfp4_post_rotation", lambda x, s, **k: x)
    from Native_NVFP4_HiF4_Linear_Puncture.src import semantic_model as sm_mod

    monkeypatch.setattr(sm_mod, "qdq_nvfp4_post_rotation", lambda x, s, **k: x)
    native = _native_linear(64, 32, "mock.q_proj")
    x = torch.randn(2, 5, 64, dtype=torch.bfloat16)
    y0 = native(x).clone()
    wrap = SwitchableNVHiF4Linear(
        native,
        diag_state=_fusable_state(),
        proj="q_proj",
        diag_mode="fusable",
        use_r64=False,
        rot_order="diag_then_rot",
    )
    wrap.set_mode("native_nvfp4")
    y1 = wrap(x)
    assert torch.equal(y0, y1)


def test_weight_qdq_once_per_forward_and_invalidates_after_z_update(monkeypatch):
    calls = {"n": 0}

    def spy(w, use_ste):
        calls["n"] += 1
        return w.to(torch.bfloat16)

    monkeypatch.setattr(sem_mod, "quant_weight", spy)
    monkeypatch.setattr(sem_mod, "quant_activation", lambda x, use_ste: x.to(torch.bfloat16))

    wrap = SwitchableNVHiF4Linear(
        _native_linear(64, 64, "mock.q_proj"),
        diag_state=_fusable_state(),
        proj="q_proj",
        diag_mode="fusable",
        use_r64=False,
        rot_order="diag_then_rot",
    )
    wrap.set_mode("hif4_train_ste")
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    _ = wrap(x)
    assert calls["n"] == 1
    _ = wrap(x)
    assert calls["n"] == 2
    with torch.no_grad():
        wrap.diag_state.z_qkv.add_(0.1)
    _ = wrap(x)
    assert calls["n"] == 3


def test_hif4_train_ste_has_nonzero_finite_grad_to_z(monkeypatch):
    monkeypatch.setattr(sem_mod, "quant_activation", lambda x, use_ste: x.to(torch.bfloat16))
    monkeypatch.setattr(sem_mod, "quant_weight", lambda w, use_ste: w.to(torch.bfloat16))
    st = _fusable_state()
    wrap = SwitchableNVHiF4Linear(
        _native_linear(64, 32, "mock.q_proj"),
        diag_state=st,
        proj="q_proj",
        diag_mode="fusable",
        use_r64=False,
        rot_order="diag_then_rot",
    )
    wrap.set_mode("hif4_train_ste")
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    y = wrap(x)
    y.float().pow(2).mean().backward()
    assert st.z_qkv.grad is not None
    assert torch.isfinite(st.z_qkv.grad).all()
    assert st.z_qkv.grad.abs().sum() > 0


class _Cfg:
    hidden_size = 64
    num_attention_heads = 1
    num_key_value_heads = 1
    head_dim = 64
    intermediate_size = 64


class _Attn(nn.Module):
    def __init__(self, layer: int):
        super().__init__()
        self.q_proj = _native_linear(64, 64, f"model.layers.{layer}.self_attn.q_proj")
        self.k_proj = _native_linear(64, 64, f"model.layers.{layer}.self_attn.k_proj")
        self.v_proj = _native_linear(64, 64, f"model.layers.{layer}.self_attn.v_proj")
        self.o_proj = _native_linear(64, 64, f"model.layers.{layer}.self_attn.o_proj")


class _MLP(nn.Module):
    def __init__(self, layer: int):
        super().__init__()
        self.gate_proj = _native_linear(64, 64, f"model.layers.{layer}.mlp.gate_proj")
        self.up_proj = _native_linear(64, 64, f"model.layers.{layer}.mlp.up_proj")
        self.down_proj = _native_linear(64, 64, f"model.layers.{layer}.mlp.down_proj")


class _Layer(nn.Module):
    def __init__(self, layer: int):
        super().__init__()
        self.self_attn = _Attn(layer)
        self.mlp = _MLP(layer)


class _TinyModel(nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.config = _Cfg()
        inner = nn.Module()
        inner.layers = nn.ModuleList([_Layer(i) for i in range(n_layers)])
        self.model = inner


def test_upgrade_wraps_all_target_linears_and_keeps_native_outputs(monkeypatch):
    monkeypatch.setattr(sem_mod, "qdq_nvfp4_post_rotation", lambda x, s, **k: x)
    from Native_NVFP4_HiF4_Linear_Puncture.src import semantic_model as sm_mod

    monkeypatch.setattr(sm_mod, "qdq_nvfp4_post_rotation", lambda x, s, **k: x)
    model = _TinyModel(2)
    x = torch.randn(1, 3, 64, dtype=torch.bfloat16)
    before = {}
    for i, layer in enumerate(model.model.layers):
        before[(i, "q_proj")] = layer.self_attn.q_proj(x).clone()
        before[(i, "down_proj")] = layer.mlp.down_proj(x).clone()
    n = upgrade_semantic_model_inplace(model, E2ETrainConfig.for_test())
    assert n == 14
    wraps = [m for m in model.modules() if isinstance(m, SwitchableNVHiF4Linear)]
    assert len(wraps) == 14
    set_layer_runtime_mode(model.model.layers[0], "native_nvfp4")
    set_layer_runtime_mode(model.model.layers[1], "native_nvfp4")
    assert torch.equal(model.model.layers[0].self_attn.q_proj(x), before[(0, "q_proj")])
    assert torch.equal(model.model.layers[1].mlp.down_proj(x), before[(1, "down_proj")])
