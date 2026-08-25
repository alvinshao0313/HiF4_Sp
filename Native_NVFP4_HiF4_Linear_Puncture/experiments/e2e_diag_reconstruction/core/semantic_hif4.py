"""Switchable native NVFP4 / HiF4 Linear runtime. Reuses transform math from transforms.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
    FUSABLE_COMPONENT_MAP,
    TARGET_LINEARS_PER_LAYER,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.transforms import (
    apply_block_right_fp32,
    apply_r64,
    expand_vo_scale,
    fusable_weight_transform,
    online_weight_transform,
    scale_from_log2,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct
from Native_NVFP4_HiF4_Linear_Puncture.src.native_nvfp4 import qdq_nvfp4_post_rotation
from Native_NVFP4_HiF4_Linear_Puncture.src.rotation import apply_block_rotation
from Native_NVFP4_HiF4_Linear_Puncture.src.semantic_model import NativeNVFP4SemanticLinear

RuntimeMode = Literal["native_nvfp4", "hif4_train_ste", "hif4_eval", "folded"]
ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJS = ("gate_proj", "up_proj", "down_proj")
ALL_PROJS = ATTN_PROJS + MLP_PROJS
ONLINE_Z_NAMES = {
    "q_proj": "z_q",
    "k_proj": "z_k",
    "v_proj": "z_v",
    "o_proj": "z_o",
    "gate_proj": "z_gate",
    "up_proj": "z_up",
    "down_proj": "z_down",
}
FUSABLE_EXPLICIT_INPUT_D = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
FUSABLE_OUTPUT_D = {"v_proj": "z_vo", "up_proj": "z_ud"}


def qdq_hif4_ste_bf16(x: torch.Tensor) -> torch.Tensor:
    xf = x.to(torch.float32)
    with torch.no_grad():
        recon = qdq_hif4_direct(xf.detach(), output_dtype=torch.bfloat16).to(torch.float32)
    y = xf + (recon - xf).detach()
    return y.to(torch.bfloat16)


def quant_activation(x: torch.Tensor, use_ste: bool) -> torch.Tensor:
    if use_ste:
        return qdq_hif4_ste_bf16(x)
    return qdq_hif4_direct(x, output_dtype=torch.bfloat16)


def quant_weight(w: torch.Tensor, use_ste: bool) -> torch.Tensor:
    if use_ste:
        return qdq_hif4_ste_bf16(w)
    return qdq_hif4_direct(w, output_dtype=torch.bfloat16)


def _head_dim(config: Any) -> int:
    if hasattr(config, "head_dim") and config.head_dim is not None:
        return int(config.head_dim)
    hidden = int(config.hidden_size)
    n_heads = int(config.num_attention_heads)
    if hidden % n_heads != 0:
        raise ValueError(
            f"hidden_size={hidden} is not divisible by num_attention_heads={n_heads}"
        )
    return hidden // n_heads


class LayerDiagState(nn.Module):
    def __init__(
        self,
        *,
        diag_mode: str,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        k_dims: dict[str, int],
    ) -> None:
        super().__init__()
        self.diag_mode = diag_mode
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.intermediate_size = int(intermediate_size)
        self.k_dims = dict(k_dims)
        if diag_mode == "fusable":
            self.z_qkv = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
            self.z_vo = nn.Parameter(
                torch.zeros(num_key_value_heads * head_dim, dtype=torch.float32)
            )
            self.z_gu = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
            self.z_ud = nn.Parameter(torch.zeros(intermediate_size, dtype=torch.float32))
        elif diag_mode == "online":
            for proj in ALL_PROJS:
                name = ONLINE_Z_NAMES[proj]
                self.register_parameter(
                    name,
                    nn.Parameter(torch.zeros(k_dims[proj], dtype=torch.float32)),
                )
        else:
            raise ValueError(f"invalid diag_mode={diag_mode!r}")

    def scale(self, z: torch.Tensor) -> torch.Tensor:
        return scale_from_log2(z)

    def zero_all(self) -> None:
        with torch.no_grad():
            for p in self.parameters():
                p.zero_()

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {name: p.detach().cpu().to(torch.float32).clone() for name, p in self.named_parameters()}

    def load_snapshot(self, snapshot: dict[str, torch.Tensor]) -> None:
        missing = [name for name, _ in self.named_parameters() if name not in snapshot]
        extra = [name for name in snapshot if name not in dict(self.named_parameters())]
        if missing or extra:
            raise ValueError(f"diag snapshot mismatch missing={missing} extra={extra}")
        with torch.no_grad():
            for name, p in self.named_parameters():
                p.copy_(snapshot[name].to(device=p.device, dtype=torch.float32))

    def project_log2_clamp(self, bounds: tuple[float, float] | None) -> None:
        if bounds is None:
            return
        lo, hi = bounds
        with torch.no_grad():
            for p in self.parameters():
                p.clamp_(lo, hi)

    def d_qkv(self) -> torch.Tensor:
        return self.scale(self.z_qkv)

    def d_vo(self) -> torch.Tensor:
        return self.scale(self.z_vo)

    def d_vo_expanded(self) -> torch.Tensor:
        return expand_vo_scale(
            self.d_vo(),
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )

    def d_gu(self) -> torch.Tensor:
        return self.scale(self.z_gu)

    def d_ud(self) -> torch.Tensor:
        return self.scale(self.z_ud)

    def d_online(self, proj: str) -> torch.Tensor:
        return self.scale(getattr(self, ONLINE_Z_NAMES[proj]))

    def named_z(self) -> dict[str, torch.Tensor]:
        return dict(self.named_parameters())

    def configure_fusable_components(self, preset: str) -> None:
        if self.diag_mode != "fusable":
            if preset != "all":
                raise ValueError(
                    "online diag_mode only allows fusable_diag_components=all"
                )
            return
        if preset not in FUSABLE_COMPONENT_MAP:
            raise ValueError(f"invalid fusable_diag_components={preset!r}")
        active = FUSABLE_COMPONENT_MAP[preset]
        self.z_qkv.requires_grad_("qkv" in active)
        self.z_vo.requires_grad_("vo" in active)
        self.z_gu.requires_grad_("gu" in active)
        self.z_ud.requires_grad_("ud" in active)


@dataclass
class LinearRole:
    proj: str
    apply_input_d: bool
    output_d_name: str | None


def _fusable_role(proj: str) -> LinearRole:
    return LinearRole(
        proj=proj,
        apply_input_d=proj in FUSABLE_EXPLICIT_INPUT_D,
        output_d_name=FUSABLE_OUTPUT_D.get(proj),
    )


class SwitchableNVHiF4Linear(nn.Module):
    def __init__(
        self,
        native: NativeNVFP4SemanticLinear,
        *,
        diag_state: LayerDiagState,
        proj: str,
        diag_mode: str,
        use_r64: bool,
        rot_order: str,
    ) -> None:
        super().__init__()
        if proj not in ALL_PROJS:
            raise ValueError(f"unknown projection {proj!r}")
        self.module_name = native.module_name
        self.proj = proj
        self.diag_mode = diag_mode
        self.use_r64 = bool(use_r64)
        self.rot_order = rot_order
        self.rotation_group_size = int(native.rotation_group_size)
        self.activation_group_size = int(native.activation_group_size)
        self.weight = native.weight
        if native.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = native.bias
        self.register_buffer("input_global_scale", native.input_global_scale, persistent=True)
        self.register_buffer("rotation_matrix", native.rotation_matrix, persistent=True)
        self._diag_state_cell = [diag_state]
        self._mode: RuntimeMode = "native_nvfp4"
        self._cache_weight = False
        self._cached_w_h: torch.Tensor | None = None
        self._folded_weight_fp32: torch.Tensor | None = None
        self._folded_bias_fp32: torch.Tensor | None = None
        self._weight_qdq_calls = 0
        self._role = _fusable_role(proj) if diag_mode == "fusable" else LinearRole(proj, True, None)

    @property
    def diag_state(self) -> LayerDiagState:
        return self._diag_state_cell[0]

    def set_mode(self, mode: RuntimeMode) -> None:
        if mode not in ("native_nvfp4", "hif4_train_ste", "hif4_eval", "folded"):
            raise ValueError(f"invalid runtime mode {mode!r}")
        self._mode = mode
        if mode == "hif4_train_ste":
            self.clear_weight_cache()
            self._cache_weight = False

    def enable_weight_cache(self) -> None:
        if self._mode == "hif4_train_ste":
            raise RuntimeError("weight cache is forbidden in hif4_train_ste")
        self._cache_weight = True

    def clear_weight_cache(self) -> None:
        self._cached_w_h = None
        self._cache_weight = False

    def _h16_fp32(self) -> torch.Tensor:
        return self.rotation_matrix.to(dtype=torch.float32)

    def _fusable_d_in_out(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.diag_state
        out_features = int(self.weight.shape[0])
        in_features = int(self.weight.shape[1])
        ones_out = torch.ones(out_features, device=self.weight.device, dtype=torch.float32)
        if self.proj in ("q_proj", "k_proj"):
            return state.d_qkv(), ones_out
        if self.proj == "v_proj":
            return state.d_qkv(), state.d_vo()
        if self.proj == "o_proj":
            return state.d_vo_expanded(), ones_out
        if self.proj == "gate_proj":
            return state.d_gu(), ones_out
        if self.proj == "up_proj":
            return state.d_gu(), state.d_ud()
        if self.proj == "down_proj":
            return state.d_ud(), ones_out
        raise ValueError(self.proj)

    def _online_d(self) -> torch.Tensor:
        return self.diag_state.d_online(self.proj)

    def transformed_master_weight(self) -> torch.Tensor:
        w_n = self.weight.to(torch.float32)
        if self.diag_mode == "fusable":
            d_in, d_out = self._fusable_d_in_out()
            return fusable_weight_transform(
                w_n, self._h16_fp32(), d_in, d_out, use_r64=self.use_r64
            )
        d = self._online_d()
        return online_weight_transform(w_n, d, use_r64=self.use_r64, rot_order=self.rot_order)

    def _quantize_weight(self, w_t: torch.Tensor, use_ste: bool) -> torch.Tensor:
        if self._cache_weight and self._cached_w_h is not None and not use_ste:
            return self._cached_w_h
        self._weight_qdq_calls += 1
        w_h = quant_weight(w_t, use_ste=use_ste)
        if self._cache_weight and not use_ste:
            self._cached_w_h = w_h.detach()
        return w_h

    def _native_forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rot = apply_block_rotation(x, self.rotation_matrix, self.rotation_group_size)
        a_n = qdq_nvfp4_post_rotation(x_rot, self.input_global_scale)
        return F.linear(a_n.to(dtype=self.weight.dtype), self.weight, self.bias)

    def _prepare_activation_fp32(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.to(torch.float32)
        h16 = self._h16_fp32()
        if self._mode == "folded" or (
            self.diag_mode == "fusable" and not self._role.apply_input_d
        ):
            y = apply_block_right_fp32(xf, h16, self.rotation_group_size)
            if self.use_r64:
                y = apply_r64(y)
            return y
        if self.diag_mode == "fusable":
            d_in, _ = self._fusable_d_in_out()
            y = apply_block_right_fp32(xf * d_in, h16, self.rotation_group_size)
            if self.use_r64:
                y = apply_r64(y)
            return y
        y = apply_block_right_fp32(xf, h16, self.rotation_group_size)
        d = self._online_d()
        if self.rot_order == "diag_then_rot":
            y = y * d
            if self.use_r64:
                y = apply_r64(y)
            return y
        if self.rot_order == "rot_then_diag":
            if self.use_r64:
                y = apply_r64(y)
            return y * d
        raise ValueError(self.rot_order)

    def _maybe_scale_bias(self, d_out: torch.Tensor | None) -> torch.Tensor | None:
        if self.bias is None:
            return None
        b = self.bias
        if d_out is None:
            return b
        return b.to(torch.float32) * d_out

    def _hif4_forward(self, x: torch.Tensor, use_ste: bool) -> torch.Tensor:
        x_pre = self._prepare_activation_fp32(x)
        a_h = quant_activation(x_pre, use_ste=use_ste)
        if self._mode == "folded":
            if self._folded_weight_fp32 is None:
                raise RuntimeError(f"{self.module_name}: folded mode missing FP32 master weight")
            w_t = self._folded_weight_fp32
            w_h = self._quantize_weight(w_t, use_ste=False)
            bias = self._folded_bias_fp32
            if bias is not None:
                bias = bias.to(dtype=w_h.dtype)
            return F.linear(a_h.to(dtype=w_h.dtype), w_h, bias)
        w_t = self.transformed_master_weight()
        w_h = self._quantize_weight(w_t, use_ste=use_ste)
        bias = self.bias
        if self.diag_mode == "fusable" and self._role.output_d_name is not None:
            _, d_out = self._fusable_d_in_out()
            bias = self._maybe_scale_bias(d_out)
        if bias is not None:
            bias = bias.to(dtype=w_h.dtype)
        return F.linear(a_h.to(dtype=w_h.dtype), w_h, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mode == "native_nvfp4":
            return self._native_forward(x)
        if self._mode == "hif4_train_ste":
            return self._hif4_forward(x, use_ste=True)
        if self._mode in ("hif4_eval", "folded"):
            return self._hif4_forward(x, use_ste=False)
        raise RuntimeError(f"unhandled mode {self._mode}")

    def replace_master_weight_(self, w_preq: torch.Tensor) -> None:
        if tuple(w_preq.shape) != tuple(self.weight.shape):
            raise ValueError(
                f"{self.module_name}: folded weight shape {tuple(w_preq.shape)} "
                f"!= master {tuple(self.weight.shape)}"
            )
        self._folded_weight_fp32 = w_preq.detach().to(
            device=self.weight.device, dtype=torch.float32
        ).contiguous()
        with torch.no_grad():
            self.weight.copy_(self._folded_weight_fp32.to(dtype=self.weight.dtype))
        self.clear_weight_cache()

    def set_folded_bias_fp32(self, bias: torch.Tensor | None) -> None:
        if bias is None:
            self._folded_bias_fp32 = None
            return
        self._folded_bias_fp32 = bias.detach().to(
            device=self.weight.device, dtype=torch.float32
        ).contiguous()


def _layer_k_dims(layer: nn.Module) -> dict[str, int]:
    dims = {}
    for proj in ATTN_PROJS:
        mod = getattr(layer.self_attn, proj)
        dims[proj] = int(mod.weight.shape[1])
    for proj in MLP_PROJS:
        mod = getattr(layer.mlp, proj)
        dims[proj] = int(mod.weight.shape[1])
    return dims


def _replace_linear(
    parent: nn.Module,
    leaf: str,
    native: NativeNVFP4SemanticLinear,
    diag_state: LayerDiagState,
    cfg: E2ETrainConfig,
) -> SwitchableNVHiF4Linear:
    wrapped = SwitchableNVHiF4Linear(
        native,
        diag_state=diag_state,
        proj=leaf,
        diag_mode=cfg.diag_mode,
        use_r64=cfg.use_r64,
        rot_order=cfg.rot_order,
    )
    setattr(parent, leaf, wrapped)
    return wrapped


def upgrade_semantic_model_inplace(model: nn.Module, cfg: E2ETrainConfig) -> int:
    config = model.config
    hidden_size = int(config.hidden_size)
    num_attention_heads = int(config.num_attention_heads)
    num_key_value_heads = int(config.num_key_value_heads)
    head_dim = _head_dim(config)
    intermediate_size = int(config.intermediate_size)
    wrapped = 0
    layers = model.model.layers
    for layer in layers:
        k_dims = _layer_k_dims(layer)
        device = layer.self_attn.q_proj.weight.device
        state = LayerDiagState(
            diag_mode=cfg.diag_mode,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            k_dims=k_dims,
        ).to(device)
        state.configure_fusable_components(cfg.fusable_diag_components)
        layer.diag_state = state
        for proj in ATTN_PROJS:
            native = getattr(layer.self_attn, proj)
            if not isinstance(native, NativeNVFP4SemanticLinear):
                raise TypeError(f"{native} is not NativeNVFP4SemanticLinear")
            _replace_linear(layer.self_attn, proj, native, state, cfg)
            wrapped += 1
        for proj in MLP_PROJS:
            native = getattr(layer.mlp, proj)
            if not isinstance(native, NativeNVFP4SemanticLinear):
                raise TypeError(f"{native} is not NativeNVFP4SemanticLinear")
            _replace_linear(layer.mlp, proj, native, state, cfg)
            wrapped += 1
    expected = len(layers) * TARGET_LINEARS_PER_LAYER
    if wrapped != expected:
        raise RuntimeError(f"wrapped={wrapped} != expected={expected}")
    return wrapped


def iter_switchable_linears(model: nn.Module) -> list[SwitchableNVHiF4Linear]:
    return [m for _, m in model.named_modules() if isinstance(m, SwitchableNVHiF4Linear)]


def set_layer_runtime_mode(layer: nn.Module, mode: RuntimeMode) -> None:
    for m in layer.modules():
        if isinstance(m, SwitchableNVHiF4Linear):
            m.set_mode(mode)


def set_model_runtime_mode(model: nn.Module, mode: RuntimeMode) -> None:
    for m in iter_switchable_linears(model):
        m.set_mode(mode)


def enable_eval_weight_cache(layer: nn.Module) -> None:
    for m in layer.modules():
        if isinstance(m, SwitchableNVHiF4Linear):
            m.enable_weight_cache()


def snapshot_layer_diag(layer: nn.Module) -> dict[str, torch.Tensor]:
    return layer.diag_state.snapshot()


def load_layer_diag_snapshot(layer: nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    layer.diag_state.load_snapshot(snapshot)
    for m in layer.modules():
        if isinstance(m, SwitchableNVHiF4Linear):
            m.clear_weight_cache()


def diag_parameter_stats(state: LayerDiagState) -> dict[str, float]:
    zs = [p.detach().float().reshape(-1) for p in state.parameters()]
    z_cat = torch.cat(zs)
    d_cat = torch.exp2(z_cat)
    return {
        "z_min": float(z_cat.min().item()),
        "z_max": float(z_cat.max().item()),
        "d_min": float(d_cat.min().item()),
        "d_max": float(d_cat.max().item()),
    }


def clamp_hit_ratios(state: LayerDiagState, bounds: tuple[float, float] | None) -> dict[str, float]:
    if bounds is None:
        return {"clamp_low_hit_ratio": 0.0, "clamp_high_hit_ratio": 0.0}
    lo, hi = bounds
    z_cat = torch.cat([p.detach().float().reshape(-1) for p in state.parameters()])
    n = max(int(z_cat.numel()), 1)
    return {
        "clamp_low_hit_ratio": float((z_cat <= lo).sum().item() / n),
        "clamp_high_hit_ratio": float((z_cat >= hi).sum().item() / n),
    }
