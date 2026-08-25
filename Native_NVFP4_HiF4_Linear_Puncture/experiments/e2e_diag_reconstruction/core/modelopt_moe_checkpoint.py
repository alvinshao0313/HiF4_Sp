"""Layer-scoped loader for the native Qwen3-MoE ModelOpt NVFP4 checkpoint.

Only one decoder layer is dequantized at a time.  The numeric decode is the
existing vLLM NVFP4 emulation primitive; this module owns checkpoint naming and
the exact global-scale collapse performed by vLLM during loading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import (
    ATTENTION_PROJECTIONS,
    EXPERT_PROJECTIONS,
    Qwen3MoeModelSpec,
    load_qwen3_moe_model_spec,
)


@dataclass
class NativeNvfp4LinearMetadata:
    input_global_scale_inv: torch.Tensor
    weight_global_scale: torch.Tensor


@dataclass
class MoEExpertMasterState:
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
    gate_metadata: NativeNvfp4LinearMetadata
    up_metadata: NativeNvfp4LinearMetadata
    down_metadata: NativeNvfp4LinearMetadata


@dataclass
class MoELayerMasterState:
    layer_idx: int
    spec: Qwen3MoeModelSpec
    input_layernorm_weight: torch.Tensor
    post_attention_layernorm_weight: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    router_weight: torch.Tensor
    attention: dict[str, torch.Tensor]
    attention_metadata: dict[str, NativeNvfp4LinearMetadata]
    experts: list[MoEExpertMasterState]


def _read_layer_tensors(snapshot: Path, layer_idx: int) -> dict[str, torch.Tensor]:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    prefix = f"model.layers.{layer_idx}."
    keys = [key for key in index["weight_map"] if key.startswith(prefix)]
    if not keys:
        raise ValueError(f"layer {layer_idx} is absent from checkpoint index")
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index["weight_map"][key], []).append(key)
    tensors: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def _scalar(tensor: torch.Tensor, key: str) -> torch.Tensor:
    if tensor.numel() != 1:
        raise ValueError(f"{key} must be scalar, got shape={tuple(tensor.shape)}")
    return tensor.reshape(()).to(dtype=torch.float32)


def _decode_weight(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype,
    )

    # This is the same vLLM LUT handle used by the emulation kernel.  On SM80
    # its Python reference decoder indexes the handle directly, so it must
    # live alongside the packed tensor before decoding.
    if device.type == "cuda":
        nvfp4_emulation_utils.kE2M1ToFloat_handle.val = (
            nvfp4_emulation_utils.kE2M1ToFloat_handle.val.to(device)
        )
    weight = dequantize_to_dtype(
        packed.to(device),
        block_scale.to(device),
        weight_global_scale.to(device=device, dtype=torch.float32),
        dtype=torch.float32,
        block_size=16,
        swizzle=False,
    )
    return weight.contiguous()


def _linear_parts(tensors: dict[str, torch.Tensor], prefix: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = {suffix: f"{prefix}.{suffix}" for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")}
    missing = [key for key in keys.values() if key not in tensors]
    if missing:
        raise KeyError(f"missing ModelOpt tensor(s): {missing}")
    return tuple(tensors[keys[suffix]] for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"))  # type: ignore[return-value]


def load_qwen3_moe_layer_state(
    snapshot: str | Path,
    layer_idx: int,
    device: torch.device | str,
) -> MoELayerMasterState:
    """Load and dequantize exactly one MoE layer using vLLM scale semantics."""
    root = Path(snapshot)
    spec = load_qwen3_moe_model_spec(str(root))
    if not 0 <= layer_idx <= spec.last_layer:
        raise ValueError(f"layer_idx={layer_idx} outside [0, {spec.last_layer}]")
    target_device = torch.device(device)
    tensors = _read_layer_tensors(root, layer_idx)
    base = f"model.layers.{layer_idx}"

    attention_raw = {
        proj: _linear_parts(tensors, f"{base}.self_attn.{proj}")
        for proj in ATTENTION_PROJECTIONS
    }
    # QKVParallelLinear collapses all three input and weight global scales.
    qkv_input = torch.stack([_scalar(attention_raw[p][3], f"{p}.input_scale") for p in ("q_proj", "k_proj", "v_proj")]).max()
    qkv_weight = torch.stack([_scalar(attention_raw[p][2], f"{p}.weight_scale_2") for p in ("q_proj", "k_proj", "v_proj")]).max()
    attention: dict[str, torch.Tensor] = {}
    attention_metadata: dict[str, NativeNvfp4LinearMetadata] = {}
    for proj, (packed, scale, weight_scale_2, input_scale) in attention_raw.items():
        global_weight = qkv_weight if proj in {"q_proj", "k_proj", "v_proj"} else _scalar(weight_scale_2, f"{proj}.weight_scale_2")
        global_input = qkv_input if proj in {"q_proj", "k_proj", "v_proj"} else _scalar(input_scale, f"{proj}.input_scale")
        attention[proj] = _decode_weight(packed, scale, global_weight, device=target_device)
        attention_metadata[proj] = NativeNvfp4LinearMetadata(
            input_global_scale_inv=(1.0 / global_input).to(device=target_device, dtype=torch.float32),
            weight_global_scale=global_weight.to(device=target_device, dtype=torch.float32),
        )

    expert_raw = [
        {proj: _linear_parts(tensors, f"{base}.mlp.experts.{expert}.{proj}") for proj in EXPERT_PROJECTIONS}
        for expert in range(spec.num_experts)
    ]
    # ModelOpt fused-MoE emulation globally collapses activation scales.  W13
    # uses the gate branch global weight scale for both gate and up.
    a13 = torch.stack([
        _scalar(raw[proj][3], f"expert.{proj}.input_scale")
        for raw in expert_raw for proj in ("gate_proj", "up_proj")
    ]).max()
    a2 = torch.stack([
        _scalar(raw["down_proj"][3], "expert.down_proj.input_scale")
        for raw in expert_raw
    ]).max()
    experts: list[MoEExpertMasterState] = []
    for raw in expert_raw:
        gate_w, gate_s, gate_g, _ = raw["gate_proj"]
        up_w, up_s, _, _ = raw["up_proj"]
        down_w, down_s, down_g, _ = raw["down_proj"]
        w13_global = _scalar(gate_g, "gate_proj.weight_scale_2")
        w2_global = _scalar(down_g, "down_proj.weight_scale_2")
        experts.append(MoEExpertMasterState(
            gate_proj=_decode_weight(gate_w, gate_s, w13_global, device=target_device),
            up_proj=_decode_weight(up_w, up_s, w13_global, device=target_device),
            down_proj=_decode_weight(down_w, down_s, w2_global, device=target_device),
            gate_metadata=NativeNvfp4LinearMetadata((1.0 / a13).to(target_device), w13_global.to(target_device)),
            up_metadata=NativeNvfp4LinearMetadata((1.0 / a13).to(target_device), w13_global.to(target_device)),
            down_metadata=NativeNvfp4LinearMetadata((1.0 / a2).to(target_device), w2_global.to(target_device)),
        ))

    def bf16(name: str) -> torch.Tensor:
        if name not in tensors:
            raise KeyError(f"missing BF16 layer tensor {name}")
        return tensors[name].to(device=target_device, dtype=torch.bfloat16).contiguous()

    return MoELayerMasterState(
        layer_idx=layer_idx,
        spec=spec,
        input_layernorm_weight=bf16(f"{base}.input_layernorm.weight"),
        post_attention_layernorm_weight=bf16(f"{base}.post_attention_layernorm.weight"),
        q_norm_weight=bf16(f"{base}.self_attn.q_norm.weight"),
        k_norm_weight=bf16(f"{base}.self_attn.k_norm.weight"),
        router_weight=bf16(f"{base}.mlp.gate.weight"),
        attention=attention,
        attention_metadata=attention_metadata,
        experts=experts,
    )


def release_qwen3_moe_layer_state(state: MoELayerMasterState) -> None:
    """Drop every master reference held by a layer state before the next layer."""
    state.attention.clear()
    state.attention_metadata.clear()
    state.experts.clear()
    state.input_layernorm_weight = torch.empty(0)
    state.post_attention_layernorm_weight = torch.empty(0)
    state.q_norm_weight = torch.empty(0)
    state.k_norm_weight = torch.empty(0)
    state.router_weight = torch.empty(0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
