from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    load_qwen3_moe_layer_state,
    release_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import QWEN3_30B_A3B_NVFP4
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    NativeQwen3MoELayerRuntime,
    qdq_native_nvfp4,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_layer_runtime import (
    build_qwen3_moe_layer_call,
    call_native_qwen3_moe_layer,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant_dequant


DENSE_ATOL = 0.0
DENSE_RTOL = 0.0
MOE_ATOL = 0.05
MOE_RTOL = 0.01


def _load_keys(snapshot: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index["weight_map"][key], []).append(key)
    out: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as handle:
            for key in shard_keys:
                out[key] = handle.get_tensor(key)
    return out


def _vllm_dense_emulation(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_global_scale: torch.Tensor,
) -> torch.Tensor:
    from torch.nn.parameter import Parameter

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.kernel import KernelConfig
    from vllm.model_executor.kernels.linear.nvfp4.emulation import EmulationNvFp4LinearKernel
    from vllm.model_executor.kernels.linear.nvfp4.select import init_nvfp4_linear_kernel

    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend="emulation"))
    ):
        kernel = init_nvfp4_linear_kernel()
        assert isinstance(kernel, EmulationNvFp4LinearKernel)
        layer = torch.nn.Module()
        layer.weight = Parameter(packed_weight.to(x.device), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.to(x.device), requires_grad=False)
        layer.input_global_scale_inv = Parameter(
            input_global_scale_inv.to(device=x.device, dtype=torch.float32).reshape(()),
            requires_grad=False,
        )
        layer.weight_global_scale = Parameter(
            weight_global_scale.to(device=x.device, dtype=torch.float32).reshape(()),
            requires_grad=False,
        )
        kernel.process_weights_after_loading(layer)
        return kernel.apply_weights(layer, x, bias=None)


def test_native_activation_qdq_is_vllm_emulation():
    torch.manual_seed(7)
    x = torch.randn(3, 32, dtype=torch.bfloat16)
    global_scale_inv = torch.tensor(8192.0, dtype=torch.float32)
    got = qdq_native_nvfp4(x, global_scale_inv)
    expected = ref_nvfp4_quant_dequant(x, global_scale_inv, block_size=16)
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real native teacher gates require CUDA")
def test_native_attention_projections_match_vllm_emulation_puncture():
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import _rms_norm

    snapshot = Path(resolve_local_snapshot(QWEN3_30B_A3B_NVFP4))
    state = load_qwen3_moe_layer_state(snapshot, 0, "cuda")
    try:
        runtime = NativeQwen3MoELayerRuntime(state).cuda().eval()
        torch.manual_seed(0)
        x = torch.randn(1, 2, 2048, device="cuda", dtype=torch.bfloat16)
        call = build_qwen3_moe_layer_call(str(snapshot), x)
        with torch.no_grad():
            attn_input = _rms_norm(x, state.input_layernorm_weight, runtime.rms_norm_eps)
            got = runtime.attention_projections(attn_input, None, call.position_embeddings)

        prefixes = {proj: f"model.layers.0.self_attn.{proj}" for proj in ("q_proj", "k_proj", "v_proj", "o_proj")}
        keys = [f"{prefix}.{suffix}" for prefix in prefixes.values() for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")]
        tensors = _load_keys(snapshot, keys)
        qkv_input = torch.stack([tensors[f"{prefixes[p]}.input_scale"].reshape(()).float() for p in ("q_proj", "k_proj", "v_proj")]).max()
        qkv_weight = torch.stack([tensors[f"{prefixes[p]}.weight_scale_2"].reshape(()).float() for p in ("q_proj", "k_proj", "v_proj")]).max()
        flat = attn_input.reshape(-1, attn_input.shape[-1])
        for proj, observed in (("q_proj", got.q), ("k_proj", got.k), ("v_proj", got.v)):
            prefix = prefixes[proj]
            expected = _vllm_dense_emulation(
                flat,
                tensors[f"{prefix}.weight"],
                tensors[f"{prefix}.weight_scale"],
                1.0 / qkv_input,
                qkv_weight,
            ).reshape_as(observed)
            torch.testing.assert_close(observed, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)

        prefix = prefixes["o_proj"]
        expected_o = _vllm_dense_emulation(
            got.o_input.reshape(-1, got.o_input.shape[-1]),
            tensors[f"{prefix}.weight"],
            tensors[f"{prefix}.weight_scale"],
            1.0 / tensors[f"{prefix}.input_scale"].reshape(()).float(),
            tensors[f"{prefix}.weight_scale_2"].reshape(()).float(),
        ).reshape_as(got.o)
        torch.testing.assert_close(got.o, expected_o, atol=DENSE_ATOL, rtol=DENSE_RTOL)
    finally:
        release_qwen3_moe_layer_state(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real native teacher gates require CUDA")
def test_native_router_logits_and_topk_match_hf_reference():
    snapshot = Path(resolve_local_snapshot(QWEN3_30B_A3B_NVFP4))
    state = load_qwen3_moe_layer_state(snapshot, 0, "cuda")
    try:
        runtime = NativeQwen3MoELayerRuntime(state).cuda().eval()
        torch.manual_seed(1)
        x = torch.randn(1, 4, 2048, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            got = runtime.routed_moe(x)
            flat = x.reshape(-1, x.shape[-1])
            logits = F.linear(flat, state.router_weight)
            probs = torch.softmax(logits, dtype=torch.float32, dim=-1)
            weights, indices = torch.topk(probs, state.spec.top_k, dim=-1)
            weights = (weights / weights.sum(dim=-1, keepdim=True)).to(logits.dtype)
        torch.testing.assert_close(got.router_logits, logits, atol=0.0, rtol=0.0)
        torch.testing.assert_close(got.routing_weights, weights, atol=0.0, rtol=0.0)
        assert torch.equal(got.selected_experts, indices)
    finally:
        release_qwen3_moe_layer_state(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real native teacher gates require CUDA")
def test_native_routed_moe_output_matches_vllm_emulation_puncture():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
        nvfp4_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
        Nvfp4QuantizationEmulationTritonExperts,
    )

    snapshot = Path(resolve_local_snapshot(QWEN3_30B_A3B_NVFP4))
    state = load_qwen3_moe_layer_state(snapshot, 0, "cuda")
    try:
        runtime = NativeQwen3MoELayerRuntime(state).cuda().eval()
        torch.manual_seed(2)
        x = torch.randn(1, 4, 2048, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            got = runtime.routed_moe(x)

        keys: list[str] = []
        for expert in range(state.spec.num_experts):
            base = f"model.layers.0.mlp.experts.{expert}"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                    keys.append(f"{base}.{proj}.{suffix}")
        tensors = _load_keys(snapshot, keys)

        w1, w1_scale, w1_gscale = [], [], []
        w2, w2_scale, w2_gscale = [], [], []
        a13_scales, a2_scales = [], []
        for expert in range(state.spec.num_experts):
            base = f"model.layers.0.mlp.experts.{expert}"
            gate = f"{base}.gate_proj"
            up = f"{base}.up_proj"
            down = f"{base}.down_proj"
            w1.append(torch.cat([tensors[f"{gate}.weight"], tensors[f"{up}.weight"]], dim=0))
            w1_scale.append(torch.cat([tensors[f"{gate}.weight_scale"], tensors[f"{up}.weight_scale"]], dim=0))
            w1_gscale.append(tensors[f"{gate}.weight_scale_2"].reshape(()).float())
            w2.append(tensors[f"{down}.weight"])
            w2_scale.append(tensors[f"{down}.weight_scale"])
            w2_gscale.append(tensors[f"{down}.weight_scale_2"].reshape(()).float())
            a13_scales.extend([
                tensors[f"{gate}.input_scale"].reshape(()).float(),
                tensors[f"{up}.input_scale"].reshape(()).float(),
            ])
            a2_scales.append(tensors[f"{down}.input_scale"].reshape(()).float())

        w1_t = torch.stack(w1).cuda()
        w1_scale_t = torch.stack(w1_scale).cuda()
        w1_gscale_t = torch.stack(w1_gscale).cuda()
        w2_t = torch.stack(w2).cuda()
        w2_scale_t = torch.stack(w2_scale).cuda()
        w2_gscale_t = torch.stack(w2_gscale).cuda()
        a1_gscale = (1.0 / torch.stack(a13_scales).max()).cuda()
        a2_gscale = (1.0 / torch.stack(a2_scales).max()).cuda()

        moe_config = FusedMoEConfig(
            num_experts=state.spec.num_experts,
            experts_per_token=state.spec.top_k,
            hidden_dim=state.spec.hidden_size,
            intermediate_size_per_partition=state.spec.moe_intermediate_size,
            num_local_experts=state.spec.num_experts,
            num_logical_experts=state.spec.num_experts,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            activation=MoEActivation.SILU,
            in_dtype=torch.bfloat16,
            device="cuda",
            routing_method=RoutingMethodType.TopK,
            moe_backend="emulation",
        )
        experts = Nvfp4QuantizationEmulationTritonExperts(
            moe_config=moe_config,
            quant_config=nvfp4_moe_quant_config(
                g1_alphas=w1_gscale_t,
                g2_alphas=w2_gscale_t,
                a1_gscale=a1_gscale,
                a2_gscale=a2_gscale,
                w1_scale=w1_scale_t,
                w2_scale=w2_scale_t,
            ),
        )
        flat = x.reshape(-1, x.shape[-1])
        output = torch.zeros_like(flat)
        ws13 = torch.zeros(
            flat.shape[0] * state.spec.top_k * max(state.spec.moe_intermediate_size, state.spec.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        ws2 = torch.zeros_like(ws13)
        experts.apply(
            output=output,
            hidden_states=flat,
            w1=w1_t,
            w2=w2_t,
            topk_weights=got.routing_weights.reshape(flat.shape[0], state.spec.top_k),
            topk_ids=got.selected_experts.reshape(flat.shape[0], state.spec.top_k).to(torch.int32),
            activation=MoEActivation.SILU,
            global_num_experts=state.spec.num_experts,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=ws13,
            workspace2=ws2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        torch.testing.assert_close(got.output.reshape_as(output), output, atol=MOE_ATOL, rtol=MOE_RTOL)
    finally:
        release_qwen3_moe_layer_state(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native MoE runtime smoke requires CUDA")
def test_native_moe_layer_runtime_smoke():
    snapshot = Path(resolve_local_snapshot(QWEN3_30B_A3B_NVFP4))
    state = load_qwen3_moe_layer_state(snapshot, 0, "cuda")
    try:
        runtime = NativeQwen3MoELayerRuntime(state).cuda().eval()
        x = torch.randn(1, 2, 2048, device="cuda", dtype=torch.bfloat16)
        call = build_qwen3_moe_layer_call(str(snapshot), x)
        with torch.no_grad():
            result = call_native_qwen3_moe_layer(runtime, call)
        assert result.output.shape == x.shape
        assert result.router_logits.shape == (2, 128)
        assert result.selected_experts.shape == (2, 8)
        assert torch.isfinite(result.output).all()
    finally:
        release_qwen3_moe_layer_state(state)
