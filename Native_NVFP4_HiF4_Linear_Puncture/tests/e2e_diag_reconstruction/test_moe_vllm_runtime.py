from __future__ import annotations

import pytest
import torch


def test_hif4_runtime_sidecar_cache_and_dense_hook(monkeypatch, tmp_path):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization import hif4_runtime

    sidecar = tmp_path / "hif4_runtime_spec.pt"
    trace = tmp_path / "trace.jsonl"
    torch.save({"model_type": "qwen3_moe", "variant": "direct"}, sidecar)
    hif4_runtime._SIDECAR_CACHE.clear()
    hif4_runtime._TRACE_SEEN.clear()
    monkeypatch.setenv("HIF4_RUNTIME_TRACE_JSONL", str(trace))
    calls = {"n": 0}

    def fake_quant(x):
        calls["n"] += 1
        return x + 1

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.hif4_fake.hif4_fake_quantize_hifx4",
        fake_quant,
    )
    with set_current_vllm_config(
        VllmConfig(additional_config={"hif4_runtime_spec_path": str(sidecar)})
    ):
        x = torch.zeros(2, 4)
        y = hif4_runtime.apply_dense_hif4_runtime("model.layers.0.self_attn.q_proj", x)
        z = hif4_runtime.apply_dense_hif4_runtime("model.layers.0.self_attn.k_proj", x)
    assert torch.equal(y, torch.ones_like(x))
    assert torch.equal(z, torch.ones_like(x))
    assert calls["n"] == 2
    assert list(hif4_runtime._SIDECAR_CACHE) == [str(sidecar)]
    trace_text = trace.read_text()
    assert '"event": "sidecar_load"' in trace_text
    assert '"event": "dense_apply"' in trace_text


def test_hif4_runtime_absent_sidecar_is_noop():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization import hif4_runtime

    x = torch.randn(2, 4)
    with set_current_vllm_config(VllmConfig(additional_config={})):
        y = hif4_runtime.apply_dense_hif4_runtime("model.layers.0.self_attn.q_proj", x)
    assert y is x


def _scalar_hif4_fused_moe(hidden, w13, w2, topk_weights, topk_ids, quant):
    ref = torch.zeros_like(hidden)
    x_q = quant(hidden)
    for t in range(hidden.shape[0]):
        acc = torch.zeros(hidden.shape[1], device=hidden.device, dtype=torch.float32)
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k].item())
            gu = torch.nn.functional.linear(x_q[t], w13[e])
            gate, up = gu.chunk(2, dim=-1)
            act_q = quant((torch.nn.functional.silu(gate) * up).unsqueeze(0)).squeeze(0)
            down = torch.nn.functional.linear(act_q, w2[e])
            acc += down.float() * float(topk_weights[t, k].item())
        ref[t] = acc.to(ref.dtype)
    return ref


def test_hif4_fused_moe_matches_manual_reference(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts.hif4_emulation_moe import (
        apply_hif4_fused_moe,
    )

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.hif4_fake.hif4_fake_quantize_hifx4",
        lambda x: x,
    )
    for experts in (2, 8, 128):
        torch.manual_seed(experts)
        hidden = torch.randn(4, 16, dtype=torch.bfloat16)
        w13 = torch.randn(experts, 24, 16, dtype=torch.bfloat16) * 0.01
        w2 = torch.randn(experts, 16, 12, dtype=torch.bfloat16) * 0.01
        topk_ids = torch.tensor([[0, 0], [min(1, experts - 1), 0], [experts - 1, 0], [0, min(1, experts - 1)]])
        topk_weights = torch.full((4, 2), 0.5, dtype=torch.float32)
        got = apply_hif4_fused_moe(hidden, w13, w2, topk_weights, topk_ids)
        ref = _scalar_hif4_fused_moe(hidden, w13, w2, topk_weights, topk_ids, lambda x: x)
        torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)


def test_hif4_fused_moe_grouped_matches_scalar_with_real_hif4():
    from vllm.model_executor.layers.fused_moe.experts.hif4_emulation_moe import (
        apply_hif4_fused_moe,
    )
    from vllm.model_executor.layers.quantization.hif4_fake import hif4_fake_quantize_hifx4

    torch.manual_seed(0)
    hidden = torch.randn(8, 64, dtype=torch.bfloat16)
    w13 = torch.randn(16, 32, 64, dtype=torch.bfloat16) * 0.01
    w2 = torch.randn(16, 64, 16, dtype=torch.bfloat16) * 0.01
    topk_ids = torch.tensor(
        [
            [0, 1],
            [1, 1],
            [15, 0],
            [3, 7],
            [3, 7],
            [8, 2],
            [0, 15],
            [9, 9],
        ]
    )
    topk_weights = torch.tensor(
        [
            [0.25, 0.75],
            [0.5, 0.5],
            [1.0, 0.0],
            [0.1, 0.9],
            [0.4, 0.6],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.55, 0.45],
        ],
        dtype=torch.float32,
    )
    got = apply_hif4_fused_moe(hidden, w13, w2, topk_weights, topk_ids)
    ref = _scalar_hif4_fused_moe(
        hidden, w13, w2, topk_weights, topk_ids, hif4_fake_quantize_hifx4
    )
    torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hif4_triton_qdq_matches_torch_reference():
    from vllm.model_executor.layers.quantization.hif4_fake import (
        _hif4_fake_quantize_hifx4_torch,
    )
    from vllm.model_executor.layers.quantization.hif4_triton import (
        hif4_quantize_hifx4_triton,
    )

    torch.manual_seed(17)
    for cols in (64, 65, 128, 192):
        x = torch.randn(5, cols, device="cuda", dtype=torch.bfloat16)
        got = hif4_quantize_hifx4_triton(x)
        ref = _hif4_fake_quantize_hifx4_torch(x)
        torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hif4_triton_grouped_moe_matches_scalar_reference():
    from vllm.model_executor.layers.fused_moe.experts.hif4_emulation_moe import (
        apply_hif4_fused_moe,
    )
    from vllm.model_executor.layers.quantization.hif4_fake import (
        _hif4_fake_quantize_hifx4_torch,
    )

    torch.manual_seed(23)
    experts = 8
    hidden = torch.randn(6, 64, device="cuda", dtype=torch.bfloat16)
    w13 = (
        torch.randn(experts, 128, 64, device="cuda", dtype=torch.bfloat16)
        * 0.01
    )
    w2 = (
        torch.randn(experts, 64, 64, device="cuda", dtype=torch.bfloat16)
        * 0.01
    )
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [7, 0], [3, 4], [5, 7], [6, 2]],
        device="cuda",
        dtype=torch.int64,
    )
    topk_weights = torch.tensor(
        [[0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.1, 0.9], [0.5, 0.5], [0.8, 0.2]],
        device="cuda",
        dtype=torch.float32,
    )
    got = apply_hif4_fused_moe(hidden, w13, w2, topk_weights, topk_ids)
    ref = _scalar_hif4_fused_moe(
        hidden,
        w13,
        w2,
        topk_weights,
        topk_ids,
        _hif4_fake_quantize_hifx4_torch,
    )
    torch.testing.assert_close(got, ref, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hif4_r64_qdq_matches_fp32_reference():
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.diag_gradient.r64_transform import (
        apply_r64_g64,
    )
    from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct
    from vllm.model_executor.layers.quantization.hif4_transform_triton import (
        hif4_r64_quantize_hifx4_triton,
    )

    torch.manual_seed(31)
    for cols in (64, 128, 2048):
        x = torch.randn(5, cols, device="cuda", dtype=torch.bfloat16)
        got = hif4_r64_quantize_hifx4_triton(x)
        rotated = apply_r64_g64(
            x.float(), dim=-1, compute_dtype=torch.float32, output_dtype=torch.float32
        )
        ref = qdq_hif4_direct(rotated, output_dtype=torch.bfloat16)
        torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("use_r64", "rot_order"),
    [
        (False, "diag_then_rot"),
        (True, "diag_then_rot"),
        (True, "rot_then_diag"),
    ],
)
def test_hif4_online_dense_qdq_matches_training_semantics(use_r64, rot_order):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_transforms import (
        apply_r64_no_cross_head,
    )
    from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct
    from vllm.model_executor.layers.quantization.hif4_transform_triton import (
        hif4_online_dense_qdq_triton,
    )

    torch.manual_seed(37)
    x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    d = torch.exp2(torch.linspace(-0.7, 0.8, 128, device="cuda"))
    got = hif4_online_dense_qdq_triton(
        x,
        d,
        use_r64=use_r64,
        rot_order=rot_order,
    )
    xf = x.float()
    if rot_order == "diag_then_rot":
        transformed = xf * d
        if use_r64:
            transformed = apply_r64_no_cross_head(transformed)
    else:
        transformed = apply_r64_no_cross_head(xf) if use_r64 else xf
        transformed = transformed * d
    ref = qdq_hif4_direct(transformed, output_dtype=torch.bfloat16)
    torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("use_r64", "rot_order"),
    [
        (False, "diag_then_rot"),
        (True, "diag_then_rot"),
        (True, "rot_then_diag"),
    ],
)
def test_hif4_online_routed_qdq_preserves_expert_specific_diag(use_r64, rot_order):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_transforms import (
        apply_r64_no_cross_head,
    )
    from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct
    from vllm.model_executor.layers.quantization.hif4_transform_triton import (
        hif4_online_routed_qdq_triton,
    )

    torch.manual_seed(41)
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    route_ids = torch.tensor(
        [[0, 2], [1, 3], [3, 0], [2, 1]], device="cuda", dtype=torch.int64
    )
    z = torch.stack(
        [torch.linspace(-0.9 + 0.2 * e, 0.4 + 0.1 * e, 128) for e in range(4)]
    ).to(device="cuda", dtype=torch.float32)
    expert_d = torch.exp2(z).contiguous()
    got = hif4_online_routed_qdq_triton(
        x,
        route_ids,
        expert_d,
        source_top_k=2,
        use_r64=use_r64,
        rot_order=rot_order,
    )

    refs = []
    for route_row, expert in enumerate(route_ids.reshape(-1).tolist()):
        xf = x[route_row // 2].float()
        d = expert_d[expert]
        if rot_order == "diag_then_rot":
            transformed = xf * d
            if use_r64:
                transformed = apply_r64_no_cross_head(transformed)
        else:
            transformed = apply_r64_no_cross_head(xf) if use_r64 else xf
            transformed = transformed * d
        refs.append(qdq_hif4_direct(transformed, output_dtype=torch.bfloat16))
    ref = torch.stack(refs)
    torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)


def test_hif4_online_reference_moe_matches_manual_reference(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts.hif4_online_reference_moe import (
        apply_hif4_online_reference_moe,
    )

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.hif4_fake.hif4_fake_quantize_hifx4",
        lambda x: x,
    )
    torch.manual_seed(0)
    hidden = torch.randn(3, 16, dtype=torch.bfloat16)
    gate = torch.randn(4, 12, 16, dtype=torch.bfloat16) * 0.01
    up = torch.randn(4, 12, 16, dtype=torch.bfloat16) * 0.01
    down = torch.randn(4, 16, 12, dtype=torch.bfloat16) * 0.01
    z_gate = torch.zeros(4, 16)
    z_up = torch.zeros(4, 16)
    z_down = torch.zeros(4, 12)
    topk_ids = torch.tensor([[0, 1], [1, 1], [3, 0]])
    topk_weights = torch.tensor([[0.25, 0.75], [0.5, 0.5], [1.0, 0.0]])
    got = apply_hif4_online_reference_moe(
        hidden, gate, up, down, topk_weights, topk_ids, z_gate, z_up, z_down
    )
    ref = torch.zeros_like(hidden)
    for t in range(hidden.shape[0]):
        acc = torch.zeros(16)
        for k in range(2):
            e = int(topk_ids[t, k])
            g = torch.nn.functional.linear(hidden[t], gate[e])
            u = torch.nn.functional.linear(hidden[t], up[e])
            d = torch.nn.functional.linear(torch.nn.functional.silu(g) * u, down[e])
            acc += d.float() * float(topk_weights[t, k])
        ref[t] = acc.to(ref.dtype)
    torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)
