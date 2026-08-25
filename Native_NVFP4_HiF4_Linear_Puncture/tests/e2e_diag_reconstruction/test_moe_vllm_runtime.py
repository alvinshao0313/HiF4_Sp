from __future__ import annotations

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
        ref = torch.zeros_like(hidden)
        for t in range(hidden.shape[0]):
            acc = torch.zeros(16)
            for k in range(2):
                e = int(topk_ids[t, k])
                gu = torch.nn.functional.linear(hidden[t], w13[e])
                gate, up = gu.chunk(2, dim=-1)
                down = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, w2[e])
                acc += down.float() * 0.5
            ref[t] = acc.to(ref.dtype)
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
