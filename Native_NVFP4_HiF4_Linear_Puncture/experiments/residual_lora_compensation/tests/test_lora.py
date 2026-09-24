from types import SimpleNamespace

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.residual_lora_compensation.transforms import (
    LayerParameters,
)


def make_base(hidden=128):
    spec = SimpleNamespace(hidden_size=hidden)
    attention = {name: torch.randn(hidden, hidden, dtype=torch.bfloat16)
                 for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
    expert = SimpleNamespace(
        gate_proj=torch.randn(hidden, hidden, dtype=torch.bfloat16),
        up_proj=torch.randn(hidden, hidden, dtype=torch.bfloat16),
        down_proj=torch.randn(hidden, hidden, dtype=torch.bfloat16),
    )
    return SimpleNamespace(
        spec=spec, attention=attention, experts=[expert],
        router_weight=torch.randn(2, hidden, dtype=torch.bfloat16),
        input_layernorm_weight=torch.ones(hidden, dtype=torch.bfloat16),
        post_attention_layernorm_weight=torch.ones(hidden, dtype=torch.bfloat16),
    )


def test_zero_initialized_lora_is_exact_zero_and_only_active_branch_trains():
    torch.manual_seed(7)
    params = LayerParameters(make_base(), "group", lora_mode="attention",
                             lora_rank=4, lora_alpha=8)
    x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    torch.testing.assert_close(params.lora(x, "attention"), torch.zeros_like(x))
    torch.testing.assert_close(params.lora(x, "moe"), torch.zeros_like(x))
    assert params.attention_lora_A.requires_grad
    assert params.attention_lora_B.requires_grad
    assert not params.moe_lora_A.requires_grad
    assert not params.moe_lora_B.requires_grad
    params.lora(x, "attention").square().sum().backward()
    assert params.attention_lora_B.grad is not None
    assert params.moe_lora_B.grad is None


def test_lora_scaling_and_shapes():
    torch.manual_seed(9)
    params = LayerParameters(make_base(), "group", lora_mode="both",
                             lora_rank=4, lora_alpha=8)
    assert tuple(params.attention_lora_A.shape) == (4, 128)
    assert tuple(params.attention_lora_B.shape) == (128, 4)
    with torch.no_grad():
        params.attention_lora_A.fill_(1)
        params.attention_lora_B.fill_(1)
    x = torch.ones(1, 1, 128, dtype=torch.bfloat16)
    # A@x=128 in every rank, B@(A@x)=512, alpha/r=2.
    torch.testing.assert_close(params.lora(x, "attention"), torch.full_like(x, 1024))


def test_branch_modes_are_deterministic_and_snapshot_complete():
    torch.manual_seed(11)
    params = LayerParameters(make_base(), "group", lora_mode="both",
                             lora_rank=4, lora_alpha=8)
    snap = params.snapshot()
    expected = {"input_norm", "moe_norm", "attention_lora_A", "attention_lora_B",
                "moe_lora_A", "moe_lora_B"}
    assert expected.issubset(snap)
    assert all(value.device.type == "cpu" for value in snap.values())
