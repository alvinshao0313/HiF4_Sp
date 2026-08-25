from __future__ import annotations

from pathlib import Path

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import (
    QWEN3_30B_A3B_NVFP4,
    load_qwen3_moe_model_spec,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot


def test_real_qwen3_moe_nvfp4_contract():
    snapshot = resolve_local_snapshot(QWEN3_30B_A3B_NVFP4)
    assert Path(snapshot).is_dir()
    spec = load_qwen3_moe_model_spec(QWEN3_30B_A3B_NVFP4)
    assert spec.architecture == "Qwen3MoeForCausalLM"
    assert spec.model_type == "qwen3_moe"
    assert (spec.num_layers, spec.num_experts, spec.top_k) == (48, 128, 8)
    assert (spec.hidden_size, spec.num_attention_heads, spec.num_key_value_heads, spec.head_dim) == (2048, 32, 4, 128)
    assert spec.moe_intermediate_size == 768
    assert spec.attention_linear_count == 192
    assert spec.expert_linear_count == 18432
    assert spec.target_linear_count == 18624
    assert spec.last_layer == 47
    assert spec.kv_cache_dtype == "bfloat16"
    assert not spec.native_has_online_rotation
