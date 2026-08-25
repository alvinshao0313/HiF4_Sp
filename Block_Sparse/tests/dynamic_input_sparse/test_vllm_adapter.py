from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.dynamic_input_sparse.common import is_target_mlp_prefix  # noqa: E402
from Block_Sparse.dynamic_input_sparse.config import (  # noqa: E402
    DynamicInputMaskMethod,
    DynamicInputSparseConfig,
)
from Block_Sparse.dynamic_input_sparse.vllm_adapter import (  # noqa: E402
    mask_linear_input,
    setup_dynamic_input_sparse_for_layer,
    validate_parallelism_for_dynamic_input,
)


@pytest.mark.parametrize(
    "prefix,expect",
    [
        ("model.layers.0.mlp.gate_up_proj", True),
        ("model.layers.3.mlp.down_proj", True),
        ("model.layers.0.self_attn.qkv_proj", False),
        ("model.layers.0.self_attn.q_proj", False),
        ("model.layers.0.self_attn.k_proj", False),
        ("model.layers.0.self_attn.v_proj", False),
        ("model.layers.0.self_attn.o_proj", False),
        ("model.layers.0.linear_attn.in_proj_qkvz", False),
        ("lm_head", False),
        ("model.embed_tokens", False),
        ("model.layers.0.mlp.experts.0.down_proj", False),
    ],
)
def test_prefix_gates(prefix, expect):
    assert is_target_mlp_prefix(prefix) is expect


def test_tp_gt_1_fails():
    with pytest.raises(RuntimeError, match="TP=1"):
        validate_parallelism_for_dynamic_input(
            tensor_parallel_size=2, method=DynamicInputMaskMethod.M8_ENERGY
        )


def test_setup_and_mask_target():
    cfg = DynamicInputSparseConfig(
        method=DynamicInputMaskMethod.M8_ENERGY, keep_ratio=0.5
    )
    layer = nn.Linear(256, 64, bias=False)
    layer.prefix = "model.layers.0.mlp.gate_up_proj"
    assert setup_dynamic_input_sparse_for_layer(layer, cfg)
    x = torch.randn(3, 256)
    y = mask_linear_input(layer, x, cfg)
    assert y.shape == x.shape
    assert not torch.equal(y, x)


def test_nontarget_passthrough():
    cfg = DynamicInputSparseConfig(
        method=DynamicInputMaskMethod.M8_ENERGY, keep_ratio=0.5
    )
    layer = nn.Linear(256, 64, bias=False)
    layer.prefix = "model.layers.0.self_attn.o_proj"
    assert not setup_dynamic_input_sparse_for_layer(layer, cfg)
    x = torch.randn(3, 256)
    y = mask_linear_input(layer, x, cfg)
    assert torch.equal(y, x)
