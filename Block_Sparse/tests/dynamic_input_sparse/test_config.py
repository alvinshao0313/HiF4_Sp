from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.dynamic_input_sparse.config import (  # noqa: E402
    DynamicInputMaskMethod,
    DynamicInputSparseConfig,
)


def test_valid_config():
    cfg = DynamicInputSparseConfig(
        method=DynamicInputMaskMethod.M8_ENERGY, keep_ratio=0.5
    )
    assert cfg.k_block_size == 64
    assert cfg.m1_token_chunk_size == 8


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1])
def test_invalid_keep_ratio(ratio):
    with pytest.raises(ValueError):
        DynamicInputSparseConfig(method=DynamicInputMaskMethod.M8_ENERGY, keep_ratio=ratio)


def test_invalid_block_size():
    with pytest.raises(ValueError):
        DynamicInputSparseConfig(
            method=DynamicInputMaskMethod.M8_ENERGY,
            keep_ratio=0.5,
            k_block_size=32,
        )


def test_invalid_granularity():
    with pytest.raises(ValueError):
        DynamicInputSparseConfig(
            method=DynamicInputMaskMethod.M8_ENERGY,
            keep_ratio=0.5,
            mask_granularity="shared_32",
        )
