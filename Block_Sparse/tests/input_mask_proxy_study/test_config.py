from __future__ import annotations

import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.input_mask_proxy_study.config import (  # noqa: E402
    ExperimentConfig,
    MethodId,
    load_config,
    ratio_to_keep_count,
)


def _valid_config_dict(**overrides) -> dict:
    base = {
        "model_path": "Qwen/Qwen3.5-4B",
        "dataset_hf_id": "simplescaling/s1K-1.1_tokenized",
        "seed": 31,
        "num_samples": 8,
        "max_seq_len": 1024,
        "max_activation_blocks": 256,
        "layer_index": 15,
        "projection": "up_proj",
        "activation_block_rows": 32,
        "k_block_size": 64,
        "output_block_cols": 32,
        "output_keep_ratios": [0.25, 0.5, 0.75],
        "input_keep_ratios": [0.25, 0.5, 0.75],
        "model_dtype": "bfloat16",
        "compute_dtype": "float32",
        "warmup": 5,
        "fast_repeats": 30,
        "exact_repeats": 5,
    }
    base.update(overrides)
    return base


def test_method_ids_exactly_eight():
    ids = [m.value for m in MethodId]
    assert ids == [
        "full_exact_ref",
        "xproxy_exact_own_output",
        "xproxy_energy_own_output",
        "full_energy_ref_output",
        "xwproxy_exact_ref_output",
        "xwproxy_exact_own_output",
        "xproxy_s0mean_energy_own_output",
        "xproxy_energy_unconditioned_own_output",
    ]
    assert len(MethodId) == 8


def test_load_config_accepts_valid(tmp_path: Path):
    path = tmp_path / "full.json"
    path.write_text(json.dumps(_valid_config_dict()), encoding="utf-8")
    cfg = load_config(path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.layer_index == 15
    assert cfg.projection == "up_proj"
    assert cfg.activation_block_rows == 32
    assert cfg.k_block_size == 64
    assert cfg.output_block_cols == 32
    assert cfg.model_dtype == "bfloat16"
    assert cfg.compute_dtype == "float32"
    assert cfg.output_keep_ratios == (0.25, 0.5, 0.75)


@pytest.mark.parametrize(
    "overrides",
    [
        {"layer_index": 14},
        {"projection": "gate_proj"},
        {"activation_block_rows": 16},
        {"k_block_size": 32},
        {"output_block_cols": 64},
        {"model_dtype": "float16"},
        {"compute_dtype": "bfloat16"},
        {"output_keep_ratios": [0.5, 0.25, 0.75]},
        {"input_keep_ratios": [0.0, 0.5]},
        {"output_keep_ratios": [1.5]},
        {"num_samples": 0},
        {"max_seq_len": 1000},  # not divisible by 32
        {"max_activation_blocks": 32},  # != 8*(1024/32)=256
    ],
)
def test_load_config_rejects_invalid(tmp_path: Path, overrides: dict):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(_valid_config_dict(**overrides)), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    "ratio,total,expected",
    [
        (0.25, 4, 1),
        (0.5, 4, 2),
        (0.75, 4, 3),
        (0.25, 3, 1),  # 0.75 -> ROUND_HALF_UP -> 1
        (0.5, 1, 1),
        (1.0, 7, 7),
        (0.125, 4, 1),  # 0.5 -> 1 via half-up, then max(1, ...)
    ],
)
def test_ratio_to_keep_count_round_half_up(ratio: float, total: int, expected: int):
    assert ratio_to_keep_count(ratio, total) == expected
    raw = Decimal(str(ratio)) * Decimal(total)
    rounded = int(raw.to_integral_value(rounding=ROUND_HALF_UP))
    assert max(1, rounded) == expected


def test_ratio_to_keep_count_never_zero():
    assert ratio_to_keep_count(0.01, 10) == 1
