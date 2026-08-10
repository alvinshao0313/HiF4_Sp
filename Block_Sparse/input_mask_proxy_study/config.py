from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from pathlib import Path
from typing import Any


class MethodId(str, Enum):
    FULL_EXACT_REF = "full_exact_ref"
    XPROXY_EXACT_OWN_OUTPUT = "xproxy_exact_own_output"
    XPROXY_ENERGY_OWN_OUTPUT = "xproxy_energy_own_output"
    FULL_ENERGY_REF_OUTPUT = "full_energy_ref_output"
    XWPROXY_EXACT_REF_OUTPUT = "xwproxy_exact_ref_output"
    XWPROXY_EXACT_OWN_OUTPUT = "xwproxy_exact_own_output"
    XPROXY_S0MEAN_ENERGY_OWN_OUTPUT = "xproxy_s0mean_energy_own_output"
    XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT = "xproxy_energy_unconditioned_own_output"


@dataclass(frozen=True)
class ExperimentConfig:
    model_path: str
    dataset_hf_id: str
    seed: int
    num_samples: int
    max_seq_len: int
    max_activation_blocks: int
    layer_index: int
    projection: str
    activation_block_rows: int
    k_block_size: int
    output_block_cols: int
    output_keep_ratios: tuple[float, ...]
    input_keep_ratios: tuple[float, ...]
    model_dtype: str
    compute_dtype: str
    warmup: int
    fast_repeats: int
    exact_repeats: int

    def __post_init__(self) -> None:
        _validate_config(self)


def _as_float_tuple(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(f"{field} must be a non-empty list/tuple of floats")
    out = tuple(float(v) for v in value)
    return out


def _validate_ratios(ratios: tuple[float, ...], field: str) -> None:
    if any(not (0.0 < r <= 1.0) for r in ratios):
        raise ValueError(f"{field} values must satisfy 0 < r <= 1, got {ratios}")
    if any(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1)):
        raise ValueError(f"{field} must be strictly ascending, got {ratios}")


def _validate_config(cfg: ExperimentConfig) -> None:
    if not isinstance(cfg.model_path, str) or not cfg.model_path.strip():
        raise ValueError("model_path must be a non-empty string")
    if not isinstance(cfg.dataset_hf_id, str) or not cfg.dataset_hf_id.strip():
        raise ValueError("dataset_hf_id must be a non-empty string")
    if int(cfg.seed) < 0:
        raise ValueError(f"seed must be >= 0, got {cfg.seed}")
    if int(cfg.num_samples) < 1:
        raise ValueError(f"num_samples must be >= 1, got {cfg.num_samples}")
    if int(cfg.max_seq_len) < 1:
        raise ValueError(f"max_seq_len must be >= 1, got {cfg.max_seq_len}")
    if int(cfg.max_activation_blocks) < 1:
        raise ValueError(
            f"max_activation_blocks must be >= 1, got {cfg.max_activation_blocks}"
        )
    if int(cfg.layer_index) != 15:
        raise ValueError(f"layer_index must be 15, got {cfg.layer_index}")
    if cfg.projection != "up_proj":
        raise ValueError(f"projection must be 'up_proj', got {cfg.projection!r}")
    if int(cfg.activation_block_rows) != 32:
        raise ValueError(
            f"activation_block_rows must be 32, got {cfg.activation_block_rows}"
        )
    if int(cfg.max_seq_len) % int(cfg.activation_block_rows) != 0:
        raise ValueError(
            f"max_seq_len={cfg.max_seq_len} must be divisible by "
            f"activation_block_rows={cfg.activation_block_rows}"
        )
    blocks_per_sample = int(cfg.max_seq_len) // int(cfg.activation_block_rows)
    expected_blocks = int(cfg.num_samples) * blocks_per_sample
    if int(cfg.max_activation_blocks) != expected_blocks:
        raise ValueError(
            "max_activation_blocks must equal num_samples * "
            f"(max_seq_len / activation_block_rows) = {expected_blocks}, "
            f"got {cfg.max_activation_blocks}"
        )
    if int(cfg.k_block_size) != 64:
        raise ValueError(f"k_block_size must be 64, got {cfg.k_block_size}")
    if int(cfg.output_block_cols) != 32:
        raise ValueError(f"output_block_cols must be 32, got {cfg.output_block_cols}")
    if cfg.model_dtype != "bfloat16":
        raise ValueError(f"model_dtype must be 'bfloat16', got {cfg.model_dtype!r}")
    if cfg.compute_dtype != "float32":
        raise ValueError(f"compute_dtype must be 'float32', got {cfg.compute_dtype!r}")
    if int(cfg.warmup) < 0:
        raise ValueError(f"warmup must be >= 0, got {cfg.warmup}")
    if int(cfg.fast_repeats) < 1:
        raise ValueError(f"fast_repeats must be >= 1, got {cfg.fast_repeats}")
    if int(cfg.exact_repeats) < 1:
        raise ValueError(f"exact_repeats must be >= 1, got {cfg.exact_repeats}")
    _validate_ratios(cfg.output_keep_ratios, "output_keep_ratios")
    _validate_ratios(cfg.input_keep_ratios, "input_keep_ratios")


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a JSON object, got {type(raw)}")

    required = {
        "model_path",
        "dataset_hf_id",
        "seed",
        "num_samples",
        "max_seq_len",
        "max_activation_blocks",
        "layer_index",
        "projection",
        "activation_block_rows",
        "k_block_size",
        "output_block_cols",
        "output_keep_ratios",
        "input_keep_ratios",
        "model_dtype",
        "compute_dtype",
        "warmup",
        "fast_repeats",
        "exact_repeats",
    }
    missing = required - set(raw)
    extra = set(raw) - required
    if missing:
        raise ValueError(f"config missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"config has unknown fields: {sorted(extra)}")

    return ExperimentConfig(
        model_path=str(raw["model_path"]),
        dataset_hf_id=str(raw["dataset_hf_id"]),
        seed=int(raw["seed"]),
        num_samples=int(raw["num_samples"]),
        max_seq_len=int(raw["max_seq_len"]),
        max_activation_blocks=int(raw["max_activation_blocks"]),
        layer_index=int(raw["layer_index"]),
        projection=str(raw["projection"]),
        activation_block_rows=int(raw["activation_block_rows"]),
        k_block_size=int(raw["k_block_size"]),
        output_block_cols=int(raw["output_block_cols"]),
        output_keep_ratios=_as_float_tuple(raw["output_keep_ratios"], "output_keep_ratios"),
        input_keep_ratios=_as_float_tuple(raw["input_keep_ratios"], "input_keep_ratios"),
        model_dtype=str(raw["model_dtype"]),
        compute_dtype=str(raw["compute_dtype"]),
        warmup=int(raw["warmup"]),
        fast_repeats=int(raw["fast_repeats"]),
        exact_repeats=int(raw["exact_repeats"]),
    )


def ratio_to_keep_count(ratio: float, total: int) -> int:
    if not (0.0 < float(ratio) <= 1.0):
        raise ValueError(f"ratio must satisfy 0 < ratio <= 1, got {ratio}")
    if int(total) < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    product = Decimal(str(ratio)) * Decimal(int(total))
    keep = int(product.to_integral_value(rounding=ROUND_HALF_UP))
    return max(1, keep)
