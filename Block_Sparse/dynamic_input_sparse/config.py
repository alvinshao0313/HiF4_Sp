from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DynamicInputMaskMethod(str, Enum):
    NONE = "none"
    M1_ORACLE = "m1_oracle"
    M8_ENERGY = "m8_energy"


@dataclass(frozen=True)
class DynamicInputSparseConfig:
    method: DynamicInputMaskMethod
    keep_ratio: float
    k_block_size: int = 64
    output_energy_block_size: int = 32
    mask_granularity: str = "per_token"
    m1_token_chunk_size: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.method, DynamicInputMaskMethod):
            object.__setattr__(
                self, "method", DynamicInputMaskMethod(str(self.method))
            )
        ratio = float(self.keep_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"keep_ratio must satisfy 0 < ratio <= 1, got {ratio}")
        if int(self.k_block_size) != 64:
            raise ValueError(f"k_block_size must be 64, got {self.k_block_size}")
        if int(self.output_energy_block_size) != 32:
            raise ValueError(
                f"output_energy_block_size must be 32, got {self.output_energy_block_size}"
            )
        if str(self.mask_granularity) != "per_token":
            raise ValueError(
                f"mask_granularity must be 'per_token', got {self.mask_granularity}"
            )
        if int(self.m1_token_chunk_size) != 8:
            raise ValueError(
                f"m1_token_chunk_size must be 8, got {self.m1_token_chunk_size}"
            )


def config_from_additional(additional_config: dict | None) -> DynamicInputSparseConfig | None:
    """Build runtime config from vLLM additional_config, or None if disabled."""
    cfg = additional_config or {}
    method_raw = cfg.get("dynamic_input_sparse_method", "none")
    method = DynamicInputMaskMethod(str(method_raw))
    if method == DynamicInputMaskMethod.NONE:
        return None
    return DynamicInputSparseConfig(
        method=method,
        keep_ratio=float(cfg.get("dynamic_input_keep_ratio", 1.0)),
        k_block_size=int(cfg.get("dynamic_input_k_block_size", 64)),
        output_energy_block_size=int(
            cfg.get("dynamic_input_output_energy_block_size", 32)
        ),
        mask_granularity=str(cfg.get("dynamic_input_mask_granularity", "per_token")),
        m1_token_chunk_size=int(cfg.get("dynamic_input_m1_token_chunk_size", 8)),
    )
