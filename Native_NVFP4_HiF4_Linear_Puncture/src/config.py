"""Experiment configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    EXPERIMENT_ROOT / "configs" / "qwen3_8b_native_nvfp4_linear_puncture.yaml"
)

TARGET_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Checkpoint physical key -> semantic field (fixed once for this experiment).
CHECKPOINT_KEY_SCHEMA = {
    "weight_packed": "qweight",
    "weight_scale": "scales",
    "weight_global_scale": "weight_global_scale",
    "input_global_scale": "act_global_scale",
    "rotation_matrix": "forward_hadamard_matrix",
}

SMOKE_MODULES = (
    "model.layers.2.self_attn.q_proj",
    "model.layers.18.mlp.down_proj",
    "model.layers.34.self_attn.o_proj",
)


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    expected_architecture: str
    expected_num_layers: int
    target_projections: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    formal_layers: tuple[int, ...]
    max_seq_len: int
    token_rows_per_prompt: int
    seed: int
    phase: str


@dataclass(frozen=True)
class Nvfp4Config:
    activation_group_size: int
    expected_payload: str
    expected_local_scale: str


@dataclass(frozen=True)
class Mxfp8Config:
    block_size: int
    payload: str
    shared_scale: str


@dataclass(frozen=True)
class Hif4Config:
    group_size: int
    s0_divisor: float
    e8_threshold: float
    e4_threshold: float
    s0_mode: str


@dataclass(frozen=True)
class WeightGreedyConfig:
    budget: str
    enumerate_e8_e4: bool
    memory_budget_fraction: float


@dataclass(frozen=True)
class DiagonalSearchConfig:
    parameterization: str
    coarse_log2_offsets: tuple[float, ...]
    refine_log2_offsets: tuple[float, ...]
    num_coarse_sweeps: int
    num_refine_sweeps: int
    log2_scale_min: float
    log2_scale_max: float
    search_token_rows_per_module: int
    search_output_channels_per_module: int
    full_calibration_rescore: bool
    eps: float
    group_size: int = 64


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    experiment: ExperimentConfig
    nvfp4: Nvfp4Config
    mxfp8: Mxfp8Config
    hif4: Hif4Config
    weight_greedy: WeightGreedyConfig
    diagonal_search: DiagonalSearchConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def formal_module_names(self) -> list[str]:
        names: list[str] = []
        for layer in self.experiment.formal_layers:
            for proj in self.model.target_projections:
                if proj in {"q_proj", "k_proj", "v_proj", "o_proj"}:
                    names.append(f"model.layers.{layer}.self_attn.{proj}")
                else:
                    names.append(f"model.layers.{layer}.mlp.{proj}")
        return names


def _as_tuple(values: list[Any]) -> tuple[Any, ...]:
    return tuple(values)


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model = ModelConfig(
        model_id=raw["model"]["model_id"],
        expected_architecture=raw["model"]["expected_architecture"],
        expected_num_layers=int(raw["model"]["expected_num_layers"]),
        target_projections=_as_tuple(raw["model"]["target_projections"]),
    )
    experiment = ExperimentConfig(
        formal_layers=_as_tuple(int(x) for x in raw["experiment"]["formal_layers"]),
        max_seq_len=int(raw["experiment"]["max_seq_len"]),
        token_rows_per_prompt=int(raw["experiment"]["token_rows_per_prompt"]),
        seed=int(raw["experiment"]["seed"]),
        phase=str(raw["experiment"]["phase"]),
    )
    nvfp4 = Nvfp4Config(
        activation_group_size=int(raw["nvfp4"]["activation_group_size"]),
        expected_payload=str(raw["nvfp4"]["expected_payload"]),
        expected_local_scale=str(raw["nvfp4"]["expected_local_scale"]),
    )
    mxfp8 = Mxfp8Config(
        block_size=int(raw["mxfp8"]["block_size"]),
        payload=str(raw["mxfp8"]["payload"]),
        shared_scale=str(raw["mxfp8"]["shared_scale"]),
    )
    hif4 = Hif4Config(
        group_size=int(raw["hif4"]["group_size"]),
        s0_divisor=float(raw["hif4"]["s0_divisor"]),
        e8_threshold=float(raw["hif4"]["e8_threshold"]),
        e4_threshold=float(raw["hif4"]["e4_threshold"]),
        s0_mode=str(raw["hif4"]["s0_mode"]),
    )
    weight_greedy = WeightGreedyConfig(
        budget=str(raw["weight_greedy"]["budget"]),
        enumerate_e8_e4=bool(raw["weight_greedy"]["enumerate_e8_e4"]),
        memory_budget_fraction=float(raw["weight_greedy"]["memory_budget_fraction"]),
    )
    ds = raw["diagonal_search"]
    diagonal_search = DiagonalSearchConfig(
        parameterization=str(ds["parameterization"]),
        coarse_log2_offsets=_as_tuple(float(x) for x in ds["coarse_log2_offsets"]),
        refine_log2_offsets=_as_tuple(float(x) for x in ds["refine_log2_offsets"]),
        num_coarse_sweeps=int(ds["num_coarse_sweeps"]),
        num_refine_sweeps=int(ds["num_refine_sweeps"]),
        log2_scale_min=float(ds["log2_scale_min"]),
        log2_scale_max=float(ds["log2_scale_max"]),
        search_token_rows_per_module=int(ds["search_token_rows_per_module"]),
        search_output_channels_per_module=int(ds["search_output_channels_per_module"]),
        full_calibration_rescore=bool(ds["full_calibration_rescore"]),
        eps=float(ds["eps"]),
        group_size=int(raw["hif4"]["group_size"]),
    )
    return AppConfig(
        model=model,
        experiment=experiment,
        nvfp4=nvfp4,
        mxfp8=mxfp8,
        hif4=hif4,
        weight_greedy=weight_greedy,
        diagonal_search=diagonal_search,
        raw=raw,
    )


def results_dir(run_id: str) -> Path:
    return EXPERIMENT_ROOT / "results" / run_id


def validate_forward_dtype(forward_dtype: str) -> None:
    if forward_dtype != "nvfp4":
        raise ValueError(f"forward_dtype must be nvfp4, got {forward_dtype!r}")
