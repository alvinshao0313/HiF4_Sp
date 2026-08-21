"""Experiment configuration loading and validation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from Inference_Paradigm_Conversion.ipc_analysis import SOURCE_SEMANTICS_ALLOWED

REPO_ROOT = Path("/home/shaoyuantian/program/HiF4_Sp")
IPC_ROOT = REPO_ROOT / "Inference_Paradigm_Conversion"
FORMAL_CONFIG_PATH = IPC_ROOT / "configs" / "qwen3_8b_nvfp4_qat_formal.yaml"
ACTIVATION_SCALES_FILENAME = "nvfp4_activation_scales.safetensors"

LINEAR_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class ModelConfig:
    source_checkpoint: str
    source_semantics: str
    optional_fp32_storage_probe: str
    original_training_source: str
    hadamard_runtime: str
    hadamard_reason: str


@dataclass(frozen=True)
class ActivationNvfp4Config:
    block_size: int
    global_scale_dtype: str
    global_scale_granularity: str
    scale_file_resolution: str
    calibration_dataset: str
    calibration_nsamples: int
    calibration_seqlen: int
    calibration_slice_mode: str
    calibration_slice_offset: int


@dataclass(frozen=True)
class AnalysisConfig:
    representative_layer_policy: str
    representative_layer_count: int
    token_sample_per_module_per_phase: int
    raw_token_sample_per_module_per_phase: int
    token_chunk_size: int
    samples_per_prompt_family: int
    discovery_samples_per_prompt_family: int
    validation_samples_per_prompt_family: int
    network_scan_samples: int
    network_scan_max_length: int
    sequence_length_buckets: list[int]
    prequant_activation_dtype: str
    qdq_output_dtype: str
    accumulator_dtype: str


@dataclass(frozen=True)
class ActivationConfig:
    phases: list[str]
    decode_steps_per_prompt_for_analysis: int
    exclude_modules: list[str]
    p2_use_matched_coverage: bool


@dataclass(frozen=True)
class BootstrapConfig:
    repeats: int
    confidence: float


@dataclass(frozen=True)
class E2EConfig:
    mmlu_pro_max_samples: int
    arc_num_fewshot: int
    aime25_repeats: int


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    repo_root: Path
    model: ModelConfig
    activation_nvfp4: ActivationNvfp4Config
    analysis: AnalysisConfig
    activation: ActivationConfig
    bootstrap: BootstrapConfig
    e2e: E2EConfig

    def source_checkpoint_path(self) -> Path:
        p = Path(self.model.source_checkpoint)
        if not p.is_absolute():
            p = self.repo_root / p
        return p.resolve()

    def optional_fp32_probe_path(self) -> Path:
        p = Path(self.model.optional_fp32_storage_probe)
        if not p.is_absolute():
            p = self.repo_root / p
        return p.resolve()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["repo_root"] = str(self.repo_root)
        return d


def resolve_representative_layers(num_hidden_layers: int) -> list[int]:
    """early=floor(L/8), middle=floor(L/2), late=L-2; must be distinct and valid."""
    if num_hidden_layers < 3:
        raise ValueError(f"num_hidden_layers must be >= 3, got {num_hidden_layers}")
    early = num_hidden_layers // 8
    middle = num_hidden_layers // 2
    late = num_hidden_layers - 2
    layers = [early, middle, late]
    if len(set(layers)) != 3:
        raise ValueError(
            f"representative layers not distinct for L={num_hidden_layers}: {layers}"
        )
    if any(i < 0 or i >= num_hidden_layers for i in layers):
        raise ValueError(
            f"representative layers out of range for L={num_hidden_layers}: {layers}"
        )
    return layers


def resolve_activation_scale_file(checkpoint_dir: Path) -> Path:
    """Mirror main.py / vLLM NVFP4 loader: <checkpoint>/nvfp4_activation_scales.safetensors."""
    path = (checkpoint_dir / ACTIVATION_SCALES_FILENAME).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "NVFP4 activation scale file not found (existing_repo_nvfp4_loader): "
            f"{path}"
        )
    return path


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"config.{key} must be a mapping")
    return value


def validate_experiment_config(cfg: ExperimentConfig) -> None:
    if cfg.model.source_semantics not in SOURCE_SEMANTICS_ALLOWED:
        raise ValueError(
            "source_semantics must be nvfp4_qat_fake_dequant_bf16 "
            f"(BF16 is only the storage container dtype); got {cfg.model.source_semantics!r}"
        )
    if cfg.model.source_semantics in {"bf16", "bfloat16", "native_bf16"}:
        raise ValueError("source_semantics must not be ordinary bf16")
    if cfg.model.hadamard_runtime != "disabled":
        raise ValueError(
            f"hadamard_runtime must be disabled, got {cfg.model.hadamard_runtime!r}"
        )
    if cfg.activation_nvfp4.block_size != 16:
        raise ValueError("NVFP4 activation block_size must be 16")
    if cfg.e2e.mmlu_pro_max_samples != 300:
        raise ValueError("e2e.mmlu_pro_max_samples must be 300")
    if cfg.analysis.representative_layer_policy != "early_middle_late":
        raise ValueError("representative_layer_policy must be early_middle_late")


def load_experiment_config(path: Path | str | None = None) -> ExperimentConfig:
    cfg_path = Path(path) if path is not None else FORMAL_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be mapping: {cfg_path}")

    model_raw = _require_mapping(raw, "model")
    act_nv_raw = _require_mapping(raw, "activation_nvfp4")
    analysis_raw = _require_mapping(raw, "analysis")
    activation_raw = _require_mapping(raw, "activation")
    bootstrap_raw = _require_mapping(raw, "bootstrap")
    e2e_raw = _require_mapping(raw, "e2e")

    cfg = ExperimentConfig(
        seed=int(raw["seed"]),
        repo_root=Path(raw["repo_root"]).resolve(),
        model=ModelConfig(
            source_checkpoint=str(model_raw["source_checkpoint"]),
            source_semantics=str(model_raw["source_semantics"]),
            optional_fp32_storage_probe=str(model_raw["optional_fp32_storage_probe"]),
            original_training_source=str(model_raw["original_training_source"]),
            hadamard_runtime=str(model_raw["hadamard_runtime"]),
            hadamard_reason=str(model_raw["hadamard_reason"]),
        ),
        activation_nvfp4=ActivationNvfp4Config(
            block_size=int(act_nv_raw["block_size"]),
            global_scale_dtype=str(act_nv_raw["global_scale_dtype"]),
            global_scale_granularity=str(act_nv_raw["global_scale_granularity"]),
            scale_file_resolution=str(act_nv_raw["scale_file_resolution"]),
            calibration_dataset=str(act_nv_raw["calibration_dataset"]),
            calibration_nsamples=int(act_nv_raw["calibration_nsamples"]),
            calibration_seqlen=int(act_nv_raw["calibration_seqlen"]),
            calibration_slice_mode=str(act_nv_raw["calibration_slice_mode"]),
            calibration_slice_offset=int(act_nv_raw["calibration_slice_offset"]),
        ),
        analysis=AnalysisConfig(
            representative_layer_policy=str(analysis_raw["representative_layer_policy"]),
            representative_layer_count=int(analysis_raw["representative_layer_count"]),
            token_sample_per_module_per_phase=int(
                analysis_raw["token_sample_per_module_per_phase"]
            ),
            raw_token_sample_per_module_per_phase=int(
                analysis_raw["raw_token_sample_per_module_per_phase"]
            ),
            token_chunk_size=int(analysis_raw["token_chunk_size"]),
            samples_per_prompt_family=int(analysis_raw["samples_per_prompt_family"]),
            discovery_samples_per_prompt_family=int(
                analysis_raw["discovery_samples_per_prompt_family"]
            ),
            validation_samples_per_prompt_family=int(
                analysis_raw["validation_samples_per_prompt_family"]
            ),
            network_scan_samples=int(analysis_raw["network_scan_samples"]),
            network_scan_max_length=int(analysis_raw["network_scan_max_length"]),
            sequence_length_buckets=[int(x) for x in analysis_raw["sequence_length_buckets"]],
            prequant_activation_dtype=str(analysis_raw["prequant_activation_dtype"]),
            qdq_output_dtype=str(analysis_raw["qdq_output_dtype"]),
            accumulator_dtype=str(analysis_raw["accumulator_dtype"]),
        ),
        activation=ActivationConfig(
            phases=[str(x) for x in activation_raw["phases"]],
            decode_steps_per_prompt_for_analysis=int(
                activation_raw["decode_steps_per_prompt_for_analysis"]
            ),
            exclude_modules=[str(x) for x in activation_raw["exclude_modules"]],
            p2_use_matched_coverage=bool(activation_raw["p2_use_matched_coverage"]),
        ),
        bootstrap=BootstrapConfig(
            repeats=int(bootstrap_raw["repeats"]),
            confidence=float(bootstrap_raw["confidence"]),
        ),
        e2e=E2EConfig(
            mmlu_pro_max_samples=int(e2e_raw["mmlu_pro_max_samples"]),
            arc_num_fewshot=int(e2e_raw["arc_num_fewshot"]),
            aime25_repeats=int(e2e_raw["aime25_repeats"]),
        ),
    )
    validate_experiment_config(cfg)
    return cfg


def layer_role_name(layer_idx: int, resolved: list[int]) -> str | None:
    names = ("early", "middle", "late")
    for name, idx in zip(names, resolved, strict=True):
        if layer_idx == idx:
            return name
    return None


# silence unused import if math needed later for docs
_ = math
