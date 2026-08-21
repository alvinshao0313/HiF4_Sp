"""Canonical result schemas. Downstream tasks must not invent synonym fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


SCHEMA_VERSION = 1


@dataclass
class RunManifest:
    run_id: str
    git_commit: str
    device: str
    torch_version: str
    status: str
    notes: str
    source_checkpoint: str
    source_semantics: str
    source_weight_dtype: str
    source_weight_tensor_count: int
    num_hidden_layers: int
    resolved_representative_layers: list[int]
    hadamard_runtime: str
    nvfp4_activation_scale_file: str
    nvfp4_activation_scale_count: int
    matched_activation_module_count: int
    seed: int = 20260810
    analysis_seed: int = 20260809
    schema_version: int = SCHEMA_VERSION
    path_ids: list[str] = field(default_factory=list)
    discovery_sample_ids: dict[str, list[str]] = field(default_factory=dict)
    validation_sample_ids: dict[str, list[str]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunManifest":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        unknown = {k: v for k, v in data.items() if k not in known}
        if unknown:
            extra = dict(kwargs.get("extra") or {})
            extra.update(unknown)
            kwargs["extra"] = extra
        return cls(**kwargs)


@dataclass
class TensorMetricRecord:
    path_id: str
    layer_idx: int
    module_name: str
    projection: str
    phase: str
    reference_name: str
    target_name: str
    nmse: float
    sqnr_db: float
    cosine: float
    mae: float
    mean_signed_error: float
    max_abs_error: float
    relative_norm_change: float
    reference_energy: float
    target_energy: float
    error_energy: float
    error_p50: float
    error_p90: float
    error_p99: float
    error_p99_9: float
    top1pct_error_energy_share: float
    numel: int
    evidence_class: str = "observational_correlation"
    schema_version: int = SCHEMA_VERSION
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorMetricRecord":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


@dataclass
class LinearDecompositionRecord:
    path_id: str
    layer_idx: int
    module_name: str
    projection: str
    phase: str
    sample_id: str
    energy_wn_an: float
    energy_delta_w_an: float
    energy_wn_delta_a: float
    energy_delta_w_delta_a: float
    cross_dw_an_wn_da: float
    cross_dw_an_dw_da: float
    cross_wn_da_dw_da: float
    residual_rel: float
    evidence_class: str = "controlled_causal_evidence"
    schema_version: int = SCHEMA_VERSION
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearDecompositionRecord":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


@dataclass
class RootCauseRecord:
    cause_id: str
    hypothesis_id: str
    mechanism: str
    path_id: str
    evidence_class: str
    recoverable_error_fraction: float
    affected_scope: str
    metric_name: str
    metric_value: float
    confounded_by: list[str] = field(default_factory=list)
    optimization_implication: str = ""
    notes: str = ""
    schema_version: int = SCHEMA_VERSION
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RootCauseRecord":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)
