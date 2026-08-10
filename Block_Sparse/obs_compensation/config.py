from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_CALIBRATION_DATASETS = frozenset({"s1k", "wikitext2"})
_DTYPES = frozenset({"bfloat16", "float16", "float32"})
_ORDER_POLICIES = frozenset({"auto", "standard", "permutation_aware"})


@dataclass(frozen=True)
class OBSCompensationConfig:
    model_path: str
    source_artifacts_dir: Path
    output_dir: Path
    calibration_dataset: str
    calibration_samples: int
    sequence_length: int
    obs_percdamp: float
    solver_block_size: int
    obs_order_policy: str
    dtype: str
    device: str
    seed: int
    trust_remote_code: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_artifacts_dir", Path(self.source_artifacts_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("model_path must be a non-empty string")
        if self.calibration_dataset not in _CALIBRATION_DATASETS:
            raise ValueError(
                f"Unsupported calibration_dataset={self.calibration_dataset!r}; "
                f"choose from {sorted(_CALIBRATION_DATASETS)}"
            )
        if self.calibration_samples < 1:
            raise ValueError(
                f"calibration_samples must be >= 1, got {self.calibration_samples}"
            )
        if self.sequence_length < 2:
            raise ValueError(
                f"sequence_length must be >= 2, got {self.sequence_length}"
            )
        if not (0.0 < float(self.obs_percdamp) <= 1.0):
            raise ValueError(
                f"obs_percdamp must satisfy 0 < obs_percdamp <= 1, got {self.obs_percdamp}"
            )
        if int(self.solver_block_size) < 1:
            raise ValueError(
                f"solver_block_size must be >= 1, got {self.solver_block_size}"
            )
        if self.obs_order_policy not in _ORDER_POLICIES:
            raise ValueError(
                f"Unsupported obs_order_policy={self.obs_order_policy!r}; "
                f"choose from {sorted(_ORDER_POLICIES)}"
            )
        if self.dtype not in _DTYPES:
            raise ValueError(
                f"Unsupported dtype={self.dtype!r}; choose from {sorted(_DTYPES)}"
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.source_artifacts_dir.resolve() == self.output_dir.resolve():
            raise ValueError(
                "source_artifacts_dir and output_dir must differ"
            )

    def validate_paths(self, require_source_exists: bool = True) -> None:
        if require_source_exists:
            if not self.source_artifacts_dir.is_dir():
                raise FileNotFoundError(
                    f"source_artifacts_dir does not exist: {self.source_artifacts_dir}"
                )
        if self.output_dir.exists():
            if any(self.output_dir.iterdir()):
                raise ValueError(
                    f"output_dir exists and is non-empty: {self.output_dir}"
                )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OBS compensation initialization for fixed MLP block masks"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional. Must exactly match source pruning_summary.json model_path.",
    )
    parser.add_argument(
        "--source_artifacts_dir",
        type=str,
        required=True,
        help="Directory containing pruning_summary.json and block_masks.pt",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--calibration_dataset",
        type=str,
        default="s1k",
        choices=sorted(_CALIBRATION_DATASETS),
    )
    parser.add_argument("--calibration_samples", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=1024)
    parser.add_argument("--obs_percdamp", type=float, default=0.01)
    parser.add_argument("--solver_block_size", type=int, default=128)
    parser.add_argument(
        "--obs_order_policy",
        type=str,
        default="auto",
        choices=sorted(_ORDER_POLICIES),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=sorted(_DTYPES),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace, source_model_path: str) -> OBSCompensationConfig:
    if not isinstance(source_model_path, str) or not source_model_path.strip():
        raise ValueError("source_model_path must be a non-empty string")
    explicit = getattr(args, "model_path", None)
    if explicit is None or (isinstance(explicit, str) and not explicit.strip()):
        model_path = source_model_path
    else:
        if explicit != source_model_path:
            raise ValueError(
                f"--model_path {explicit!r} does not exactly match source "
                f"summary model_path {source_model_path!r}"
            )
        model_path = explicit

    cfg = OBSCompensationConfig(
        model_path=model_path,
        source_artifacts_dir=Path(args.source_artifacts_dir),
        output_dir=Path(args.output_dir),
        calibration_dataset=str(args.calibration_dataset),
        calibration_samples=int(args.calibration_samples),
        sequence_length=int(args.sequence_length),
        obs_percdamp=float(args.obs_percdamp),
        solver_block_size=int(args.solver_block_size),
        obs_order_policy=str(args.obs_order_policy),
        dtype=str(args.dtype),
        device=str(args.device),
        seed=int(args.seed),
        trust_remote_code=bool(args.trust_remote_code),
    )
    return cfg
