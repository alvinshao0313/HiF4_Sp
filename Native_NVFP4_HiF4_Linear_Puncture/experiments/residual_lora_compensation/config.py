"""Experiment configuration; no dependency on the legacy training configuration."""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/residual_lora_compensation"
LEGACY_RESULTS = ROOT / "results/e2e_diag_reconstruction"
DEFAULT_INIT = LEGACY_RESULTS / "phaseA_refactor_20260825T035730Z/E4_fusable_r64/checkpoint/final_model/conversion_state.pt"


@dataclass(frozen=True)
class Config:
    output_dir: str
    matrix_sharing: str = "group"
    lora_mode: str = "both"
    lora_rank: int = 4
    lora_alpha: float = 8.0
    router_loss: str = "top_mass"
    init_artifact: str = str(DEFAULT_INIT)
    model_path: str = "nvidia/Qwen3-30B-A3B-NVFP4"
    epochs: int = 20
    batch_size: int = 4
    lr: float = 1e-4
    router_top_k: int = 8
    router_temperature: float = 1.0
    router_loss_weight: float = 1.0
    calib_source: str = "s1k_original"
    calib_nsamples: int = 128
    calib_val_nsamples: int = 32
    calib_seed: int = 42
    calib_seqlen: int = 1024
    calib_cache_dir: str = str(LEGACY_RESULTS / "shared_calibration/s1k_original_n128_v32_seed42")

    def validate(self) -> None:
        if self.matrix_sharing != "group":
            raise ValueError("this experiment requires per-G64 group matrices")
        if self.lora_mode not in {"attention", "moe", "both", "none"}:
            raise ValueError("lora_mode must be attention, moe, both, or none")
        if self.router_loss != "top_mass":
            raise ValueError("this experiment uses top_mass")
        if self.calib_source != "s1k_original":
            raise ValueError("this experiment uses s1k_original")
        for name in ("epochs", "batch_size", "router_top_k", "calib_nsamples", "calib_val_nsamples", "lora_rank"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("lr", "router_temperature", "lora_alpha"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.router_loss_weight) or self.router_loss_weight < 0:
            raise ValueError("router_loss_weight must be finite and nonnegative")
        if self.lora_rank >= 2048:
            raise ValueError("lora_rank must be smaller than hidden size")

    def to_dict(self) -> dict:
        return asdict(self)


def train_args() -> tuple[Config, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    defaults = Config(output_dir="")
    for name, value in defaults.to_dict().items():
        if name != "output_dir":
            parser.add_argument(f"--{name}", type=type(value), default=value)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = args.pop("resume")
    cfg = Config(**args)
    cfg.validate()
    return cfg, resume
