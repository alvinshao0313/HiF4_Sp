"""Frozen protocol for the progressive error-cancellation experiment."""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_RESULTS = ROOT / "results/e2e_diag_reconstruction"
DEFAULT_INIT = LEGACY_RESULTS / "phaseA_refactor_20260825T035730Z/E4_fusable_r64/checkpoint/final_model/conversion_state.pt"
DEFAULT_CALIB = LEGACY_RESULTS / "shared_calibration/s1k_original_n128_v32_seed42"


@dataclass(frozen=True)
class Config:
    output_dir: str
    method: str = "baseline"
    lambda_direction: float = 0.0
    matrix_sharing: str = "group"
    router_loss: str = "top_mass"
    router_top_k: int = 8
    router_temperature: float = 1.0
    router_loss_weight: float = 1.0
    init_artifact: str = str(DEFAULT_INIT)
    model_path: str = "nvidia/Qwen3-30B-A3B-NVFP4"
    epochs: int = 20
    batch_size: int = 4
    lr: float = 1e-4
    calib_source: str = "s1k_original"
    calib_nsamples: int = 128
    calib_val_nsamples: int = 32
    calib_holdout_nsamples: int = 32
    calib_seed: int = 42
    calib_seqlen: int = 1024
    calib_cache_dir: str = str(DEFAULT_CALIB)
    layers: str = "0:48"
    shuffle_seed: int = 4242

    @property
    def layer_ids(self) -> tuple[int, ...]:
        value = self.layers.strip()
        if ":" in value:
            start, stop = (int(x) for x in value.split(":", 1))
            result = tuple(range(start, stop))
        else:
            result = tuple(int(x) for x in value.split(",") if x.strip())
        return result

    def validate(self) -> None:
        if self.method not in {"baseline", "direct", "jvp", "shuffled"}:
            raise ValueError("method must be baseline, direct, jvp, or shuffled")
        if self.method == "baseline" and self.lambda_direction != 0:
            raise ValueError("baseline must use lambda_direction=0")
        if not math.isfinite(self.lambda_direction) or self.lambda_direction < 0:
            raise ValueError("lambda_direction must be finite and nonnegative")
        if self.matrix_sharing != "group":
            raise ValueError("this protocol fixes matrix_sharing=group")
        if self.router_loss != "top_mass":
            raise ValueError("this protocol fixes router_loss=top_mass")
        if self.calib_source != "s1k_original":
            raise ValueError("this protocol uses s1k_original")
        if self.calib_nsamples <= 0 or self.calib_val_nsamples <= 0 or self.calib_holdout_nsamples <= 0:
            raise ValueError("all calibration split sizes must be positive")
        if self.epochs <= 0 or self.batch_size <= 0 or self.router_top_k <= 0:
            raise ValueError("epochs, batch_size, and router_top_k must be positive")
        for name in ("lr", "router_temperature"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.router_loss_weight) or self.router_loss_weight < 0:
            raise ValueError("router_loss_weight must be finite and nonnegative")
        layers = self.layer_ids
        if not layers or len(layers) > 48 or layers != tuple(sorted(set(layers))):
            raise ValueError("layers must be an increasing nonempty subset")
        if layers[0] < 0 or layers[-1] >= 48:
            raise ValueError("layers must be in [0, 48)")

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
