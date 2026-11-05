"""Fixed calibration and phase budgets for the confirmed ablation matrix."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

from .variants import get_variant

ROOT = Path(__file__).resolve().parents[2]
NATIVE_SNAPSHOT = Path("/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3")
CALIB_CACHE = ROOT / "results/e2e_diag_reconstruction/shared_calibration/s1k_original_n128_v32_seed42"
LR_MULTIPLIERS = (0.3, 1.0, 3.0)


@dataclass(frozen=True)
class Config:
    output_dir: str
    variant: str
    phase: str
    lr_multiplier: float = 1.0
    model_path: str = str(NATIVE_SNAPSHOT)
    batch_size: int = 4
    router_loss: str = "top_mass"
    router_top_k: int = 8
    router_temperature: float = 1.0
    calib_source: str = "s1k_original"
    calib_nsamples: int = 128
    calib_val_nsamples: int = 32
    calib_seed: int = 42
    calib_seqlen: int = 1024  # Cache metadata only; full S1K text is not truncated.
    calib_cache_dir: str = str(CALIB_CACHE)

    @property
    def epochs(self):
        return {"smoke": 2, "screen": 3, "formal": 20}[self.phase]

    @property
    def layers(self):
        return {"smoke": 2, "screen": 1, "formal": 48}[self.phase]

    @property
    def router_loss_weight(self):
        return get_variant(self.variant).kl_weight

    def validate(self):
        if self.phase not in {"smoke", "screen", "formal"}:
            raise ValueError("phase must be smoke, screen or formal")
        if not get_variant(self.variant).needs_training:
            raise ValueError("only trained variants use this entrypoint")
        if not math.isfinite(self.lr_multiplier) or self.lr_multiplier not in LR_MULTIPLIERS:
            raise ValueError(f"lr_multiplier must be one of {LR_MULTIPLIERS}")
        if Path(self.model_path).resolve() != NATIVE_SNAPSHOT.resolve():
            raise ValueError("this experiment starts from the fixed original NVFP4 snapshot")
        if (self.batch_size, self.router_loss, self.router_top_k, self.router_temperature,
            self.calib_source, self.calib_nsamples, self.calib_val_nsamples, self.calib_seed) != (
                4, "top_mass", 8, 1.0, "s1k_original", 128, 32, 42):
            raise ValueError("the confirmed calibration/loss protocol cannot be overridden")

    def to_dict(self):
        return {**asdict(self), "epochs": self.epochs, "layers": self.layers,
                "router_loss_weight": self.router_loss_weight}


def parameter_groups(student, multiplier):
    """Scale each parameterization around its own LR; never tune on tasks."""
    groups = {}
    for name, parameter in student.learned.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f"frozen tensor {name} should be a buffer")
        family = name.split(".", 1)[0]
        base_lr = {"norms": 1e-4, "matrices": 1e-4, "rotations": 1e-3, "diag": 5e-3}[family]
        group = groups.setdefault(family, {"params": [], "lr": base_lr * multiplier, "name": family})
        group["params"].append(parameter)
    if not groups:
        raise ValueError("training requires selected parameters")
    return list(groups.values())
