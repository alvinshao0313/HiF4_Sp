from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = "nvidia/Qwen3-30B-A3B-NVFP4"
DEFAULT_PHASEA_ROOT = (
    REPO_ROOT
    / "Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/phaseA_refactor_20260825T035730Z"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability"
)

POSITION_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("d0000_0127", 0, 128),
    ("d0128_0511", 128, 512),
    ("d0512_2047", 512, 2048),
    ("d2048_8191", 2048, 8192),
    ("d8192_plus", 8192, None),
)

DEFAULT_PROBE_LAYERS = tuple(range(48))
DEFAULT_PROBES_PER_BIN = 4
DEFAULT_DIVERGENCE_OFFSETS = (-2, -1, 0, 1)
DEFAULT_MAX_PROBE_DECODE_INDEX = 12287
DEFAULT_CAUSAL_REPLAY_SAMPLES = 16
DEFAULT_FREE_RUN_SAMPLES = 64
DEFAULT_FREE_RUN_MAX_NEW_TOKENS = 16384


@dataclass(frozen=True)
class Variant:
    name: str
    eval_variant: str
    phasea_subdir: str
    uses_artifact: bool = False

    def phasea_run_dir(self, phasea_root: Path) -> Path:
        path = phasea_root / self.phasea_subdir
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path

    def artifact_path(self, phasea_root: Path) -> Path | None:
        if not self.uses_artifact:
            return None
        path = self.phasea_run_dir(phasea_root) / "checkpoint/final_model/conversion_state.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


VARIANTS: dict[str, Variant] = {
    "E0": Variant("E0", "native_nvfp4", "E0_native_nvfp4"),
    "E1": Variant("E1", "direct_hif4", "E1_direct_hif4"),
    "E2": Variant("E2", "r64_only", "E2_r64_only"),
    "E3": Variant("E3", "artifact", "E3_fusable", True),
    "E4": Variant("E4", "artifact", "E4_fusable_r64", True),
}


def resolve_variant(name: str) -> Variant:
    key = name.upper()
    if key not in VARIANTS:
        raise ValueError(f"unknown variant={name!r}; expected one of {sorted(VARIANTS)}")
    return VARIANTS[key]


def decode_bin(index: int) -> str:
    if index < 0:
        raise ValueError(f"decode index must be non-negative, got {index}")
    for name, lo, hi in POSITION_BINS:
        if index >= lo and (hi is None or index < hi):
            return name
    raise AssertionError(index)
