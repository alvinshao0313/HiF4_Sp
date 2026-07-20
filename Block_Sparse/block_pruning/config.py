from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


_CALIBRATION_DATASETS = frozenset({"wikitext2", "c4", "ptb", "s1k"})
_BLOCK_SIZE_RE = re.compile(r"^(\d+)(?:[xX](\d+))?$")


def parse_block_size(spec: str | int) -> tuple[int, int]:
    """Parse block size spec into (height, width).

    Accepts:
      - int 128           -> (128, 128)
      - "128"             -> (128, 128)
      - "64x128" / "64X128" -> (64, 128)

    height is along weight dim0 (d_out), width along dim1 (d_in).
    """
    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError(f"block_size must be > 0, got {spec}")
        return spec, spec

    text = str(spec).strip()
    match = _BLOCK_SIZE_RE.fullmatch(text)
    if match is None:
        raise ValueError(
            f"Invalid block_size '{spec}'. Use '128' or 'HxW' (e.g. '64x128')."
        )
    height = int(match.group(1))
    width = int(match.group(2)) if match.group(2) is not None else height
    if height <= 0 or width <= 0:
        raise ValueError(f"block_size dimensions must be > 0, got {height}x{width}")
    return height, width


@dataclass
class GradientBlockPruningConfig:
    model_path: str = "Qwen/Qwen3.5-27B"
    calibration_dataset: str = "wikitext2"
    output_dir: str = "Block_Sparse/outputs/default"

    # Single public knob: "128" or "64x128". Parsed into block_height/block_width.
    block_size: str = "128"
    block_height: int = field(init=False, default=128)
    block_width: int = field(init=False, default=128)

    target_block_sparsity: float = 0.30

    calibration_samples: int = 128
    sequence_length: int = 2048
    score_batch_size: int = 1

    score_type: str = "fisher"  # fisher | magnitude | random | fisher_budget_wanda
    selection_mode: str = "global_constrained"

    max_prune_ratio_per_matrix: float = 0.60
    min_keep_blocks_per_matrix: int = 1

    share_up_gate_mask: bool = False
    pruning_rounds: int = 1

    seed: int = 42
    score_accumulation_dtype: str = "float64"

    dtype: str = "bfloat16"
    device: str = "cuda"
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        self.block_height, self.block_width = parse_block_size(self.block_size)

    def validate(self) -> None:
        self.block_height, self.block_width = parse_block_size(self.block_size)
        if not (0.0 < self.target_block_sparsity < 1.0):
            raise ValueError(
                f"target_block_sparsity must be in (0, 1), got {self.target_block_sparsity}"
            )
        if self.score_type not in {
            "fisher",
            "magnitude",
            "random",
            "fisher_budget_wanda",
        }:
            raise ValueError(f"Unsupported score_type: {self.score_type}")
        if self.selection_mode != "global_constrained":
            raise ValueError(f"Unsupported selection_mode: {self.selection_mode}")
        if self.score_batch_size != 1:
            raise ValueError(
                f"score_batch_size must be 1 for comparable Fisher scores, "
                f"got {self.score_batch_size}"
            )
        if self.pruning_rounds < 1:
            raise ValueError(f"pruning_rounds must be >= 1, got {self.pruning_rounds}")
        if not (0.0 < self.max_prune_ratio_per_matrix <= 1.0):
            raise ValueError(
                f"max_prune_ratio_per_matrix must be in (0, 1], "
                f"got {self.max_prune_ratio_per_matrix}"
            )
        if self.min_keep_blocks_per_matrix < 1:
            raise ValueError(
                f"min_keep_blocks_per_matrix must be >= 1, "
                f"got {self.min_keep_blocks_per_matrix}"
            )
        if self.calibration_dataset not in _CALIBRATION_DATASETS:
            raise ValueError(
                f"Unsupported calibration_dataset: {self.calibration_dataset}. "
                f"Choose from {sorted(_CALIBRATION_DATASETS)}."
            )

    def requires_calibration(self) -> bool:
        return self.score_type in {"fisher", "fisher_budget_wanda"}

    def requires_gradient_checkpointing(self) -> bool:
        return self.score_type in {"fisher", "fisher_budget_wanda"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROJECTION_TYPES = ("gate_proj", "up_proj", "down_proj")
CALIBRATION_DATASETS = _CALIBRATION_DATASETS
