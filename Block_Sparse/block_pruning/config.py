from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


_CALIBRATION_DATASETS = frozenset({"wikitext2", "c4", "ptb", "s1k"})
_BLOCK_SIZE_RE = re.compile(r"^(\d+)(?:[xX](\d+))?$")
PROJECTION_TYPES = ("gate_proj", "up_proj", "down_proj")


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


def parse_projection_prune_shares(raw: str) -> dict[str, float]:
    """Parse ``gate_proj=1,up_proj=1,down_proj=2`` into a shares dict.

    Values must be > 0. Keys must be exactly the three MLP projection types.
    Returned values are the raw (unnormalized) shares.
    """
    text = str(raw).strip()
    if not text:
        raise ValueError("projection_prune_shares string is empty")
    shares: dict[str, float] = {}
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(
                f"Invalid projection_prune_shares item {piece!r}; "
                "expected key=value (e.g. gate_proj=1)"
            )
        key, value_s = piece.split("=", 1)
        key = key.strip()
        value_s = value_s.strip()
        if key not in PROJECTION_TYPES:
            raise ValueError(
                f"Unknown projection type {key!r} in projection_prune_shares; "
                f"expected one of {list(PROJECTION_TYPES)}"
            )
        if key in shares:
            raise ValueError(f"Duplicate projection type in shares: {key}")
        try:
            value = float(value_s)
        except ValueError as exc:
            raise ValueError(
                f"Invalid share value for {key}: {value_s!r}"
            ) from exc
        if not (value > 0.0):
            raise ValueError(f"Share for {key} must be > 0, got {value}")
        shares[key] = value
    missing = [p for p in PROJECTION_TYPES if p not in shares]
    if missing:
        raise ValueError(
            f"projection_prune_shares missing keys: {missing}. "
            f"Required: {list(PROJECTION_TYPES)}"
        )
    if len(shares) != len(PROJECTION_TYPES):
        raise ValueError(
            f"projection_prune_shares must contain exactly {list(PROJECTION_TYPES)}"
        )
    return {p: shares[p] for p in PROJECTION_TYPES}


def normalize_projection_prune_shares(shares: dict[str, float]) -> dict[str, float]:
    """Return shares that sum to 1.0, keyed in PROJECTION_TYPES order."""
    if set(shares) != set(PROJECTION_TYPES):
        raise ValueError(
            f"shares keys must be exactly {list(PROJECTION_TYPES)}, got {sorted(shares)}"
        )
    total = sum(float(shares[p]) for p in PROJECTION_TYPES)
    if total <= 0.0:
        raise ValueError(f"share sum must be > 0, got {total}")
    return {p: float(shares[p]) / total for p in PROJECTION_TYPES}


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
    # Optional: split global prune budget across projection types.
    # None = legacy global ranking across all u/g/d.
    projection_prune_shares: dict[str, float] | None = None
    pruning_rounds: int = 1
    mlp_permutation: str = "none"  # none | wanda_shared
    residual_permutation: str = "none"  # none | block_loss
    residual_perm_search_steps: int = 2000
    # π0 residual-channel aggregation:
    # equal | layer_fisher | matrix_fisher | raw_wanda |
    # sparsity_raw_wanda | density_raw_wanda
    residual_channel_agg: str = "equal"

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
        if self.mlp_permutation not in {"none", "wanda_shared"}:
            raise ValueError(
                f"Unsupported mlp_permutation: {self.mlp_permutation}. "
                f"Choose from ['none', 'wanda_shared']."
            )
        if self.residual_permutation not in {"none", "block_loss"}:
            raise ValueError(
                f"Unsupported residual_permutation: {self.residual_permutation}. "
                f"Choose from ['none', 'block_loss']."
            )
        if self.residual_perm_search_steps < 0:
            raise ValueError(
                f"residual_perm_search_steps must be >= 0, "
                f"got {self.residual_perm_search_steps}"
            )
        if self.residual_channel_agg not in {
            "equal",
            "layer_fisher",
            "matrix_fisher",
            "raw_wanda",
            "sparsity_raw_wanda",
            "density_raw_wanda",
        }:
            raise ValueError(
                f"Unsupported residual_channel_agg: {self.residual_channel_agg}. "
                f"Choose from ['equal', 'layer_fisher', 'matrix_fisher', "
                f"'raw_wanda', 'sparsity_raw_wanda', 'density_raw_wanda']."
            )
        if self.residual_channel_agg in {
            "layer_fisher",
            "matrix_fisher",
            "sparsity_raw_wanda",
            "density_raw_wanda",
        } and (
            self.residual_permutation == "block_loss"
            and self.score_type not in {"fisher", "fisher_budget_wanda"}
        ):
            raise ValueError(
                f"residual_channel_agg={self.residual_channel_agg} requires "
                f"score_type in ['fisher', 'fisher_budget_wanda'], "
                f"got {self.score_type!r}"
            )
        if (
            self.residual_permutation == "block_loss"
            and self.score_type == "random"
        ):
            raise ValueError(
                "residual_permutation=block_loss is incompatible with score_type=random"
            )
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
        if self.projection_prune_shares is not None:
            shares = self.projection_prune_shares
            if set(shares) != set(PROJECTION_TYPES):
                raise ValueError(
                    f"projection_prune_shares keys must be exactly "
                    f"{list(PROJECTION_TYPES)}, got {sorted(shares)}"
                )
            for proj in PROJECTION_TYPES:
                value = float(shares[proj])
                if not (value > 0.0):
                    raise ValueError(
                        f"projection_prune_shares[{proj}] must be > 0, got {value}"
                    )
            if self.share_up_gate_mask:
                g = float(shares["gate_proj"])
                u = float(shares["up_proj"])
                if abs(g - u) > 1e-12:
                    raise ValueError(
                        "share_up_gate_mask requires gate_proj and up_proj "
                        f"shares to be equal, got gate_proj={g}, up_proj={u}"
                    )
            # Store normalized shares for allocate-time use
            self.projection_prune_shares = normalize_projection_prune_shares(shares)

    def requires_calibration(self) -> bool:
        return (
            self.score_type in {"fisher", "fisher_budget_wanda"}
            or self.mlp_permutation == "wanda_shared"
            or self.residual_permutation == "block_loss"
        )

    def requires_gradient_checkpointing(self) -> bool:
        return self.score_type in {"fisher", "fisher_budget_wanda"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CALIBRATION_DATASETS = _CALIBRATION_DATASETS
