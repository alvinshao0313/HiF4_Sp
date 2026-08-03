"""Global residual-hidden permutation optimized for block-pruning loss."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tqdm import tqdm

from block_pruning.block_utils import (
    reduce_weight_magnitude_to_blocks,
    reduce_weight_wanda_to_blocks,
)
from block_pruning.config import GradientBlockPruningConfig
from block_pruning.gradient_scorer import BlockScoreRecord, _make_record
from block_pruning.mask_allocator import (
    allocate_block_masks,
    allocate_masks_by_module_budget,
    extract_module_prune_budgets,
)
from block_pruning.mlp_permutation import group_mlp_projection_triplets
from block_pruning.mlp_registry import MLPLinearTarget, initialize_all_one_masks
from block_pruning.wanda_scorer import InputRMSRecord, collect_mlp_input_rms


# Parameters that may touch hidden_size-shaped buffers but are NOT residual axes.
_IGNORE_NAME_SUBSTRINGS = (
    ".conv1d.",
    ".conv1d",
    "A_log",
    "dt_bias",
    ".q_norm.",
    ".k_norm.",
    "linear_attn.norm",
)


@dataclass(frozen=True)
class ResidualMount:
    name: str
    param: nn.Parameter
    dim: int


@dataclass
class ResidualPermutationRecord:
    hidden_size: int
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    channel_score: torch.Tensor
    loss_init: float
    loss_final: float
    search_steps: int
    accepted_swaps: int
    mount_names: list[str] = field(default_factory=list)
    search_score_mode: str = "wanda"
    channel_agg: str = "equal"


@dataclass
class _MLPSearchCache:
    """CPU float32 MLP weights + RMS for virtual residual reordering."""

    targets: list[MLPLinearTarget]
    weights: dict[str, torch.Tensor]  # float32 CPU
    input_rms: dict[str, torch.Tensor]  # float64/float32 CPU
    residual_side: dict[str, str]  # "in" | "out"
    hidden_size: int
    block_height: int
    block_width: int
    masks: dict[str, torch.Tensor]


def resolve_hidden_size(
    model: nn.Module,
    targets: list[MLPLinearTarget],
) -> int:
    # gate/up: d_in = hidden; down: d_out = hidden
    gate_up = [
        int(t.module.weight.shape[1])
        for t in targets
        if t.projection_type in ("gate_proj", "up_proj")
    ]
    down = [
        int(t.module.weight.shape[0])
        for t in targets
        if t.projection_type == "down_proj"
    ]
    if not gate_up or not down:
        raise RuntimeError("Cannot resolve hidden_size: missing gate/up or down targets")
    hidden = gate_up[0]
    if any(s != hidden for s in gate_up) or any(s != hidden for s in down):
        raise RuntimeError(
            f"Inconsistent residual sizes across MLP targets: "
            f"gate_up={sorted(set(gate_up))} down={sorted(set(down))}"
        )
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg_hidden = getattr(cfg, "hidden_size", None)
        if cfg_hidden is None and hasattr(cfg, "text_config"):
            cfg_hidden = getattr(cfg.text_config, "hidden_size", None)
        if cfg_hidden is not None and int(cfg_hidden) != hidden:
            raise RuntimeError(
                f"config.hidden_size={cfg_hidden} != MLP residual size {hidden}"
            )
    return hidden


def _should_ignore_param(name: str) -> bool:
    if name.endswith("A_log") or ".A_log" in name:
        return True
    if name.endswith("dt_bias") or ".dt_bias" in name:
        return True
    for sub in _IGNORE_NAME_SUBSTRINGS:
        if sub in name:
            return True
    return False


def _classify_residual_dim(name: str, shape: torch.Size, hidden_size: int) -> int | None:
    """Return permute dim if this tensor is on the residual axis, else None.

    Returns None only for ignored tensors. Raises for unknown residual-sized tensors.
    """
    if _should_ignore_param(name):
        return None

    dims_matching = [i for i, s in enumerate(shape) if int(s) == hidden_size]
    if not dims_matching:
        return None

    leaf = name.rsplit(".", 1)[-1]
    # Embedding / lm_head: [vocab, H]
    if leaf == "weight" and (
        name.endswith("embed_tokens.weight")
        or name.endswith("lm_head.weight")
        or ".embed_tokens.weight" in name
    ):
        if shape[-1] != hidden_size:
            raise ValueError(f"Unexpected embed/lm_head shape for {name}: {tuple(shape)}")
        return len(shape) - 1

    # RMSNorm gains on residual: [H]
    if leaf == "weight" and len(shape) == 1 and int(shape[0]) == hidden_size:
        if any(
            key in name
            for key in (
                "input_layernorm",
                "post_attention_layernorm",
                ".norm.weight",
                "model.norm",
            )
        ) or name.endswith("norm.weight"):
            # Exclude head-wise norms already ignored; residual norms remain.
            if "q_norm" in name or "k_norm" in name or "linear_attn.norm" in name:
                return None
            return 0

    # Linear / projection weights: 2D
    if leaf == "weight" and len(shape) == 2:
        d0, d1 = int(shape[0]), int(shape[1])
        # Prefer name-based side when square: intermediate activations are NOT residual.
        out_names = (
            "out_proj.weight",
            "o_proj.weight",
            "down_proj.weight",
        )
        in_names = (
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_a.weight",
            "in_proj_b.weight",
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "gate_proj.weight",
            "up_proj.weight",
        )
        if any(name.endswith(suf) for suf in out_names):
            if d0 != hidden_size:
                raise ValueError(
                    f"Output-side projection {name} d_out={d0} != hidden {hidden_size}"
                )
            return 0
        if any(name.endswith(suf) for suf in in_names):
            if d1 != hidden_size:
                raise ValueError(
                    f"Input-side projection {name} d_in={d1} != hidden {hidden_size}"
                )
            return 1
        if d0 == hidden_size and d1 == hidden_size:
            raise ValueError(
                f"Square residual-sized weight {name} shape {tuple(shape)} "
                f"has no known in/out role; refusing ambiguous PWP^T"
            )
        if d1 == hidden_size and d0 != hidden_size:
            return 1  # input / column side
        if d0 == hidden_size and d1 != hidden_size:
            return 0  # output / row side

    # Bias on residual output
    if leaf == "bias" and len(shape) == 1 and int(shape[0]) == hidden_size:
        return 0

    raise ValueError(
        f"Parameter {name} has residual-sized dim(s) {dims_matching} "
        f"in shape {tuple(shape)} but is not a known residual mount. "
        f"Refusing to skip (let it crash)."
    )


def discover_residual_mounts(
    model: nn.Module,
    hidden_size: int,
) -> list[ResidualMount]:
    """Find every parameter that must absorb the residual permutation."""
    mounts: list[ResidualMount] = []
    seen_ptrs: set[int] = set()
    for name, param in model.named_parameters():
        dim = _classify_residual_dim(name, param.shape, hidden_size)
        if dim is None:
            continue
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            # Tied weights (e.g. lm_head == embed): permute storage once.
            continue
        seen_ptrs.add(ptr)
        mounts.append(ResidualMount(name=name, param=param, dim=dim))
    if not mounts:
        raise RuntimeError(
            f"No residual mounts found for hidden_size={hidden_size}"
        )
    return mounts


def _validate_permutation(perm: torch.Tensor, hidden_size: int, ctx: str) -> None:
    if perm.dtype != torch.int64:
        raise TypeError(f"{ctx}: permutation dtype must be int64, got {perm.dtype}")
    if perm.ndim != 1 or perm.numel() != hidden_size:
        raise ValueError(
            f"{ctx}: permutation length {perm.numel()} != hidden_size {hidden_size}"
        )
    if int(perm.min().item()) < 0 or int(perm.max().item()) >= hidden_size:
        raise ValueError(f"{ctx}: permutation values out of range")
    if int(torch.unique(perm).numel()) != hidden_size:
        raise ValueError(f"{ctx}: permutation is not a bijection")


def _index_select_inplace(param: nn.Parameter, dim: int, index: torch.Tensor) -> None:
    with torch.no_grad():
        selected = param.detach().index_select(dim, index)
        param.copy_(selected)


def apply_residual_permutation(
    model: nn.Module,
    mounts: list[ResidualMount],
    perm: torch.Tensor,
    hidden_size: int,
) -> None:
    """Absorb residual permutation into listed mounts. ``perm[k]`` = old index at new k."""
    _validate_permutation(perm, hidden_size, "apply_residual_permutation")
    for mount in mounts:
        index = perm.to(device=mount.param.device)
        param_id = id(mount.param)
        if mount.dim == -1:
            raise RuntimeError(
                f"Unexpected sentinel dim=-1 for mount {mount.name}; "
                "square maps must be classified as in- or out-side"
            )
        _index_select_inplace(mount.param, mount.dim, index)
        if id(mount.param) != param_id:
            raise RuntimeError(
                f"Parameter identity changed while permuting {mount.name}"
            )


def undo_residual_permutation(
    model: nn.Module,
    mounts: list[ResidualMount],
    inverse_perm: torch.Tensor,
    hidden_size: int,
) -> None:
    """Tests only: invert a previously applied residual permutation."""
    apply_residual_permutation(model, mounts, inverse_perm, hidden_size)


def _projection_residual_side(projection_type: str) -> str:
    if projection_type in ("gate_proj", "up_proj"):
        return "in"
    if projection_type == "down_proj":
        return "out"
    raise ValueError(f"Unknown projection_type: {projection_type}")


def build_mlp_search_cache(
    targets: list[MLPLinearTarget],
    input_rms_records: dict[str, InputRMSRecord],
    config: GradientBlockPruningConfig,
    hidden_size: int,
) -> _MLPSearchCache:
    """Cache float32 MLP weights/RMS on each parameter's device for fast search."""
    weights: dict[str, torch.Tensor] = {}
    rms: dict[str, torch.Tensor] = {}
    sides: dict[str, str] = {}
    for target in targets:
        name = target.module_name
        if name not in input_rms_records:
            raise KeyError(f"Missing InputRMSRecord for {name}")
        device = target.module.weight.device
        w = target.module.weight.detach().float().contiguous().to(device)
        side = _projection_residual_side(target.projection_type)
        if side == "in" and w.shape[1] != hidden_size:
            raise ValueError(
                f"{name}: expected d_in={hidden_size}, got {w.shape[1]}"
            )
        if side == "out" and w.shape[0] != hidden_size:
            raise ValueError(
                f"{name}: expected d_out={hidden_size}, got {w.shape[0]}"
            )
        weights[name] = w
        rms[name] = (
            input_rms_records[name]
            .input_rms.detach()
            .float()
            .contiguous()
            .to(device)
        )
        sides[name] = side
    masks = initialize_all_one_masks(
        targets, config.block_height, config.block_width
    )
    return _MLPSearchCache(
        targets=targets,
        weights=weights,
        input_rms=rms,
        residual_side=sides,
        hidden_size=hidden_size,
        block_height=config.block_height,
        block_width=config.block_width,
        masks=masks,
    )


def release_mlp_search_cache(cache: _MLPSearchCache) -> None:
    """Drop search weight/RMS buffers and free CUDA caching allocator if used."""
    used_cuda = False
    for name in list(cache.weights):
        tensor = cache.weights.pop(name)
        if tensor.is_cuda:
            used_cuda = True
        del tensor
    for name in list(cache.input_rms):
        tensor = cache.input_rms.pop(name)
        if tensor.is_cuda:
            used_cuda = True
        del tensor
    cache.weights.clear()
    cache.input_rms.clear()
    if used_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _residual_axis_channel_raw(
    weight: torch.Tensor,
    rms: torch.Tensor,
    side: str,
    name: str,
) -> torch.Tensor:
    """Wanda-style residual-axis channel mass (CPU float64)."""
    a = rms
    if a.device != weight.device:
        a = a.to(device=weight.device)
    if side == "in":
        if a.numel() != weight.shape[1]:
            raise ValueError(
                f"{name}: rms len {a.numel()} != d_in {weight.shape[1]}"
            )
        return (weight.abs().sum(dim=0) * a).double().cpu()
    if a.numel() != weight.shape[1]:
        raise ValueError(
            f"{name}: rms len {a.numel()} != d_in {weight.shape[1]}"
        )
    return (weight.abs() * a.unsqueeze(0)).sum(dim=1).double().cpu()


def compute_residual_channel_scores(
    cache: _MLPSearchCache,
    *,
    agg: str = "equal",
    matrix_fisher_weights: dict[str, float] | None = None,
    layer_fisher_weights: dict[int, float] | None = None,
    matrix_sparsity_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Aggregate per-channel Wanda mass on the residual axis across all MLP mats.

    agg:
      equal              — L1-normalize each matrix, sum u/g/d and layers equally
      layer_fisher       — L1 per matrix, sum in-layer, weight by layer Fisher total
      matrix_fisher      — L1 per matrix, weight by that matrix Fisher total
      raw_wanda          — sum raw residual-axis Wanda mass (no L1)
      sparsity_raw_wanda — raw Wanda × per-matrix Fisher prune sparsity ρ_m (no L1)
      density_raw_wanda  — raw Wanda × (1 − ρ_m) keep rate (no L1)
    """
    if agg not in {
        "equal",
        "layer_fisher",
        "matrix_fisher",
        "raw_wanda",
        "sparsity_raw_wanda",
        "density_raw_wanda",
    }:
        raise ValueError(f"Unknown residual channel agg: {agg!r}")
    if agg == "matrix_fisher":
        if matrix_fisher_weights is None:
            raise ValueError("matrix_fisher agg requires matrix_fisher_weights")
    if agg == "layer_fisher":
        if layer_fisher_weights is None:
            raise ValueError("layer_fisher agg requires layer_fisher_weights")
    if agg in {"sparsity_raw_wanda", "density_raw_wanda"}:
        if matrix_sparsity_weights is None:
            raise ValueError(f"{agg} agg requires matrix_sparsity_weights")

    hidden = cache.hidden_size
    accum = torch.zeros(hidden, dtype=torch.float64)
    by_layer: dict[int, list[MLPLinearTarget]] = {}
    for target in cache.targets:
        by_layer.setdefault(target.layer_index, []).append(target)

    for layer_index, layer_targets in by_layer.items():
        layer_score = torch.zeros(hidden, dtype=torch.float64)
        for target in layer_targets:
            name = target.module_name
            raw = _residual_axis_channel_raw(
                cache.weights[name],
                cache.input_rms[name],
                cache.residual_side[name],
                name,
            )
            total = float(raw.sum().item())
            if not (total > 0.0):
                raise ValueError(
                    f"layer {layer_index} {target.projection_type}: "
                    f"non-positive residual channel score total {total}"
                )
            if agg == "raw_wanda":
                contrib = raw
            elif agg in {"sparsity_raw_wanda", "density_raw_wanda"}:
                if name not in matrix_sparsity_weights:
                    raise KeyError(
                        f"matrix_sparsity_weights missing module {name}"
                    )
                rho = float(matrix_sparsity_weights[name])
                if not (0.0 <= rho <= 1.0):
                    raise ValueError(
                        f"matrix sparsity for {name} must be in [0, 1], got {rho}"
                    )
                w = rho if agg == "sparsity_raw_wanda" else (1.0 - rho)
                contrib = raw * w
            else:
                contrib = raw / total
                if agg == "matrix_fisher":
                    if name not in matrix_fisher_weights:
                        raise KeyError(
                            f"matrix_fisher_weights missing module {name}"
                        )
                    w = float(matrix_fisher_weights[name])
                    if not (w > 0.0):
                        raise ValueError(
                            f"matrix Fisher weight for {name} must be > 0, got {w}"
                        )
                    contrib = contrib * w
            layer_score = layer_score + contrib

        if agg == "layer_fisher":
            if layer_index not in layer_fisher_weights:
                raise KeyError(
                    f"layer_fisher_weights missing layer {layer_index}"
                )
            w_layer = float(layer_fisher_weights[layer_index])
            if not (w_layer > 0.0):
                raise ValueError(
                    f"layer Fisher weight for layer {layer_index} must be > 0, "
                    f"got {w_layer}"
                )
            layer_score = layer_score * w_layer

        accum = accum + layer_score

    if not torch.isfinite(accum).all():
        raise ValueError("Aggregated residual channel scores contain non-finite values")
    if float(accum.sum().item()) <= 0.0:
        raise ValueError("Aggregated residual channel scores sum to non-positive")
    return accum


def compute_init_residual_permutation(
    channel_score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.argsort(channel_score, descending=True, stable=True).to(dtype=torch.int64)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), dtype=torch.int64)
    return perm, inverse


def _virtual_weight_and_rms(
    cache: _MLPSearchCache,
    name: str,
    perm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = cache.weights[name]
    rms = cache.input_rms[name]
    side = cache.residual_side[name]
    index = perm if perm.device == w.device else perm.to(device=w.device)
    if side == "in":
        return w.index_select(1, index), rms.index_select(0, index)
    return w.index_select(0, index), rms


def _score_mode_for_config(config: GradientBlockPruningConfig) -> str:
    if config.score_type == "magnitude":
        return "magnitude"
    # fisher and fisher_budget_wanda both search with Wanda proxy / Wanda positions.
    return "wanda"


def build_virtual_score_records(
    cache: _MLPSearchCache,
    perm: torch.Tensor,
    config: GradientBlockPruningConfig,
    score_mode: str,
) -> dict[str, BlockScoreRecord]:
    _validate_permutation(perm, cache.hidden_size, "build_virtual_score_records")
    h, w_blk = cache.block_height, cache.block_width
    records: dict[str, BlockScoreRecord] = {}
    for target in cache.targets:
        name = target.module_name
        weight_p, rms_p = _virtual_weight_and_rms(cache, name, perm)
        if score_mode == "magnitude":
            block = (
                reduce_weight_magnitude_to_blocks(weight_p, h, w_blk)
                .double()
                .cpu()
            )
            zeros = torch.zeros_like(block)
            records[name] = _make_record(
                target=target,
                config=config,
                fisher=block,
                abs_taylor=zeros,
                signed_mean=zeros.clone(),
                current_mask=cache.masks[name].clone(),
            )
        elif score_mode == "wanda":
            block = (
                reduce_weight_wanda_to_blocks(weight_p, rms_p, h, w_blk)
                .double()
                .cpu()
            )
            zeros = torch.zeros_like(block)
            records[name] = _make_record(
                target=target,
                config=config,
                fisher=zeros,
                abs_taylor=zeros.clone(),
                signed_mean=zeros.clone(),
                current_mask=cache.masks[name].clone(),
                wanda=block,
            )
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")
    return records


def pruned_block_score_sum(
    score_records: dict[str, BlockScoreRecord],
    masks: dict[str, torch.Tensor],
    ranking_score_type: str,
) -> float:
    total = 0.0
    for name, mask in masks.items():
        score = score_records[name].primary_score(ranking_score_type)
        pruned = score[~mask]
        total += float(pruned.sum().item())
    return total


def _masks_are_all_ones(masks: dict[str, torch.Tensor]) -> bool:
    return all(bool(mask.all().item()) for mask in masks.values())


def _fast_loss_module_budgets_independent(
    score_records: dict[str, BlockScoreRecord],
    module_budgets: dict[str, int],
    ranking_score_type: str,
) -> float:
    """Sum of K_m lowest block scores per module (all-ones mask, independent)."""
    total = 0.0
    for name, budget in module_budgets.items():
        k = int(budget)
        if k < 0:
            raise ValueError(f"Negative budget for {name}: {k}")
        if k == 0:
            continue
        score = score_records[name].primary_score(ranking_score_type).reshape(-1)
        if k > int(score.numel()):
            raise RuntimeError(
                f"Budget {k} exceeds block count {score.numel()} for {name}"
            )
        total += float(torch.topk(score, k, largest=False).values.sum().item())
    return total


def _fast_loss_global_independent(
    score_records: dict[str, BlockScoreRecord],
    current_masks: dict[str, torch.Tensor],
    config: GradientBlockPruningConfig,
    ranking_score_type: str,
) -> float:
    """Vectorized global lowest-score selection under per-matrix caps (all-ones)."""
    import numpy as np

    from block_pruning.mask_allocator import _max_prunable

    total_blocks = sum(mask.numel() for mask in current_masks.values())
    target_pruned = int(total_blocks * config.target_block_sparsity)
    if target_pruned <= 0:
        return 0.0

    names = sorted(current_masks)
    score_parts: list[np.ndarray] = []
    module_parts: list[np.ndarray] = []
    max_prunable = np.zeros(len(names), dtype=np.int64)
    for mid, name in enumerate(names):
        score = (
            score_records[name]
            .primary_score(ranking_score_type)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        score_parts.append(score.astype(np.float64, copy=False))
        module_parts.append(np.full(score.shape[0], mid, dtype=np.int64))
        max_prunable[mid] = int(
            _max_prunable(
                current_masks[name],
                config.max_prune_ratio_per_matrix,
                config.min_keep_blocks_per_matrix,
            )
        )

    scores = np.concatenate(score_parts)
    modules = np.concatenate(module_parts)
    positions = np.arange(scores.shape[0], dtype=np.int64)
    # Match allocator key: (score, module_name, out, in) ~ (score, mid, flat_pos)
    order = np.lexsort((positions, modules, scores))

    pruned_per = np.zeros(len(names), dtype=np.int64)
    selected = 0
    loss = 0.0
    for idx in order:
        if selected >= target_pruned:
            break
        mid = int(modules[idx])
        if pruned_per[mid] >= max_prunable[mid]:
            continue
        pruned_per[mid] += 1
        selected += 1
        loss += float(scores[idx])

    if selected != target_pruned:
        raise RuntimeError(
            "Fast global loss could not reach target sparsity under caps: "
            f"selected={selected}, target_pruned={target_pruned}"
        )
    return loss


def evaluate_residual_permutation_loss(
    cache: _MLPSearchCache,
    perm: torch.Tensor,
    config: GradientBlockPruningConfig,
    *,
    score_mode: str,
    module_budgets: dict[str, int] | None,
) -> float:
    records = build_virtual_score_records(cache, perm, config, score_mode)
    ranking = "magnitude" if score_mode == "magnitude" else "wanda"
    use_fast = (
        _masks_are_all_ones(cache.masks)
        and not config.share_up_gate_mask
        and config.projection_prune_shares is None
    )

    if module_budgets is not None:
        if use_fast:
            return _fast_loss_module_budgets_independent(
                records, module_budgets, "wanda"
            )
        allocation = allocate_masks_by_module_budget(
            score_records=records,
            target_pruned_per_module=module_budgets,
            config=config,
            current_masks=cache.masks,
            ranking_score_type="wanda",
        )
        return pruned_block_score_sum(records, allocation.masks, "wanda")

    if use_fast:
        return _fast_loss_global_independent(
            records, cache.masks, config, ranking
        )

    allocation = allocate_block_masks(
        score_records=records,
        config=config,
        current_masks=cache.masks,
        ranking_score_type=ranking,
    )
    return pruned_block_score_sum(records, allocation.masks, ranking)


def local_search_residual_permutation(
    perm0: torch.Tensor,
    cache: _MLPSearchCache,
    config: GradientBlockPruningConfig,
    *,
    score_mode: str,
    module_budgets: dict[str, int] | None,
) -> tuple[torch.Tensor, float, float, int]:
    """Greedy channel-swap search minimizing pruned block score sum.

    Returns (best_perm, loss_init, loss_final, accepted_swaps).
    """
    steps = int(config.residual_perm_search_steps)
    if steps < 0:
        raise ValueError(
            f"residual_perm_search_steps must be >= 0, got {steps}"
        )
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(config.seed))

    current = perm0.clone()
    loss_init = evaluate_residual_permutation_loss(
        cache, current, config, score_mode=score_mode, module_budgets=module_budgets
    )
    best = current.clone()
    best_loss = loss_init
    accepted = 0
    hidden = cache.hidden_size

    if steps == 0:
        return best, float(loss_init), float(best_loss), accepted

    log_every = max(1, min(100, steps // 10))
    pbar = tqdm(range(steps), desc="[prune] residual-perm search", unit="step")
    for step in pbar:
        i = int(torch.randint(0, hidden, (1,), generator=rng).item())
        j = int(torch.randint(0, hidden, (1,), generator=rng).item())
        if i == j:
            pbar.set_postfix(L=f"{best_loss:.6g}", accepted=accepted)
            continue
        trial = current.clone()
        trial[i], trial[j] = trial[j].clone(), trial[i].clone()
        loss = evaluate_residual_permutation_loss(
            cache, trial, config, score_mode=score_mode, module_budgets=module_budgets
        )
        if loss < best_loss:
            current = trial
            best = trial
            best_loss = loss
            accepted += 1
        pbar.set_postfix(L=f"{best_loss:.6g}", accepted=accepted)
        if (step + 1) % log_every == 0 or step + 1 == steps:
            print(
                f"[prune] residual-perm search step={step + 1}/{steps} "
                f"L_best={best_loss:.6g} accepted={accepted}",
                flush=True,
            )

    return best, float(loss_init), float(best_loss), accepted


def _collect_fisher_records_for_residual(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    targets: list[MLPLinearTarget],
    config: GradientBlockPruningConfig,
) -> dict[str, BlockScoreRecord]:
    from block_pruning.gradient_scorer import collect_mlp_block_scores

    masks = initialize_all_one_masks(
        targets, config.block_height, config.block_width
    )
    return collect_mlp_block_scores(
        model=model,
        batches=batches,
        targets=targets,
        config=config,
        current_masks=masks,
    )


def _module_budgets_from_fisher_records(
    fisher_records: dict[str, BlockScoreRecord],
    targets: list[MLPLinearTarget],
    config: GradientBlockPruningConfig,
) -> dict[str, int]:
    masks = initialize_all_one_masks(
        targets, config.block_height, config.block_width
    )
    reference = allocate_block_masks(
        score_records=fisher_records,
        config=config,
        current_masks=masks,
        ranking_score_type="fisher",
    )
    budgets = extract_module_prune_budgets(reference.masks)
    if sum(budgets.values()) != reference.num_pruned_blocks:
        raise RuntimeError(
            "Fisher budget sum mismatch vs reference pruned count: "
            f"{sum(budgets.values())} vs {reference.num_pruned_blocks}"
        )
    return budgets


def _matrix_fisher_weight_dict(
    fisher_records: dict[str, BlockScoreRecord],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, rec in fisher_records.items():
        total = float(rec.fisher.sum().item())
        if not (total > 0.0):
            raise ValueError(f"Non-positive Fisher total for {name}: {total}")
        out[name] = total
    return out


def _layer_fisher_weight_dict(
    fisher_records: dict[str, BlockScoreRecord],
    targets: list[MLPLinearTarget],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for target in targets:
        name = target.module_name
        if name not in fisher_records:
            raise KeyError(f"Fisher record missing for {name}")
        total = float(fisher_records[name].fisher.sum().item())
        if not (total > 0.0):
            raise ValueError(f"Non-positive Fisher total for {name}: {total}")
        out[target.layer_index] = out.get(target.layer_index, 0.0) + total
    for layer_index, w in out.items():
        if not (w > 0.0):
            raise ValueError(
                f"Non-positive layer Fisher total for layer {layer_index}: {w}"
            )
    return out


def _matrix_sparsity_weight_dict(
    module_budgets: dict[str, int],
    masks: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Per-matrix prune sparsity ρ_m = K_m / N_m from Fisher allocation budgets."""
    out: dict[str, float] = {}
    for name, mask in masks.items():
        if name not in module_budgets:
            raise KeyError(f"module_budgets missing module {name}")
        n_blocks = int(mask.numel())
        if n_blocks <= 0:
            raise ValueError(f"Non-positive block count for {name}: {n_blocks}")
        k = int(module_budgets[name])
        if k < 0 or k > n_blocks:
            raise ValueError(
                f"Budget {k} out of range for {name} with {n_blocks} blocks"
            )
        out[name] = float(k) / float(n_blocks)
    if not out:
        raise ValueError("Empty matrix sparsity weights")
    if not any(rho > 0.0 for rho in out.values()):
        raise ValueError(
            "All matrix sparsities are zero; cannot weight residual channels"
        )
    return out


def _freeze_fisher_module_budgets(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    targets: list[MLPLinearTarget],
    config: GradientBlockPruningConfig,
) -> dict[str, int]:
    fisher_records = _collect_fisher_records_for_residual(
        model, batches, targets, config
    )
    return _module_budgets_from_fisher_records(fisher_records, targets, config)


def prepare_and_apply_residual_permutation(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]] | None,
    targets: list[MLPLinearTarget],
    config: GradientBlockPruningConfig,
) -> ResidualPermutationRecord:
    """Search residual π minimizing pruned-block score sum, then absorb into model."""
    if config.residual_permutation != "block_loss":
        raise ValueError(
            "prepare_and_apply_residual_permutation requires "
            f"residual_permutation='block_loss', got {config.residual_permutation!r}"
        )
    if config.score_type == "random":
        raise ValueError(
            "residual_permutation=block_loss is incompatible with score_type=random"
        )
    if batches is None:
        raise ValueError(
            "residual_permutation=block_loss requires calibration batches; got None"
        )

    # Validate MLP triplets early (shape consistency).
    group_mlp_projection_triplets(targets)
    hidden_size = resolve_hidden_size(model, targets)
    mounts = discover_residual_mounts(model, hidden_size)
    channel_agg = config.residual_channel_agg

    print(
        f"[prune] residual perm: discovering mounts hidden_size={hidden_size} "
        f"n_mounts={len(mounts)} channel_agg={channel_agg}",
        flush=True,
    )
    input_rms_records = collect_mlp_input_rms(
        model,
        batches,
        targets,
        progress_desc="[prune] residual-perm rms",
    )

    module_budgets: dict[str, int] | None = None
    matrix_fisher_weights: dict[str, float] | None = None
    layer_fisher_weights: dict[int, float] | None = None
    matrix_sparsity_weights: dict[str, float] | None = None
    need_fisher_for_agg = channel_agg in {
        "layer_fisher",
        "matrix_fisher",
        "sparsity_raw_wanda",
        "density_raw_wanda",
    }
    need_fisher_for_budget = (
        config.score_type == "fisher_budget_wanda"
        or channel_agg in {"sparsity_raw_wanda", "density_raw_wanda"}
    )

    fisher_records: dict[str, BlockScoreRecord] | None = None
    if need_fisher_for_budget or need_fisher_for_agg:
        print(
            "[prune] residual perm: collecting Fisher for "
            f"budget={need_fisher_for_budget} channel_agg={channel_agg}",
            flush=True,
        )
        fisher_records = _collect_fisher_records_for_residual(
            model, batches, targets, config
        )
        if need_fisher_for_budget:
            module_budgets = _module_budgets_from_fisher_records(
                fisher_records, targets, config
            )
        if channel_agg == "matrix_fisher":
            matrix_fisher_weights = _matrix_fisher_weight_dict(fisher_records)
        elif channel_agg == "layer_fisher":
            layer_fisher_weights = _layer_fisher_weight_dict(
                fisher_records, targets
            )

    cache = build_mlp_search_cache(
        targets, input_rms_records, config, hidden_size
    )
    try:
        cache_devices = sorted({str(w.device) for w in cache.weights.values()})
        print(
            f"[prune] residual perm: search cache devices={cache_devices}",
            flush=True,
        )
        if channel_agg in {"sparsity_raw_wanda", "density_raw_wanda"}:
            if module_budgets is None:
                raise RuntimeError(
                    f"{channel_agg} requires module_budgets from Fisher allocation"
                )
            matrix_sparsity_weights = _matrix_sparsity_weight_dict(
                module_budgets, cache.masks
            )
        channel_score = compute_residual_channel_scores(
            cache,
            agg=channel_agg,
            matrix_fisher_weights=matrix_fisher_weights,
            layer_fisher_weights=layer_fisher_weights,
            matrix_sparsity_weights=matrix_sparsity_weights,
        )
        perm0, _inv0 = compute_init_residual_permutation(channel_score)
        score_mode = _score_mode_for_config(config)

        print(
            f"[prune] residual perm: search mode={score_mode} "
            f"steps={config.residual_perm_search_steps}",
            flush=True,
        )
        best_perm, loss_init, loss_final, accepted = local_search_residual_permutation(
            perm0,
            cache,
            config,
            score_mode=score_mode,
            module_budgets=module_budgets,
        )
    finally:
        release_mlp_search_cache(cache)

    inverse = torch.empty_like(best_perm)
    inverse[best_perm] = torch.arange(hidden_size, dtype=torch.int64)

    print(
        f"[prune] residual perm: L_init={loss_init:.6g} L_final={loss_final:.6g} "
        f"accepted={accepted} channel_agg={channel_agg}",
        flush=True,
    )
    apply_residual_permutation(model, mounts, best_perm, hidden_size)

    return ResidualPermutationRecord(
        hidden_size=hidden_size,
        permutation=best_perm.cpu().contiguous(),
        inverse_permutation=inverse.cpu().contiguous(),
        channel_score=channel_score.cpu().contiguous(),
        loss_init=loss_init,
        loss_final=loss_final,
        search_steps=int(config.residual_perm_search_steps),
        accepted_swaps=accepted,
        mount_names=[m.name for m in mounts],
        search_score_mode=score_mode,
        channel_agg=channel_agg,
    )
