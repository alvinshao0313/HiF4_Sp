from __future__ import annotations

from dataclasses import dataclass

import torch

from block_pruning.block_utils import active_block_indices
from block_pruning.config import GradientBlockPruningConfig
from block_pruning.gradient_scorer import BlockScoreRecord


@dataclass
class MaskAllocationResult:
    masks: dict[str, torch.Tensor]
    num_total_blocks: int
    num_pruned_blocks: int
    actual_block_sparsity: float
    newly_pruned: int
    target_pruned: int


def _clone_masks(masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in masks.items()}


def _pruned_count_per_matrix(masks: dict[str, torch.Tensor]) -> dict[str, int]:
    return {name: int((~mask).sum().item()) for name, mask in masks.items()}


def _max_prunable(
    mask: torch.Tensor,
    max_prune_ratio_per_matrix: float,
    min_keep_blocks_per_matrix: int,
) -> int:
    total = mask.numel()
    max_by_ratio = int(total * max_prune_ratio_per_matrix)
    max_by_keep_floor = total - min_keep_blocks_per_matrix
    return min(max_by_ratio, max_by_keep_floor)


def _pair_key(module_name: str) -> tuple[str, str] | None:
    """Return (prefix, role) for up/gate sharing; role in {up, gate}."""
    if module_name.endswith(".up_proj"):
        return module_name[: -len("up_proj")], "up"
    if module_name.endswith(".gate_proj"):
        return module_name[: -len("gate_proj")], "gate"
    return None


def extract_module_prune_budgets(
    reference_masks: dict[str, torch.Tensor],
) -> dict[str, int]:
    """Return cumulative pruned-block count for every module."""
    if not reference_masks:
        raise ValueError("reference_masks is empty")
    budgets: dict[str, int] = {}
    for module_name, mask in reference_masks.items():
        if mask.dtype != torch.bool:
            raise ValueError(
                f"Mask for {module_name} must be bool, got {mask.dtype}"
            )
        if mask.ndim != 2:
            raise ValueError(
                f"Mask for {module_name} must be rank 2, got shape {tuple(mask.shape)}"
            )
        budget = int((~mask).sum().item())
        if budget < 0 or budget > mask.numel():
            raise ValueError(
                f"Invalid budget {budget} for {module_name} "
                f"(numel={mask.numel()})"
            )
        budgets[module_name] = budget
    return budgets


def allocate_block_masks(
    score_records: dict[str, BlockScoreRecord],
    config: GradientBlockPruningConfig,
    current_masks: dict[str, torch.Tensor],
    cumulative_target_sparsity: float | None = None,
    ranking_score_type: str | None = None,
) -> MaskAllocationResult:
    """Global ranking + per-matrix pruning cap. Raises if target unreachable."""
    if config.selection_mode != "global_constrained":
        raise ValueError(f"Unsupported selection_mode: {config.selection_mode}")

    target_sparsity = (
        config.target_block_sparsity
        if cumulative_target_sparsity is None
        else cumulative_target_sparsity
    )
    if not (0.0 <= target_sparsity < 1.0):
        raise ValueError(f"Invalid cumulative_target_sparsity: {target_sparsity}")

    score_type = ranking_score_type or config.score_type

    if config.projection_prune_shares is not None:
        return _allocate_by_projection_shares(
            score_records=score_records,
            config=config,
            current_masks=current_masks,
            target_sparsity=target_sparsity,
            score_type=score_type,
        )

    if config.share_up_gate_mask:
        return _allocate_shared_up_gate(
            score_records=score_records,
            config=config,
            current_masks=current_masks,
            target_sparsity=target_sparsity,
            score_type=score_type,
        )
    return _allocate_independent(
        score_records=score_records,
        config=config,
        current_masks=current_masks,
        target_sparsity=target_sparsity,
        score_type=score_type,
    )


def _distribute_integer_budget(
    total: int,
    normalized_shares: dict[str, float],
    order: tuple[str, ...],
) -> dict[str, int]:
    """Largest-remainder allocation so sum(budgets) == total."""
    if total < 0:
        raise ValueError(f"total budget must be >= 0, got {total}")
    if set(normalized_shares) != set(order):
        raise ValueError(
            f"share keys {sorted(normalized_shares)} != order {list(order)}"
        )
    raw = {k: total * float(normalized_shares[k]) for k in order}
    floors = {k: int(raw[k]) for k in order}
    assigned = sum(floors.values())
    remainder = total - assigned
    # Prefer types with larger fractional parts, then larger share, then name.
    ranked = sorted(
        order,
        key=lambda k: (-(raw[k] - floors[k]), -float(normalized_shares[k]), k),
    )
    budgets = dict(floors)
    for i in range(remainder):
        budgets[ranked[i]] += 1
    if sum(budgets.values()) != total:
        raise RuntimeError(
            f"Budget distribution failed: sum={sum(budgets.values())} total={total}"
        )
    return budgets


def _allocate_by_projection_shares(
    score_records: dict[str, BlockScoreRecord],
    config: GradientBlockPruningConfig,
    current_masks: dict[str, torch.Tensor],
    target_sparsity: float,
    score_type: str,
) -> MaskAllocationResult:
    """Split global prune budget across projection types, then allocate within each."""
    from block_pruning.config import (
        PROJECTION_TYPES,
        normalize_projection_prune_shares,
    )

    shares = config.projection_prune_shares
    if shares is None:
        raise RuntimeError("projection_prune_shares is None in share allocate path")
    shares = normalize_projection_prune_shares(shares)

    total_blocks = sum(mask.numel() for mask in current_masks.values())
    target_pruned = int(total_blocks * target_sparsity)
    type_targets = _distribute_integer_budget(
        target_pruned, shares, PROJECTION_TYPES
    )

    by_type: dict[str, list[str]] = {p: [] for p in PROJECTION_TYPES}
    for name, rec in score_records.items():
        proj = rec.projection_type
        if proj not in by_type:
            raise ValueError(f"Unexpected projection_type {proj!r} for {name}")
        by_type[proj].append(name)

    for proj in PROJECTION_TYPES:
        if not by_type[proj]:
            raise RuntimeError(f"No modules found for projection type {proj}")

    # share_up_gate only makes sense when allocating gate+up together.
    # With per-type pools they are separate; require equal shares (validated in config)
    # and still allocate each type independently unless share flag is on — when on,
    # gate and up must be allocated jointly with equal budgets.
    merged_masks = _clone_masks(current_masks)
    newly_pruned = 0

    if config.share_up_gate_mask:
        if type_targets["gate_proj"] != type_targets["up_proj"]:
            raise RuntimeError(
                "share_up_gate_mask requires equal gate/up prune budgets after "
                f"share distribution, got gate={type_targets['gate_proj']}, "
                f"up={type_targets['up_proj']}"
            )
        # Allocate gate+up jointly with combined pool sparsity, then down alone.
        ug_names = by_type["gate_proj"] + by_type["up_proj"]
        ug_scores = {n: score_records[n] for n in ug_names}
        ug_masks = {n: current_masks[n] for n in ug_names}
        ug_target = type_targets["gate_proj"] + type_targets["up_proj"]
        ug_result = _allocate_shared_up_gate(
            score_records=ug_scores,
            config=config,
            current_masks=ug_masks,
            target_sparsity=0.0,
            score_type=score_type,
            exact_target_pruned=ug_target,
        )
        for n, mask in ug_result.masks.items():
            merged_masks[n] = mask
        newly_pruned += ug_result.newly_pruned

        down_names = by_type["down_proj"]
        down_scores = {n: score_records[n] for n in down_names}
        down_masks = {n: current_masks[n] for n in down_names}
        down_target = type_targets["down_proj"]
        down_result = _allocate_independent(
            score_records=down_scores,
            config=config,
            current_masks=down_masks,
            target_sparsity=0.0,
            score_type=score_type,
            exact_target_pruned=down_target,
        )
        for n, mask in down_result.masks.items():
            merged_masks[n] = mask
        newly_pruned += down_result.newly_pruned
    else:
        for proj in PROJECTION_TYPES:
            names = by_type[proj]
            sub_scores = {n: score_records[n] for n in names}
            sub_masks = {n: current_masks[n] for n in names}
            t_pruned = type_targets[proj]
            sub_result = _allocate_independent(
                score_records=sub_scores,
                config=config,
                current_masks=sub_masks,
                target_sparsity=0.0,
                score_type=score_type,
                exact_target_pruned=t_pruned,
            )
            actual_type_pruned = sum(
                int((~sub_result.masks[n]).sum().item()) for n in names
            )
            if actual_type_pruned != t_pruned:
                raise RuntimeError(
                    f"Projection share unmet for {proj}: "
                    f"target_pruned={t_pruned}, actual={actual_type_pruned}"
                )
            for n, mask in sub_result.masks.items():
                merged_masks[n] = mask
            newly_pruned += sub_result.newly_pruned

    num_pruned = sum(int((~mask).sum().item()) for mask in merged_masks.values())
    if num_pruned != target_pruned:
        raise RuntimeError(
            f"Global pruned count {num_pruned} != target {target_pruned} "
            f"after projection-share allocation"
        )
    return MaskAllocationResult(
        masks=merged_masks,
        num_total_blocks=total_blocks,
        num_pruned_blocks=num_pruned,
        actual_block_sparsity=num_pruned / total_blocks if total_blocks else 0.0,
        newly_pruned=newly_pruned,
        target_pruned=target_pruned,
    )


def _allocate_independent(
    score_records: dict[str, BlockScoreRecord],
    config: GradientBlockPruningConfig,
    current_masks: dict[str, torch.Tensor],
    target_sparsity: float,
    score_type: str,
    exact_target_pruned: int | None = None,
) -> MaskAllocationResult:
    total_blocks = sum(mask.numel() for mask in current_masks.values())
    target_pruned = (
        int(exact_target_pruned)
        if exact_target_pruned is not None
        else int(total_blocks * target_sparsity)
    )
    if exact_target_pruned is not None:
        if not (0 <= target_pruned <= total_blocks):
            raise ValueError(
                f"exact_target_pruned={target_pruned} out of range "
                f"[0, {total_blocks}]"
            )
    already_pruned = sum(int((~mask).sum().item()) for mask in current_masks.values())
    additional_needed = target_pruned - already_pruned

    new_masks = _clone_masks(current_masks)
    if additional_needed <= 0:
        num_pruned = already_pruned
        return MaskAllocationResult(
            masks=new_masks,
            num_total_blocks=total_blocks,
            num_pruned_blocks=num_pruned,
            actual_block_sparsity=num_pruned / total_blocks if total_blocks else 0.0,
            newly_pruned=0,
            target_pruned=target_pruned,
        )

    candidates: list[tuple[float, str, int, int]] = []
    for module_name, record in score_records.items():
        mask = current_masks[module_name]
        score = record.primary_score(score_type)
        for out_b, in_b in active_block_indices(mask):
            candidates.append(
                (float(score[out_b, in_b].item()), module_name, out_b, in_b)
            )

    # Stable: score asc, then module_name, then indices
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    pruned_per_matrix = _pruned_count_per_matrix(new_masks)
    max_prunable_per_matrix = {
        name: _max_prunable(
            mask,
            config.max_prune_ratio_per_matrix,
            config.min_keep_blocks_per_matrix,
        )
        for name, mask in new_masks.items()
    }

    selected = 0
    for _score, module_name, out_b, in_b in candidates:
        if selected >= additional_needed:
            break
        if pruned_per_matrix[module_name] >= max_prunable_per_matrix[module_name]:
            continue
        if not new_masks[module_name][out_b, in_b]:
            continue
        new_masks[module_name][out_b, in_b] = False
        pruned_per_matrix[module_name] += 1
        selected += 1

    if selected != additional_needed:
        max_possible = already_pruned + selected
        max_sparsity = max_possible / total_blocks if total_blocks else 0.0
        raise RuntimeError(
            "Cannot reach target sparsity under current matrix constraints. "
            f"needed_additional={additional_needed}, selected={selected}, "
            f"max_achievable_sparsity={max_sparsity:.6f}, "
            f"target_sparsity={target_sparsity:.6f}."
        )

    num_pruned = sum(int((~mask).sum().item()) for mask in new_masks.values())
    return MaskAllocationResult(
        masks=new_masks,
        num_total_blocks=total_blocks,
        num_pruned_blocks=num_pruned,
        actual_block_sparsity=num_pruned / total_blocks,
        newly_pruned=selected,
        target_pruned=target_pruned,
    )


def _allocate_shared_up_gate(
    score_records: dict[str, BlockScoreRecord],
    config: GradientBlockPruningConfig,
    current_masks: dict[str, torch.Tensor],
    target_sparsity: float,
    score_type: str,
    exact_target_pruned: int | None = None,
) -> MaskAllocationResult:
    """Prune up/gate as joint pairs; cost counts as two physical blocks."""
    total_blocks = sum(mask.numel() for mask in current_masks.values())
    target_pruned = (
        int(exact_target_pruned)
        if exact_target_pruned is not None
        else int(total_blocks * target_sparsity)
    )
    if exact_target_pruned is not None:
        if not (0 <= target_pruned <= total_blocks):
            raise ValueError(
                f"exact_target_pruned={target_pruned} out of range "
                f"[0, {total_blocks}]"
            )
    already_pruned = sum(int((~mask).sum().item()) for mask in current_masks.values())
    additional_needed = target_pruned - already_pruned

    new_masks = _clone_masks(current_masks)
    if additional_needed <= 0:
        return MaskAllocationResult(
            masks=new_masks,
            num_total_blocks=total_blocks,
            num_pruned_blocks=already_pruned,
            actual_block_sparsity=already_pruned / total_blocks if total_blocks else 0.0,
            newly_pruned=0,
            target_pruned=target_pruned,
        )

    if additional_needed % 2 != 0:
        raise RuntimeError(
            "share_up_gate_mask requires even additional_needed because each "
            f"pair costs 2 blocks; additional_needed={additional_needed}."
        )

    # Group up/gate by prefix
    pairs: dict[str, dict[str, str]] = {}
    down_modules: list[str] = []
    for name in current_masks:
        pk = _pair_key(name)
        if pk is None:
            down_modules.append(name)
            continue
        prefix, role = pk
        pairs.setdefault(prefix, {})[role] = name

    for prefix, roles in pairs.items():
        if "up" not in roles or "gate" not in roles:
            raise RuntimeError(f"Incomplete up/gate pair for prefix={prefix}: {roles}")

    pair_candidates: list[tuple[float, str, str, str, int, int]] = []
    # (score, up_name, gate_name, prefix, out_b, in_b)
    for prefix, roles in pairs.items():
        up_name = roles["up"]
        gate_name = roles["gate"]
        up_mask = current_masks[up_name]
        gate_mask = current_masks[gate_name]
        if up_mask.shape != gate_mask.shape:
            raise RuntimeError(
                f"up/gate mask shape mismatch: {up_name} {tuple(up_mask.shape)} vs "
                f"{gate_name} {tuple(gate_mask.shape)}"
            )
        up_score = score_records[up_name].primary_score(score_type)
        gate_score = score_records[gate_name].primary_score(score_type)
        joint = up_score + gate_score
        # Only prune where both are still kept
        both_kept = up_mask & gate_mask
        for out_b, in_b in active_block_indices(both_kept):
            pair_candidates.append(
                (
                    float(joint[out_b, in_b].item()),
                    up_name,
                    gate_name,
                    prefix,
                    out_b,
                    in_b,
                )
            )

    # Independent down_proj candidates (cost 1)
    down_candidates: list[tuple[float, str, int, int]] = []
    for name in down_modules:
        mask = current_masks[name]
        score = score_records[name].primary_score(score_type)
        for out_b, in_b in active_block_indices(mask):
            down_candidates.append((float(score[out_b, in_b].item()), name, out_b, in_b))

    # Unified candidate stream: (score, cost, kind, payload)
    # Prefer lower score; for ties prefer smaller cost then name.
    unified: list[tuple] = []
    for score, up_name, gate_name, prefix, out_b, in_b in pair_candidates:
        unified.append((score, 2, "pair", up_name, gate_name, out_b, in_b))
    for score, name, out_b, in_b in down_candidates:
        unified.append((score, 1, "single", name, out_b, in_b))

    unified.sort(key=lambda x: (x[0], x[1], str(x[3]), x[-2], x[-1]))

    pruned_per_matrix = _pruned_count_per_matrix(new_masks)
    max_prunable_per_matrix = {
        name: _max_prunable(
            mask,
            config.max_prune_ratio_per_matrix,
            config.min_keep_blocks_per_matrix,
        )
        for name, mask in new_masks.items()
    }

    selected = 0
    for item in unified:
        if selected >= additional_needed:
            break
        if item[2] == "pair":
            _score, cost, _kind, up_name, gate_name, out_b, in_b = item
            if selected + cost > additional_needed:
                continue
            if pruned_per_matrix[up_name] >= max_prunable_per_matrix[up_name]:
                continue
            if pruned_per_matrix[gate_name] >= max_prunable_per_matrix[gate_name]:
                continue
            if (not new_masks[up_name][out_b, in_b]) or (not new_masks[gate_name][out_b, in_b]):
                continue
            new_masks[up_name][out_b, in_b] = False
            new_masks[gate_name][out_b, in_b] = False
            pruned_per_matrix[up_name] += 1
            pruned_per_matrix[gate_name] += 1
            selected += cost
        else:
            _score, cost, _kind, name, out_b, in_b = item
            if selected + cost > additional_needed:
                continue
            if pruned_per_matrix[name] >= max_prunable_per_matrix[name]:
                continue
            if not new_masks[name][out_b, in_b]:
                continue
            new_masks[name][out_b, in_b] = False
            pruned_per_matrix[name] += 1
            selected += cost

    if selected != additional_needed:
        max_possible = already_pruned + selected
        max_sparsity = max_possible / total_blocks if total_blocks else 0.0
        raise RuntimeError(
            "Cannot reach target sparsity under shared up/gate constraints. "
            f"needed_additional={additional_needed}, selected={selected}, "
            f"max_achievable_sparsity={max_sparsity:.6f}, "
            f"target_sparsity={target_sparsity:.6f}."
        )

    num_pruned = sum(int((~mask).sum().item()) for mask in new_masks.values())
    return MaskAllocationResult(
        masks=new_masks,
        num_total_blocks=total_blocks,
        num_pruned_blocks=num_pruned,
        actual_block_sparsity=num_pruned / total_blocks,
        newly_pruned=selected,
        target_pruned=target_pruned,
    )


def allocate_masks_by_module_budget(
    score_records: dict[str, BlockScoreRecord],
    target_pruned_per_module: dict[str, int],
    config: GradientBlockPruningConfig,
    current_masks: dict[str, torch.Tensor],
    ranking_score_type: str = "wanda",
) -> MaskAllocationResult:
    """Exact per-module cumulative budgets with local score ranking."""
    if config.share_up_gate_mask:
        return _allocate_by_budget_shared(
            score_records=score_records,
            target_pruned_per_module=target_pruned_per_module,
            config=config,
            current_masks=current_masks,
            ranking_score_type=ranking_score_type,
        )
    return _allocate_by_budget_independent(
        score_records=score_records,
        target_pruned_per_module=target_pruned_per_module,
        config=config,
        current_masks=current_masks,
        ranking_score_type=ranking_score_type,
    )


def _validate_budget_module_sets(
    score_records: dict[str, BlockScoreRecord],
    target_pruned_per_module: dict[str, int],
    current_masks: dict[str, torch.Tensor],
) -> None:
    keys_scores = set(score_records)
    keys_budgets = set(target_pruned_per_module)
    keys_masks = set(current_masks)
    if keys_scores != keys_budgets or keys_scores != keys_masks:
        raise ValueError(
            "Module key mismatch across score_records / budgets / masks: "
            f"scores={sorted(keys_scores)}, budgets={sorted(keys_budgets)}, "
            f"masks={sorted(keys_masks)}"
        )


def _prune_module_to_budget(
    module_name: str,
    score: torch.Tensor,
    current_mask: torch.Tensor,
    target_pruned: int,
    config: GradientBlockPruningConfig,
) -> tuple[torch.Tensor, int]:
    """Return (new_mask, newly_pruned) for one module."""
    if score.shape != current_mask.shape:
        raise ValueError(
            f"Score/mask shape mismatch for {module_name}: "
            f"{tuple(score.shape)} vs {tuple(current_mask.shape)}"
        )
    current_pruned = int((~current_mask).sum().item())
    max_prunable = _max_prunable(
        current_mask,
        config.max_prune_ratio_per_matrix,
        config.min_keep_blocks_per_matrix,
    )
    if not (0 <= current_pruned <= target_pruned <= max_prunable):
        raise RuntimeError(
            f"Unreachable budget for {module_name}: "
            f"current_pruned={current_pruned}, target_pruned={target_pruned}, "
            f"max_prunable={max_prunable}"
        )
    additional_needed = target_pruned - current_pruned
    new_mask = current_mask.clone()
    if additional_needed == 0:
        return new_mask, 0

    candidates: list[tuple[float, int, int]] = []
    for out_b, in_b in active_block_indices(current_mask):
        candidates.append((float(score[out_b, in_b].item()), out_b, in_b))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))

    if len(candidates) < additional_needed:
        raise RuntimeError(
            f"Not enough active blocks for {module_name}: "
            f"need {additional_needed}, have {len(candidates)}"
        )

    for i in range(additional_needed):
        _s, out_b, in_b = candidates[i]
        new_mask[out_b, in_b] = False
    return new_mask, additional_needed


def _allocate_by_budget_independent(
    score_records: dict[str, BlockScoreRecord],
    target_pruned_per_module: dict[str, int],
    config: GradientBlockPruningConfig,
    current_masks: dict[str, torch.Tensor],
    ranking_score_type: str,
) -> MaskAllocationResult:
    _validate_budget_module_sets(
        score_records, target_pruned_per_module, current_masks
    )
    new_masks = _clone_masks(current_masks)
    newly_pruned = 0
    for module_name in sorted(current_masks):
        score = score_records[module_name].primary_score(ranking_score_type)
        new_mask, added = _prune_module_to_budget(
            module_name=module_name,
            score=score,
            current_mask=current_masks[module_name],
            target_pruned=target_pruned_per_module[module_name],
            config=config,
        )
        new_masks[module_name] = new_mask
        newly_pruned += added

    for name, target in target_pruned_per_module.items():
        actual = int((~new_masks[name]).sum().item())
        if actual != target:
            raise RuntimeError(
                f"Budget mismatch for {name}: expected {target}, got {actual}"
            )

    total_blocks = sum(mask.numel() for mask in new_masks.values())
    num_pruned = sum(int((~mask).sum().item()) for mask in new_masks.values())
    target_pruned = sum(target_pruned_per_module.values())
    return MaskAllocationResult(
        masks=new_masks,
        num_total_blocks=total_blocks,
        num_pruned_blocks=num_pruned,
        actual_block_sparsity=num_pruned / total_blocks if total_blocks else 0.0,
        newly_pruned=newly_pruned,
        target_pruned=target_pruned,
    )


def _allocate_by_budget_shared(
    score_records: dict[str, BlockScoreRecord],
    target_pruned_per_module: dict[str, int],
    config: GradientBlockPruningConfig,
    current_masks: dict[str, torch.Tensor],
    ranking_score_type: str,
) -> MaskAllocationResult:
    _validate_budget_module_sets(
        score_records, target_pruned_per_module, current_masks
    )

    pairs: dict[str, dict[str, str]] = {}
    down_modules: list[str] = []
    for name in current_masks:
        pk = _pair_key(name)
        if pk is None:
            down_modules.append(name)
            continue
        prefix, role = pk
        pairs.setdefault(prefix, {})[role] = name

    for prefix, roles in pairs.items():
        if "up" not in roles or "gate" not in roles:
            raise RuntimeError(
                f"Incomplete up/gate pair for prefix={prefix}: {roles}"
            )

    new_masks = _clone_masks(current_masks)
    newly_pruned = 0

    for prefix, roles in pairs.items():
        up_name = roles["up"]
        gate_name = roles["gate"]
        up_budget = target_pruned_per_module[up_name]
        gate_budget = target_pruned_per_module[gate_name]
        if up_budget != gate_budget:
            raise RuntimeError(
                f"Unequal up/gate budgets for {prefix}: "
                f"up={up_budget}, gate={gate_budget}"
            )
        up_mask = current_masks[up_name]
        gate_mask = current_masks[gate_name]
        if not torch.equal(up_mask, gate_mask):
            raise RuntimeError(
                f"Current up/gate masks differ for {prefix}: "
                f"{up_name} vs {gate_name}"
            )
        up_score = score_records[up_name].primary_score(ranking_score_type)
        gate_score = score_records[gate_name].primary_score(ranking_score_type)
        if up_score.shape != gate_score.shape:
            raise RuntimeError(
                f"Unequal up/gate score shapes for {prefix}: "
                f"{tuple(up_score.shape)} vs {tuple(gate_score.shape)}"
            )
        joint = up_score + gate_score
        new_up, added = _prune_module_to_budget(
            module_name=up_name,
            score=joint,
            current_mask=up_mask,
            target_pruned=up_budget,
            config=config,
        )
        new_masks[up_name] = new_up
        new_masks[gate_name] = new_up.clone()
        # Pair coordinate costs two physical blocks.
        newly_pruned += added * 2

    for name in down_modules:
        score = score_records[name].primary_score(ranking_score_type)
        new_mask, added = _prune_module_to_budget(
            module_name=name,
            score=score,
            current_mask=current_masks[name],
            target_pruned=target_pruned_per_module[name],
            config=config,
        )
        new_masks[name] = new_mask
        newly_pruned += added

    for name, target in target_pruned_per_module.items():
        actual = int((~new_masks[name]).sum().item())
        if actual != target:
            raise RuntimeError(
                f"Budget mismatch for {name}: expected {target}, got {actual}"
            )

    total_blocks = sum(mask.numel() for mask in new_masks.values())
    num_pruned = sum(int((~mask).sum().item()) for mask in new_masks.values())
    target_pruned = sum(target_pruned_per_module.values())
    return MaskAllocationResult(
        masks=new_masks,
        num_total_blocks=total_blocks,
        num_pruned_blocks=num_pruned,
        actual_block_sparsity=num_pruned / total_blocks if total_blocks else 0.0,
        newly_pruned=newly_pruned,
        target_pruned=target_pruned,
    )
