from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from block_pruning.block_utils import (
    block_grid_shape,
    reduce_weight_gradient_to_blocks,
    reduce_weight_magnitude_to_blocks,
)
from block_pruning.calibration import move_batch_to_device
from block_pruning.config import GradientBlockPruningConfig
from block_pruning.mlp_registry import MLPLinearTarget
from block_pruning.model_loader import resolve_model_input_device


@dataclass
class BlockScoreRecord:
    module_name: str
    layer_index: int
    projection_type: str
    weight_shape: tuple[int, int]
    block_size: str
    block_height: int
    block_width: int
    fisher: torch.Tensor
    abs_taylor: torch.Tensor
    signed_mean: torch.Tensor
    current_mask: torch.Tensor
    wanda: torch.Tensor | None = None

    def primary_score(self, score_type: str) -> torch.Tensor:
        if score_type == "fisher":
            return self.fisher
        if score_type == "magnitude":
            return self.fisher
        if score_type == "random":
            return self.fisher
        if score_type == "wanda":
            if self.wanda is None:
                raise ValueError(
                    f"Wanda score missing for module: {self.module_name}"
                )
            return self.wanda
        raise ValueError(f"Unknown score_type: {score_type}")


def freeze_all_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def enable_mlp_weight_grads(targets: list[MLPLinearTarget]) -> None:
    for target in targets:
        target.module.weight.requires_grad_(True)


def _empty_accumulators(
    targets: list[MLPLinearTarget],
    block_height: int,
    block_width: int,
) -> dict[str, dict[str, torch.Tensor]]:
    acc: dict[str, dict[str, torch.Tensor]] = {}
    for target in targets:
        n_out, n_in = block_grid_shape(
            tuple(target.module.weight.shape), block_height, block_width
        )
        zeros = torch.zeros(n_out, n_in, dtype=torch.float64)
        acc[target.module_name] = {
            "score_sq": zeros.clone(),
            "score_abs": zeros.clone(),
            "score_signed": zeros.clone(),
        }
    return acc


def _make_record(
    target: MLPLinearTarget,
    config: GradientBlockPruningConfig,
    fisher: torch.Tensor,
    abs_taylor: torch.Tensor,
    signed_mean: torch.Tensor,
    current_mask: torch.Tensor,
    wanda: torch.Tensor | None = None,
) -> BlockScoreRecord:
    return BlockScoreRecord(
        module_name=target.module_name,
        layer_index=target.layer_index,
        projection_type=target.projection_type,
        weight_shape=tuple(target.module.weight.shape),
        block_size=str(config.block_size),
        block_height=config.block_height,
        block_width=config.block_width,
        fisher=fisher,
        abs_taylor=abs_taylor,
        signed_mean=signed_mean,
        current_mask=current_mask,
        wanda=wanda,
    )


def offload_weight_grad_to_block_accumulators(
    weight: torch.Tensor,
    grad: torch.Tensor,
    module_name: str,
    block_height: int,
    block_width: int,
    current_masks: dict[str, torch.Tensor],
    accumulators: dict[str, dict[str, torch.Tensor]],
) -> None:
    """Reduce W⊙G to blocks, accumulate on CPU. Does not retain GPU grad."""
    block_signal = reduce_weight_gradient_to_blocks(
        weight=weight.detach(),
        grad=grad.detach(),
        block_height=block_height,
        block_width=block_width,
    )
    active_mask = current_masks[module_name].to(device=block_signal.device)
    block_signal = block_signal * active_mask.to(block_signal.dtype)
    signal_cpu = block_signal.detach().double().cpu()
    accumulators[module_name]["score_sq"] += signal_cpu.square()
    accumulators[module_name]["score_abs"] += signal_cpu.abs()
    accumulators[module_name]["score_signed"] += signal_cpu


def _register_fisher_grad_offload_hooks(
    targets: list[MLPLinearTarget],
    block_height: int,
    block_width: int,
    current_masks: dict[str, torch.Tensor],
    accumulators: dict[str, dict[str, torch.Tensor]],
    seen: set[str],
) -> list:
    """Register post-accumulate hooks that offload block Fisher signal to CPU.

    Requires ``Tensor.register_post_accumulate_grad_hook`` so each leaf grad can
    be freed during backward instead of retaining all MLP grads until the end.
    """
    handles = []
    for target in targets:
        weight = target.module.weight
        if not hasattr(weight, "register_post_accumulate_grad_hook"):
            raise RuntimeError(
                "Fisher grad offload requires "
                "Tensor.register_post_accumulate_grad_hook (PyTorch >= 2.1)"
            )

        def _make_hook(module_name: str = target.module_name):
            def hook(param: torch.nn.Parameter) -> None:
                grad = param.grad
                if grad is None:
                    raise RuntimeError(
                        f"No gradient for target module: {module_name}"
                    )
                offload_weight_grad_to_block_accumulators(
                    weight=param,
                    grad=grad,
                    module_name=module_name,
                    block_height=block_height,
                    block_width=block_width,
                    current_masks=current_masks,
                    accumulators=accumulators,
                )
                seen.add(module_name)
                param.grad = None

            return hook

        handles.append(weight.register_post_accumulate_grad_hook(_make_hook()))
    return handles


def collect_mlp_block_scores(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    targets: list[MLPLinearTarget],
    config: GradientBlockPruningConfig,
    current_masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, BlockScoreRecord]:
    """Collect Block Empirical Fisher scores from causal LM loss gradients.

    One full forward/backward per calibration batch. MLP weight grads are reduced
    to block scores and moved to CPU inside post-accumulate hooks, then freed so
    all-layer grads never reside on GPU at once.
    """
    if config.score_batch_size != 1:
        raise ValueError("score_batch_size must be 1")
    if not batches:
        raise ValueError("calibration batches is empty")

    h, w = config.block_height, config.block_width
    if current_masks is None:
        from block_pruning.mlp_registry import initialize_all_one_masks

        current_masks = initialize_all_one_masks(targets, h, w)

    freeze_all_parameters(model)
    enable_mlp_weight_grads(targets)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    accumulators = _empty_accumulators(targets, h, w)
    device = resolve_model_input_device(model)
    num_batches = 0
    expected = {t.module_name for t in targets}

    for batch in batches:
        model.zero_grad(set_to_none=True)
        batch_dev = move_batch_to_device(batch, device)
        seen: set[str] = set()
        handles = _register_fisher_grad_offload_hooks(
            targets, h, w, current_masks, accumulators, seen
        )
        try:
            with torch.autocast(
                device_type=device.type if isinstance(device, torch.device) else "cuda",
                dtype=torch.bfloat16 if config.dtype == "bfloat16" else torch.float16,
                enabled=device.type == "cuda" or str(device).startswith("cuda"),
            ):
                outputs = model(
                    input_ids=batch_dev["input_ids"],
                    attention_mask=batch_dev["attention_mask"],
                    labels=batch_dev["labels"],
                    use_cache=False,
                )
                loss = outputs.loss

            if loss is None:
                raise RuntimeError("Model returned loss=None; labels may be missing")
            loss.backward()
        finally:
            for handle in handles:
                handle.remove()

        missing = expected - seen
        if missing:
            raise RuntimeError(
                "No gradient received via offload hooks for modules: "
                + ", ".join(sorted(missing))
            )
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("No calibration batches were processed")

    records: dict[str, BlockScoreRecord] = {}
    for target in targets:
        acc = accumulators[target.module_name]
        records[target.module_name] = _make_record(
            target=target,
            config=config,
            fisher=acc["score_sq"] / num_batches,
            abs_taylor=acc["score_abs"] / num_batches,
            signed_mean=acc["score_signed"] / num_batches,
            current_mask=current_masks[target.module_name].clone(),
        )

    model.zero_grad(set_to_none=True)
    freeze_all_parameters(model)
    return records


def collect_magnitude_block_scores(
    targets: list[MLPLinearTarget],
    config: GradientBlockPruningConfig,
    current_masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, BlockScoreRecord]:
    """Score blocks by ||W_b||_F^2 (stored in fisher field for allocator reuse)."""
    h, w = config.block_height, config.block_width
    if current_masks is None:
        from block_pruning.mlp_registry import initialize_all_one_masks

        current_masks = initialize_all_one_masks(targets, h, w)

    records: dict[str, BlockScoreRecord] = {}
    for target in targets:
        mag = reduce_weight_magnitude_to_blocks(
            target.module.weight.detach(), h, w
        ).double().cpu()
        mask = current_masks[target.module_name]
        mag = mag * mask.to(dtype=mag.dtype)
        zeros = torch.zeros_like(mag)
        records[target.module_name] = _make_record(
            target=target,
            config=config,
            fisher=mag,
            abs_taylor=zeros,
            signed_mean=zeros,
            current_mask=mask.clone(),
        )
    return records


def collect_random_block_scores(
    targets: list[MLPLinearTarget],
    config: GradientBlockPruningConfig,
    current_masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, BlockScoreRecord]:
    """Assign i.i.d. random scores (reproducible via config.seed)."""
    h, w = config.block_height, config.block_width
    if current_masks is None:
        from block_pruning.mlp_registry import initialize_all_one_masks

        current_masks = initialize_all_one_masks(targets, h, w)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(config.seed)

    records: dict[str, BlockScoreRecord] = {}
    for target in targets:
        n_out, n_in = block_grid_shape(tuple(target.module.weight.shape), h, w)
        scores = torch.rand(n_out, n_in, dtype=torch.float64, generator=rng)
        mask = current_masks[target.module_name]
        scores = torch.where(mask, scores, torch.full_like(scores, float("inf")))
        zeros = torch.zeros_like(scores)
        records[target.module_name] = _make_record(
            target=target,
            config=config,
            fisher=scores,
            abs_taylor=zeros,
            signed_mean=zeros,
            current_mask=mask.clone(),
        )
    return records
