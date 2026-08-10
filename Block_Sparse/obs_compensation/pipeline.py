from __future__ import annotations

from pathlib import Path

import torch

from block_pruning.config import GradientBlockPruningConfig
from block_pruning.mlp_registry import collect_mlp_linears
from block_pruning.model_loader import load_model_and_tokenizer
from obs_compensation.artifacts import (
    load_source_artifacts,
    validate_source_artifacts_against_targets,
)
from obs_compensation.calibration import build_calibration_samples
from obs_compensation.config import OBSCompensationConfig
from obs_compensation.layerwise import run_layerwise_mlp_obs
from obs_compensation.model_adapter import capture_first_decoder_layer_inputs
from obs_compensation.permutation import (
    apply_saved_mlp_permutations,
    group_mlp_projection_triplets,
)
from obs_compensation.serialization import (
    save_obs_package_atomically,
    validate_atomic_output_paths,
    verify_fixed_masks_and_weights,
)
from obs_compensation.solver import resolve_obs_order_policy


def _build_loader_config(
    config: OBSCompensationConfig,
    block_size: str,
    target_block_sparsity: float,
) -> GradientBlockPruningConfig:
    loader_config = GradientBlockPruningConfig(
        model_path=config.model_path,
        calibration_dataset="s1k",
        output_dir=str(config.output_dir),
        block_size=block_size,
        target_block_sparsity=target_block_sparsity,
        calibration_samples=1,
        sequence_length=2,
        score_type="magnitude",
        pruning_rounds=1,
        mlp_permutation="none",
        residual_permutation="none",
        dtype=config.dtype,
        device=config.device,
        gradient_checkpointing=False,
        trust_remote_code=config.trust_remote_code,
    )
    loader_config.validate()
    return loader_config


def run_obs_compensation(config: OBSCompensationConfig) -> Path:
    config.validate_paths(require_source_exists=True)
    validate_atomic_output_paths(config.output_dir)

    artifacts = load_source_artifacts(config.source_artifacts_dir)
    if config.model_path != artifacts.metadata.model_path:
        raise ValueError(
            f"config.model_path {config.model_path!r} != source "
            f"model_path {artifacts.metadata.model_path!r}"
        )

    order_policy = resolve_obs_order_policy(
        requested_policy=config.obs_order_policy,
        mlp_permutation=artifacts.metadata.mlp_permutation,
    )
    print(
        f"[obs] requested_policy={order_policy.requested_policy} "
        f"resolved_obs_order_policy={order_policy.resolved_policy} "
        f"gate_up={order_policy.gate_up_direction} "
        f"down={order_policy.down_direction}",
        flush=True,
    )

    loader_config = _build_loader_config(
        config,
        block_size=artifacts.metadata.block_size,
        target_block_sparsity=artifacts.metadata.target_block_sparsity,
    )
    model, tokenizer = load_model_and_tokenizer(loader_config)
    model.eval()

    targets = collect_mlp_linears(
        model,
        block_height=artifacts.metadata.block_height,
        block_width=artifacts.metadata.block_width,
    )
    validate_source_artifacts_against_targets(artifacts, targets)
    triplets = group_mlp_projection_triplets(targets)

    if artifacts.metadata.mlp_permutation == "wanda_shared":
        if artifacts.permutation_payload is None:
            raise RuntimeError("wanda_shared source missing permutation payload")
        apply_saved_mlp_permutations(triplets, artifacts.permutation_payload)
    elif artifacts.permutation_payload is not None:
        raise RuntimeError(
            "mlp_permutation=none but permutation payload is present"
        )

    samples = build_calibration_samples(tokenizer, config)
    captured = capture_first_decoder_layer_inputs(model, samples)
    layerwise_result = run_layerwise_mlp_obs(
        model=model,
        captured=captured,
        triplets=triplets,
        masks=artifacts.masks,
        config=config,
        order_policy=order_policy,
        block_height=artifacts.metadata.block_height,
        block_width=artifacts.metadata.block_width,
    )

    verify_fixed_masks_and_weights(
        masks=artifacts.masks,
        targets=targets,
        block_height=artifacts.metadata.block_height,
        block_width=artifacts.metadata.block_width,
    )

    with torch.no_grad():
        input_device = next(model.parameters()).device
        if hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                input_device = emb.weight.device
        out = model(
            input_ids=samples[0].input_ids.to(input_device),
            attention_mask=samples[0].attention_mask.to(input_device),
            use_cache=False,
        )
        logits = out.logits if hasattr(out, "logits") else out[0]
        if not torch.isfinite(logits).all():
            raise RuntimeError("final full-model forward produced non-finite logits")

    saved_output = save_obs_package_atomically(
        model=model,
        tokenizer=tokenizer,
        config=config,
        artifacts=artifacts,
        order_policy=order_policy,
        layerwise_result=layerwise_result,
    )
    print(
        f"[obs] saved compensated model to {saved_output} "
        f"actual_block_sparsity={artifacts.metadata.actual_block_sparsity}",
        flush=True,
    )
    return saved_output
