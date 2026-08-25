"""Train orchestration: parse config, reconstruct layers, save compact artifact."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import (
    assert_resume_artifacts,
    load_layer_diag_file,
    save_conversion_artifact,
    save_layer_artifacts,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    TARGET_LINEAR_COUNT,
    parse_train_args,
    resolved_config_lines,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.fold import (
    fold_fusable_layer_inplace,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.semantic_hif4 import (
    iter_switchable_linears,
    load_layer_diag_snapshot,
    upgrade_semantic_model_inplace,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    build_or_load_calibration,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.teacher_traces import (
    require_qwen3_chat_template,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.layer_runtime import (
    ProgressiveHiddenCache,
    build_initial_hidden_cache,
    propagate_native_layer,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.trainer import (
    finalize_layer_runtime,
    propagate_accepted_layer,
    run_fold_gate,
    train_layer_joint,
    train_layer_linear_independent,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_trainer_resume import (
    train_qwen3_moe_lazy,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import ensure_dir, write_json
from Native_NVFP4_HiF4_Linear_Puncture.src.semantic_model import load_native_nvfp4_semantic_model


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("e2e diag reconstruction training requires CUDA")
    return torch.device("cuda")


def _print_resolved(cfg, extra: dict) -> None:
    for line in resolved_config_lines(cfg, extra):
        print(line, flush=True)


def _replay_prefix(cfg, model, samples, collator, device, x_cache: ProgressiveHiddenCache) -> ProgressiveHiddenCache:
    for i in range(cfg.start_layer):
        layer = model.model.layers[i]
        if cfg.calib_input_mode == "teacher":
            nxt = propagate_native_layer(
                model=model,
                layer=layer,
                samples=samples,
                collator=collator,
                x_cache=x_cache,
                device=device,
                batch_size=cfg.diag_batch_size,
            )
            z = load_layer_diag_file(cfg.output_dir, i)
            load_layer_diag_snapshot(layer, z)
            if cfg.diag_mode == "fusable":
                fold_fusable_layer_inplace(layer, layer.diag_state, cfg.use_r64)
            else:
                finalize_layer_runtime(cfg, layer)
            x_cache.clear()
            x_cache = nxt
            continue
        z = load_layer_diag_file(cfg.output_dir, i)
        load_layer_diag_snapshot(layer, z)
        if cfg.diag_mode == "fusable":
            fold_fusable_layer_inplace(layer, layer.diag_state, cfg.use_r64)
            finalize = lambda _layer: None
        else:
            finalize = lambda layer: finalize_layer_runtime(cfg, layer)
        x_cache = propagate_accepted_layer(
            model=model,
            layer=layer,
            samples=samples,
            collator=collator,
            x_cache=x_cache,
            device=device,
            batch_size=cfg.diag_batch_size,
            finalize=finalize,
        )
    return x_cache


def main(argv: list[str] | None = None) -> None:
    cfg = parse_train_args(argv)
    device = _require_cuda()
    if cfg.model_type == "qwen3_moe":
        train_qwen3_moe_lazy(cfg, device)
        return
    output_dir = ensure_dir(cfg.output_dir)
    assert_resume_artifacts(cfg)

    snapshot = resolve_local_snapshot(cfg.model_path)
    extra = {
        "source_snapshot": str(snapshot),
        "coverage": f"36/{TARGET_LINEAR_COUNT}",
    }
    _print_resolved(cfg, extra)
    if cfg.start_layer == 0:
        write_json(output_dir / "config.json", {**cfg.to_dict(), **extra})

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no pad_token_id or eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.calib_source == "s1k_teacher_cot":
        require_qwen3_chat_template(tokenizer)

    model, _index = load_native_nvfp4_semantic_model(snapshot, device=device)
    n_wrap = upgrade_semantic_model_inplace(model, cfg)
    if n_wrap != TARGET_LINEAR_COUNT:
        raise RuntimeError(f"coverage {n_wrap} != {TARGET_LINEAR_COUNT}")
    if len(iter_switchable_linears(model)) != TARGET_LINEAR_COUNT:
        raise RuntimeError("switchable linear coverage mismatch")

    train_samples, val_samples = build_or_load_calibration(
        cfg, tokenizer, model, Path(cfg.calib_cache_dir) if cfg.calib_cache_dir else output_dir
    )
    collator = DynamicCalibrationCollator(pad_token_id=int(tokenizer.pad_token_id))
    all_samples = list(train_samples) + list(val_samples)
    x_cache = build_initial_hidden_cache(
        model, all_samples, collator, device, cfg.diag_batch_size
    )
    if cfg.start_layer > 0:
        x_cache = _replay_prefix(cfg, model, all_samples, collator, device, x_cache)

    layer_records: dict[int, dict] = {}
    summaries = []
    for layer_idx in range(cfg.start_layer, cfg.end_layer + 1):
        print(f"=== layer {layer_idx} ===", flush=True)
        layer = model.model.layers[layer_idx]
        native_next = None
        if cfg.calib_input_mode == "teacher":
            native_next = propagate_native_layer(
                model=model,
                layer=layer,
                samples=all_samples,
                collator=collator,
                x_cache=x_cache,
                device=device,
                batch_size=cfg.diag_batch_size,
            )
        if cfg.diag_train_scope == "linear_independent":
            result = train_layer_linear_independent(
                model=model,
                layer_idx=layer_idx,
                cfg=cfg,
                train_samples=train_samples,
                val_samples=val_samples,
                collator=collator,
                x_cache=x_cache,
                device=device,
            )
        else:
            result = train_layer_joint(
                model=model,
                layer_idx=layer_idx,
                cfg=cfg,
                train_samples=train_samples,
                val_samples=val_samples,
                collator=collator,
                x_cache=x_cache,
                device=device,
            )
        if cfg.diag_mode == "fusable":
            gate = run_fold_gate(
                model=model,
                layer=layer,
                samples=val_samples,
                collator=collator,
                x_cache=x_cache,
                device=device,
                batch_size=cfg.diag_batch_size,
            )
            result.metrics["fold_gate_rel_l2"] = gate
            finalize = lambda _layer: None
        else:
            finalize = lambda lyr: finalize_layer_runtime(cfg, lyr)
        save_layer_artifacts(
            output_dir,
            layer_idx,
            z=result.snapshot,
            metrics=result.metrics,
            train_log=result.train_log,
        )
        layer_records[layer_idx] = {
            "accepted": result.accepted,
            "rollback": result.rollback,
            "best_epoch": result.best_epoch,
            "z": result.snapshot,
        }
        summaries.append(result.metrics)
        if cfg.calib_input_mode == "teacher":
            finalize(layer)
            x_cache.clear()
            x_cache = native_next
        else:
            x_cache = propagate_accepted_layer(
                model=model,
                layer=layer,
                samples=all_samples,
                collator=collator,
                x_cache=x_cache,
                device=device,
                batch_size=cfg.diag_batch_size,
                finalize=finalize,
            )
        print(json.dumps(result.metrics, ensure_ascii=False), flush=True)

    save_conversion_artifact(cfg=cfg, layer_records=layer_records, out_dir=output_dir)
    write_json(
        Path(output_dir) / "summary.json",
        {
            "accepted": sum(1 for m in summaries if m["accepted"]),
            "rollback": sum(1 for m in summaries if m["rollback"]),
            "layers": summaries,
        },
    )


if __name__ == "__main__":
    main()
