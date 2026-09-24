#!/usr/bin/env python3
"""HiF4 S0 ScaleTuning 训练入口：kd_top_1000 + adaptive_top_3 hidden alignment。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from torch import nn

_SCALE_TUNING_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCALE_TUNING_DIR.parent
_DEFAULT_VAELLM_ROOT = Path("/home/shaoyuantian/program/VAELLM")

for path in (_SCALE_TUNING_DIR, _REPO_ROOT / "ChuanCi"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _ensure_vaellm_on_path(vaellm_root: Path) -> None:
    root = str(vaellm_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


from hif4_scale_linear import HiF4ScaleLinear  # noqa: E402
from wrap_model import (  # noqa: E402
    collect_hif4_scale_linears,
    freeze_non_s0_parameters,
    parse_csv_set,
    parse_layer_indices,
    wrap_model_for_scale_tuning,
)

logger = logging.getLogger("scale_tuning")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HiF4 S0 ScaleTuning QAD")
    parser.add_argument("--model_path", type=str, required=True, help="BF16 原始 HF 模型路径")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--vaellm_root",
        type=str,
        default=str(_DEFAULT_VAELLM_ROOT),
        help="VAELLM 仓库根目录（提供 CustomSFTTrainer / distill 数据）",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="替换为 HiF4ScaleLinear 的模块叶名，逗号分隔",
    )
    parser.add_argument(
        "--tune_modules",
        type=str,
        default="",
        help="可训 S0 的模块叶名；空表示与 target_modules 相同",
    )
    parser.add_argument("--target_layers", type=str, default="", help="替换的层号；空表示全部层")
    parser.add_argument("--tune_layers", type=str, default="", help="可训 S0 的层号；空表示与 target_layers 相同策略")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--deterministic", type=str, default="false")
    parser.add_argument(
        "--distill_dataset",
        type=str,
        default="edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009",
    )
    parser.add_argument("--distill_steps", type=int, default=2000)
    parser.add_argument("--distill_batch_size", type=int, default=8)
    parser.add_argument("--distill_lr", type=float, default=2e-5)
    parser.add_argument("--distill_weight_decay", type=float, default=0.001)
    parser.add_argument("--distill_log_every", type=int, default=10)
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--distill_loss_alpha", type=float, default=0.5)
    parser.add_argument("--distill_loss_type", type=str, default="kd_top_1000")
    parser.add_argument("--distill_hidden_loss_weight", type=float, default=0.1)
    parser.add_argument("--distill_pre_mlp_hidden_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--distill_hidden_alignment_layer_weighting",
        type=str,
        default="adaptive_top_3",
    )
    parser.add_argument("--distill_eakld_confidence_k", type=int, default=16)
    parser.add_argument("--distill_teacher_logits_cpu_staging", type=str, default="true")
    parser.add_argument("--distill_gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--distill_gradient_checkpointing", type=str, default="true")
    parser.add_argument(
        "--distill_gradient_checkpointing_kwargs",
        type=str,
        default='{"use_reentrant": false}',
    )
    parser.add_argument("--distill_optim", type=str, default="adamw_torch")
    parser.add_argument("--distill_max_grad_norm", type=float, default=1.3)
    parser.add_argument("--distill_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--distill_group_by_length", type=str, default="false")
    parser.add_argument("--distill_lr_scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument("--distill_model_max_length", type=int, default=1024)
    parser.add_argument("--distill_dataloader_num_workers", type=int, default=-1)
    parser.add_argument("--bf16", type=str, default="true")
    parser.add_argument("--fp16", type=str, default="false")
    parser.add_argument(
        "--export_reconstructed_model",
        type=str,
        default="false",
        help="训练结束后导出用最终 S0 重建权重的 HF 模型",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="torchrun 注入")
    return parser.parse_args(argv)


def _str_to_bool(raw: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid bool string: {raw!r}")


def _is_main_process() -> bool:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return rank == 0


def _setup_logging() -> None:
    level = logging.INFO if _is_main_process() else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
    return trainable, total


def _assert_only_s0_trainable(model: nn.Module) -> list[str]:
    trainable_names: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not name.endswith("s0_continuous"):
            raise RuntimeError(f"Unexpected trainable parameter: {name}")
        trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError("No trainable s0_continuous parameters found")
    return trainable_names


def _save_s0_checkpoint(model: nn.Module, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "s0_continuous": {},
        "s0_e6m2": {},
        "trainable": [],
        "wrapped": [],
    }
    for name, module in collect_hif4_scale_linears(model):
        payload["wrapped"].append(name)
        payload["s0_continuous"][name] = module.s0_continuous.detach().cpu()
        payload["s0_e6m2"][name] = module.s0_e6m2.cpu()
        if module.s0_continuous.requires_grad:
            payload["trainable"].append(name)
    path = output_dir / "s0_scales.pt"
    torch.save(payload, path)
    meta = {
        "num_wrapped": len(payload["wrapped"]),
        "num_trainable": len(payload["trainable"]),
        "wrapped": payload["wrapped"],
        "trainable": payload["trainable"],
    }
    (output_dir / "s0_scales_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def _unwrap_model(model: nn.Module) -> nn.Module:
    unwrapped = model
    for attr in ("module",):
        if hasattr(unwrapped, attr) and isinstance(getattr(unwrapped, attr), nn.Module):
            candidate = getattr(unwrapped, attr)
            # DDP / DataParallel
            if type(unwrapped).__name__ in {"DistributedDataParallel", "DataParallel"}:
                unwrapped = candidate
    return unwrapped


def _export_reconstructed_hf_model(model: nn.Module, tokenizer, output_dir: Path) -> None:
    """把 HiF4ScaleLinear 融成普通 Linear（student 重建权重），再 save_pretrained。"""
    export_model = _unwrap_model(model)
    replacements: list[tuple[str, HiF4ScaleLinear]] = collect_hif4_scale_linears(export_model)
    for name, module in replacements:
        parent_name, _, attr = name.rpartition(".")
        parent = export_model.get_submodule(parent_name) if parent_name else export_model
        with torch.no_grad():
            weight = module.reconstruct_weight().detach().to(dtype=module.teacher_weight.dtype)
            linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            linear = linear.to(dtype=weight.dtype)
            linear.weight.data.copy_(weight.cpu())
            if module.bias is not None:
                linear.bias.data.copy_(module.bias.detach().cpu())
        setattr(parent, attr, linear)
    export_dir = output_dir / "reconstructed_model"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_model.to("cpu")
    export_model.save_pretrained(export_dir, safe_serialization=True)
    tokenizer.save_pretrained(export_dir)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _setup_logging()

    vaellm_root = Path(args.vaellm_root)
    if not vaellm_root.is_dir():
        raise FileNotFoundError(f"vaellm_root does not exist: {vaellm_root}")
    _ensure_vaellm_on_path(vaellm_root)

    from e2e_common.lazy_datasets import (  # noqa: WPS433
        build_edgerazor_data_collator,
        default_dataloader_num_workers,
        dataset_length_or_none,
        is_iterable_training_dataset,
    )
    from train_utils.lora_data import prepare_distill_datasets  # noqa: WPS433
    from train_utils.lora_training import (  # noqa: WPS433
        CustomSFTTrainer,
        parse_distill_hidden_alignment_layer_weighting,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    from nvfp4_hif4_torch import HiF4Config

    output_dir = Path(args.output_dir)
    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _str_to_bool(args.deterministic):
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    target_modules = sorted(parse_csv_set(args.target_modules))
    tune_modules_raw = parse_csv_set(args.tune_modules)
    tune_modules = sorted(tune_modules_raw) if tune_modules_raw else None
    target_layers = parse_layer_indices(args.target_layers)
    tune_layers = parse_layer_indices(args.tune_layers)
    if tune_layers is None and args.tune_layers.strip() == "" and target_layers is not None:
        # 未显式传 tune_layers 时：与 target_layers 相同（wrap_model 内 tune_layers=None 表示不额外限制）
        tune_layers = None

    hidden_weighting = parse_distill_hidden_alignment_layer_weighting(
        args.distill_hidden_alignment_layer_weighting
    )

    logger.info("Loading tokenizer/model from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if _str_to_bool(args.bf16) else torch.float32,
        trust_remote_code=True,
    )
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    hif4_config = HiF4Config(group_size=64, group_dim=-1, scale_mode="hardware")
    wrapped_names = wrap_model_for_scale_tuning(
        model,
        target_modules=target_modules,
        tune_modules=tune_modules,
        target_layers=target_layers,
        tune_layers=tune_layers,
        config=hif4_config,
    )
    freeze_non_s0_parameters(model)
    trainable_names = _assert_only_s0_trainable(model)
    trainable_n, total_n = _count_trainable(model)
    logger.info(
        "Wrapped %d modules; trainable s0 tensors=%d params=%d / %d",
        len(wrapped_names),
        len(trainable_names),
        trainable_n,
        total_n,
    )
    if _is_main_process():
        (output_dir / "wrapped_modules.json").write_text(
            json.dumps({"wrapped": wrapped_names, "trainable": trainable_names}, indent=2),
            encoding="utf-8",
        )

    if hasattr(model, "config"):
        model.config.use_cache = False
    if _str_to_bool(args.distill_gradient_checkpointing) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    dataset_mix_spec, source_stats, train_ds, eval_ds, _ = prepare_distill_datasets(
        args.distill_dataset,
        seed=seed,
        tokenizer=tokenizer,
        max_seq_len=int(args.distill_model_max_length),
    )
    train_is_iterable = is_iterable_training_dataset(train_ds)
    train_len = dataset_length_or_none(train_ds)
    logger.info(
        "Dataset mix=%s iterable=%s len=%s sources=%d",
        dataset_mix_spec,
        train_is_iterable,
        "unknown" if train_len is None else train_len,
        len(source_stats),
    )
    if train_len == 0:
        raise RuntimeError("Distill dataset is empty")

    gc_kwargs = None
    raw_gc = str(args.distill_gradient_checkpointing_kwargs).strip()
    if raw_gc:
        gc_kwargs = json.loads(raw_gc)
        if not isinstance(gc_kwargs, dict):
            raise ValueError("--distill_gradient_checkpointing_kwargs must be a JSON object")

    num_workers = int(args.distill_dataloader_num_workers)
    if num_workers < 0:
        num_workers = int(default_dataloader_num_workers())

    training_kwargs = dict(
        output_dir=str(output_dir / "trainer_state"),
        per_device_train_batch_size=int(args.distill_batch_size),
        gradient_accumulation_steps=int(args.distill_gradient_accumulation_steps),
        gradient_checkpointing=_str_to_bool(args.distill_gradient_checkpointing),
        gradient_checkpointing_kwargs=gc_kwargs,
        optim=str(args.distill_optim),
        logging_strategy="steps",
        logging_steps=max(1, int(args.distill_log_every)),
        logging_first_step=True,
        learning_rate=float(args.distill_lr),
        weight_decay=float(args.distill_weight_decay),
        fp16=_str_to_bool(args.fp16),
        bf16=_str_to_bool(args.bf16),
        max_grad_norm=float(args.distill_max_grad_norm),
        max_steps=int(args.distill_steps),
        warmup_ratio=float(args.distill_warmup_ratio),
        group_by_length=_str_to_bool(args.distill_group_by_length) and not train_is_iterable,
        lr_scheduler_type=str(args.distill_lr_scheduler_type),
        report_to=[],
        disable_tqdm=not _is_main_process(),
        save_strategy="no",
        seed=seed,
        data_seed=seed,
        full_determinism=_str_to_bool(args.deterministic),
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
    )
    sft_args = TrainingArguments(**training_kwargs)

    max_seq_len = int(args.distill_model_max_length)
    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
        processing_class=tokenizer,
        data_collator=build_edgerazor_data_collator(tokenizer, max_seq_len=max_seq_len),
        loss_type=str(args.distill_loss_type).strip().lower(),
        temperature=float(args.distill_temperature),
        loss_alpha=float(args.distill_loss_alpha),
        hidden_loss_weight=float(args.distill_hidden_loss_weight),
        pre_mlp_hidden_loss_weight=float(args.distill_pre_mlp_hidden_loss_weight),
        hidden_alignment_layer_weighting=hidden_weighting,
        eakld_confidence_k=int(args.distill_eakld_confidence_k),
        teacher_logits_cpu_staging=_str_to_bool(args.distill_teacher_logits_cpu_staging),
    )

    logger.info(
        "Start ScaleTuning: loss=%s hidden_w=%.4f layer_weighting=%s steps=%d",
        args.distill_loss_type,
        float(args.distill_hidden_loss_weight),
        hidden_weighting,
        int(args.distill_steps),
    )
    trainer.train()

    if _is_main_process():
        raw_model = _unwrap_model(trainer.model)
        if hasattr(trainer, "accelerator") and trainer.accelerator is not None:
            raw_model = trainer.accelerator.unwrap_model(trainer.model)
        s0_path = _save_s0_checkpoint(raw_model, output_dir)
        logger.info("Saved S0 scales to %s", s0_path)
        if _str_to_bool(args.export_reconstructed_model):
            _export_reconstructed_hf_model(raw_model, tokenizer, output_dir)
            logger.info("Exported reconstructed HF model to %s", output_dir / "reconstructed_model")


if __name__ == "__main__":
    main()
