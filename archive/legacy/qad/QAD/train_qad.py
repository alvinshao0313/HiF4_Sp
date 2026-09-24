#!/usr/bin/env python3
"""HiF4 × EdgeRazor QAD 训练入口：S1K-1.1 + EAKLD/LAFD，仅依赖本仓库。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from torch import nn

_QAD_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _QAD_DIR.parent
_SCALE_TUNING_DIR = _REPO_ROOT / "ScaleTuning"
_CHUANCI_DIR = _REPO_ROOT / "ChuanCi"

for path in (_QAD_DIR, _SCALE_TUNING_DIR, _CHUANCI_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data import DEFAULT_DATASET_NAME, ChatSFTCollator, build_reasoning_train_dataset  # noqa: E402
from hif4_fixed_s0 import apply_e6m2_ste  # noqa: E402
from hif4_frozen_b import HiF4FrozenBLinear, collect_hif4_frozen_b_linears  # noqa: E402
from parallel import build_fsdp_training_kwargs, load_model_for_qad  # noqa: E402
from trainer import (  # noqa: E402
    HiF4QADTrainer,
    assert_only_s0_trainable,
    clone_lm_head_for_loss,
)
from wrap_frozen_b import (  # noqa: E402
    freeze_non_s0_parameters,
    parse_csv_set,
    parse_layer_indices,
    wrap_model_for_qad_frozen_b,
)

logger = logging.getLogger("qad")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HiF4 × EdgeRazor QAD")
    parser.add_argument("--model_path", type=str, required=True, help="BF16 原始 HF 模型路径（教师）")
    parser.add_argument(
        "--pseudo_quant_model_path",
        type=str,
        default="",
        help="可选：GPTQ/RTN 等伪量化 HF 模型路径；用其 Linear 权重初始化 frozen_b+s0。"
        "为空则对 BF16 teacher_weight 做一次 HiF4 初始化",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="HuggingFace dataset name（默认本地已缓存的 s1K-1.1_tokenized）",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="本地数据集路径；非空时优先于 dataset_name",
    )
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--trace_source",
        type=str,
        default="deepseek",
        help="reasoning 轨迹来源；当前仅支持 deepseek",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default=(
            "q_proj,k_proj,v_proj,o_proj,"
            "in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,"
            "gate_proj,up_proj,down_proj"
        ),
    )
    parser.add_argument("--tune_modules", type=str, default="")
    parser.add_argument("--target_layers", type=str, default="")
    parser.add_argument("--tune_layers", type=str, default="")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--deterministic", type=str, default="false")

    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.3)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--model_max_length", type=int, default=32768)
    parser.add_argument("--allow_truncate", type=str, default="false")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", type=str, default="true")
    parser.add_argument(
        "--gradient_checkpointing_kwargs",
        type=str,
        default='{"use_reentrant": false}',
    )
    parser.add_argument("--bf16", type=str, default="true")
    parser.add_argument("--fp16", type=str, default="false")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="sdpa / flash_attention_2 / eager",
    )

    # EdgeRazor QAD loss weights
    parser.add_argument("--task_alpha", type=float, default=0.05)
    parser.add_argument("--eakld_alpha", type=float, default=2.0)
    parser.add_argument("--lafd_alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--confidence_k", type=int, default=16)
    parser.add_argument("--lafd_topk", type=int, default=3)
    parser.add_argument("--logit_chunk_size", type=int, default=512)
    parser.add_argument(
        "--kl_mode",
        type=str,
        default="eakld",
        choices=["eakld", "eakld_topk", "kl_topk"],
        help="KL 损失：eakld=全词表（默认）；eakld_topk=top-k 版 EAKLD；kl_topk=仅 forward top-k KL",
    )
    parser.add_argument(
        "--kl_topk",
        type=int,
        default=128,
        help="top-k KL 的 k（kl_mode!=eakld 时生效）",
    )
    parser.add_argument(
        "--kl_post_attn",
        type=str,
        default="false",
        help="true=全词表 softmax 后 gather 部分 KL；false=k 维重归一化（默认）",
    )

    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="layer",
        choices=["fsdp", "layer", "ddp", "none"],
        help="layer=按层 device_map（默认，拟合实测通过）；fsdp=FULL_SHARD；ddp=每卡整模",
    )
    parser.add_argument(
        "--prefer_longest_sample",
        type=str,
        default="false",
        help="把最长样本排到前面（fit 验收用）",
    )
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


def _unwrap_model(model: nn.Module) -> nn.Module:
    unwrapped = model
    if type(unwrapped).__name__ in {"DistributedDataParallel", "DataParallel"}:
        unwrapped = unwrapped.module
    return unwrapped


def _save_s0_checkpoint(model: nn.Module, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "s0_continuous": {},
        "s0_e6m2": {},
        "trainable": [],
        "wrapped": [],
        "scheme": "frozen_b_ste_s0",
    }
    for name, module in collect_hif4_frozen_b_linears(model):
        payload["wrapped"].append(name)
        payload["s0_continuous"][name] = module.s0_continuous.detach().cpu()
        with torch.no_grad():
            s0_hw = apply_e6m2_ste(
                module.s0_continuous.detach(), scale_mode=module.config.scale_mode
            )
        payload["s0_e6m2"][name] = s0_hw.cpu()
        if module.s0_continuous.requires_grad:
            payload["trainable"].append(name)
    path = output_dir / "s0_scales.pt"
    torch.save(payload, path)
    meta = {
        "num_wrapped": len(payload["wrapped"]),
        "num_trainable": len(payload["trainable"]),
        "wrapped": payload["wrapped"],
        "trainable": payload["trainable"],
        "scheme": "frozen_b_ste_s0",
    }
    (output_dir / "s0_scales_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def _export_reconstructed_hf_model(model: nn.Module, tokenizer, output_dir: Path) -> None:
    export_model = _unwrap_model(model)
    for name, module in collect_hif4_frozen_b_linears(export_model):
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

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    from nvfp4_hif4_torch import HiF4Config

    parallel_mode = str(args.parallel_mode).strip().lower()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if parallel_mode == "layer" and world_size > 1:
        raise RuntimeError(
            "parallel_mode=layer 是单进程按层切分，请勿用多进程 torchrun "
            f"(WORLD_SIZE={world_size})"
        )
    if parallel_mode == "fsdp" and world_size < 2:
        logger.warning(
            "parallel_mode=fsdp 但 WORLD_SIZE=%d；单进程 FSDP 无参数分片收益，建议 torchrun 多卡",
            world_size,
        )

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

    logger.info("Loading tokenizer/model from %s parallel_mode=%s", args.model_path, parallel_mode)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model_for_qad(
        args.model_path,
        parallel_mode=parallel_mode,
        bf16=_str_to_bool(args.bf16),
        attn_implementation=str(args.attn_implementation),
    )
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    hif4_config = HiF4Config(group_size=64, group_dim=-1, scale_mode="hardware")
    pseudo_path = str(args.pseudo_quant_model_path).strip()
    init_weight_model = None
    if pseudo_path:
        logger.info(
            "Loading pseudo-quant init model on CPU for B/S0 init: %s",
            pseudo_path,
        )
        init_weight_model = AutoModelForCausalLM.from_pretrained(
            pseudo_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if _str_to_bool(args.bf16) else torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        init_weight_model.eval()
        for p in init_weight_model.parameters():
            p.requires_grad_(False)

    logger.info(
        "Wrapping with HiF4FrozenBLinear (freeze B, train S0 only; init_from=%s)...",
        pseudo_path if pseudo_path else "teacher_bf16",
    )
    wrapped_names = wrap_model_for_qad_frozen_b(
        model,
        target_modules=target_modules,
        tune_modules=tune_modules,
        target_layers=target_layers,
        tune_layers=tune_layers,
        config=hif4_config,
        init_weight_model=init_weight_model,
    )
    if init_weight_model is not None:
        del init_weight_model
        import gc

        gc.collect()
    freeze_non_s0_parameters(model)
    trainable_names = assert_only_s0_trainable(model)
    # 校验：frozen_b / teacher_weight 不可训；s0 必须 float32
    for name, module in collect_hif4_frozen_b_linears(model):
        if module.teacher_weight.requires_grad:
            raise RuntimeError(f"teacher_weight must be frozen: {name}")
        if getattr(module, "frozen_b").requires_grad:
            raise RuntimeError(f"frozen_b must be frozen: {name}")
        if module.s0_continuous.dtype != torch.float32:
            raise RuntimeError(
                f"s0_continuous must be float32, got {module.s0_continuous.dtype} ({name})"
            )
    trainable_n, total_n = _count_trainable(model)
    logger.info(
        "Wrapped %d FrozenB modules; trainable s0 tensors=%d params=%d / %d",
        len(wrapped_names),
        len(trainable_names),
        trainable_n,
        total_n,
    )
    if _is_main_process():
        (output_dir / "wrapped_modules.json").write_text(
            json.dumps(
                {
                    "wrapped": wrapped_names,
                    "trainable": trainable_names,
                    "parallel_mode": parallel_mode,
                    "model_path": str(args.model_path),
                    "pseudo_quant_model_path": pseudo_path or None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if hasattr(model, "config"):
        model.config.use_cache = False
    if _str_to_bool(args.gradient_checkpointing) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # layer：用模型自带 lm_head；fsdp：克隆完整 lm_head 供分块 logits
    loss_lm_head = None
    if parallel_mode == "fsdp":
        loss_lm_head = clone_lm_head_for_loss(model)
        logger.info(
            "Cloned loss lm_head: in=%d out=%d (~%.2f GiB bf16)",
            loss_lm_head.in_features,
            loss_lm_head.out_features,
            loss_lm_head.weight.numel() * 2 / (1024**3),
        )
    else:
        logger.info("parallel_mode=%s: use in-model lm_head for chunked logits", parallel_mode)

    dataset_path = str(args.dataset_path).strip() or None
    train_ds = build_reasoning_train_dataset(
        dataset_name=str(args.dataset_name),
        dataset_path=dataset_path,
        split=str(args.dataset_split),
        trace_source=str(args.trace_source),
    )
    if _str_to_bool(args.prefer_longest_sample):
        train_ds = _reorder_longest_first(train_ds, tokenizer, max_length=int(args.model_max_length))

    collator = ChatSFTCollator(
        tokenizer=tokenizer,
        max_length=int(args.model_max_length),
        allow_truncate=_str_to_bool(args.allow_truncate),
    )
    sample_batch = collator([train_ds[0]])
    logger.info(
        "Sample tokenized length=%d max_length=%d",
        int(sample_batch["attention_mask"].sum().item()),
        int(args.model_max_length),
    )

    gc_kwargs = None
    raw_gc = str(args.gradient_checkpointing_kwargs).strip()
    if raw_gc:
        gc_kwargs = json.loads(raw_gc)
        if not isinstance(gc_kwargs, dict):
            raise ValueError("--gradient_checkpointing_kwargs must be a JSON object")

    ta_kwargs: dict = {
        "output_dir": str(output_dir / "trainer_state"),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "gradient_checkpointing": _str_to_bool(args.gradient_checkpointing),
        "gradient_checkpointing_kwargs": gc_kwargs,
        "optim": str(args.optim),
        "logging_strategy": "steps",
        "logging_steps": max(1, int(args.logging_steps)),
        "logging_first_step": True,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "fp16": _str_to_bool(args.fp16),
        "bf16": _str_to_bool(args.bf16),
        "max_grad_norm": float(args.max_grad_norm),
        "max_steps": int(args.max_steps),
        "warmup_ratio": float(args.warmup_ratio),
        "lr_scheduler_type": str(args.lr_scheduler_type),
        "report_to": [],
        "disable_tqdm": not _is_main_process(),
        "save_strategy": "no",
        "seed": seed,
        "data_seed": seed,
        "full_determinism": _str_to_bool(args.deterministic),
        "dataloader_num_workers": int(args.dataloader_num_workers),
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": True,
    }
    if parallel_mode == "fsdp":
        ta_kwargs.update(build_fsdp_training_kwargs())
        # 激活重计算走 fsdp_config.activation_checkpointing，避免双重 AllGather
        ta_kwargs["gradient_checkpointing"] = False
    elif parallel_mode == "layer":
        # 单进程多卡；禁止 Trainer 再做 DDP
        ta_kwargs["use_cpu"] = False

    training_args = TrainingArguments(**ta_kwargs)

    trainer = HiF4QADTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
        task_alpha=float(args.task_alpha),
        eakld_alpha=float(args.eakld_alpha),
        lafd_alpha=float(args.lafd_alpha),
        temperature=float(args.temperature),
        confidence_k=int(args.confidence_k),
        lafd_topk=int(args.lafd_topk),
        logit_chunk_size=int(args.logit_chunk_size),
        kl_mode=str(args.kl_mode),
        kl_topk=int(args.kl_topk),
        kl_post_attn=_str_to_bool(args.kl_post_attn),
        loss_lm_head=loss_lm_head,
    )

    logger.info(
        "Start QAD: dataset=%s parallel=%s task_a=%.3f eakld_a=%.3f lafd_a=%.3f steps=%d max_len=%d"
        " kl_mode=%s kl_topk=%d kl_post_attn=%s",
        dataset_path or args.dataset_name,
        parallel_mode,
        float(args.task_alpha),
        float(args.eakld_alpha),
        float(args.lafd_alpha),
        int(args.max_steps),
        int(args.model_max_length),
        str(args.kl_mode),
        int(args.kl_topk),
        str(args.kl_post_attn),
    )
    trainer.train()

    if _is_main_process():
        raw_model = _unwrap_model(trainer.model)
        if hasattr(trainer, "accelerator") and trainer.accelerator is not None:
            raw_model = trainer.accelerator.unwrap_model(trainer.model)
        s0_path = _save_s0_with_full_params(trainer.model, raw_model, output_dir)
        logger.info("Saved S0 scales to %s", s0_path)
        if _str_to_bool(args.export_reconstructed_model):
            _export_reconstructed_hf_model(raw_model, tokenizer, output_dir)
            logger.info("Exported reconstructed HF model to %s", output_dir / "reconstructed_model")


def _save_s0_with_full_params(trainer_model: nn.Module, raw_model: nn.Module, output_dir: Path) -> Path:
    """FSDP 下 gather 完整参数再保存 s0。"""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        return _save_s0_checkpoint(raw_model, output_dir)

    root = trainer_model
    if isinstance(root, FSDP):
        with FSDP.summon_full_params(root, writeback=False):
            return _save_s0_checkpoint(raw_model, output_dir)
    return _save_s0_checkpoint(raw_model, output_dir)


def _reorder_longest_first(dataset, tokenizer, *, max_length: int):
    """按 chat 模板长度降序重排（仅用于短跑验收）。"""
    from torch.utils.data import Subset

    lengths: list[tuple[int, int]] = []
    for i in range(len(dataset)):
        messages = dataset[i]["messages"]
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = list(ids["input_ids"])
        n = len(ids)
        if n > int(max_length):
            raise ValueError(f"sample {i} length {n} exceeds max_length={max_length}")
        lengths.append((n, i))
    lengths.sort(key=lambda x: x[0], reverse=True)
    order = [i for _, i in lengths]
    logger.info(
        "prefer_longest_sample: max_len=%d idx=%d; min_len=%d",
        lengths[0][0],
        lengths[0][1],
        lengths[-1][0],
    )
    return Subset(dataset, order)


if __name__ == "__main__":
    main()
