#!/usr/bin/env python3
"""SFT recovery for block-pruned MLP via peft Masked LoRA (S1K chat).

Loads a pruned HF checkpoint + pruning_artifacts/block_masks.pt, attaches peft
LoRA only on gate/up/down with the same block mask, trains with S1K chat data,
then merge_and_unload + verify before exporting a dense HF checkpoint.

Distillation mode (--teacher_model_dir): loads the unpruned model as a frozen
teacher and replaces the plain CE loss with the EdgeRazor-style QAD objective
(CE + EAKLD/kl-topk + LAFD, same conventions as QAD: pre-final-norm hidden,
chunked lm_head). Known boundaries: LAFD is always computed when distilling
(even with --lafd_alpha 0); fsdp is not adapted; teacher and student weights
are both resident in memory, OOM raises.

All implementation lives under Block_Sparse (no QAD imports; the QAD loss and
hook modules are verbatim copies under block_pruning/).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, Trainer, TrainingArguments

BLOCK_SPARSE_ROOT = Path(__file__).resolve().parents[1]
if str(BLOCK_SPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(BLOCK_SPARSE_ROOT))

from block_pruning.distill_losses import (  # noqa: E402
    compute_edgerazor_qad_loss_chunked,
)
from block_pruning.hidden_hooks import (  # noqa: E402
    SelectedHiddenCapture,
    StreamingTeacherSelector,
)
from block_pruning.model_loader import load_causal_lm_for_training  # noqa: E402
from block_pruning.peft_masked_lora import (  # noqa: E402
    assert_only_lora_trainable,
    merge_and_verify,
    wrap_pruned_mlp_with_peft_lora,
)
from block_pruning.serialization import load_masks  # noqa: E402
from block_pruning.sft_data import (  # noqa: E402
    DEFAULT_DATASET_NAME,
    ChatSFTCollator,
    build_reasoning_train_dataset,
)

logger = logging.getLogger("mlp_lora_sft")


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pruned_model_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--artifacts_dir",
        type=str,
        default="",
        help="Defaults to <pruned_model_dir>/pruning_artifacts",
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    p.add_argument("--dataset_path", type=str, default="")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--trace_source", type=str, default="deepseek")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--optim", type=str, default="adamw_torch")
    p.add_argument("--model_max_length", type=int, default=32768)
    p.add_argument("--allow_truncate", type=str, default="false")
    p.add_argument("--logit_chunk_size", type=int, default=512)
    p.add_argument(
        "--teacher_model_dir",
        type=str,
        default="",
        help="Unpruned HF dir as distillation teacher; empty = plain CE SFT",
    )
    p.add_argument("--task_alpha", type=float, default=0.05)
    p.add_argument("--eakld_alpha", type=float, default=2.0)
    p.add_argument("--lafd_alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--confidence_k", type=int, default=16)
    p.add_argument("--lafd_topk", type=int, default=3)
    p.add_argument(
        "--kl_mode",
        type=str,
        default="eakld",
        choices=["eakld", "eakld_topk", "kl_topk"],
    )
    p.add_argument("--kl_topk", type=int, default=0)
    p.add_argument("--kl_post_attn", type=str, default="false")
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--gradient_checkpointing", type=str, default="true")
    p.add_argument(
        "--gradient_checkpointing_kwargs",
        type=str,
        default='{"use_reentrant": false}',
    )
    p.add_argument("--bf16", type=str, default="true")
    p.add_argument("--fp16", type=str, default="false")
    p.add_argument("--attn_implementation", type=str, default="sdpa")
    p.add_argument(
        "--parallel_mode",
        type=str,
        default="layer",
        choices=["fsdp", "layer", "ddp", "none"],
    )
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args(argv)


def _load_pruning_meta(artifacts_dir: Path) -> tuple[int, int, dict[str, Any]]:
    summary_path = artifacts_dir / "pruning_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing pruning summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    block_h = int(summary["block_height"])
    block_w = int(summary["block_width"])
    if block_h <= 0 or block_w <= 0:
        raise ValueError(f"Invalid block size in summary: {block_h}x{block_w}")
    return block_h, block_w, summary


def _resolve_causal_lm(model: nn.Module) -> nn.Module:
    """PeftModel -> LoraModel.model (CausalLM) or identity."""
    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
        return base
    return model


def _resolve_backbone_and_lm_head(causal_lm: nn.Module) -> tuple[nn.Module, nn.Module]:
    if not hasattr(causal_lm, "model") or not hasattr(causal_lm, "lm_head"):
        raise TypeError(
            f"Expected CausalLM with .model and .lm_head, got {type(causal_lm).__name__}"
        )
    return causal_lm.model, causal_lm.lm_head


def compute_chunked_causal_lm_loss(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Backbone forward + chunked lm_head CE (sum/count), equivalent to full-seq mean CE."""
    causal_lm = _resolve_causal_lm(model)
    backbone, lm_head = _resolve_backbone_and_lm_head(causal_lm)

    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    if hidden.ndim != 3:
        raise ValueError(f"Expected last_hidden_state rank 3, got {tuple(hidden.shape)}")

    shift_hidden = hidden[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    seq_len = int(shift_hidden.shape[1])
    vocab = int(lm_head.out_features)
    chunk = max(1, int(chunk_size))

    total_loss = torch.zeros((), device=shift_hidden.device, dtype=torch.float32)
    total_tokens = torch.zeros((), device=shift_hidden.device, dtype=torch.float32)

    for start in range(0, seq_len, chunk):
        end = min(seq_len, start + chunk)
        logits = lm_head(shift_hidden[:, start:end, :])
        chunk_labels = shift_labels[:, start:end]
        loss_sum = F.cross_entropy(
            logits.reshape(-1, vocab).float(),
            chunk_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_valid = (chunk_labels != -100).sum().to(dtype=torch.float32)
        total_loss = total_loss + loss_sum.to(dtype=torch.float32)
        total_tokens = total_tokens + n_valid
        del logits

    if int(total_tokens.item()) == 0:
        raise RuntimeError("No valid label tokens in batch (all -100)")
    return total_loss / total_tokens


class ChunkedCESFTTrainer(Trainer):
    def __init__(self, *args, logit_chunk_size: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit_chunk_size = int(logit_chunk_size)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("labels are required for ChunkedCESFTTrainer")
        loss = compute_chunked_causal_lm_loss(
            model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
            chunk_size=self.logit_chunk_size,
        )
        if return_outputs:
            return loss, None
        return loss


def _empty_cuda_cache() -> None:
    if not torch.cuda.is_available():
        return
    # multi-GPU forwards may still have queued kernels on non-current devices;
    # sync all before cudaFree
    for idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(idx)
    torch.cuda.empty_cache()


def clone_lm_head_for_loss(model: nn.Module) -> nn.Linear:
    """Clone lm_head as a standalone module for loss computation.

    The in-model lm_head is managed by accelerate device_map hooks (and may be
    tied to embed_tokens); calling it on arbitrary-device hiddens fights the
    hook's input relocation. A detached CPU clone can be moved freely.
    """
    root = model
    if type(root).__name__ in {"DistributedDataParallel", "DataParallel"}:
        root = root.module
    if not hasattr(root, "lm_head"):
        raise RuntimeError(f"Model has no lm_head: {type(root).__name__}")
    src = root.lm_head
    if not isinstance(src, nn.Linear):
        raise RuntimeError(f"Expected nn.Linear lm_head, got {type(src).__name__}")
    cloned = nn.Linear(
        src.in_features,
        src.out_features,
        bias=src.bias is not None,
        dtype=src.weight.dtype,
    )
    with torch.no_grad():
        cloned.weight.copy_(src.weight.detach().cpu())
        if src.bias is not None:
            cloned.bias.copy_(src.bias.detach().cpu())
    cloned.weight.requires_grad_(False)
    if cloned.bias is not None:
        cloned.bias.requires_grad_(False)
    return cloned


class ChunkedDistillSFTTrainer(Trainer):
    """Frozen unpruned teacher + pruned Masked-LoRA student, QAD-style losses.

    Per step: teacher pass 1 (streaming cosine layer selection), teacher pass 2
    (capture selected-layer hiddens), student backbone forward with capture,
    then chunked CE + EAKLD/kl-topk + LAFD. Hidden convention matches QAD:
    last_hidden is the last decoder layer output (pre-final-norm). The loss
    lm_head is a hook-free clone (pruning never touches lm_head).
    """

    def __init__(
        self,
        *args,
        teacher_model: nn.Module,
        loss_lm_head: nn.Linear,
        task_alpha: float = 0.05,
        eakld_alpha: float = 2.0,
        lafd_alpha: float = 0.5,
        temperature: float = 1.0,
        confidence_k: int = 16,
        lafd_topk: int = 3,
        logit_chunk_size: int = 512,
        kl_mode: str = "eakld",
        kl_topk: int = 0,
        kl_post_attn: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.loss_lm_head = loss_lm_head
        self.task_alpha = float(task_alpha)
        self.eakld_alpha = float(eakld_alpha)
        self.lafd_alpha = float(lafd_alpha)
        self.temperature = float(temperature)
        self.confidence_k = int(confidence_k)
        self.lafd_topk = int(lafd_topk)
        self.logit_chunk_size = int(logit_chunk_size)
        self.kl_mode = str(kl_mode)
        self.kl_topk = int(kl_topk)
        self.kl_post_attn = bool(kl_post_attn)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):
        del num_items_in_batch
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("labels are required for ChunkedDistillSFTTrainer")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        teacher = self.teacher_model
        if hasattr(teacher, "module"):
            teacher = teacher.module
        teacher_backbone, _ = _resolve_backbone_and_lm_head(teacher)

        _empty_cuda_cache()
        # ---- Teacher pass 1: streaming cosine selection ----
        selector = StreamingTeacherSelector(attention_mask=attention_mask)
        selector.attach(teacher)
        with torch.no_grad():
            _ = teacher_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        selected = selector.selected_indices(self.lafd_topk)
        teacher_last = selector.last_hidden
        if teacher_last is None:
            raise RuntimeError("Teacher pass1 did not capture last_hidden")
        teacher_last = teacher_last.detach()
        selector.close()
        del selector
        _empty_cuda_cache()

        # ---- Teacher pass 2: capture selected layers only ----
        t_cap = SelectedHiddenCapture(
            selected=selected, attention_mask=attention_mask, keep_grad=False
        )
        t_cap.attach(teacher)
        with torch.no_grad():
            _ = teacher_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        teacher_lafd = t_cap.ordered_hiddens()
        if t_cap.last_hidden is not None:
            teacher_last = t_cap.last_hidden.detach()
        t_cap.close()
        del t_cap

        with torch.no_grad():
            teacher_last_cpu = teacher_last.detach().to("cpu").contiguous()
            teacher_lafd_cpu = [h.detach().to("cpu").contiguous() for h in teacher_lafd]
        del teacher_last, teacher_lafd
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _empty_cuda_cache()

        # ---- Student: pruned backbone + Masked LoRA ----
        causal_lm = _resolve_causal_lm(model)
        backbone, _ = _resolve_backbone_and_lm_head(causal_lm)
        lm_head = self.loss_lm_head
        s_cap = SelectedHiddenCapture(
            selected=selected, attention_mask=attention_mask, keep_grad=True
        )
        s_cap.attach(causal_lm)
        _ = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        student_last = s_cap.last_hidden
        student_lafd = s_cap.ordered_hiddens()
        s_cap.close()
        del s_cap
        if student_last is None:
            raise RuntimeError("Student did not capture last_hidden")
        _empty_cuda_cache()

        target_device = student_last.device
        if next(lm_head.parameters()).device != target_device:
            lm_head = lm_head.to(device=target_device)

        def _lm_head_fn(h: torch.Tensor) -> torch.Tensor:
            if h.device != target_device:
                h = h.to(device=target_device, non_blocking=True)
            return lm_head(h)

        # teacher hiddens stay on CPU; the loss moves them back per chunk/layer
        total, loss_dict = compute_edgerazor_qad_loss_chunked(
            student_hidden=student_last,
            teacher_hidden=teacher_last_cpu,
            lm_head=_lm_head_fn,
            labels=labels,
            attention_mask=attention_mask,
            teacher_lafd_hiddens=teacher_lafd_cpu,
            student_lafd_hiddens=student_lafd,
            chunk_size=self.logit_chunk_size,
            task_alpha=self.task_alpha,
            eakld_alpha=self.eakld_alpha,
            lafd_alpha=self.lafd_alpha,
            temperature=self.temperature,
            confidence_k=self.confidence_k,
            kl_mode=self.kl_mode,
            kl_topk=self.kl_topk,
            kl_post_attn=self.kl_post_attn,
        )
        del teacher_last_cpu, teacher_lafd_cpu
        input_device = input_ids.device
        if total.device != input_device:
            total = total.to(device=input_device)
        if not torch.isfinite(total).all():
            raise RuntimeError(
                "Distill loss is not finite: "
                + ", ".join(
                    f"{k}={float(v.detach().float().item())}"
                    for k, v in loss_dict.items()
                )
            )

        should_log = (
            self.state.global_step == 0
            or (
                self.args.logging_steps > 0
                and self.state.global_step % int(self.args.logging_steps) == 0
            )
        )
        if should_log and self.args.local_rank in (-1, 0):
            self.log(
                {
                    "ce": float(loss_dict["ce"].float().item()),
                    "eakld": float(loss_dict["eakld"].float().item()),
                    "lafd": float(loss_dict["lafd"].float().item()),
                    "qad_total": float(loss_dict["total"].float().item()),
                }
            )

        if return_outputs:
            return total, {"loss": total}
        return total


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _setup_logging()

    if os.environ.get("CONDA_DEFAULT_ENV", "") != "hif4":
        raise RuntimeError(
            f"Must run in conda env hif4, got {os.environ.get('CONDA_DEFAULT_ENV')!r}"
        )

    pruned_dir = Path(args.pruned_model_dir)
    if not pruned_dir.is_dir():
        raise FileNotFoundError(f"pruned_model_dir does not exist: {pruned_dir}")
    artifacts_dir = (
        Path(args.artifacts_dir)
        if str(args.artifacts_dir).strip()
        else pruned_dir / "pruning_artifacts"
    )
    masks_path = artifacts_dir / "block_masks.pt"
    if not masks_path.is_file():
        raise FileNotFoundError(f"Missing block masks: {masks_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    block_h, block_w, pruning_summary = _load_pruning_meta(artifacts_dir)
    masks = load_masks(masks_path)
    logger.info(
        "Loaded masks from %s (%d matrices, block=%dx%d, sparsity=%.4f)",
        masks_path,
        len(masks),
        block_h,
        block_w,
        float(pruning_summary.get("actual_block_sparsity", float("nan"))),
    )

    seed = int(args.seed)
    torch.manual_seed(seed)

    parallel_mode = str(args.parallel_mode).strip().lower()
    logger.info("Loading pruned model from %s (parallel_mode=%s)", pruned_dir, parallel_mode)
    model = load_causal_lm_for_training(
        str(pruned_dir),
        parallel_mode=parallel_mode,
        dtype="bfloat16" if _str_to_bool(args.bf16) else "float32",
        attn_implementation=str(args.attn_implementation),
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(pruned_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    teacher_model_dir = str(args.teacher_model_dir).strip()
    teacher_model = None
    if teacher_model_dir:
        logger.info(
            "Loading frozen teacher from %s (parallel_mode=%s)",
            teacher_model_dir,
            parallel_mode,
        )
        teacher_model = load_causal_lm_for_training(
            teacher_model_dir,
            parallel_mode=parallel_mode,
            dtype="bfloat16" if _str_to_bool(args.bf16) else "float32",
            attn_implementation=str(args.attn_implementation),
            trust_remote_code=True,
        )
        teacher_model.requires_grad_(False)
        teacher_model.eval()
        loss_lm_head = clone_lm_head_for_loss(model)

    model = wrap_pruned_mlp_with_peft_lora(
        model,
        masks,
        block_height=block_h,
        block_width=block_w,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
    )
    trainable, total = assert_only_lora_trainable(model)
    logger.info(
        "Trainable params: %d / %d (%.4f%%)",
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )
    if _str_to_bool(args.gradient_checkpointing):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=json.loads(args.gradient_checkpointing_kwargs)
            if str(args.gradient_checkpointing_kwargs).strip()
            else None
        )

    dataset_path = str(args.dataset_path).strip() or None
    train_ds = build_reasoning_train_dataset(
        dataset_name=str(args.dataset_name),
        dataset_path=dataset_path,
        split=str(args.dataset_split),
        trace_source=str(args.trace_source),
    )
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

    ta_kwargs: dict[str, Any] = {
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
        "dataloader_num_workers": int(args.dataloader_num_workers),
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
    }
    if parallel_mode == "layer":
        ta_kwargs["use_cpu"] = False

    training_args = TrainingArguments(**ta_kwargs)
    if teacher_model is not None:
        logger.info(
            "Distillation enabled: kl_mode=%s task_alpha=%g eakld_alpha=%g lafd_alpha=%g",
            str(args.kl_mode),
            float(args.task_alpha),
            float(args.eakld_alpha),
            float(args.lafd_alpha),
        )
        trainer = ChunkedDistillSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=collator,
            processing_class=tokenizer,
            teacher_model=teacher_model,
            loss_lm_head=loss_lm_head,
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
        )
    else:
        trainer = ChunkedCESFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=collator,
            processing_class=tokenizer,
            logit_chunk_size=int(args.logit_chunk_size),
        )

    logger.info("Starting Masked LoRA SFT for %d steps", int(args.max_steps))
    trainer.train()

    logger.info("Merging masked LoRA and verifying pruned blocks remain zero")
    merged = merge_and_verify(
        model,
        masks,
        block_height=block_h,
        block_width=block_w,
    )

    # Keep CausalLM export compatible with vLLM / eval scripts
    if hasattr(merged, "config") and getattr(merged.config, "model_type", None) == "qwen3_5_text":
        merged.config.architectures = ["Qwen3_5ForCausalLM"]
        merged.config.use_cache = True

    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "pruned_model_dir": str(pruned_dir),
        "artifacts_dir": str(artifacts_dir),
        "output_dir": str(output_dir),
        "block_height": block_h,
        "block_width": block_w,
        "actual_block_sparsity": pruning_summary.get("actual_block_sparsity"),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "max_steps": int(args.max_steps),
        "learning_rate": float(args.learning_rate),
        "model_max_length": int(args.model_max_length),
        "logit_chunk_size": int(args.logit_chunk_size),
        "dataset_name": str(args.dataset_name),
        "dataset_path": dataset_path,
        "trainable_params": trainable,
        "total_params": total,
        "parallel_mode": parallel_mode,
        "teacher_model_dir": teacher_model_dir,
        "task_alpha": float(args.task_alpha),
        "eakld_alpha": float(args.eakld_alpha),
        "lafd_alpha": float(args.lafd_alpha),
        "temperature": float(args.temperature),
        "confidence_k": int(args.confidence_k),
        "lafd_topk": int(args.lafd_topk),
        "kl_mode": str(args.kl_mode),
        "kl_topk": int(args.kl_topk),
        "kl_post_attn": _str_to_bool(args.kl_post_attn),
        "seed": seed,
    }
    (output_dir / "lora_train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Preserve masks next to the recovered checkpoint for later audits
    torch.save({k: v.cpu().clone() for k, v in masks.items()}, output_dir / "block_masks.pt")
    logger.info("Saved merged HF model to %s", output_dir)


if __name__ == "__main__":
    main()
