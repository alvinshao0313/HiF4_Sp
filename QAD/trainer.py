"""HiF4 QAD Trainer：冻 B + 只训 S0；串行 teacher/student；hook 选择性 LAFD。"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn
from transformers import Trainer

from distill_losses import compute_edgerazor_qad_loss_chunked
from hidden_hooks import SelectedHiddenCapture, StreamingTeacherSelector
from hif4_frozen_b import HiF4FrozenBLinear

logger = logging.getLogger("qad.trainer")

__all__ = [
    "HiF4QADTrainer",
    "assert_only_s0_trainable",
    "set_model_temporary",
    "clone_lm_head_for_loss",
]


def set_model_temporary(model: nn.Module, temporary: bool) -> None:
    for module in model.modules():
        if hasattr(module, "set_temporary"):
            module.set_temporary(bool(temporary))


def assert_only_s0_trainable(model: nn.Module) -> list[str]:
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


def clone_lm_head_for_loss(model: nn.Module) -> nn.Linear:
    root = model
    if type(root).__name__ in {"DistributedDataParallel", "DataParallel"}:
        root = root.module
    if not hasattr(root, "lm_head"):
        raise RuntimeError(f"Model has no lm_head: {type(root).__name__}")
    src = root.lm_head
    if not isinstance(src, nn.Linear):
        raise RuntimeError(f"Expected nn.Linear lm_head, got {type(src).__name__}")
    cloned = nn.Linear(src.in_features, src.out_features, bias=src.bias is not None)
    with torch.no_grad():
        cloned.weight.copy_(src.weight.detach().cpu())
        if src.bias is not None:
            cloned.bias.copy_(src.bias.detach().cpu())
    cloned.weight.requires_grad_(False)
    if cloned.bias is not None:
        cloned.bias.requires_grad_(False)
    return cloned


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unwrap_root(model: nn.Module) -> nn.Module:
    root = model
    if type(root).__name__ in {"DistributedDataParallel", "DataParallel"}:
        root = root.module
    return root


class HiF4QADTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        task_alpha: float = 0.05,
        eakld_alpha: float = 2.0,
        lafd_alpha: float = 0.5,
        temperature: float = 1.0,
        confidence_k: int = 16,
        lafd_topk: int = 3,
        logit_chunk_size: int = 512,
        loss_lm_head: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.task_alpha = float(task_alpha)
        self.eakld_alpha = float(eakld_alpha)
        self.lafd_alpha = float(lafd_alpha)
        self.temperature = float(temperature)
        self.confidence_k = int(confidence_k)
        self.lafd_topk = int(lafd_topk)
        self.logit_chunk_size = int(logit_chunk_size)
        self.loss_lm_head = loss_lm_head
        self._last_loss_dict: dict[str, float] = {}

    def _resolve_loss_lm_head(self, model: nn.Module) -> nn.Module:
        if self.loss_lm_head is not None:
            return self.loss_lm_head
        root = model
        if hasattr(self, "accelerator") and self.accelerator is not None:
            root = self.accelerator.unwrap_model(model)
        root = _unwrap_root(root)
        if not hasattr(root, "lm_head"):
            raise RuntimeError("lm_head missing")
        return root.lm_head

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        del num_items_in_batch

        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("labels are required for QAD training")

        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        attention_mask = inputs.get("attention_mask")
        forward_kwargs = dict(
            output_hidden_states=False,
            use_cache=False,
            logits_to_keep=1,
        )
        _empty_cuda_cache()

        # ---- Teacher pass 1: 流式 cosine 选型 ----
        set_model_temporary(model, False)
        selector = StreamingTeacherSelector(attention_mask=attention_mask)
        selector.attach(model)
        with torch.no_grad():
            _ = model(**model_inputs, **forward_kwargs)
        selected = selector.selected_indices(self.lafd_topk)
        teacher_last = selector.last_hidden
        if teacher_last is None:
            raise RuntimeError("Teacher pass1 did not capture last_hidden")
        teacher_last = teacher_last.detach()
        selector.close()
        del selector
        _empty_cuda_cache()

        # ---- Teacher pass 2: 只抓 selected 层 hidden ----
        t_cap = SelectedHiddenCapture(
            selected=selected, attention_mask=attention_mask, keep_grad=False
        )
        t_cap.attach(model)
        with torch.no_grad():
            _ = model(**model_inputs, **forward_kwargs)
        teacher_lafd = t_cap.ordered_hiddens()
        if t_cap.last_hidden is not None:
            teacher_last = t_cap.last_hidden.detach()
        t_cap.close()
        del t_cap

        # 教师标签落到 CPU，释放教师激活显存，再跑学生
        with torch.no_grad():
            teacher_last_cpu = teacher_last.detach().to("cpu").contiguous()
            teacher_lafd_cpu = [h.detach().to("cpu").contiguous() for h in teacher_lafd]
        del teacher_last, teacher_lafd
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _empty_cuda_cache()

        # ---- Student: HiF4 伪量化 Ŵ=STE(S0)⊙B ----
        set_model_temporary(model, True)
        s_cap = SelectedHiddenCapture(
            selected=selected, attention_mask=attention_mask, keep_grad=True
        )
        s_cap.attach(model)
        _ = model(**model_inputs, **forward_kwargs)
        student_last = s_cap.last_hidden
        student_lafd = s_cap.ordered_hiddens()
        s_cap.close()
        del s_cap
        if student_last is None:
            raise RuntimeError("Student did not capture last_hidden")
        _empty_cuda_cache()

        lm_head = self._resolve_loss_lm_head(model)
        target_device = student_last.device
        if next(lm_head.parameters()).device != target_device:
            lm_head = lm_head.to(device=target_device)

        def _lm_head_fn(h: torch.Tensor) -> torch.Tensor:
            if h.device != target_device:
                h = h.to(device=target_device, non_blocking=True)
            return lm_head(h)

        # teacher_last 保持在 CPU：loss 内按 chunk 搬回，避免整段再占显存
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
        )
        del teacher_last_cpu, teacher_lafd_cpu
        # HF Trainer 要求 loss 在 inputs 所在设备（通常是第一张卡）
        input_device = model_inputs["input_ids"].device
        if total.device != input_device:
            total = total.to(device=input_device)
        if not torch.isfinite(total).all():
            raise RuntimeError(
                "QAD loss is not finite: "
                + ", ".join(f"{k}={float(v.detach().float().item())}" for k, v in loss_dict.items())
            )

        self._last_loss_dict = {k: float(v.detach().float().item()) for k, v in loss_dict.items()}
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
                    "ce": self._last_loss_dict["ce"],
                    "eakld": self._last_loss_dict["eakld"],
                    "lafd": self._last_loss_dict["lafd"],
                    "qad_total": self._last_loss_dict["total"],
                }
            )

        if return_outputs:
            return total, {"loss": total}
        return total
