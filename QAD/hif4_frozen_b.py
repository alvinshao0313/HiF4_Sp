"""冻 B + 只训 S0 的伪量化 Linear（QAD 专用，不改 ScaleTuning）。

初始化一次 HiF4：B = sign * 2^(e8+e4) * payload（冻），S0 可训。
学生：Ŵ = STE(S0) ⊙ B；教师：W = teacher_weight（BF16）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_QAD_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _QAD_DIR.parent
_SCALE_TUNING_DIR = _REPO_ROOT / "ScaleTuning"
_CHUANCI_DIR = _REPO_ROOT / "ChuanCi"
for _p in (_SCALE_TUNING_DIR, _CHUANCI_DIR):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from nvfp4_hif4_torch import HiF4Config, quantize_hif4  # noqa: E402

from hif4_fixed_s0 import apply_e6m2_ste  # noqa: E402

__all__ = [
    "HiF4FrozenBLinear",
    "build_frozen_b_and_s0",
    "collect_hif4_frozen_b_linears",
]


def build_frozen_b_and_s0(
    weight: torch.Tensor,
    *,
    config: HiF4Config = HiF4Config(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """一次量化：返回 (s0_init, frozen_B)，B 与 weight 同形状。"""
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got {tuple(weight.shape)}")
    group_size = int(config.group_size)
    out_f, in_f = int(weight.shape[0]), int(weight.shape[1])
    if in_f % group_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by group_size={group_size}")

    with torch.no_grad():
        # 在 CPU 上量化，避免 GPU 上再开一份 float32 峰值
        w_cpu = weight.detach().to(device="cpu", dtype=torch.float32)
        result = quantize_hif4(w_cpu, config=config)
        s0 = result.top_scale.detach().to(dtype=torch.float32)
        # recon = s0 * B  =>  B = recon / s0（零组 recon=0）
        s0_exp = s0.unsqueeze(-1).expand(out_f, in_f // group_size, group_size).reshape(out_f, in_f)
        recon = result.values
        frozen_b = torch.where(
            s0_exp.abs() > 0,
            recon / s0_exp,
            torch.zeros_like(recon),
        )
        frozen_b = frozen_b.to(dtype=weight.dtype)
        del w_cpu, result, recon, s0_exp
    return s0, frozen_b


class HiF4FrozenBLinear(nn.Module):
    """教师 BF16 / 学生 STE(S0)⊙B 伪量化。"""

    def __init__(
        self,
        teacher_weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        init_weight: torch.Tensor | None = None,
        trainable_s0: bool = False,
        config: HiF4Config = HiF4Config(),
    ) -> None:
        """Args:
        teacher_weight: 教师前向用的 BF16 权重。
        init_weight: 可选；若给，则用它对学生做一次 HiF4，初始化 frozen_b + s0
            （例如 GPTQ 伪量化 HF 权重）。默认与 teacher_weight 相同。
        """
        super().__init__()
        if teacher_weight.ndim != 2:
            raise ValueError(f"teacher_weight must be 2D, got {tuple(teacher_weight.shape)}")
        out_features, in_features = int(teacher_weight.shape[0]), int(teacher_weight.shape[1])
        if in_features % int(config.group_size) != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by group_size={config.group_size}"
            )
        if init_weight is not None:
            if init_weight.ndim != 2:
                raise ValueError(f"init_weight must be 2D, got {tuple(init_weight.shape)}")
            if tuple(init_weight.shape) != (out_features, in_features):
                raise ValueError(
                    f"init_weight shape {tuple(init_weight.shape)} != "
                    f"teacher_weight shape {(out_features, in_features)}"
                )

        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.group_size = int(config.group_size)
        self.temporary = True

        # 不 clone：由 wrap 侧移交 storage，避免双权重替换时 3 倍峰值
        self.register_parameter(
            "teacher_weight",
            nn.Parameter(teacher_weight.detach(), requires_grad=False),
        )
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.register_parameter(
                "bias",
                nn.Parameter(bias.detach(), requires_grad=False),
            )

        quant_src = self.teacher_weight if init_weight is None else init_weight.detach()
        s0_init, frozen_b = build_frozen_b_and_s0(quant_src, config=config)
        # S0 训练态固定 float32，避免 bf16 梯度溢出
        self.s0_continuous = nn.Parameter(
            s0_init.detach().to(dtype=torch.float32),
            requires_grad=bool(trainable_s0),
        )
        # 冻 B：S1/S2/FP4 反量化底
        self.register_buffer("frozen_b", frozen_b, persistent=True)

    def to(self, *args, **kwargs):  # type: ignore[override]
        """搬设备/dtype 后强制 s0_continuous 回到 float32。"""
        super().to(*args, **kwargs)
        if self.s0_continuous.dtype != torch.float32:
            self.s0_continuous.data = self.s0_continuous.data.to(dtype=torch.float32)
        return self

    def set_temporary(self, temporary: bool = True) -> None:
        self.temporary = bool(temporary)

    def set_s0_trainable(self, trainable: bool) -> None:
        self.s0_continuous.requires_grad = bool(trainable)

    def reconstruct_weight(self) -> torch.Tensor:
        """Ŵ = STE(S0) ⊙ B（按 group 广播）。"""
        s0 = apply_e6m2_ste(self.s0_continuous, scale_mode=self.config.scale_mode)
        g = self.group_size
        # s0: [out, in/g] -> [out, in]
        s0_exp = s0.unsqueeze(-1).expand(self.out_features, self.in_features // g, g)
        s0_exp = s0_exp.reshape(self.out_features, self.in_features)
        # 乘在 float32 上更稳，再 cast
        w = s0_exp.to(dtype=torch.float32) * self.frozen_b.to(dtype=torch.float32)
        return w.to(dtype=self.teacher_weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if bool(self.temporary):
            weight = self.reconstruct_weight()
        else:
            weight = self.teacher_weight
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        bias = self.bias
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


def collect_hif4_frozen_b_linears(model: nn.Module) -> list[tuple[str, HiF4FrozenBLinear]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, HiF4FrozenBLinear)
    ]
