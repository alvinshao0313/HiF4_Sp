"""可训练 S0 的 HiF4 Linear：冻结权重，仅训 s0_continuous。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHUANCI_DIR = _REPO_ROOT / "ChuanCi"
if str(_CHUANCI_DIR) not in sys.path:
    sys.path.insert(0, str(_CHUANCI_DIR))

from nvfp4_hif4_torch import HiF4Config  # noqa: E402

from hif4_fixed_s0 import apply_e6m2_ste, init_s0_from_weight, quantize_hif4_with_fixed_s0  # noqa: E402

__all__ = ["HiF4ScaleLinear"]


class HiF4ScaleLinear(nn.Module):
    """Teacher/student 切换的 HiF4 S0 可调 Linear。

    temporary=False -> 原始 BF16 teacher_weight
    temporary=True  -> 用 s0_continuous（STE→E6M2）重建的 HiF4 权重
    """

    def __init__(
        self,
        teacher_weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        trainable_s0: bool = False,
        config: HiF4Config = HiF4Config(),
    ) -> None:
        super().__init__()
        if teacher_weight.ndim != 2:
            raise ValueError(f"teacher_weight must be 2D [out, in], got shape {tuple(teacher_weight.shape)}")
        out_features, in_features = int(teacher_weight.shape[0]), int(teacher_weight.shape[1])
        if in_features % int(config.group_size) != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by group_size={config.group_size}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.temporary = True

        weight_param = nn.Parameter(teacher_weight.detach().clone(), requires_grad=False)
        self.register_parameter("teacher_weight", weight_param)

        if bias is None:
            self.register_parameter("bias", None)
        else:
            bias_param = nn.Parameter(bias.detach().clone(), requires_grad=False)
            if tuple(bias_param.shape) != (out_features,):
                raise ValueError(f"bias shape {tuple(bias_param.shape)} != ({out_features},)")
            self.register_parameter("bias", bias_param)

        s0_init = init_s0_from_weight(weight_param, config=config)
        self.s0_continuous = nn.Parameter(s0_init, requires_grad=bool(trainable_s0))

    def set_temporary(self, temporary: bool = True) -> None:
        self.temporary = bool(temporary)

    def set_s0_trainable(self, trainable: bool) -> None:
        self.s0_continuous.requires_grad = bool(trainable)

    @property
    def s0_e6m2(self) -> torch.Tensor:
        """当前连续 S0 对应的 E6M2 离散值（无梯度）。"""
        with torch.no_grad():
            return apply_e6m2_ste(self.s0_continuous.detach(), scale_mode=self.config.scale_mode)

    def reconstruct_weight(self) -> torch.Tensor:
        """用当前 s0_continuous（STE）重建 HiF4 权重。"""
        result = quantize_hif4_with_fixed_s0(
            self.teacher_weight,
            self.s0_continuous,
            config=self.config,
            apply_ste=True,
        )
        return result.values

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
