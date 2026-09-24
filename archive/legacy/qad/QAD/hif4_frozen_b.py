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

from nvfp4_hif4_torch import (  # noqa: E402
    E6M2_VALUES,
    HiF4Config,
    quantize_hif4,
    round_bfloat16,
    round_e6m2,
)

from hif4_fixed_s0 import apply_e6m2_ste  # noqa: E402

__all__ = [
    "HiF4FrozenBLinear",
    "build_frozen_b_and_s0",
    "collect_hif4_frozen_b_linears",
]


def _e6m2_step_up(values: torch.Tensor, steps: int) -> torch.Tensor:
    """把 e6m2 网格上的正值向上走 steps 格；超出码本顶端返回 inf。"""
    book = E6M2_VALUES.to(device=values.device, dtype=torch.float32)
    idx = torch.searchsorted(book, values.clamp_min(book[0]))
    idx = idx.clamp_max(book.numel() - 1)
    if not bool((book[idx] == values).all()):
        raise ValueError("s0 estimate is not on the e6m2 codebook")
    nxt = idx + int(steps)
    out = torch.full_like(values, torch.inf)
    ok = nxt < book.numel()
    out[ok] = book[nxt[ok]]
    return out


def _on_canonical_hif4_grid(b: torch.Tensor) -> torch.Tensor:
    """逐元素检查 b 是否为 canonical payload×2^e：|b| ∈ {0} ∪ {p×2^m, p∈{0.25..1.75 步长0.25}, m∈{0,1,2}}。"""
    a = b.abs()
    zero = a == 0
    m0 = (a >= 0.25) & (a <= 1.75) & ((a * 4.0) == torch.round(a * 4.0))
    m1 = (a >= 2.0) & (a <= 3.5) & ((a * 2.0) == torch.round(a * 2.0))
    m2 = (a >= 4.0) & (a <= 7.0) & (a == torch.round(a))
    return zero | m0 | m1 | m2


def _recover_s0_and_b_exact(
    weight: torch.Tensor,
    *,
    config: HiF4Config,
    max_step_up: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """已在 HiF4 网格上的权重的精确分解：e6m2(s0) ⊙ B == weight，B 为 canonical payload×2^e。

    量化器本身不幂等（bf16 倒数使 e8/e4 阈值在边界翻转），不能靠重新量化恢复网格；
    这里按组从 s0 公式估计值向上步进搜索，取满足精确分解的最大 s0。
    输入不在 canonical HiF4 网格上时直接报错。
    """
    if int(config.group_dim) != -1:
        raise ValueError(f"exact grid recovery only supports group_dim=-1, got {config.group_dim}")
    if config.hierarchy_format != "s1p2" or config.payload_format != "s1p2":
        raise ValueError(
            f"exact grid recovery only supports s1p2 hierarchy/payload, got "
            f"hierarchy={config.hierarchy_format} payload={config.payload_format}"
        )
    group_size = int(config.group_size)
    out_f, in_f = int(weight.shape[0]), int(weight.shape[1])
    ng = in_f // group_size
    w = weight.detach().to(device="cpu", dtype=torch.float32)
    groups = w.reshape(out_f, ng, group_size)
    amax = groups.abs().amax(dim=-1)
    nonzero = amax > 0

    # 与 quantize_hif4 hardware 模式相同的 s0 估计：e6m2(bf16(amax * bf16(1/7)))
    recip = round_bfloat16(torch.tensor(1.0 / 7.0))
    s0_est = round_e6m2(round_bfloat16(amax * recip))
    s0_est = torch.where(nonzero, s0_est, torch.ones_like(s0_est))

    chosen = s0_est.clone()
    resolved = ~nonzero  # 零组：s0=1, B=0
    cand = s0_est
    for _ in range(int(max_step_up) + 1):
        cand = torch.where(resolved, chosen, cand)
        b = groups / cand.unsqueeze(-1)
        exact = round_bfloat16(b) * cand.unsqueeze(-1) == groups
        valid = (exact & _on_canonical_hif4_grid(b)).all(dim=-1)
        chosen = torch.where(valid, cand, chosen)
        resolved = resolved | valid
        if bool(resolved.all()):
            break
        cand = _e6m2_step_up(cand, 1)
    if not bool(resolved.all()):
        bad = (~resolved).nonzero()[0].tolist()
        raise ValueError(
            f"exact HiF4 grid recovery failed at group (row={bad[0]}, group={bad[1]}): "
            "init weight is not on the canonical HiF4 grid "
            "(only hif4-format pseudo-quant ckpts are supported; hif4-1 等其它格式请在量化侧对齐)"
        )

    frozen_b = round_bfloat16(groups / chosen.unsqueeze(-1)).reshape(out_f, in_f)
    recon = (chosen.unsqueeze(-1) * frozen_b.reshape(out_f, ng, group_size).float()).reshape(out_f, in_f)
    if not torch.equal(recon, w):
        raise ValueError("exact HiF4 grid recovery internal error: e6m2(s0)*B != weight")
    return chosen.to(dtype=torch.float32), frozen_b.to(dtype=weight.dtype)


def build_frozen_b_and_s0(
    weight: torch.Tensor,
    *,
    config: HiF4Config = HiF4Config(),
    exact_grid: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """初始化分解：返回 (s0_init, frozen_B)，B 与 weight 同形状。

    exact_grid=False：对 BF16 权重做一次 HiF4 量化（RTN 初始化，量化损失符合预期）。
    exact_grid=True：输入必须是已在 canonical HiF4 网格上的伪量化 ckpt，
        逐组恢复 s0 并精确分解，保证 e6m2(s0)⊙B == weight 逐 bit 相等；否则报错。
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got {tuple(weight.shape)}")
    group_size = int(config.group_size)
    out_f, in_f = int(weight.shape[0]), int(weight.shape[1])
    if in_f % group_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by group_size={group_size}")

    with torch.no_grad():
        if exact_grid:
            return _recover_s0_and_b_exact(weight, config=config)
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
        # 伪量化 ckpt 初始化必须精确恢复网格（训练起点 ≡ ckpt）；
        # 非 HiF4 网格的 ckpt（如 hif4-1）会在恢复时报错
        s0_init, frozen_b = build_frozen_b_and_s0(
            quant_src, config=config, exact_grid=init_weight is not None
        )
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
