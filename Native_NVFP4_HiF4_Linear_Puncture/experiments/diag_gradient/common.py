"""diag_gradient 共享工具：加载、STE、基线/参考前向、指标。

逐通道 DIAG（run.py）与 H4+组共享 DIAG（run_h4_group.py）共用本模块，
避免两套加载逻辑漂移。改数据路径 / STE 定义时优先改这里。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import load_packed_linear_state
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import load_pt, module_capture_stem
from Native_NVFP4_HiF4_Linear_Puncture.src.metrics import error_energy, reference_energy
from Native_NVFP4_HiF4_Linear_Puncture.src.native_nvfp4 import (
    PackedNVFP4LinearState,
    dequantize_packed_weight,
)

# 未量化乘积等价性检查
EQUIV_CHECK_ROWS = 32
EQUIV_REL_L2_MAX = 1e-6

DEFAULT_CAPTURE_RUN_ID = "20260812T103800Z_native_nvfp4_hif4_linear_puncture"
DEFAULT_LR = 0.05
DEFAULT_STEPS = 200
LOG2_MIN = -4.0
LOG2_MAX = 4.0


def log2_bound_meta(log2_clamp: bool) -> dict[str, Any]:
    """落盘用：是否钳位 z，以及对应 log2 上下界（无约束时为 null）。"""
    return {
        "log2_clamp": log2_clamp,
        "log2_min": LOG2_MIN if log2_clamp else None,
        "log2_max": LOG2_MAX if log2_clamp else None,
    }


def aggregate_scale_extremes(
    ds: list[torch.Tensor],
    log2_ds: list[torch.Tensor],
) -> dict[str, float]:
    """跨 module 汇总 d / log2(d) 极值。"""
    if not ds:
        return {
            "max_abs_log2_d": float("nan"),
            "min_d": float("nan"),
            "max_d": float("nan"),
        }
    d_cat = torch.cat([t.reshape(-1).float() for t in ds])
    z_cat = torch.cat([t.reshape(-1).float() for t in log2_ds])
    return {
        "max_abs_log2_d": float(z_cat.abs().max().item()),
        "min_d": float(d_cat.min().item()),
        "max_d": float(d_cat.max().item()),
    }


def linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """标准 Linear：y = x @ w^T + b。"""
    return F.linear(x, w, bias)


def qdq_hif4_ste(x: torch.Tensor) -> torch.Tensor:
    """HiF4 伪量化 + STE：前向真量化，反传当作恒等。"""
    y = qdq_hif4_direct(x, output_dtype=torch.float32)
    return x + (y - x).detach()


def nmse_from_outputs(y_hat: torch.Tensor, y_ref: torch.Tensor) -> tuple[float, float, float]:
    """返回 (error_energy, reference_energy, nmse)。"""
    err = error_energy(y_hat, y_ref)
    ref = reference_energy(y_ref)
    return err, ref, float(err / max(ref, 1e-30))


def forward_y_nn(
    a_n: torch.Tensor,
    w_n: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """参考：激活 NV + 权重 NV。"""
    return linear(a_n, w_n, bias)


def forward_y_base(
    x: torch.Tensor,
    w_n: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    use_ste: bool,
) -> torch.Tensor:
    """无 DIAG / 无 H4 基线：Q_H(X) @ Q_H(W_N)。"""
    if use_ste:
        a_h = qdq_hif4_ste(x)
        w_h = qdq_hif4_ste(w_n)
    else:
        a_h = qdq_hif4_direct(x, output_dtype=torch.float32)
        w_h = qdq_hif4_direct(w_n, output_dtype=torch.float32)
    return linear(a_h, w_h, bias)


def load_x_and_an(
    capture_run_dir: Path,
    module_name: str,
    split: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """加载未量化 X 与离线 A_N，转到 device、float32。"""
    stem = module_capture_stem(module_name)
    cap_path = capture_run_dir / "captures" / f"{stem}_{split}.pt"
    an_path = capture_run_dir / "nvfp4_qdq" / f"{stem}_{split}.pt"
    if not cap_path.is_file():
        raise FileNotFoundError(f"缺少 capture: {cap_path}")
    if not an_path.is_file():
        raise FileNotFoundError(f"缺少 nvfp4_qdq: {an_path}")

    cap = load_pt(cap_path, map_location="cpu")
    an_pack = load_pt(an_path, map_location="cpu")
    if cap.get("module_name") != module_name:
        raise RuntimeError(f"capture module_name 不符: {cap_path}")
    if an_pack.get("module_name") != module_name:
        raise RuntimeError(f"nvfp4_qdq module_name 不符: {an_path}")

    x = cap["x_rot_bf16"].to(device=device, dtype=torch.float32)
    a_n = an_pack["a_n_bf16"].to(device=device, dtype=torch.float32)
    if tuple(x.shape) != tuple(a_n.shape):
        raise RuntimeError(
            f"X 与 A_N shape 不一致: {module_name} {split} "
            f"X={tuple(x.shape)} A_N={tuple(a_n.shape)}"
        )
    return x, a_n


def load_w_n_bias(
    snapshot: Path,
    weight_map: dict[str, str],
    module_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """从 ckpt 解出 W_N（float32）与 bias。"""
    packed = load_packed_linear_state(snapshot, weight_map, module_name)
    state = PackedNVFP4LinearState(
        module_name=module_name,
        weight_packed=packed["weight_packed"],  # type: ignore[arg-type]
        weight_scale=packed["weight_scale"],  # type: ignore[arg-type]
        weight_global_scale=packed["weight_global_scale"].to(torch.float32),  # type: ignore[union-attr]
        input_global_scale=packed["input_global_scale"].to(torch.float32),  # type: ignore[union-attr]
        rotation_matrix=packed["rotation_matrix"].to(torch.bfloat16),  # type: ignore[union-attr]
        bias=packed["bias"],
    )
    w_n = dequantize_packed_weight(state).to(device=device, dtype=torch.float32)
    bias = (
        state.bias.to(device=device, dtype=torch.float32)
        if state.bias is not None
        else None
    )
    return w_n, bias
