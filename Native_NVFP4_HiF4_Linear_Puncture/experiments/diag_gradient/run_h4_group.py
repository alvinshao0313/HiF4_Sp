#!/usr/bin/env python3
"""H4 块旋转 + 四元组共享 DIAG 的梯度优化。

公式约定
--------
R4 = H4/2（正交；禁止直接用未归一化 H4），R = blockdiag(R4,...,R4)。

可学习 d_g 长度 K/4，展开为每标量重复 4 次得到 d（作用在旋转后坐标系）。

参考 B：
    Y_B = Y_NN = Linear(A_N, W_N)

无变换基线：
    Y_base = Linear(Q_H(X), Q_H(W_N))

仅 H4（d=1）：
    Y_H4 = Linear(Q_H(X R), Q_H(W_N R))

优化路径 A：
    Y_A = Linear(Q_H((X R) * d), Q_H((W_N R) / d))

损失（cal）：L = ||Y_A - Y_B||^2 / ||Y_B||^2

共享工具见 common.py；逐通道 DIAG 见 run.py。
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from typing import Any

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.diag_gradient.common import (
    DEFAULT_CAPTURE_RUN_ID,
    DEFAULT_LR,
    DEFAULT_STEPS,
    EQUIV_CHECK_ROWS,
    EQUIV_REL_L2_MAX,
    LOG2_MAX,
    LOG2_MIN,
    forward_y_base,
    forward_y_nn,
    linear,
    load_w_n_bias,
    load_x_and_an,
    nmse_from_outputs,
    qdq_hif4_ste,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.h4_block_rotation.h4_transform import (
    H4_GROUP_SIZE,
    apply_h4_g4,
    assert_r4_orthogonal,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from Native_NVFP4_HiF4_Linear_Puncture.src.config import load_config, results_dir
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import (
    ensure_dir,
    module_capture_stem,
    save_pt,
    write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.metrics import (
    aggregate_global_nmse,
    error_energy,
    recovery_ratio,
)


def expand_group_scale(d_g: torch.Tensor) -> torch.Tensor:
    """把长度 K/4 的组缩放展开为长度 K：每个值重复 4 次。"""
    if d_g.ndim != 1:
        raise ValueError(f"d_g 必须是一维，got shape={tuple(d_g.shape)}")
    return d_g.repeat_interleave(H4_GROUP_SIZE)


def forward_y_h4_group(
    x: torch.Tensor,
    w_n: torch.Tensor,
    d_g: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    use_ste: bool,
) -> torch.Tensor:
    """Y_A：先 G4 右乘 R4，再组共享 DIAG，再两侧 HiF4。"""
    d = expand_group_scale(d_g)
    x_r = apply_h4_g4(x, compute_dtype=torch.float32, output_dtype=torch.float32)
    w_r = apply_h4_g4(w_n, compute_dtype=torch.float32, output_dtype=torch.float32)
    if use_ste:
        a_h = qdq_hif4_ste(x_r * d)
        w_h = qdq_hif4_ste(w_r / d)
    else:
        a_h = qdq_hif4_direct(x_r * d, output_dtype=torch.float32)
        w_h = qdq_hif4_direct(w_r / d, output_dtype=torch.float32)
    return linear(a_h, w_h, bias)


def assert_unquant_h4_group_equivalence(
    x: torch.Tensor,
    w_n: torch.Tensor,
    d_g: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    """门禁：无量化时 Linear((XR)*d, (WR)/d) ≈ Linear(X, W)。"""
    n = min(EQUIV_CHECK_ROWS, x.shape[0])
    d = expand_group_scale(d_g)
    x_s = x[:n]
    x_r = apply_h4_g4(x_s, compute_dtype=torch.float32, output_dtype=torch.float32)
    w_r = apply_h4_g4(w_n, compute_dtype=torch.float32, output_dtype=torch.float32)
    y0 = linear(x_s, w_n, bias)
    y1 = linear(x_r * d, w_r / d, bias)
    rel = float((y0 - y1).norm().item() / max(float(y0.norm().item()), 1e-30))
    if rel > EQUIV_REL_L2_MAX:
        raise RuntimeError(
            f"未量化 H4+组DIAG 乘积不等价: rel_L2={rel} > {EQUIV_REL_L2_MAX}"
        )


def optimize_one_module(
    *,
    module_name: str,
    x_cal: torch.Tensor,
    a_n_cal: torch.Tensor,
    x_val: torch.Tensor,
    a_n_val: torch.Tensor,
    w_n: torch.Tensor,
    bias: torch.Tensor | None,
    lr: float,
    steps: int,
) -> dict[str, Any]:
    """cal 上优化 z_g=log2(d_g)；val 上报基线 / 仅H4 / 优化结果。"""
    k = x_cal.shape[-1]
    if k % H4_GROUP_SIZE != 0:
        raise ValueError(f"{module_name}: K={k} 不能被 {H4_GROUP_SIZE} 整除")
    n_g = k // H4_GROUP_SIZE
    device = x_cal.device

    with torch.no_grad():
        y_nn_cal = forward_y_nn(a_n_cal, w_n, bias).detach()
        y_nn_val = forward_y_nn(a_n_val, w_n, bias).detach()
        y_base_val = forward_y_base(x_val, w_n, bias, use_ste=False).detach()
        e_base, ref_e, nmse_base = nmse_from_outputs(y_base_val, y_nn_val)

        d_g_one = torch.ones(n_g, device=device, dtype=torch.float32)
        assert_unquant_h4_group_equivalence(x_cal, w_n, d_g_one, bias)
        y_h4_val = forward_y_h4_group(x_val, w_n, d_g_one, bias, use_ste=False).detach()
        e_h4, _, nmse_h4 = nmse_from_outputs(y_h4_val, y_nn_val)

    z_g = torch.nn.Parameter(torch.zeros(n_g, device=device, dtype=torch.float32))
    opt = torch.optim.Adam([z_g], lr=lr)

    history: list[float] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        d_g = torch.pow(torch.tensor(2.0, device=device), z_g)
        y = forward_y_h4_group(x_cal, w_n, d_g, bias, use_ste=True)
        err = ((y - y_nn_cal) ** 2).sum()
        ref = (y_nn_cal ** 2).sum().clamp_min(1e-30)
        loss = err / ref
        loss.backward()
        opt.step()
        with torch.no_grad():
            z_g.clamp_(LOG2_MIN, LOG2_MAX)
        history.append(float(loss.detach().item()))
        if step == 0 or step + 1 == steps or (step + 1) % 50 == 0:
            print(
                f"  [{module_name}] step {step+1}/{steps}  cal_nmse={history[-1]:.6e}",
                flush=True,
            )

    with torch.no_grad():
        d_g_final = torch.pow(torch.tensor(2.0, device=device), z_g.detach())
        assert_unquant_h4_group_equivalence(x_cal, w_n, d_g_final, bias)
        y_grad_val = forward_y_h4_group(
            x_val, w_n, d_g_final, bias, use_ste=False
        )
        e_grad, _, nmse_grad = nmse_from_outputs(y_grad_val, y_nn_val)
        recovery = recovery_ratio(e_base, e_grad)
        recovery_h4 = recovery_ratio(e_base, e_h4)
        d_expanded = expand_group_scale(d_g_final).detach().cpu().float()

    return {
        "module_name": module_name,
        "k_dim": k,
        "n_groups": n_g,
        "d_g": d_g_final.detach().cpu().float(),
        "log2_d_g": z_g.detach().cpu().float(),
        "d": d_expanded,
        "cal_nmse_final": history[-1] if history else float("nan"),
        "val_error_energy_base": e_base,
        "val_error_energy_h4_only": e_h4,
        "val_error_energy_grad": e_grad,
        "val_reference_energy": ref_e,
        "val_nmse_base": nmse_base,
        "val_nmse_h4_only": nmse_h4,
        "val_nmse_grad": nmse_grad,
        "val_recovery_vs_base": recovery,
        "val_recovery_h4_only_vs_base": recovery_h4,
    }


def run(
    *,
    capture_run_id: str,
    run_id: str,
    config_path: str | None,
    device: str,
    lr: float,
    steps: int,
    modules: list[str] | None = None,
) -> dict[str, Any]:
    assert_r4_orthogonal()

    config = load_config(config_path)
    capture_dir = results_dir(capture_run_id)
    out_dir = ensure_dir(results_dir(run_id))
    scales_dir = ensure_dir(out_dir / "group_scales_grad")

    snapshot = resolve_local_snapshot(config.model.model_id)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)

    module_names = modules if modules is not None else config.formal_module_names
    rows: list[dict[str, Any]] = []

    for module_name in module_names:
        print(f"[h4_group_diag] {module_name}", flush=True)
        x_cal, a_n_cal = load_x_and_an(capture_dir, module_name, "cal", torch_device)
        x_val, a_n_val = load_x_and_an(capture_dir, module_name, "val", torch_device)
        w_n, bias = load_w_n_bias(snapshot, weight_map, module_name, torch_device)

        result = optimize_one_module(
            module_name=module_name,
            x_cal=x_cal,
            a_n_cal=a_n_cal,
            x_val=x_val,
            a_n_val=a_n_val,
            w_n=w_n,
            bias=bias,
            lr=lr,
            steps=steps,
        )

        stem = module_capture_stem(module_name)
        save_pt(
            scales_dir / f"{stem}.pt",
            {
                "module_name": module_name,
                "d_g": result["d_g"],
                "log2_d_g": result["log2_d_g"],
                "d": result["d"],
                "lr": lr,
                "steps": steps,
                "log2_min": LOG2_MIN,
                "log2_max": LOG2_MAX,
                "capture_run_id": capture_run_id,
            },
        )
        rows.append(
            {
                "module_name": module_name,
                "k_dim": result["k_dim"],
                "n_groups": result["n_groups"],
                "cal_nmse_final": result["cal_nmse_final"],
                "val_nmse_base": result["val_nmse_base"],
                "val_nmse_h4_only": result["val_nmse_h4_only"],
                "val_nmse_grad": result["val_nmse_grad"],
                "val_recovery_vs_base": result["val_recovery_vs_base"],
                "val_recovery_h4_only_vs_base": result["val_recovery_h4_only_vs_base"],
                "val_error_energy_base": result["val_error_energy_base"],
                "val_error_energy_h4_only": result["val_error_energy_h4_only"],
                "val_error_energy_grad": result["val_error_energy_grad"],
                "val_reference_energy": result["val_reference_energy"],
            }
        )
        print(
            f"  val nmse_base={result['val_nmse_base']:.6e} "
            f"h4_only={result['val_nmse_h4_only']:.6e} "
            f"grad={result['val_nmse_grad']:.6e} "
            f"recovery={result['val_recovery_vs_base']:.4f}",
            flush=True,
        )

        del x_cal, a_n_cal, x_val, a_n_val, w_n, bias
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = out_dir / "h4_group_diag_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    global_nmse_base = aggregate_global_nmse(
        [r["val_error_energy_base"] for r in rows],
        [r["val_reference_energy"] for r in rows],
    )
    global_nmse_h4 = aggregate_global_nmse(
        [r["val_error_energy_h4_only"] for r in rows],
        [r["val_reference_energy"] for r in rows],
    )
    global_nmse_grad = aggregate_global_nmse(
        [r["val_error_energy_grad"] for r in rows],
        [r["val_reference_energy"] for r in rows],
    )
    e_base_sum = float(sum(r["val_error_energy_base"] for r in rows))
    e_h4_sum = float(sum(r["val_error_energy_h4_only"] for r in rows))
    e_grad_sum = float(sum(r["val_error_energy_grad"] for r in rows))
    summary = {
        "run_id": run_id,
        "capture_run_id": capture_run_id,
        "model_id": config.model.model_id,
        "num_modules": len(rows),
        "lr": lr,
        "steps": steps,
        "log2_min": LOG2_MIN,
        "log2_max": LOG2_MAX,
        "global_val_nmse_base": global_nmse_base,
        "global_val_nmse_h4_only": global_nmse_h4,
        "global_val_nmse_grad": global_nmse_grad,
        "global_val_recovery_vs_base": recovery_ratio(e_base_sum, e_grad_sum),
        "global_val_recovery_h4_only_vs_base": recovery_ratio(e_base_sum, e_h4_sum),
        "notes": {
            "reference": "Y_B = Linear(A_N, W_N)",
            "baseline": "Y_base = Linear(Q_H(X), Q_H(W_N))",
            "h4_only": "Y_H4 = Linear(Q_H(X R), Q_H(W_N R))",
            "optimized": "Y_A = Linear(Q_H((X R)*d), Q_H((W_N R)/d))",
            "R4": "H4/2 block-diagonal G4",
            "d_g": "one scale per 4 channels after rotation",
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(f"DONE -> {out_dir}", flush=True)
    print(
        f"global val nmse_base={global_nmse_base:.6e} "
        f"h4_only={global_nmse_h4:.6e} "
        f"grad={global_nmse_grad:.6e} "
        f"recovery={summary['global_val_recovery_vs_base']:.4f}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="H4 + 组共享 DIAG：HiF4 路径对齐 NV 参考输出"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--capture-run-id", type=str, default=DEFAULT_CAPTURE_RUN_ID)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--modules", type=str, nargs="*", default=None)
    args = parser.parse_args(argv)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_h4_group_diag_gradient"
    )
    run(
        capture_run_id=args.capture_run_id,
        run_id=run_id,
        config_path=args.config,
        device=args.device,
        lr=args.lr,
        steps=args.steps,
        modules=args.modules,
    )


if __name__ == "__main__":
    main()
