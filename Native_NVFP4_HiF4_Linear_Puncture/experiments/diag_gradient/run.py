#!/usr/bin/env python3
"""梯度下降优化 DIAG：让 HiF4 路径逼近 NVFP4 参考输出。

公式约定
--------
参考（MSE 对齐目标）：
    Y_NN   = Linear(A_N, W_N, b)

无 DIAG 基线：
    Y_base = Linear(Q_H(X), Q_H(W_N), b)

待优化路径（激活乘 d，权重除 d，再各自 HiF4）：
    Y(d)   = Linear(Q_H(X * d), Q_H(W_N / d), b)

损失（仅在 cal 上优化）：
    L = ||Y(d) - Y_NN||^2 / ||Y_NN||^2

共享加载 / STE / 基线见同目录 common.py。
H4 + 组共享 DIAG 见 run_h4_group.py。
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
    aggregate_scale_extremes,
    forward_y_base,
    forward_y_nn,
    linear,
    load_w_n_bias,
    load_x_and_an,
    log2_bound_meta,
    nmse_from_outputs,
    qdq_hif4_ste,
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


def forward_y_diag(
    x: torch.Tensor,
    w_n: torch.Tensor,
    d: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    use_ste: bool,
) -> torch.Tensor:
    """带 DIAG：Q_H(X*d) @ Q_H(W_N/d)^T。"""
    if use_ste:
        a_h = qdq_hif4_ste(x * d)
        w_h = qdq_hif4_ste(w_n / d)
    else:
        a_h = qdq_hif4_direct(x * d, output_dtype=torch.float32)
        w_h = qdq_hif4_direct(w_n / d, output_dtype=torch.float32)
    return linear(a_h, w_h, bias)


def assert_unquant_equivalence(
    x: torch.Tensor,
    w_n: torch.Tensor,
    d: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    """门禁：无量化时 Linear(X*d, W/d) 必须 ≈ Linear(X, W)。"""
    n = min(EQUIV_CHECK_ROWS, x.shape[0])
    x_s = x[:n]
    y0 = linear(x_s, w_n, bias)
    y1 = linear(x_s * d, w_n / d, bias)
    diff = (y0 - y1).float()
    rel = float(diff.norm().item() / max(float(y0.float().norm().item()), 1e-30))
    if rel > EQUIV_REL_L2_MAX:
        raise RuntimeError(
            f"未量化 DIAG 乘积不等价: rel_L2={rel} > {EQUIV_REL_L2_MAX}"
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
    log2_clamp: bool = True,
) -> dict[str, Any]:
    """在 cal 上 Adam 优化 z=log2(d)，在 val 上评基线与优化结果。"""
    k = x_cal.shape[-1]
    device = x_cal.device

    with torch.no_grad():
        y_nn_cal = forward_y_nn(a_n_cal, w_n, bias).detach()
        y_nn_val = forward_y_nn(a_n_val, w_n, bias).detach()
        y_base_val = forward_y_base(x_val, w_n, bias, use_ste=False).detach()
        e_base, ref_e, nmse_base = nmse_from_outputs(y_base_val, y_nn_val)

    z = torch.nn.Parameter(torch.zeros(k, device=device, dtype=torch.float32))
    opt = torch.optim.Adam([z], lr=lr)

    with torch.no_grad():
        d_one = torch.ones(k, device=device, dtype=torch.float32)
        assert_unquant_equivalence(x_cal, w_n, d_one, bias)
        y_d1 = forward_y_diag(x_val, w_n, d_one, bias, use_ste=False)
        d1_err = error_energy(y_d1, y_base_val)
        if d1_err > 1e-6 * max(ref_e, 1.0):
            raise RuntimeError(
                f"d=1 时 Y(d) 与 Y_base 不一致: error_energy={d1_err}"
            )

    history: list[float] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        d = torch.pow(torch.tensor(2.0, device=device), z)
        y = forward_y_diag(x_cal, w_n, d, bias, use_ste=True)
        err = ((y - y_nn_cal) ** 2).sum()
        ref = (y_nn_cal ** 2).sum().clamp_min(1e-30)
        loss = err / ref
        loss.backward()
        opt.step()
        if log2_clamp:
            with torch.no_grad():
                z.clamp_(LOG2_MIN, LOG2_MAX)
        history.append(float(loss.detach().item()))
        if step == 0 or step + 1 == steps or (step + 1) % 50 == 0:
            print(
                f"  [{module_name}] step {step+1}/{steps}  cal_nmse={history[-1]:.6e}",
                flush=True,
            )

    with torch.no_grad():
        d_final = torch.pow(torch.tensor(2.0, device=device), z.detach())
        assert_unquant_equivalence(x_cal, w_n, d_final, bias)
        y_grad_val = forward_y_diag(x_val, w_n, d_final, bias, use_ste=False)
        e_grad, _, nmse_grad = nmse_from_outputs(y_grad_val, y_nn_val)
        recovery = recovery_ratio(e_base, e_grad)

    return {
        "module_name": module_name,
        "k_dim": k,
        "d": d_final.detach().cpu().float(),
        "log2_d": z.detach().cpu().float(),
        "cal_nmse_final": history[-1] if history else float("nan"),
        "cal_nmse_history": history,
        "val_error_energy_base": e_base,
        "val_error_energy_grad": e_grad,
        "val_reference_energy": ref_e,
        "val_nmse_base": nmse_base,
        "val_nmse_grad": nmse_grad,
        "val_recovery_vs_base": recovery,
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
    log2_clamp: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    capture_dir = results_dir(capture_run_id)
    out_dir = ensure_dir(results_dir(run_id))
    scales_dir = ensure_dir(out_dir / "diagonal_scales_grad")
    bound_meta = log2_bound_meta(log2_clamp)

    snapshot = resolve_local_snapshot(config.model.model_id)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)

    module_names = modules if modules is not None else config.formal_module_names
    rows: list[dict[str, Any]] = []
    scale_ds: list[torch.Tensor] = []
    scale_zs: list[torch.Tensor] = []

    for module_name in module_names:
        print(f"[diag_gradient] {module_name}", flush=True)
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
            log2_clamp=log2_clamp,
        )

        stem = module_capture_stem(module_name)
        save_pt(
            scales_dir / f"{stem}.pt",
            {
                "module_name": module_name,
                "d": result["d"],
                "log2_d": result["log2_d"],
                "lr": lr,
                "steps": steps,
                **bound_meta,
                "capture_run_id": capture_run_id,
            },
        )
        scale_ds.append(result["d"])
        scale_zs.append(result["log2_d"])
        rows.append(
            {
                "module_name": module_name,
                "k_dim": result["k_dim"],
                "cal_nmse_final": result["cal_nmse_final"],
                "val_nmse_base": result["val_nmse_base"],
                "val_nmse_grad": result["val_nmse_grad"],
                "val_recovery_vs_base": result["val_recovery_vs_base"],
                "val_error_energy_base": result["val_error_energy_base"],
                "val_error_energy_grad": result["val_error_energy_grad"],
                "val_reference_energy": result["val_reference_energy"],
            }
        )
        print(
            f"  val nmse_base={result['val_nmse_base']:.6e} "
            f"nmse_grad={result['val_nmse_grad']:.6e} "
            f"recovery={result['val_recovery_vs_base']:.4f}",
            flush=True,
        )

        del x_cal, a_n_cal, x_val, a_n_val, w_n, bias
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = out_dir / "diag_gradient_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    global_nmse_base = aggregate_global_nmse(
        [r["val_error_energy_base"] for r in rows],
        [r["val_reference_energy"] for r in rows],
    )
    global_nmse_grad = aggregate_global_nmse(
        [r["val_error_energy_grad"] for r in rows],
        [r["val_reference_energy"] for r in rows],
    )
    e_base_sum = float(sum(r["val_error_energy_base"] for r in rows))
    e_grad_sum = float(sum(r["val_error_energy_grad"] for r in rows))
    summary = {
        "run_id": run_id,
        "capture_run_id": capture_run_id,
        "model_id": config.model.model_id,
        "num_modules": len(rows),
        "lr": lr,
        "steps": steps,
        **bound_meta,
        **aggregate_scale_extremes(scale_ds, scale_zs),
        "global_val_nmse_base": global_nmse_base,
        "global_val_nmse_grad": global_nmse_grad,
        "global_val_recovery_vs_base": recovery_ratio(e_base_sum, e_grad_sum),
        "notes": {
            "reference": "Y_NN = Linear(A_N, W_N)",
            "baseline": "Y_base = Linear(Q_H(X), Q_H(W_N))",
            "optimized": "Y(d) = Linear(Q_H(X*d), Q_H(W_N/d))",
            "A_N_source": "nvfp4_qdq/",
            "X_source": "captures/",
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(f"DONE -> {out_dir}", flush=True)
    print(
        f"global val nmse_base={global_nmse_base:.6e} "
        f"nmse_grad={global_nmse_grad:.6e} "
        f"recovery={summary['global_val_recovery_vs_base']:.4f}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="梯度 DIAG：HiF4 路径对齐 NV 参考输出")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--capture-run-id", type=str, default=DEFAULT_CAPTURE_RUN_ID)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--modules", type=str, nargs="*", default=None)
    parser.add_argument(
        "--no-log2-clamp",
        action="store_true",
        help="关闭 z∈[LOG2_MIN,LOG2_MAX] 钳位（仍用 d=2^z 保正）",
    )
    args = parser.parse_args(argv)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_diag_gradient"
    )
    run(
        capture_run_id=args.capture_run_id,
        run_id=run_id,
        config_path=args.config,
        device=args.device,
        lr=args.lr,
        steps=args.steps,
        modules=args.modules,
        log2_clamp=not args.no_log2_clamp,
    )


if __name__ == "__main__":
    main()
