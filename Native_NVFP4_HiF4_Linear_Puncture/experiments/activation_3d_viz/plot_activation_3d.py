"""3D surface plots of saved val activations in each quantizer domain.

Does not recapture, does not search DIAG, and does not change HiF4 / NVFP4 math.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from Native_NVFP4_HiF4_Linear_Puncture.experiments.h4_block_rotation.h4_transform import (
    HIF4_GROUP_SIZE,
    apply_h4_g4,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.config import (
    EXPERIMENT_ROOT,
    AppConfig,
    load_config,
    results_dir,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct
from Native_NVFP4_HiF4_Linear_Puncture.src.grid_scale_validation import (
    capture_file_path,
    validate_capture_manifest,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import (
    ensure_dir,
    load_pt,
    module_capture_stem,
    read_json,
    write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.native_nvfp4 import qdq_nvfp4_post_rotation

DEFAULT_CAPTURE_RUN_ID = "20260812T103800Z_native_nvfp4_hif4_linear_puncture"
EVAL_SPLIT = "val"
NUM_TOKENS = 200
GROUP_SEED = 20260813
FULL_CSTRIDE = 4
DEFAULT_VIZ_LAYERS = (2, 10, 18)
CASES = (
    "01_xrot",
    "02_nvfp4",
    "03_diag_hif4",
    "04_h4_hif4",
    "05_hif4_direct",
    "06_residual_hif4_minus_nvfp4",
)


def _layer_idx(module_name: str) -> int:
    parts = module_name.split(".")
    return int(parts[parts.index("layers") + 1])


def select_module_names(
    config: AppConfig,
    *,
    smoke: bool,
    layers: tuple[int, ...] | list[int] | None,
) -> list[str]:
    if smoke:
        return [config.formal_module_names[0]]
    wanted = tuple(int(x) for x in (layers if layers is not None else DEFAULT_VIZ_LAYERS))
    missing_cfg = [idx for idx in wanted if idx not in config.experiment.formal_layers]
    if missing_cfg:
        raise ValueError(
            f"layers {missing_cfg} are not in config formal_layers "
            f"{list(config.experiment.formal_layers)}"
        )
    names = [name for name in config.formal_module_names if _layer_idx(name) in wanted]
    if not names:
        raise ValueError(f"no modules for layers {list(wanted)}")
    return names


def viz_results_dir(run_id: str) -> Path:
    return EXPERIMENT_ROOT / "results" / "activation_3d_viz" / run_id


def _module_seed(module_name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{module_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def select_group_id(module_name: str, num_groups: int, seed: int = GROUP_SEED) -> int:
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    rng = np.random.RandomState(_module_seed(module_name, seed))
    return int(rng.randint(0, num_groups))


def _to_numpy(matrix: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        array = matrix.detach().to(dtype=torch.float32, device="cpu").numpy()
    else:
        array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"plot input must be 2D, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise RuntimeError("plot input contains NaN or Inf")
    return array.astype(np.float64, copy=False)


def plot_3d_matrix(
    matrix: torch.Tensor | np.ndarray,
    path: Path,
    *,
    title: str,
    xlabel: str = "K",
    ylabel: str = "Token",
    zlabel: str = "Value",
    cmap: str = "coolwarm",
    group_size: int | None = HIF4_GROUP_SIZE,
    rstride: int = 1,
    cstride: int = 1,
    k_offset: int = 0,
    vmax_quantile: float | None = None,
) -> None:
    """Save a 3D surface of |matrix|. X is K, Y is token. Optional HiF4 group ticks every 64.

    If ``vmax_quantile`` is set (e.g. 0.99), color scale uses that quantile of |values|
    while the geometric Z axis still spans the true max, so rare spikes keep height but
    map into the top of the colormap.
    """
    values = np.abs(_to_numpy(matrix))
    num_tokens, k_dim = values.shape
    x_coords = np.arange(k_offset, k_offset + k_dim)
    y_coords = np.arange(num_tokens)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    fig = plt.figure(figsize=(14, 8) if k_dim > 64 else (10, 7))
    ax = fig.add_subplot(111, projection="3d")
    zmin = 0.0
    zmax = float(values.max())
    if zmax == 0.0:
        raise RuntimeError(f"{title}: absolute surface is identically zero")
    if vmax_quantile is None:
        color_max = zmax
    else:
        if not (0.0 < float(vmax_quantile) < 1.0):
            raise ValueError(f"vmax_quantile must be in (0,1), got {vmax_quantile}")
        color_max = float(np.quantile(values.reshape(-1), float(vmax_quantile)))
        if color_max <= 0.0:
            raise RuntimeError(f"{title}: vmax_quantile color max is non-positive")
    norm = Normalize(vmin=zmin, vmax=color_max)
    ax.set_zlim(zmin, zmax)
    z_floor = zmin

    ax.plot_surface(
        x_grid,
        y_grid,
        values,
        cmap=cmap,
        norm=norm,
        rstride=rstride,
        cstride=cstride,
        linewidth=0,
        antialiased=False,
        shade=True,
    )

    if group_size is not None and group_size > 0:
        start = ((k_offset + group_size - 1) // group_size) * group_size
        end = k_offset + k_dim
        boundaries = list(range(start, end + 1, group_size))
        if k_offset not in boundaries:
            boundaries.insert(0, k_offset)
        if end not in boundaries:
            boundaries.append(end)
        y0 = 0.0
        y1 = float(num_tokens - 1)
        for b in boundaries:
            ax.plot([b, b], [y0, y1], [z_floor, z_floor], color="black", linewidth=0.5, alpha=0.7)
        ax.set_xticks(boundaries)
        ax.tick_params(axis="x", labelsize=6, labelrotation=90)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _short_module(module_name: str) -> str:
    parts = module_name.split(".")
    layer = parts[parts.index("layers") + 1]
    return f"L{int(layer):02d}.{parts[-1]}"


def _assert_k_aligned(k_dim: int) -> None:
    if k_dim % HIF4_GROUP_SIZE != 0:
        raise ValueError(f"K={k_dim} is not divisible by HiF4 group size {HIF4_GROUP_SIZE}")


def build_case_tensors(
    x_rot_bf16: torch.Tensor,
    *,
    input_global_scale: torch.Tensor,
    diag_d: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    x = x_rot_bf16.to(device=device, dtype=torch.float32)
    scale = input_global_scale.to(device=device, dtype=torch.float32)
    d = diag_d.to(device=device, dtype=torch.float32)
    if d.ndim != 1 or d.numel() != x.shape[-1]:
        raise RuntimeError(f"DIAG d shape {tuple(d.shape)} != K={x.shape[-1]}")

    a_n = qdq_nvfp4_post_rotation(x_rot_bf16.to(device=device), scale).to(torch.float32)
    a_h = qdq_hif4_direct(x, output_dtype=torch.float32)
    a_diag = qdq_hif4_direct(x / d, output_dtype=torch.float32)
    x_h4 = apply_h4_g4(x, compute_dtype=torch.float32, output_dtype=torch.float32)
    a_h4 = qdq_hif4_direct(x_h4, output_dtype=torch.float32)
    residual = a_h - a_n
    for name, tensor in (
        ("xrot", x),
        ("nvfp4", a_n),
        ("diag_hif4", a_diag),
        ("h4_hif4", a_h4),
        ("hif4_direct", a_h),
        ("residual", residual),
    ):
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains NaN or Inf")
        if tuple(tensor.shape) != tuple(x.shape):
            raise RuntimeError(f"{name} shape {tuple(tensor.shape)} != {tuple(x.shape)}")
    return {
        "01_xrot": x,
        "02_nvfp4": a_n,
        "03_diag_hif4": a_diag,
        "04_h4_hif4": a_h4,
        "05_hif4_direct": a_h,
        "06_residual_hif4_minus_nvfp4": residual,
    }


CASE_TITLES = {
    "01_xrot": "|X_rot| (no quant)",
    "02_nvfp4": "|Q_NVFP4(X_rot)|",
    "03_diag_hif4": "|Q_H(X_rot / d)|",
    "04_h4_hif4": "|Q_H(X_rot R4)|",
    "05_hif4_direct": "|Q_H(X_rot)|",
    "06_residual_hif4_minus_nvfp4": "|Q_H(X_rot) - Q_NVFP4(X_rot)|",
}


STYLE_SPECS = {
    "coolwarm_max": {
        "cmap": "coolwarm",
        "vmax_quantile": None,
        "title_note": "coolwarm vmax=max",
    },
    "viridis_max": {
        "cmap": "viridis",
        "vmax_quantile": None,
        "title_note": "viridis vmax=max",
    },
    "coolwarm_p99": {
        "cmap": "coolwarm",
        "vmax_quantile": 0.99,
        "title_note": "coolwarm vmax=p99",
    },
}


def expected_png_names(group_id: int) -> list[str]:
    names: list[str] = []
    for case in CASES:
        names.append(f"{case}_full.png")
        names.append(f"{case}_group{group_id:03d}.png")
    return names


def plot_module_cases(
    *,
    module_name: str,
    tensors: dict[str, torch.Tensor],
    group_id: int,
    out_dir: Path,
    style: str = "coolwarm_max",
) -> None:
    if style not in STYLE_SPECS:
        raise ValueError(f"unknown style {style!r}, expected one of {sorted(STYLE_SPECS)}")
    style_cfg = STYLE_SPECS[style]
    cmap = str(style_cfg["cmap"])
    vmax_quantile = style_cfg["vmax_quantile"]
    title_note = str(style_cfg["title_note"])

    k_dim = int(next(iter(tensors.values())).shape[-1])
    _assert_k_aligned(k_dim)
    k0 = group_id * HIF4_GROUP_SIZE
    k1 = k0 + HIF4_GROUP_SIZE
    label = _short_module(module_name)
    for case in CASES:
        matrix = tensors[case]
        plot_3d_matrix(
            matrix,
            out_dir / f"{case}_full.png",
            title=f"{label}  {CASE_TITLES[case]}  full K={k_dim}  [{title_note}]",
            xlabel="K (tick every HiF4 group of 64)",
            ylabel="Token",
            zlabel="|value|",
            cmap=cmap,
            group_size=HIF4_GROUP_SIZE,
            rstride=1,
            cstride=FULL_CSTRIDE,
            vmax_quantile=vmax_quantile,
        )
        plot_3d_matrix(
            matrix[:, k0:k1],
            out_dir / f"{case}_group{group_id:03d}.png",
            title=f"{label}  {CASE_TITLES[case]}  group {group_id}  K[{k0}:{k1}]  [{title_note}]",
            xlabel=f"K in group {group_id}",
            ylabel="Token",
            zlabel="|value|",
            cmap=cmap,
            group_size=None,
            rstride=1,
            cstride=1,
            k_offset=k0,
            vmax_quantile=vmax_quantile,
        )


def run_activation_3d_viz(
    config: AppConfig,
    *,
    capture_run_id: str,
    run_id: str,
    device: str,
    smoke: bool,
    layers: tuple[int, ...] | list[int] | None = None,
    styles: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    capture_dir = results_dir(capture_run_id)
    manifest_path = capture_dir / "capture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"saved activations not found: {manifest_path}. Refusing to recapture."
        )
    manifest = read_json(manifest_path)
    validate_capture_manifest(config, capture_dir, manifest)

    module_names = select_module_names(config, smoke=smoke, layers=layers)
    style_names = list(styles) if styles is not None else ["coolwarm_max"]
    unknown = [s for s in style_names if s not in STYLE_SPECS]
    if unknown:
        raise ValueError(f"unknown styles {unknown}, expected one of {sorted(STYLE_SPECS)}")
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)

    out_dir = ensure_dir(viz_results_dir(run_id))
    fig_root = ensure_dir(out_dir / "figures")
    write_json(
        out_dir / "config.json",
        {
            "run_id": run_id,
            "capture_run_id": capture_run_id,
            "smoke": smoke,
            "split": EVAL_SPLIT,
            "num_tokens": NUM_TOKENS,
            "group_seed": GROUP_SEED,
            "hif4_group_size": HIF4_GROUP_SIZE,
            "full_cstride": FULL_CSTRIDE,
            "plot_absolute": True,
            "styles": style_names,
            "style_specs": {name: STYLE_SPECS[name] for name in style_names},
            "layers": sorted({_layer_idx(n) for n in module_names}),
            "cases": list(CASES),
            "modules": module_names,
            "device": str(torch_device),
            "coordinate_note": (
                "surfaces are absolute values; DIAG and H4 are in the quantizer domain; "
                "residual is |Q_H(X_rot) - Q_NVFP4(X_rot)| in X_rot coordinates; "
                "coolwarm_p99 keeps geometric zmax=max but colors with vmax=p99"
            ),
        },
    )

    selected: dict[str, Any] = {
        "seed": GROUP_SEED,
        "num_tokens": NUM_TOKENS,
        "split": EVAL_SPLIT,
        "styles": style_names,
        "modules": {},
    }

    for module_name in module_names:
        print(f"[viz] {module_name}", flush=True)
        cap_path = capture_file_path(capture_dir, module_name, EVAL_SPLIT)
        if not cap_path.is_file():
            raise FileNotFoundError(f"missing capture {cap_path}. Refusing to recapture.")
        diag_path = capture_dir / "diagonal_scales" / f"{module_capture_stem(module_name)}.pt"
        if not diag_path.is_file():
            raise FileNotFoundError(
                f"missing DIAG scale {diag_path}. Refusing to re-search DIAG."
            )
        capture = load_pt(cap_path, map_location="cpu")
        if capture["module_name"] != module_name:
            raise RuntimeError(
                f"capture module_name {capture['module_name']!r} != {module_name!r}"
            )
        x_all = capture["x_rot_bf16"]
        if x_all.ndim != 2:
            raise ValueError(f"{module_name}: expected 2D X_rot, got {tuple(x_all.shape)}")
        if x_all.shape[0] < NUM_TOKENS:
            raise RuntimeError(
                f"{module_name}: need {NUM_TOKENS} val tokens, got {x_all.shape[0]}"
            )
        k_dim = int(x_all.shape[-1])
        _assert_k_aligned(k_dim)
        num_groups = k_dim // HIF4_GROUP_SIZE
        group_id = select_group_id(module_name, num_groups, GROUP_SEED)
        stem = module_capture_stem(module_name)
        needed = expected_png_names(group_id)
        pending_styles = []
        for style in style_names:
            style_dir = ensure_dir(fig_root / style / stem)
            if all((style_dir / name).is_file() for name in needed):
                print(f"[viz] skip existing {style}/{stem}", flush=True)
            else:
                pending_styles.append(style)

        if pending_styles:
            x_rot = x_all[:NUM_TOKENS]
            diag_obj = load_pt(diag_path, map_location="cpu")
            tensors = build_case_tensors(
                x_rot,
                input_global_scale=capture["input_global_scale_fp32"],
                diag_d=diag_obj["d"].to(torch.float32),
                device=torch_device,
            )
            for style in pending_styles:
                style_dir = ensure_dir(fig_root / style / stem)
                print(f"[viz] plot {style}/{stem}", flush=True)
                plot_module_cases(
                    module_name=module_name,
                    tensors=tensors,
                    group_id=group_id,
                    out_dir=style_dir,
                    style=style,
                )
            del tensors, x_rot
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

        style_written = {
            style: sorted(p.name for p in (fig_root / style / stem).glob("*.png"))
            for style in style_names
        }
        selected["modules"][module_name] = {
            "stem": stem,
            "capture_path": str(cap_path),
            "diag_path": str(diag_path),
            "num_rows_used": NUM_TOKENS,
            "k_dim": k_dim,
            "num_groups": num_groups,
            "group_id": group_id,
            "k_slice": [group_id * HIF4_GROUP_SIZE, (group_id + 1) * HIF4_GROUP_SIZE],
            "written_by_style": style_written,
            "skipped_existing_styles": [
                s for s in style_names if s not in pending_styles
            ],
        }
        write_json(out_dir / "selected_groups.json", selected)
        del capture, x_all

    write_json(out_dir / "selected_groups.json", selected)
    expected_png = 12 * len(module_names) * len(style_names)
    written_png = list((out_dir / "figures").glob("*/*/*.png"))
    if len(written_png) != expected_png:
        raise RuntimeError(
            f"expected {expected_png} png files, found {len(written_png)}"
        )
    print(f"ACTIVATION 3D VIZ DONE smoke={smoke} -> {out_dir}", flush=True)
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D surfaces of saved activations")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--capture-run-id", type=str, default=DEFAULT_CAPTURE_RUN_ID)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_VIZ_LAYERS),
        help="Which formal layers to plot. Default: 2 10 18",
    )
    parser.add_argument(
        "--styles",
        type=str,
        nargs="+",
        default=["coolwarm_max"],
        choices=sorted(STYLE_SPECS),
        help="Color styles. Default: coolwarm_max",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    run_activation_3d_viz(
        config,
        capture_run_id=args.capture_run_id,
        run_id=args.run_id,
        device=args.device,
        smoke=bool(args.smoke),
        layers=args.layers,
        styles=args.styles,
    )


if __name__ == "__main__":
    main()
