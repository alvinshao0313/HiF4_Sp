"""Offline NVFP4 activation QDQ on saved X_rot captures using ckpt act_global_scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import (
    resolve_local_snapshot,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.config import (
    CHECKPOINT_KEY_SCHEMA,
    AppConfig,
    load_config,
    results_dir,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import (
    ensure_dir,
    load_pt,
    module_capture_stem,
    save_pt,
    write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.native_nvfp4 import qdq_nvfp4_post_rotation

SPLITS = ("cal", "val")
OUT_SUBDIR = "nvfp4_qdq"
SCHEMA_VERSION = 1


def _load_act_global_scale(
    snapshot: Path,
    weight_map: dict[str, str],
    module_name: str,
) -> torch.Tensor:
    from safetensors import safe_open

    physical = CHECKPOINT_KEY_SCHEMA["input_global_scale"]
    key = f"{module_name}.{physical}"
    if key not in weight_map:
        raise KeyError(f"missing {key}")
    shard = snapshot / weight_map[key]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        scale = handle.get_tensor(key)
    return scale.detach().reshape(()).to(torch.float32)


def _assert_scale_match(
    *,
    module_name: str,
    split: str,
    ckpt_scale: torch.Tensor,
    capture_scale: torch.Tensor,
) -> None:
    ckpt = float(ckpt_scale.reshape(()).item())
    cap = float(capture_scale.reshape(()).item())
    if ckpt != cap:
        raise RuntimeError(
            "act_global_scale mismatch between ckpt and capture: "
            f"module={module_name} split={split} ckpt={ckpt!r} capture={cap!r}"
        )


def run_fake_quant_captures(
    config: AppConfig,
    capture_run_id: str,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    run_dir = results_dir(capture_run_id)
    capture_dir = run_dir / "captures"
    if not capture_dir.is_dir():
        raise FileNotFoundError(f"captures directory missing: {capture_dir}")

    out_dir = ensure_dir(run_dir / OUT_SUBDIR)
    group_size = int(config.nvfp4.activation_group_size)

    snapshot = resolve_local_snapshot(config.model.model_id)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)

    written: list[dict[str, Any]] = []
    scale_cache: dict[str, torch.Tensor] = {}

    for module_name in config.formal_module_names:
        if module_name not in scale_cache:
            scale_cache[module_name] = _load_act_global_scale(
                snapshot, weight_map, module_name
            )
        ckpt_scale = scale_cache[module_name]
        stem = module_capture_stem(module_name)

        for split in SPLITS:
            src_name = f"{stem}_{split}.pt"
            src_path = capture_dir / src_name
            if not src_path.is_file():
                raise FileNotFoundError(f"missing capture: {src_path}")

            capture = load_pt(src_path, map_location="cpu")
            if capture.get("module_name") != module_name:
                raise RuntimeError(
                    f"module_name mismatch in {src_path}: "
                    f"expected {module_name!r}, got {capture.get('module_name')!r}"
                )

            capture_scale = capture["input_global_scale_fp32"]
            _assert_scale_match(
                module_name=module_name,
                split=split,
                ckpt_scale=ckpt_scale,
                capture_scale=capture_scale,
            )

            x_rot = capture["x_rot_bf16"].to(device=torch_device)
            scale = ckpt_scale.to(device=torch_device, dtype=torch.float32)
            a_n = qdq_nvfp4_post_rotation(
                x_rot, scale, group_size=group_size
            ).to(device="cpu", dtype=torch.bfloat16)

            if tuple(a_n.shape) != tuple(capture["x_rot_bf16"].shape):
                raise RuntimeError(
                    f"shape mismatch after QDQ for {src_name}: "
                    f"a_n={tuple(a_n.shape)} x_rot={tuple(capture['x_rot_bf16'].shape)}"
                )

            out_path = out_dir / src_name
            payload = {
                "schema_version": SCHEMA_VERSION,
                "module_name": module_name,
                "layer_idx": int(capture["layer_idx"]),
                "projection": str(capture["projection"]),
                "split": split,
                "a_n_bf16": a_n,
                "input_global_scale_fp32": ckpt_scale.detach().cpu().float().reshape(()),
                "source_capture": str(src_path.relative_to(run_dir)),
                "group_size": group_size,
            }
            save_pt(out_path, payload)
            written.append(
                {
                    "module_name": module_name,
                    "split": split,
                    "path": str(out_path.relative_to(run_dir)),
                    "shape": list(a_n.shape),
                    "input_global_scale_fp32": float(ckpt_scale.item()),
                }
            )
            print(f"[fake_quant] wrote {out_path.name} shape={tuple(a_n.shape)}", flush=True)
            del x_rot, a_n

    manifest = {
        "capture_run_id": capture_run_id,
        "model_id": config.model.model_id,
        "snapshot_path": str(snapshot),
        "group_size": group_size,
        "num_files": len(written),
        "expected_files": len(config.formal_module_names) * len(SPLITS),
        "out_subdir": OUT_SUBDIR,
        "written": written,
    }
    if manifest["num_files"] != manifest["expected_files"]:
        raise RuntimeError(
            f"expected {manifest['expected_files']} outputs, got {manifest['num_files']}"
        )
    write_json(run_dir / "nvfp4_qdq_manifest.json", manifest)
    print(f"FAKE QUANT DONE -> {out_dir} ({manifest['num_files']} files)", flush=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="NVFP4 fake-quant saved X_rot captures with ckpt act_global_scale"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--capture-run-id",
        type=str,
        default="20260812T103800Z_native_nvfp4_hif4_linear_puncture",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    run_fake_quant_captures(config, args.capture_run_id, device=args.device)


if __name__ == "__main__":
    main()
