#!/usr/bin/env python3
"""P3: dense packed-linear parity (q/k/v/o) on frozen inputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.nvfp4_operator_parity.common import (
    RESULT_ROOT,
    channel_topk_abs,
    compare_tensors,
    ensure_cuda_lut,
    load_raw_linear_parts,
    scalar_f32,
    write_jsonl,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    ref_nvfp4_quant_dequant,
    run_nvfp4_emulations,
)


ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")


def iter_packs(result_root: Path):
    manifest = json.loads((result_root / "frozen_inputs_manifest.json").read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        for entry in sample["layers"]:
            path = result_root / entry["path"]
            pack = torch.load(path, map_location="cpu", weights_only=False)
            yield pack, path


def _run_paths(
    x: torch.Tensor,
    parts: dict[str, torch.Tensor],
    input_inv: float,
    weight_global: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.to(device=device, dtype=torch.bfloat16).reshape(1, -1)
    inv = torch.tensor(input_inv, device=device, dtype=torch.float32)
    gw = torch.tensor(weight_global, device=device, dtype=torch.float32)
    packed = parts["weight"].to(device)
    scale = parts["weight_scale"].to(device)

    x_qdq = ref_nvfp4_quant_dequant(x, inv, block_size=16)
    w_fp32 = dequantize_to_dtype(
        packed.view(torch.uint8),
        scale,
        gw,
        dtype=torch.float32,
        block_size=16,
        swizzle=False,
    )
    # D0 mirrors native_nvfp4_linear: F.linear on BF16-cast FP32 weight.
    d0 = F.linear(x_qdq, w_fp32.to(dtype=x_qdq.dtype))
    d1 = run_nvfp4_emulations(
        x=x,
        input_global_scale=inv,
        weight=packed,
        weight_scale_swizzled=scale,
        weight_global_scale=gw,
        swizzle=False,
    )
    w_bf16 = dequantize_to_dtype(
        packed.view(torch.uint8),
        scale,
        gw,
        dtype=torch.bfloat16,
        block_size=16,
        swizzle=False,
    )
    d2 = torch.matmul(x_qdq, w_bf16.t())
    return d0, d1, d2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=str(RESULT_ROOT))
    p.add_argument("--model_path", default="nvidia/Qwen3-30B-A3B-NVFP4")
    args = p.parse_args()

    device = torch.device(args.device)
    ensure_cuda_lut(device)
    out = Path(args.output_dir)
    snapshot = Path(resolve_local_snapshot(args.model_path))
    rows: list[dict] = []

    # Cache packed weights per layer.
    cache: dict[int, dict] = {}

    for pack, path in iter_packs(out):
        layer = int(pack["layer"])
        if layer not in cache:
            attn = {
                proj: load_raw_linear_parts(snapshot, f"model.layers.{layer}.self_attn.{proj}")
                for proj in ATTN_PROJS
            }
            qkv_w = max(scalar_f32(attn[p]["weight_scale_2"]) for p in ("q_proj", "k_proj", "v_proj"))
            qkv_in = float(pack["scales"]["qkv_input_inv"])
            o_in = float(pack["scales"]["o_input_inv"])
            o_w = scalar_f32(attn["o_proj"]["weight_scale_2"])
            cache[layer] = {
                "attn": attn,
                "qkv_w": qkv_w,
                "qkv_in": qkv_in,
                "o_in": o_in,
                "o_w": o_w,
            }
        cfg = cache[layer]
        # Prefer pack scales (frozen) for activation; weight globals from ckpt collapse.
        qkv_in = float(pack["scales"]["qkv_input_inv"])
        o_in = float(pack["scales"]["o_input_inv"])

        for proj in ATTN_PROJS:
            x = pack["qkv_input"] if proj in ("q_proj", "k_proj", "v_proj") else pack["o_input"]
            input_inv = qkv_in if proj in ("q_proj", "k_proj", "v_proj") else o_in
            weight_global = cfg["qkv_w"] if proj in ("q_proj", "k_proj", "v_proj") else cfg["o_w"]
            d0, d1, d2 = _run_paths(x, cfg["attn"][proj], input_inv, weight_global, device)
            pairs = {
                "D0_D1": compare_tensors(d0, d1),
                "D0_D2": compare_tensors(d0, d2),
                "D1_D2": compare_tensors(d1, d2),
            }
            first_nonzero = None
            for name in ("D0_D1", "D0_D2", "D1_D2"):
                if pairs[name]["max_abs"] != 0.0 or pairs[name]["exact_fraction"] != 1.0:
                    first_nonzero = name
                    break
            row = {
                "prompt_key": pack["prompt_key"],
                "decode_index": int(pack["decode_index"]),
                "layer": layer,
                "bucket": pack["bucket"],
                "proj": proj,
                "pack_path": str(path.relative_to(out)),
                "first_nonzero_pair": first_nonzero,
                "channel_topk_D0_D1": channel_topk_abs(d0 - d1),
                "channel_topk_D0_D2": channel_topk_abs(d0 - d2),
                "channel_topk_D1_D2": channel_topk_abs(d1 - d2),
            }
            for name, metrics in pairs.items():
                for k, v in metrics.items():
                    row[f"{name}_{k}"] = v
            rows.append(row)
        print(
            f"[P3] {pack['prompt_key']} decode={pack['decode_index']} layer={layer}",
            flush=True,
        )

    write_jsonl(out / "P3_dense_linear_rows.jsonl", rows)

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)
    nonzero = [r for r in rows if r["first_nonzero_pair"] is not None]
    first_overall = None
    for name in ("D0_D1", "D0_D2", "D1_D2"):
        if any(r["first_nonzero_pair"] == name for r in rows) or any(
            r[f"{name}_max_abs"] != 0.0 for r in rows
        ):
            # Prefer earliest pair that ever differs.
            if any(r[f"{name}_max_abs"] != 0.0 or r[f"{name}_exact_fraction"] != 1.0 for r in rows):
                first_overall = name
                break

    md = [
        "# P3 dense packed-linear parity",
        "",
        f"- rows: {len(rows)}",
        f"- nonzero_rows: {len(nonzero)}",
        f"- first_nonzero_pair_overall: **{first_overall}**",
        "",
        "## Pair summary",
        "",
        "| pair | max_abs_max | rel_l2_max | exact_fraction_min |",
        "|---|---:|---:|---:|",
    ]
    for name in ("D0_D1", "D0_D2", "D1_D2"):
        md.append(
            f"| {name} | {max(r[f'{name}_max_abs'] for r in rows):.6g} | "
            f"{max(r[f'{name}_rel_l2'] for r in rows):.6g} | "
            f"{min(r[f'{name}_exact_fraction'] for r in rows):.6g} |"
        )
    md.extend(["", "## By bucket / proj", "", "| bucket | proj | n | D0_D1_max_abs | D0_D2_max_abs | D1_D2_max_abs |", "|---|---|---:|---:|---:|---:|"])
    for bucket in ("focus_low_margin", "uniform_control", "post_tp_divergence_control_only"):
        for proj in ATTN_PROJS:
            group = [r for r in by_bucket.get(bucket, []) if r["proj"] == proj]
            if not group:
                continue
            md.append(
                f"| {bucket} | {proj} | {len(group)} | "
                f"{max(r['D0_D1_max_abs'] for r in group):.6g} | "
                f"{max(r['D0_D2_max_abs'] for r in group):.6g} | "
                f"{max(r['D1_D2_max_abs'] for r in group):.6g} |"
            )
    md.append("")
    (out / "P3_dense_linear_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "first_nonzero_pair": first_overall}, indent=2))


if __name__ == "__main__":
    main()
