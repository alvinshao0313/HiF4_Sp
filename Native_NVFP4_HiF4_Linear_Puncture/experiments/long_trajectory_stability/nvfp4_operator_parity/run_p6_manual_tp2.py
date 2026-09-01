#!/usr/bin/env python3
"""P6: manual TP2 partition / reduction parity on frozen inputs."""

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
)


WORLD_SIZE = 2
COLUMN_PROJS = ("q_proj", "k_proj", "v_proj")
ROW_PROJS = ("o_proj",)


def iter_packs(result_root: Path):
    manifest = json.loads((result_root / "frozen_inputs_manifest.json").read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        for entry in sample["layers"]:
            path = result_root / entry["path"]
            pack = torch.load(path, map_location="cpu", weights_only=False)
            yield pack, path


def _dq_weight(parts: dict, weight_global: float, device: torch.device) -> torch.Tensor:
    gw = torch.tensor(weight_global, device=device, dtype=torch.float32)
    return dequantize_to_dtype(
        parts["weight"].to(device).view(torch.uint8),
        parts["weight_scale"].to(device),
        gw,
        dtype=torch.float32,
        block_size=16,
        swizzle=False,
    )


def _qdq(x: torch.Tensor, input_inv: float, device: torch.device) -> torch.Tensor:
    x = x.to(device=device, dtype=torch.bfloat16).reshape(1, -1)
    inv = torch.tensor(input_inv, device=device, dtype=torch.float32)
    return ref_nvfp4_quant_dequant(x, inv, block_size=16)


def _derive_shards(w: torch.Tensor, mode: str) -> dict:
    """Derive TP2 shard dims from dequant weight shape (out, in)."""
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    if mode == "column":
        if out_f % WORLD_SIZE != 0:
            raise ValueError(f"column-parallel out_features={out_f} not divisible by {WORLD_SIZE}")
        shard = out_f // WORLD_SIZE
        return {
            "mode": "column_parallel",
            "world_size": WORLD_SIZE,
            "weight_shape_out_in": [out_f, in_f],
            "split_dim": "out",
            "shard_out": shard,
            "shard_in": in_f,
            "derivation": f"column-parallel: split out={out_f} into {WORLD_SIZE}x{shard}; GEMM each then concat",
        }
    if in_f % WORLD_SIZE != 0:
        raise ValueError(f"row-parallel in_features={in_f} not divisible by {WORLD_SIZE}")
    shard = in_f // WORLD_SIZE
    return {
        "mode": "row_parallel",
        "world_size": WORLD_SIZE,
        "weight_shape_out_in": [out_f, in_f],
        "split_dim": "in",
        "shard_out": out_f,
        "shard_in": shard,
        "derivation": f"row-parallel: split in={in_f} into {WORLD_SIZE}x{shard}; partial GEMM then sum",
    }


def _column_tp2(x_qdq: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out_f = w.shape[0]
    shard = out_f // WORLD_SIZE
    parts = []
    for rank in range(WORLD_SIZE):
        w_s = w[rank * shard : (rank + 1) * shard]
        parts.append(F.linear(x_qdq, w_s.to(dtype=x_qdq.dtype)))
    return torch.cat(parts, dim=-1)


def _row_tp2(x_qdq: torch.Tensor, w: torch.Tensor, reduce_dtype: torch.dtype) -> torch.Tensor:
    in_f = w.shape[1]
    shard = in_f // WORLD_SIZE
    acc = None
    for rank in range(WORLD_SIZE):
        x_s = x_qdq[:, rank * shard : (rank + 1) * shard]
        w_s = w[:, rank * shard : (rank + 1) * shard]
        partial = F.linear(x_s, w_s.to(dtype=x_qdq.dtype)).to(reduce_dtype)
        acc = partial if acc is None else acc + partial
    return acc.to(x_qdq.dtype)


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
    cache: dict[int, dict] = {}

    for pack, path in iter_packs(out):
        layer = int(pack["layer"])
        if layer not in cache:
            attn = {
                proj: load_raw_linear_parts(snapshot, f"model.layers.{layer}.self_attn.{proj}")
                for proj in COLUMN_PROJS + ROW_PROJS
            }
            qkv_w = max(scalar_f32(attn[p]["weight_scale_2"]) for p in COLUMN_PROJS)
            cache[layer] = {
                "attn": attn,
                "qkv_w": qkv_w,
                "o_w": scalar_f32(attn["o_proj"]["weight_scale_2"]),
            }
        cfg = cache[layer]
        qkv_in = float(pack["scales"]["qkv_input_inv"])
        o_in = float(pack["scales"]["o_input_inv"])

        for proj in COLUMN_PROJS:
            w = _dq_weight(cfg["attn"][proj], cfg["qkv_w"], device)
            shard_meta = _derive_shards(w, "column")
            x_qdq = _qdq(pack["qkv_input"], qkv_in, device)
            full = F.linear(x_qdq, w.to(dtype=x_qdq.dtype))
            tp2 = _column_tp2(x_qdq, w)
            metrics = compare_tensors(full, tp2)
            rows.append(
                {
                    "prompt_key": pack["prompt_key"],
                    "decode_index": int(pack["decode_index"]),
                    "layer": layer,
                    "bucket": pack["bucket"],
                    "proj": proj,
                    "parallel": "column",
                    "reduce_dtype": None,
                    "pack_path": str(path.relative_to(out)),
                    **{f"shard_{k}": v for k, v in shard_meta.items()},
                    **metrics,
                }
            )

        for proj in ROW_PROJS:
            w = _dq_weight(cfg["attn"][proj], cfg["o_w"], device)
            shard_meta = _derive_shards(w, "row")
            x_qdq = _qdq(pack["o_input"], o_in, device)
            full = F.linear(x_qdq, w.to(dtype=x_qdq.dtype))
            for reduce_dtype, tag in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
                tp2 = _row_tp2(x_qdq, w, reduce_dtype)
                metrics = compare_tensors(full, tp2)
                rows.append(
                    {
                        "prompt_key": pack["prompt_key"],
                        "decode_index": int(pack["decode_index"]),
                        "layer": layer,
                        "bucket": pack["bucket"],
                        "proj": proj,
                        "parallel": "row",
                        "reduce_dtype": tag,
                        "pack_path": str(path.relative_to(out)),
                        **{f"shard_{k}": v for k, v in shard_meta.items()},
                        **metrics,
                    }
                )

        # MoE down (row-parallel on intermediate) for first routed expert as control.
        if pack["experts"]:
            expert = pack["experts"][0]
            eid = int(expert["expert_id"])
            parts = load_raw_linear_parts(
                snapshot, f"model.layers.{layer}.mlp.experts.{eid}.down_proj"
            )
            w = _dq_weight(parts, float(expert["w2_global"]), device)
            shard_meta = _derive_shards(w, "row")
            x_qdq = _qdq(expert["w2_input"], float(expert["a2_inv"]), device)
            full = F.linear(x_qdq, w.to(dtype=x_qdq.dtype))
            for reduce_dtype, tag in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
                tp2 = _row_tp2(x_qdq, w, reduce_dtype)
                metrics = compare_tensors(full, tp2)
                rows.append(
                    {
                        "prompt_key": pack["prompt_key"],
                        "decode_index": int(pack["decode_index"]),
                        "layer": layer,
                        "bucket": pack["bucket"],
                        "proj": "down_proj",
                        "expert_id": eid,
                        "parallel": "row",
                        "reduce_dtype": tag,
                        "pack_path": str(path.relative_to(out)),
                        **{f"shard_{k}": v for k, v in shard_meta.items()},
                        **metrics,
                    }
                )
        print(
            f"[P6] {pack['prompt_key']} decode={pack['decode_index']} layer={layer}",
            flush=True,
        )

    write_jsonl(out / "P6_manual_tp2_rows.jsonl", rows)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)

    md = [
        "# P6 manual TP2 partition / reduction parity",
        "",
        f"- rows: {len(rows)}",
        f"- world_size: {WORLD_SIZE}",
        "- shard dims derived from dequant weight shape (out, in)",
        "",
        "## Summary",
        "",
        "| parallel | reduce_dtype | max_abs_max | rel_l2_max | exact_fraction_min |",
        "|---|---|---:|---:|---:|",
    ]
    keys = sorted({(r["parallel"], r["reduce_dtype"]) for r in rows}, key=lambda x: (x[0], str(x[1])))
    for parallel, reduce_dtype in keys:
        group = [r for r in rows if r["parallel"] == parallel and r["reduce_dtype"] == reduce_dtype]
        md.append(
            f"| {parallel} | {reduce_dtype} | {max(r['max_abs'] for r in group):.6g} | "
            f"{max(r['rel_l2'] for r in group):.6g} | "
            f"{min(r['exact_fraction'] for r in group):.6g} |"
        )
    md.extend(
        [
            "",
            "## By bucket",
            "",
            "| bucket | n | max_abs_max |",
            "|---|---:|---:|",
        ]
    )
    for bucket in ("focus_low_margin", "uniform_control", "post_tp_divergence_control_only"):
        group = by_bucket.get(bucket, [])
        if not group:
            continue
        md.append(f"| {bucket} | {len(group)} | {max(r['max_abs'] for r in group):.6g} |")
    # Record one derivation example
    if rows:
        sample = rows[0]
        md.extend(
            [
                "",
                "## Shard derivation example",
                "",
                f"- proj: {sample['proj']}",
                f"- {sample.get('shard_derivation')}",
                f"- weight_shape_out_in: {sample.get('shard_weight_shape_out_in')}",
                "",
            ]
        )
    (out / "P6_manual_tp2_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
