#!/usr/bin/env python3
"""P1: activation QDQ parity on frozen identical inputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    qdq_native_nvfp4,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.nvfp4_operator_parity.common import (
    RESULT_ROOT,
    compare_tensors,
    ensure_cuda_lut,
    write_jsonl,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    ref_nvfp4_quant_dequant,
)


def _triton_path() -> bool:
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        _nvfp4_triton_fp8e4nv_supported,
    )

    return bool(_nvfp4_triton_fp8e4nv_supported())


def _scale_tensor(value: float, device: torch.device) -> torch.Tensor:
    return torch.tensor(float(value), device=device, dtype=torch.float32)


def _compare_one(
    *,
    name: str,
    x: torch.Tensor,
    scale_inv: float,
    device: torch.device,
    meta: dict,
) -> dict:
    x = x.to(device=device)
    scale = _scale_tensor(scale_inv, device)
    a = qdq_native_nvfp4(x, scale)
    x2d = x.reshape(-1, x.shape[-1])
    b = ref_nvfp4_quant_dequant(x2d, scale, block_size=16).reshape_as(a)
    metrics = compare_tensors(a, b)
    hard_ok = metrics["exact_fraction"] == 1.0 and metrics["max_abs"] == 0.0
    return {
        **meta,
        "tensor": name,
        "scale_inv": float(scale_inv),
        "output_dtype": str(a.dtype).replace("torch.", ""),
        "triton_path": _triton_path(),
        "hard_ok": hard_ok,
        **metrics,
    }


def iter_packs(result_root: Path):
    manifest = json.loads((result_root / "frozen_inputs_manifest.json").read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        for entry in sample["layers"]:
            path = result_root / entry["path"]
            pack = torch.load(path, map_location="cpu", weights_only=False)
            yield pack, path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=str(RESULT_ROOT))
    args = p.parse_args()

    device = torch.device(args.device)
    ensure_cuda_lut(device)
    out = Path(args.output_dir)
    rows: list[dict] = []

    for pack, path in iter_packs(out):
        meta = {
            "prompt_key": pack["prompt_key"],
            "decode_index": int(pack["decode_index"]),
            "layer": int(pack["layer"]),
            "bucket": pack["bucket"],
            "post_tp_divergence_control_only": bool(pack.get("post_tp_divergence_control_only", False)),
            "pack_path": str(path.relative_to(out)),
        }
        scales = pack["scales"]
        rows.append(
            _compare_one(
                name="qkv_input",
                x=pack["qkv_input"],
                scale_inv=scales["qkv_input_inv"],
                device=device,
                meta=meta,
            )
        )
        rows.append(
            _compare_one(
                name="o_input",
                x=pack["o_input"],
                scale_inv=scales["o_input_inv"],
                device=device,
                meta=meta,
            )
        )
        # MoE token input uses a13_inv (shared across experts in collapsed semantics).
        a13 = float(pack["experts"][0]["a13_inv"]) if pack["experts"] else None
        if a13 is not None:
            rows.append(
                _compare_one(
                    name="moe_input",
                    x=pack["moe_input"],
                    scale_inv=a13,
                    device=device,
                    meta=meta,
                )
            )
        for expert in pack["experts"]:
            rows.append(
                _compare_one(
                    name="w2_input",
                    x=expert["w2_input"],
                    scale_inv=float(expert["a2_inv"]),
                    device=device,
                    meta={
                        **meta,
                        "expert_id": int(expert["expert_id"]),
                        "slot": int(expert["slot"]),
                    },
                )
            )

    write_jsonl(out / "P1_qdq_rows.jsonl", rows)

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)

    hard_fail = [r for r in rows if not r["hard_ok"]]
    md = [
        "# P1 activation QDQ parity",
        "",
        f"- rows: {len(rows)}",
        f"- hard_ok_all: {len(hard_fail) == 0}",
        f"- hard_fail_count: {len(hard_fail)}",
        f"- triton_path: {_triton_path()}",
        "",
        "## By bucket",
        "",
        "| bucket | n | exact_fraction_min | max_abs_max | hard_ok |",
        "|---|---:|---:|---:|---|",
    ]
    for bucket in ("focus_low_margin", "uniform_control", "post_tp_divergence_control_only"):
        group = by_bucket.get(bucket, [])
        if not group:
            md.append(f"| {bucket} | 0 |  |  |  |")
            continue
        exact_min = min(r["exact_fraction"] for r in group)
        max_abs = max(r["max_abs"] for r in group)
        ok = all(r["hard_ok"] for r in group)
        md.append(f"| {bucket} | {len(group)} | {exact_min:.6g} | {max_abs:.6g} | {ok} |")
    md.extend(
        [
            "",
            "## Hard expectation",
            "",
            "- exact_fraction = 1.0",
            "- max_abs = 0",
            "",
            f"- verdict: **{'P1_QDQ_OK' if not hard_fail else 'P1_QDQ_MISMATCH'}**",
            "",
        ]
    )
    (out / "P1_qdq_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "hard_fail": len(hard_fail)}, indent=2))
    if hard_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
