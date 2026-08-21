"""Materialize HiF4-QDQ BF16 weights for vLLM semantic E2E target (P2)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor

LINEAR_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def _is_linear_weight(name: str) -> bool:
    if name == "lm_head.weight":
        return False
    return any(name.endswith(s) for s in LINEAR_SUFFIXES)


def materialize(
    src: Path,
    dst: Path,
    *,
    device: str,
) -> dict[str, int]:
    if not src.is_dir():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)

    # symlink/copy non-weight artifacts
    for p in src.iterdir():
        if p.name.startswith("model-") and p.name.endswith(".safetensors"):
            continue
        if p.name == "model.safetensors.index.json":
            continue
        target = dst / p.name
        if target.exists() or target.is_symlink():
            continue
        try:
            os.symlink(p.resolve(), target)
        except OSError:
            shutil.copy2(p, target)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = dict(index["weight_map"])
    shards = sorted(set(weight_map.values()))
    converted = 0
    copied = 0
    device_t = torch.device(device)

    for shard in shards:
        src_shard = src / shard
        dst_shard = dst / shard
        tensors = load_file(str(src_shard), device="cpu")
        out: dict[str, torch.Tensor] = {}
        for name, t in tensors.items():
            if _is_linear_weight(name) and t.ndim == 2:
                w = t.to(device=device_t, dtype=torch.float32)
                view = quantize_hif4_tensor(w, group_dim=-1, output_dtype=t.dtype)
                out[name] = view.dequantized.detach().to(device="cpu", dtype=t.dtype).contiguous()
                converted += 1
                del w, view
                if device_t.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                out[name] = t
                copied += 1
        save_file(out, str(dst_shard))
        del tensors, out

    (dst / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    )
    meta = {
        "source": str(src),
        "converted_linear_weights": converted,
        "passthrough_tensors": copied,
        "note": "HiF4 QDQ BF16 materialization for P2_matched_semantic target; lm_head/embed/norm untouched",
    }
    (dst / "ipc_hif4_materialize.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ns = ap.parse_args(argv)
    meta = materialize(ns.src, ns.dst, device=ns.device)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
