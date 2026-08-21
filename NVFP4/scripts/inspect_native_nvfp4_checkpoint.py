#!/usr/bin/env python3
"""Task 9: metadata-only preflight for a native ModelOpt NVFP4 checkpoint.

Reads config.json / hf_quant_config.json / safetensors index + file headers only.
Does NOT load tensor payloads into CPU/GPU RAM.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = (
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4"
    / "snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "NVFP4/reports/vllm_v027_nvfp4_backport/qwen3_30b_preflight.md"
)

# Plan/design assumptions for this backport target.
ASSUME_QUANT_METHOD = "modelopt"
ASSUME_QUANT_ALGO = "NVFP4"
ASSUME_GROUP_SIZE = 16
ASSUME_KV_ALGO = "FP8"
ASSUME_DENSE_SUFFIXES = ("weight", "weight_scale", "weight_scale_2", "input_scale")
ASSUME_EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    """Parse only the JSON header; do not map tensor bodies."""
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(header_len)
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header


def _dtype_str(entry: dict[str, Any]) -> str:
    return str(entry.get("dtype", "?"))


def _shape_list(entry: dict[str, Any]) -> list[int]:
    return list(entry.get("shape", []))


def _is_complete_dense(prefix: str, keys: set[str]) -> bool:
    return all(f"{prefix}.{s}" in keys for s in ASSUME_DENSE_SUFFIXES)


def _is_complete_expert(layer: int, expert: int, keys: set[str]) -> bool:
    base = f"model.layers.{layer}.mlp.experts.{expert}"
    for proj in ASSUME_EXPERT_PROJS:
        if not _is_complete_dense(f"{base}.{proj}", keys):
            return False
    return True


def _find_first_dense_sample(keys: set[str]) -> str | None:
    # Prefer attention projections; fall back to any Linear-like NVFP4 prefix.
    candidates: list[str] = []
    for k in keys:
        if not k.endswith(".weight"):
            continue
        if ".mlp.experts." in k:
            continue
        if ".mlp.gate." in k:
            continue
        prefix = k[: -len(".weight")]
        if _is_complete_dense(prefix, keys):
            candidates.append(prefix)
    if not candidates:
        return None

    def sort_key(p: str) -> tuple:
        # layers.N.self_attn.q_proj first
        parts = p.split(".")
        layer = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 10**9
        prefer = 0 if "self_attn" in p else 1
        return (layer, prefer, p)

    return sorted(candidates, key=sort_key)[0]


def _find_first_expert_sample(keys: set[str]) -> tuple[int, int] | None:
    found: list[tuple[int, int]] = []
    for k in keys:
        # model.layers.{L}.mlp.experts.{E}.gate_proj.weight
        if not k.endswith(".gate_proj.weight"):
            continue
        if ".mlp.experts." not in k:
            continue
        parts = k.split(".")
        try:
            layer = int(parts[2])
            expert = int(parts[5])
        except (IndexError, ValueError):
            continue
        if _is_complete_expert(layer, expert, keys):
            found.append((layer, expert))
    if not found:
        return None
    return sorted(found)[0]


def _tensor_meta(
    prefix: str,
    suffix: str,
    weight_map: dict[str, str],
    headers_by_file: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = f"{prefix}.{suffix}"
    shard = weight_map[key]
    entry = headers_by_file[shard][key]
    return {
        "key": key,
        "shard": shard,
        "shape": _shape_list(entry),
        "dtype": _dtype_str(entry),
    }


def inspect(ckpt: Path) -> dict[str, Any]:
    config = _read_json(ckpt / "config.json")
    hf_quant = _read_json(ckpt / "hf_quant_config.json")
    index = _read_json(ckpt / "model.safetensors.index.json")
    weight_map: dict[str, str] = index["weight_map"]
    keys = set(weight_map.keys())

    qc = config.get("quantization_config") or {}
    hq = hf_quant.get("quantization") or {}

    quant_method = qc.get("quant_method") or hq.get("quant_method")
    quant_algo = hq.get("quant_algo") or qc.get("quant_algo")
    group_size = hq.get("group_size")
    if group_size is None:
        # fallback from compressed-tensors style nested config
        try:
            group_size = (
                qc["config_groups"]["group_0"]["weights"]["group_size"]
            )
        except Exception:
            group_size = None
    kv_algo = hq.get("kv_cache_quant_algo")
    if kv_algo is None and isinstance(qc.get("kv_cache_scheme"), dict):
        # config.json may only expose bits/type; treat float8 as FP8 intent
        scheme = qc["kv_cache_scheme"]
        if scheme.get("num_bits") == 8 and scheme.get("type") == "float":
            kv_algo = "FP8 (from config.json kv_cache_scheme)"
    exclude = hq.get("exclude_modules") or qc.get("ignore") or []

    shard_names = sorted(set(weight_map.values()))
    headers_by_file: dict[str, dict[str, Any]] = {}
    for name in shard_names:
        headers_by_file[name] = _read_safetensors_header(ckpt / name)

    dense_prefix = _find_first_dense_sample(keys)
    expert_ids = _find_first_expert_sample(keys)

    dense_sample = None
    if dense_prefix is not None:
        dense_sample = {
            "prefix": dense_prefix,
            "tensors": [
                _tensor_meta(dense_prefix, s, weight_map, headers_by_file)
                for s in ASSUME_DENSE_SUFFIXES
            ],
        }

    expert_sample = None
    if expert_ids is not None:
        layer, expert = expert_ids
        base = f"model.layers.{layer}.mlp.experts.{expert}"
        expert_sample = {
            "layer": layer,
            "expert": expert,
            "w13_note": "checkpoint stores gate_proj + up_proj (fused to W13 at load)",
            "projections": {},
        }
        for proj in ASSUME_EXPERT_PROJS:
            expert_sample["projections"][proj] = [
                _tensor_meta(f"{base}.{proj}", s, weight_map, headers_by_file)
                for s in ASSUME_DENSE_SUFFIXES
            ]

    inconsistencies: list[str] = []
    if quant_method != ASSUME_QUANT_METHOD:
        inconsistencies.append(
            f"quant_method={quant_method!r} != assumed {ASSUME_QUANT_METHOD!r}"
        )
    if quant_algo != ASSUME_QUANT_ALGO:
        inconsistencies.append(
            f"quant_algo={quant_algo!r} != assumed {ASSUME_QUANT_ALGO!r}"
        )
    if group_size != ASSUME_GROUP_SIZE:
        inconsistencies.append(
            f"group_size={group_size!r} != assumed {ASSUME_GROUP_SIZE!r}"
        )
    kv_ok = isinstance(kv_algo, str) and (
        kv_algo == ASSUME_KV_ALGO or kv_algo.startswith("FP8")
    )
    if not kv_ok:
        inconsistencies.append(
            f"kv_cache_quant_algo={kv_algo!r} != assumed {ASSUME_KV_ALGO!r}"
        )
    if dense_sample is None:
        inconsistencies.append("no complete dense NVFP4 layer found in index")
    if expert_sample is None:
        inconsistencies.append("no complete expert gate/up/down NVFP4 set found")
    else:
        # packed uint8 weight expected
        for proj, tensors in expert_sample["projections"].items():
            w = next(t for t in tensors if t["key"].endswith(".weight"))
            if w["dtype"] != "U8":
                inconsistencies.append(
                    f"expert {proj} weight dtype={w['dtype']!r}, expected U8"
                )

    if dense_sample is not None:
        w = next(t for t in dense_sample["tensors"] if t["key"].endswith(".weight"))
        if w["dtype"] != "U8":
            inconsistencies.append(
                f"dense weight dtype={w['dtype']!r}, expected U8 packed NVFP4"
            )

    return {
        "checkpoint": str(ckpt),
        "architecture": (config.get("architectures") or [None])[0],
        "model_type": config.get("model_type"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_experts": config.get("num_experts"),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "hidden_size": config.get("hidden_size"),
        "moe_intermediate_size": config.get("moe_intermediate_size"),
        "quant_method": quant_method,
        "quant_algo": quant_algo,
        "group_size": group_size,
        "kv_cache_quant_algo": kv_algo,
        "exclude_modules_count": len(exclude),
        "exclude_modules_sample": exclude[:5],
        "exclude_modules": exclude,
        "num_index_keys": len(keys),
        "dense_sample": dense_sample,
        "expert_sample": expert_sample,
        "inconsistencies": inconsistencies,
        "consistent_with_plan": len(inconsistencies) == 0,
    }


def render_report(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Qwen3-30B-A3B-NVFP4 Preflight (Task 9)")
    lines.append("")
    lines.append(f"- Checkpoint: `{data['checkpoint']}`")
    lines.append(f"- Architecture: `{data['architecture']}` / `{data['model_type']}`")
    lines.append(
        f"- Layers/experts: hidden_layers={data['num_hidden_layers']}, "
        f"num_experts={data['num_experts']}, topk={data['num_experts_per_tok']}"
    )
    lines.append(f"- Index tensor keys: {data['num_index_keys']}")
    lines.append("")
    lines.append("## Quantization metadata")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"| --- | --- |")
    lines.append(f"| quant_method | `{data['quant_method']}` |")
    lines.append(f"| quant_algo | `{data['quant_algo']}` |")
    lines.append(f"| group_size | `{data['group_size']}` |")
    lines.append(f"| kv_cache_quant_algo | `{data['kv_cache_quant_algo']}` |")
    lines.append(
        f"| exclude_modules | count={data['exclude_modules_count']}; "
        f"sample={data['exclude_modules_sample']} |"
    )
    lines.append("")
    lines.append("## Dense NVFP4 sample (first complete layer from index)")
    lines.append("")
    if data["dense_sample"] is None:
        lines.append("_None found._")
    else:
        lines.append(f"- Prefix: `{data['dense_sample']['prefix']}`")
        lines.append("")
        lines.append("| key | shape | dtype | shard |")
        lines.append("| --- | --- | --- | --- |")
        for t in data["dense_sample"]["tensors"]:
            lines.append(
                f"| `{t['key']}` | `{t['shape']}` | `{t['dtype']}` | `{t['shard']}` |"
            )
    lines.append("")
    lines.append("## Expert W13/W2 packed sample (+ scales)")
    lines.append("")
    if data["expert_sample"] is None:
        lines.append("_None found._")
    else:
        es = data["expert_sample"]
        lines.append(
            f"- Layer `{es['layer']}`, expert `{es['expert']}` "
            f"({es['w13_note']})"
        )
        lines.append("")
        for proj, tensors in es["projections"].items():
            role = {"gate_proj": "W13/gate", "up_proj": "W13/up", "down_proj": "W2"}.get(
                proj, proj
            )
            lines.append(f"### {proj} ({role})")
            lines.append("")
            lines.append("| key | shape | dtype | shard |")
            lines.append("| --- | --- | --- | --- |")
            for t in tensors:
                lines.append(
                    f"| `{t['key']}` | `{t['shape']}` | `{t['dtype']}` | `{t['shard']}` |"
                )
            lines.append("")
    lines.append("## Consistency gate vs plan assumptions")
    lines.append("")
    lines.append(
        f"- Assumed: quant_method=`{ASSUME_QUANT_METHOD}`, "
        f"quant_algo=`{ASSUME_QUANT_ALGO}`, group_size=`{ASSUME_GROUP_SIZE}`, "
        f"kv=`{ASSUME_KV_ALGO}`, ModelOpt keys "
        f"`weight/weight_scale/weight_scale_2/input_scale`"
    )
    if data["consistent_with_plan"]:
        lines.append("- **PASS**: metadata consistent with plan; smoke may proceed.")
    else:
        lines.append("- **FAIL / STOP**: do not guess keys in smoke.")
        for item in data["inconsistencies"]:
            lines.append(f"  - {item}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Inspection used safetensors **headers only** "
        "(no full tensor payload load to CPU/GPU)."
    )
    lines.append(
        "- Expert weights are stored per `gate_proj`/`up_proj`/`down_proj`; "
        "vLLM fuses gate+up into W13 at load time."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional machine-readable dump next to the markdown report.",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint.resolve()
    if not ckpt.is_dir():
        print(f"checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    data = inspect(ckpt)
    report = render_report(data)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(report)
    print(f"\nWrote report: {args.report}")
    if not data["consistent_with_plan"]:
        print("PREFLIGHT GATE FAILED — stop before smoke.", file=sys.stderr)
        return 1
    print("PREFLIGHT GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
