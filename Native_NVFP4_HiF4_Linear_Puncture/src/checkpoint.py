"""Local HF snapshot resolve + packed NVFP4 checkpoint preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from Native_NVFP4_HiF4_Linear_Puncture.src.config import (
    CHECKPOINT_KEY_SCHEMA,
    TARGET_PROJECTIONS,
    AppConfig,
    load_config,
    results_dir,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import ensure_dir, write_json
from Native_NVFP4_HiF4_Linear_Puncture.src.rotation import (
    rotation_orthogonality_stats,
    rotation_sha256,
)

PHYSICAL_SUFFIXES = (
    "qweight",
    "scales",
    "weight_global_scale",
    "act_global_scale",
    "forward_hadamard_matrix",
    "backward_hadamard_matrix",
)


def _fail(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def assert_packed_triplet_coverage(coverage: dict[str, int]) -> None:
    expected = int(coverage["target_linear_count"])
    for key in (
        "packed_weight_coverage",
        "weight_scale_coverage",
        "weight_global_scale_coverage",
    ):
        if int(coverage[key]) != expected:
            raise AssertionError(f"{key}={coverage[key]} != {expected}")


def assert_activation_global_scale_coverage(coverage: dict[str, int]) -> None:
    expected = int(coverage["target_linear_count"])
    if int(coverage["activation_global_scale_coverage"]) != expected:
        raise AssertionError(
            "activation_global_scale_coverage="
            f"{coverage['activation_global_scale_coverage']} != {expected}"
        )


def validate_rotation_source(rotation_source: str | None) -> None:
    allowed = {"checkpoint_tensor", "reconstructed_from_official_config"}
    if not rotation_source or rotation_source not in allowed:
        raise ValueError(f"invalid rotation_source={rotation_source!r}")


def resolve_local_snapshot(model_id: str) -> Path:
    """Resolve local HF snapshot only; never download or fallback."""
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except Exception:  # pragma: no cover
        LocalEntryNotFoundError = Exception  # type: ignore[misc,assignment]

    try:
        path = snapshot_download(repo_id=model_id, local_files_only=True)
    except LocalEntryNotFoundError:
        _fail(
            "Native NVFP4 checkpoint is not fully available in local HF cache.\n"
            f"model_id={model_id}\n"
            "No fallback checkpoint is allowed."
        )
    except Exception as exc:
        _fail(
            "Native NVFP4 checkpoint is not fully available in local HF cache.\n"
            f"model_id={model_id}\n"
            "No fallback checkpoint is allowed.\n"
            f"detail={type(exc).__name__}:{exc}"
        )

    snapshot = Path(path)
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        _fail(
            "Native NVFP4 checkpoint is not fully available in local HF cache.\n"
            f"model_id={model_id}\n"
            "No fallback checkpoint is allowed.\n"
            "detail=missing model.safetensors.index.json"
        )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    corrupt: list[str] = []
    for shard in sorted(set(index["weight_map"].values())):
        shard_path = snapshot / shard
        if not shard_path.is_file():
            missing.append(shard)
            continue
        if shard_path.is_symlink() and not shard_path.resolve().is_file():
            missing.append(shard)
            continue
        try:
            with safe_open(str(shard_path), framework="pt") as handle:
                _ = list(handle.keys())
        except Exception as exc:  # noqa: BLE001 — incomplete/corrupt shard must fail preflight
            corrupt.append(f"{shard}:{type(exc).__name__}:{exc}")
    if missing or corrupt:
        detail_parts: list[str] = []
        if missing:
            detail_parts.append(f"missing_shards:{missing}")
        if corrupt:
            detail_parts.append(f"corrupt_shards:{corrupt}")
        _fail(
            "Native NVFP4 checkpoint is not fully available in local HF cache.\n"
            f"model_id={model_id}\n"
            "No fallback checkpoint is allowed.\n"
            f"detail={'; '.join(detail_parts)}"
        )
    return snapshot


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def enumerate_target_prefixes(
    weight_map: dict[str, str],
    *,
    num_layers: int,
    projections: tuple[str, ...] = TARGET_PROJECTIONS,
) -> list[str]:
    prefixes: list[str] = []
    for layer in range(num_layers):
        for proj in projections:
            if proj in {"q_proj", "k_proj", "v_proj", "o_proj"}:
                prefixes.append(f"model.layers.{layer}.self_attn.{proj}")
            else:
                prefixes.append(f"model.layers.{layer}.mlp.{proj}")
    # Keep only prefixes that appear in the index at least once.
    present = []
    for p in prefixes:
        if any(k.startswith(p + ".") for k in weight_map):
            present.append(p)
    return present


def build_key_schema(
    weight_map: dict[str, str],
    target_prefixes: list[str],
) -> dict[str, Any]:
    """Enumerate physical auxiliary keys per target Linear prefix."""
    per_prefix: dict[str, list[str]] = {}
    suffix_counts: dict[str, int] = defaultdict(int)
    for prefix in target_prefixes:
        suffixes = sorted(
            k[len(prefix) + 1 :]
            for k in weight_map
            if k.startswith(prefix + ".")
        )
        per_prefix[prefix] = suffixes
        for s in suffixes:
            suffix_counts[s] += 1

    # Fixed semantic mapping for this experiment (once).
    semantic_to_physical = dict(CHECKPOINT_KEY_SCHEMA)
    physical_to_semantic = {v: k for k, v in semantic_to_physical.items()}

    return {
        "semantic_to_physical": semantic_to_physical,
        "physical_to_semantic": physical_to_semantic,
        "observed_suffix_counts": dict(sorted(suffix_counts.items())),
        "target_count": len(target_prefixes),
        "per_prefix_suffixes_sample": {
            k: per_prefix[k] for k in target_prefixes[:3]
        },
        "all_prefixes_have_required": all(
            set(CHECKPOINT_KEY_SCHEMA.values()).issubset(set(per_prefix[p]))
            for p in target_prefixes
        ),
    }


def _open_tensor(snapshot: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard = snapshot / weight_map[key]
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def audit_rotations(
    snapshot: Path,
    weight_map: dict[str, str],
    target_prefixes: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prefix in target_prefixes:
        key = f"{prefix}.forward_hadamard_matrix"
        if key not in weight_map:
            raise RuntimeError(f"missing rotation tensor: {key}")
        h = _open_tensor(snapshot, weight_map, key)
        stats = rotation_orthogonality_stats(h)
        records.append(
            {
                "module_name": prefix,
                "shape": list(h.shape),
                "dtype": str(h.dtype).replace("torch.", ""),
                "sha256": rotation_sha256(h),
                **stats,
            }
        )
    return records


def run_preflight(config: AppConfig, run_id: str) -> dict[str, Any]:
    model_id = config.model.model_id
    snapshot = resolve_local_snapshot(model_id)
    out_dir = ensure_dir(results_dir(run_id))

    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]

    arch_list = cfg.get("architectures") or []
    architecture = arch_list[0] if arch_list else None
    num_layers = int(cfg.get("num_hidden_layers", -1))
    qc = cfg.get("quantization_config") or {}
    forward_dtype = qc.get("forward_dtype")
    hadamard_group_size = qc.get("hadamard_group_size")

    target_prefixes = enumerate_target_prefixes(
        weight_map,
        num_layers=config.model.expected_num_layers,
        projections=config.model.target_projections,
    )
    key_schema = build_key_schema(weight_map, target_prefixes)
    write_json(out_dir / "key_schema.json", key_schema)

    def _coverage(suffix: str) -> int:
        return sum(1 for p in target_prefixes if f"{p}.{suffix}" in weight_map)

    packed_c = _coverage("qweight")
    scale_c = _coverage("scales")
    wgs_c = _coverage("weight_global_scale")
    ags_c = _coverage("act_global_scale")
    rot_c = _coverage("forward_hadamard_matrix")

    failures: list[str] = []
    if architecture != config.model.expected_architecture:
        failures.append(
            f"architecture={architecture} != {config.model.expected_architecture}"
        )
    if num_layers != config.model.expected_num_layers:
        failures.append(
            f"num_layers={num_layers} != {config.model.expected_num_layers}"
        )
    if forward_dtype != "nvfp4":
        failures.append(f"forward_dtype={forward_dtype} != nvfp4")
    expected = config.model.expected_num_layers * len(config.model.target_projections)
    if len(target_prefixes) != expected:
        failures.append(
            f"target_linear_count={len(target_prefixes)} != {expected}"
        )
    for name, cov in [
        ("packed_weight", packed_c),
        ("weight_scale", scale_c),
        ("weight_global_scale", wgs_c),
        ("activation_global_scale", ags_c),
        ("rotation", rot_c),
    ]:
        if cov != expected:
            failures.append(f"{name}_coverage={cov} != {expected}")
    if not key_schema["all_prefixes_have_required"]:
        failures.append("key_schema missing required physical suffixes")

    rotation_source = "checkpoint_tensor" if rot_c == expected else "unknown"
    if rotation_source != "checkpoint_tensor":
        failures.append("rotation_source cannot be determined from checkpoint tensors")

    rotation_records: list[dict[str, Any]] = []
    if not failures:
        rotation_records = audit_rotations(snapshot, weight_map, target_prefixes)
        write_json(out_dir / "rotation_audit.json", {"rotations": rotation_records})

    preflight = {
        "model_id": model_id,
        "snapshot_path": str(snapshot),
        "config_sha256": _sha256_file(config_path),
        "architecture": architecture,
        "num_layers": num_layers,
        "quantization_config": qc,
        "forward_dtype": forward_dtype,
        "hadamard_group_size": hadamard_group_size,
        "target_linear_count": len(target_prefixes),
        "packed_weight_coverage": packed_c,
        "weight_scale_coverage": scale_c,
        "weight_global_scale_coverage": wgs_c,
        "activation_global_scale_coverage": ags_c,
        "rotation_source": rotation_source,
        "rotation_coverage": rot_c,
        "key_schema_semantic_to_physical": key_schema["semantic_to_physical"],
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
    }
    write_json(out_dir / "preflight.json", preflight)

    if failures:
        print("PREFLIGHT FAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        raise SystemExit(1)

    print(f"PREFLIGHT PASS -> {out_dir / 'preflight.json'}")
    return preflight


def load_packed_linear_state(
    snapshot: Path,
    weight_map: dict[str, str],
    module_name: str,
) -> dict[str, torch.Tensor | None]:
    """Load physical tensors for one target Linear (CPU)."""
    required = CHECKPOINT_KEY_SCHEMA
    out: dict[str, torch.Tensor | None] = {}
    for semantic, physical in required.items():
        key = f"{module_name}.{physical}"
        if key not in weight_map:
            raise KeyError(f"missing {key}")
        out[semantic] = _open_tensor(snapshot, weight_map, key)
    bias_key = f"{module_name}.bias"
    out["bias"] = (
        _open_tensor(snapshot, weight_map, bias_key) if bias_key in weight_map else None
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Native NVFP4 checkpoint preflight")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    run_preflight(config, args.run_id)


if __name__ == "__main__":
    main()
