"""Convert compressed-tensors NVFP4 checkpoints to BF16 weights."""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ACTIVATION_SCALES_FILE = "nvfp4_activation_scales.safetensors"
INDEX_FILE = "model.safetensors.index.json"
NVFP4_FORMAT = "nvfp4-pack-quantized"
DEFAULT_MAX_SHARD_SIZE = 5 * 1024**3

_E2M1_TO_FLOAT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


@dataclass(frozen=True)
class ConversionEstimate:
    main_bytes: int
    sidecar_bytes: int
    auxiliary_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.main_bytes + self.sidecar_bytes + self.auxiliary_bytes


def dequantize_nvfp4_weight(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    group_size: int = 16,
) -> torch.Tensor:
    """Dequantize one compressed-tensors NVFP4 weight tensor."""
    if weight_packed.dtype != torch.uint8:
        raise TypeError(f"weight_packed must be torch.uint8, got {weight_packed.dtype}")
    if weight_packed.ndim != 2:
        raise ValueError(f"weight_packed must be 2D, got {weight_packed.ndim}D")
    if weight_scale.ndim != 2:
        raise ValueError(f"weight_scale must be 2D, got {weight_scale.ndim}D")
    if weight_global_scale.numel() != 1:
        raise ValueError(
            "Only per-tensor NVFP4 weight_global_scale is supported; "
            f"got shape {tuple(weight_global_scale.shape)}"
        )

    out_features, packed_in_features = weight_packed.shape
    in_features = packed_in_features * 2
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} must be divisible by group_size={group_size}"
        )
    expected_scale_shape = (out_features, in_features // group_size)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"weight_scale shape must be {expected_scale_shape}, "
            f"got {tuple(weight_scale.shape)}"
        )

    packed = weight_packed.contiguous()
    low = packed & 0x0F
    high = (packed & 0xF0) >> 4
    combined = torch.stack((low, high), dim=-1).reshape(out_features, in_features)

    sign = torch.where(
        (combined & 0x08).bool(),
        torch.tensor(-1.0, device=combined.device),
        torch.tensor(1.0, device=combined.device),
    )
    values = _E2M1_TO_FLOAT.to(combined.device)[(combined & 0x07).long()] * sign

    scale = weight_scale.to(torch.float32) / weight_global_scale.reshape(()).to(
        torch.float32
    )
    values = values.reshape(out_features, in_features // group_size, group_size)
    dequantized = values * scale.unsqueeze(-1)
    return dequantized.reshape(out_features, in_features).to(dtype)


def convert_nvfp4_checkpoint_to_bf16(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    keep_activation_scales: bool = True,
    overwrite: bool = False,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
) -> Path:
    """Convert a compressed-tensors NVFP4 checkpoint directory to BF16."""
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    _validate_paths(input_path, output_path, overwrite)
    _validate_nvfp4_config(input_path)

    index = _load_index(input_path)
    bases = _validate_weight_groups(index["weight_map"])
    estimate = estimate_converted_checkpoint_size(
        input_path, keep_activation_scales=keep_activation_scales
    )
    _check_free_space(output_path, estimate)

    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    try:
        _copy_auxiliary_files(input_path, output_path)
        _write_dequantized_shards(
            input_path=input_path,
            output_path=output_path,
            weight_map=index["weight_map"],
            quantized_bases=bases,
            keep_activation_scales=keep_activation_scales,
            max_shard_size=max_shard_size,
        )
        _rewrite_config_as_bf16(output_path)
    except Exception:
        shutil.rmtree(output_path, ignore_errors=True)
        raise

    return output_path


def estimate_converted_checkpoint_size(
    input_dir: str | os.PathLike[str],
    keep_activation_scales: bool = True,
) -> ConversionEstimate:
    """Estimate output bytes without reading full tensors."""
    input_path = Path(input_dir).resolve()
    index = _load_index(input_path)
    weight_map = index["weight_map"]
    bases = {
        key[: -len("weight_packed")]
        for key in weight_map
        if key.endswith("weight_packed")
    }
    removed = _removed_main_keys(bases)

    main_bytes = 0
    sidecar_bytes = 0
    files = sorted(set(weight_map.values()))
    for filename in files:
        header = _read_safetensors_header(input_path / filename)
        for key, info in header.items():
            if key == "__metadata__":
                continue
            shape = info["shape"]
            dtype = info["dtype"]
            if key.endswith("weight_packed"):
                main_bytes += _numel(shape) * 2 * _dtype_nbytes("BF16")
            elif key.endswith("input_global_scale") and keep_activation_scales:
                sidecar_bytes += _numel(shape) * _dtype_nbytes(dtype)
            elif key in removed:
                continue
            else:
                main_bytes += _numel(shape) * _dtype_nbytes(dtype)

    auxiliary_bytes = 0
    for path in input_path.iterdir():
        if _is_checkpoint_file(path):
            continue
        if path.is_file() or path.is_symlink():
            auxiliary_bytes += path.stat().st_size

    return ConversionEstimate(
        main_bytes=main_bytes,
        sidecar_bytes=sidecar_bytes,
        auxiliary_bytes=auxiliary_bytes,
    )


def _validate_paths(input_path: Path, output_path: Path, overwrite: bool) -> None:
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_path}. "
            "Pass overwrite=True or --overwrite to replace it."
        )
    try:
        output_path.relative_to(input_path)
    except ValueError:
        return
    raise ValueError("output_dir must not be inside input_dir")


def _validate_nvfp4_config(input_path: Path) -> None:
    config_path = input_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise ValueError("config.json has no quantization_config")
    if quant_config.get("format") != NVFP4_FORMAT:
        raise ValueError(
            f"Only {NVFP4_FORMAT} is supported, got {quant_config.get('format')}"
        )


def _load_index(input_path: Path) -> dict:
    index_path = input_path / INDEX_FILE
    if not index_path.is_file():
        single_file = input_path / "model.safetensors"
        if not single_file.is_file():
            raise FileNotFoundError(f"Missing {INDEX_FILE}: {index_path}")
        header = _read_safetensors_header(single_file)
        weight_map = {
            key: single_file.name for key in header if key != "__metadata__"
        }
        total_size = sum(
            _numel(info["shape"]) * _dtype_nbytes(info["dtype"])
            for key, info in header.items()
            if key != "__metadata__"
        )
        return {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"{INDEX_FILE} has no weight_map")
    return index


def _validate_weight_groups(weight_map: dict[str, str]) -> set[str]:
    bases = {
        key[: -len("weight_packed")]
        for key in weight_map
        if key.endswith("weight_packed")
    }
    if not bases:
        raise ValueError("No NVFP4 weight_packed tensors found")
    for base in sorted(bases):
        for suffix in ("weight_scale", "weight_global_scale"):
            key = base + suffix
            if key not in weight_map:
                raise ValueError(f"Missing required tensor for {base}: {key}")
    return bases


def _write_dequantized_shards(
    input_path: Path,
    output_path: Path,
    weight_map: dict[str, str],
    quantized_bases: set[str],
    keep_activation_scales: bool,
    max_shard_size: int,
) -> None:
    removed = _removed_main_keys(quantized_bases)
    input_files = sorted(set(weight_map.values()))
    output_weight_map: dict[str, str] = {}
    output_total_size = 0
    output_shard: dict[str, torch.Tensor] = {}
    output_shard_bytes = 0
    output_shard_index = 1
    activation_scales: dict[str, torch.Tensor] = {}

    def flush() -> None:
        nonlocal output_shard, output_shard_bytes, output_shard_index
        if not output_shard:
            return
        filename = f"model-{output_shard_index:05d}-of-00000.safetensors"
        save_file(output_shard, output_path / filename, metadata={"format": "pt"})
        for tensor_name in output_shard:
            output_weight_map[tensor_name] = filename
        output_shard = {}
        output_shard_bytes = 0
        output_shard_index += 1

    def add_tensor(name: str, tensor: torch.Tensor) -> None:
        nonlocal output_shard_bytes, output_total_size
        tensor = tensor.contiguous()
        tensor_bytes = tensor.numel() * tensor.element_size()
        if output_shard and output_shard_bytes + tensor_bytes > max_shard_size:
            flush()
        output_shard[name] = tensor
        output_shard_bytes += tensor_bytes
        output_total_size += tensor_bytes

    for filename in input_files:
        with safe_open(input_path / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.endswith("weight_packed"):
                    base = key[: -len("weight_packed")]
                    weight = dequantize_nvfp4_weight(
                        weight_packed=handle.get_tensor(key),
                        weight_scale=_get_tensor_by_weight_map(
                            input_path,
                            weight_map,
                            base + "weight_scale",
                            current_filename=filename,
                            current_handle=handle,
                        ),
                        weight_global_scale=_get_tensor_by_weight_map(
                            input_path,
                            weight_map,
                            base + "weight_global_scale",
                            current_filename=filename,
                            current_handle=handle,
                        ),
                        dtype=torch.bfloat16,
                    )
                    add_tensor(base + "weight", weight)
                elif key.endswith("input_global_scale"):
                    if keep_activation_scales:
                        activation_scales[key] = handle.get_tensor(key).contiguous()
                elif key in removed:
                    continue
                else:
                    add_tensor(key, handle.get_tensor(key))

    flush()
    total_shards = output_shard_index - 1
    renamed_weight_map = _rename_output_shards(output_path, output_weight_map, total_shards)

    index = {
        "metadata": {"total_size": output_total_size},
        "weight_map": renamed_weight_map,
    }
    with (output_path / INDEX_FILE).open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, ensure_ascii=False)

    if keep_activation_scales and activation_scales:
        save_file(
            activation_scales,
            output_path / ACTIVATION_SCALES_FILE,
            metadata={"format": "pt"},
        )


def _get_tensor_by_weight_map(
    input_path: Path,
    weight_map: dict[str, str],
    key: str,
    current_filename: str,
    current_handle,
) -> torch.Tensor:
    filename = weight_map[key]
    if filename == current_filename:
        return current_handle.get_tensor(key)
    with safe_open(input_path / filename, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _rename_output_shards(
    output_path: Path, weight_map: dict[str, str], total_shards: int
) -> dict[str, str]:
    renamed: dict[str, str] = {}
    for old_index in range(1, total_shards + 1):
        old_name = f"model-{old_index:05d}-of-00000.safetensors"
        new_name = f"model-{old_index:05d}-of-{total_shards:05d}.safetensors"
        if old_name != new_name:
            (output_path / old_name).rename(output_path / new_name)
        for key, filename in weight_map.items():
            if filename == old_name:
                renamed[key] = new_name
    return renamed


def _copy_auxiliary_files(input_path: Path, output_path: Path) -> None:
    for src in input_path.iterdir():
        if _is_checkpoint_file(src):
            continue
        dst = output_path / src.name
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst, follow_symlinks=False)


def _rewrite_config_as_bf16(output_path: Path) -> None:
    config_path = output_path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    if "torch_dtype" in config:
        config["torch_dtype"] = "bfloat16"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _check_free_space(output_path: Path, estimate: ConversionEstimate) -> None:
    parent = output_path.parent
    while not parent.exists():
        parent = parent.parent
    free_bytes = shutil.disk_usage(parent).free
    required_bytes = estimate.total_bytes + 1024**3
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Not enough disk space for BF16 conversion: "
            f"required at least {_format_bytes(required_bytes)}, "
            f"available {_format_bytes(free_bytes)}"
        )


def _removed_main_keys(bases: set[str]) -> set[str]:
    return {
        base + suffix
        for base in bases
        for suffix in (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
            "input_global_scale",
        )
    }


def _is_checkpoint_file(path: Path) -> bool:
    return (
        path.name == INDEX_FILE
        or path.name == ACTIVATION_SCALES_FILE
        or path.suffix in {".safetensors", ".bin"}
    )


def _numel(shape: list[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = handle.read(header_len)
    return json.loads(header)


def _dtype_nbytes(dtype: str) -> int:
    normalized = dtype.upper()
    sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "F16": 2,
        "BF16": 2,
        "I16": 2,
        "U16": 2,
        "F32": 4,
        "I32": 4,
        "U32": 4,
        "F64": 8,
        "I64": 8,
        "U64": 8,
    }
    if normalized not in sizes:
        raise ValueError(f"Unsupported safetensors dtype for size estimate: {dtype}")
    return sizes[normalized]


def _format_bytes(num_bytes: int) -> str:
    gib = num_bytes / 1024**3
    return f"{gib:.2f} GiB"
