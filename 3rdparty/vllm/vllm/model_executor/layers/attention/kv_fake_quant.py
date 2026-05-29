# SPDX-License-Identifier: Apache-2.0
"""Fake quantization helpers for full-attention KV cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.quantization import hif4_fake
from NVFP4.torch_fake import (
    FP4_E2M1_MAX,
    FP8_E4M3FN_MAX,
    cast_to_fp4_e2m1,
    cast_to_fp8_e4m3fn,
)


@dataclass(frozen=True)
class KVQuantConfig:
    chunk_size: int
    sink_size: int
    target: str
    format: str = "nvfp4"
    group_size: int = 16
    quant_query: bool = False

    @property
    def quant_k(self) -> bool:
        return self.target in ("kv", "k")

    @property
    def quant_v(self) -> bool:
        return self.target in ("kv", "v")


Nvfp4KVQuantConfig = KVQuantConfig


def get_kv_quant_config(additional_config: dict[str, Any]) -> KVQuantConfig | None:
    quant_format = additional_config.get("kv_quant_format", "none")
    if quant_format == "none":
        return None
    if quant_format not in ("nvfp4", "hif4", "hif4-1"):
        raise ValueError(f"Unsupported kv_quant_format: {quant_format}")

    chunk_size = int(additional_config.get("kv_quant_chunk_size", 64))
    sink_size = int(additional_config.get("kv_quant_sink_size", 4))
    target = additional_config.get("kv_quant_target", "kv")
    quant_query = additional_config.get("kv_quant_query", "none")
    if quant_format == "nvfp4" and chunk_size < 1:
        raise ValueError("kv_quant_chunk_size must be positive.")
    if sink_size < 0:
        raise ValueError("kv_quant_sink_size must be non-negative.")
    if target not in ("kv", "k", "v"):
        raise ValueError(f"Unsupported kv_quant_target: {target}")
    if quant_query not in ("none", "enabled"):
        raise ValueError(f"Unsupported kv_quant_query: {quant_query}")

    return KVQuantConfig(
        chunk_size=chunk_size,
        sink_size=sink_size,
        target=target,
        format=quant_format,
        quant_query=quant_query == "enabled",
    )


get_nvfp4_kv_quant_config = get_kv_quant_config


def fake_quant_kv_tensor(
    x: torch.Tensor,
    config: KVQuantConfig,
) -> torch.Tensor:
    if config.format == "nvfp4":
        return fake_quant_nvfp4_per_head_chunk(x, group_size=config.group_size)
    if config.format in ("hif4", "hif4-1"):
        return fake_quant_hif4_tensor(x, config.format)
    raise ValueError(f"Unsupported kv_quant_format: {config.format}")


def fake_quant_hif4_tensor(x: torch.Tensor, quant_format: str = "hif4") -> torch.Tensor:
    """Apply HiF4 fake quant-dequant along the last dimension."""
    if not x.is_cuda:
        if quant_format == "hif4":
            return hif4_fake.hif4_fake_quantize_hifx4(x)
        if quant_format == "hif4-1":
            return hif4_fake.hif4_fake_quantize_hifx4_1(x)
        raise ValueError(f"Unsupported hif quant format: {quant_format}")
    if x.ndim != 3:
        raise ValueError(f"Expected [tokens, heads, head_dim], got {tuple(x.shape)}")
    if x.shape[0] == 0:
        return x

    out = torch.empty_like(x)
    num_tokens, num_heads, head_dim = x.shape
    _fake_quant_hif4_tensor_kernel[
        (num_tokens * num_heads, triton.cdiv(head_dim, 64))
    ](
        x,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        HIF4_1=quant_format == "hif4-1",
        BLOCK_DIMS=64,
        HEAD_DIM=head_dim,
    )
    return out


def fake_quant_nvfp4_per_head_chunk(
    x: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Fake quant-dequant one token chunk with one dynamic scale per head."""
    if x.ndim != 3:
        raise ValueError(f"Expected [tokens, heads, head_dim], got {tuple(x.shape)}")
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"head_dim {x.shape[-1]} must be divisible by group_size={group_size}"
        )
    if x.shape[0] == 0:
        return x

    out = torch.empty_like(x)
    for head_idx in range(x.shape[1]):
        out[:, head_idx, :] = _fake_quant_nvfp4_tensor(
            x[:, head_idx, :],
            group_size=group_size,
            output_dtype=x.dtype,
        )
    return out


def fake_quant_kv_query(
    query: torch.Tensor,
    attn_metadata: Any,
    config: KVQuantConfig,
) -> torch.Tensor:
    """Fake quantize Q after RoPE, before QK matmul."""
    if attn_metadata is None or not config.quant_query:
        return query
    if config.format in ("hif4", "hif4-1"):
        return fake_quant_hif4_tensor(query, config.format)
    if config.format == "nvfp4" and query.is_cuda:
        num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", query.shape[0]))
        max_query_len = int(getattr(attn_metadata, "max_query_len", num_actual_tokens))
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
        if query_start_loc is None:
            raise ValueError("NVFP4 Q fake quant requires attention query_start_loc.")
        return _fake_quant_nvfp4_query_cuda(
            query,
            query_start_loc,
            config,
            num_actual_tokens,
            max_query_len,
        )

    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    if query_start_loc is None:
        raise ValueError("KV Q fake quant requires attention query_start_loc.")

    out = query.clone()
    num_reqs = int(query_start_loc.numel()) - 1
    for req_idx in range(num_reqs):
        start = int(query_start_loc[req_idx].item())
        end = int(query_start_loc[req_idx + 1].item())
        if end > start:
            out[start:end] = fake_quant_kv_tensor(out[start:end], config)
    return out


fake_quant_nvfp4_query = fake_quant_kv_query


def fake_quant_hif4_new_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: Any,
    config: KVQuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake quantize this step's HIF4 K/V before writing to cache."""
    if config.format not in ("hif4", "hif4-1"):
        return key, value
    if attn_metadata is None:
        return key, value
    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if query_start_loc is None or seq_lens is None:
        raise ValueError("HIF4 KV fake quant requires query_start_loc and seq_lens.")

    if key.is_cuda:
        if config.quant_k:
            key = _fake_quant_hif4_new_kv_cuda(
                key,
                query_start_loc,
                seq_lens,
                attn_metadata,
                config.sink_size,
                config.format == "hif4-1",
            )
        if config.quant_v:
            value = _fake_quant_hif4_new_kv_cuda(
                value,
                query_start_loc,
                seq_lens,
                attn_metadata,
                config.sink_size,
                config.format == "hif4-1",
            )
        return key, value

    num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", key.shape[0]))
    token_positions = _new_token_positions(
        num_actual_tokens,
        query_start_loc,
        seq_lens,
        key.device,
    )
    quant_mask = torch.zeros(key.shape[0], device=key.device, dtype=torch.bool)
    quant_mask[:num_actual_tokens] = token_positions >= config.sink_size
    quant_mask = quant_mask.view(-1, 1, 1)
    if config.quant_k:
        key_quant = fake_quant_hif4_tensor(key, config.format)
        key = torch.where(quant_mask, key_quant, key)
    if config.quant_v:
        value_quant = fake_quant_hif4_tensor(value, config.format)
        value = torch.where(quant_mask, value_quant, value)
    return key, value


def rewrite_quantized_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: Any,
    config: KVQuantConfig,
) -> None:
    """Rewrite quantized KV cache entries in-place after cache update."""
    if attn_metadata is None:
        return
    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    block_table = getattr(attn_metadata, "block_table", None)
    if query_start_loc is None or seq_lens is None or block_table is None:
        raise ValueError("KV fake quant requires full attention metadata.")

    if config.format in ("hif4", "hif4-1"):
        return
    if config.format == "nvfp4" and key_cache.is_cuda:
        _rewrite_nvfp4_kv_cache_cuda(
            key_cache,
            value_cache,
            query_start_loc,
            seq_lens,
            block_table,
            attn_metadata,
            config,
        )
        return

    block_size = key_cache.shape[1]
    num_reqs = int(seq_lens.numel())

    for req_idx in range(num_reqs):
        query_start = int(query_start_loc[req_idx].item())
        query_end = int(query_start_loc[req_idx + 1].item())
        query_len = query_end - query_start
        seq_len = int(seq_lens[req_idx].item())
        context_len = seq_len - query_len
        if context_len < 0:
            raise ValueError("seq_len must be >= query_len for KV fake quant.")

        for start, end in _quantized_ranges(context_len, seq_len, config):
            block_ids, block_offsets = _cache_indices_for_positions(
                req_idx,
                start,
                end,
                block_table,
                block_size,
            )
            if config.quant_k:
                key_chunk = key_cache[block_ids, block_offsets]
                key_cache[block_ids, block_offsets] = fake_quant_kv_tensor(
                    key_chunk,
                    config,
                ).to(
                    key_cache.dtype
                )
            if config.quant_v:
                value_chunk = value_cache[block_ids, block_offsets]
                value_cache[block_ids, block_offsets] = fake_quant_kv_tensor(
                    value_chunk,
                    config,
                ).to(
                    value_cache.dtype
                )


rewrite_completed_kv_chunks = rewrite_quantized_kv


def _quantized_ranges(
    context_len: int,
    seq_len: int,
    config: KVQuantConfig,
) -> list[tuple[int, int]]:
    if config.format == "nvfp4":
        prev_chunks = _num_completed_chunks(context_len, config)
        cur_chunks = _num_completed_chunks(seq_len, config)
        return [
            (
                config.sink_size + chunk_idx * config.chunk_size,
                config.sink_size + (chunk_idx + 1) * config.chunk_size,
            )
            for chunk_idx in range(prev_chunks, cur_chunks)
        ]
    if config.format in ("hif4", "hif4-1"):
        start = max(context_len, config.sink_size)
        return [(start, seq_len)] if start < seq_len else []
    raise ValueError(f"Unsupported kv_quant_format: {config.format}")


def _new_token_positions(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    token_indices = torch.arange(num_tokens, device=device, dtype=torch.long)
    req_idx = torch.searchsorted(query_start_loc[1:], token_indices, right=False)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    context_lens = seq_lens - query_lens
    query_offsets = token_indices - query_start_loc.index_select(0, req_idx)
    return context_lens.index_select(0, req_idx) + query_offsets


def _cache_indices_for_positions(
    req_idx: int,
    start: int,
    end: int,
    block_table: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(
        start,
        end,
        device=block_table.device,
        dtype=torch.long,
    )
    block_ids = block_table[req_idx].index_select(0, positions // block_size)
    if torch.any(block_ids < 0):
        raise ValueError("KV fake quant found invalid block id.")
    return block_ids, positions % block_size


def _num_completed_chunks(seq_len: int, config: KVQuantConfig) -> int:
    if seq_len <= config.sink_size:
        return 0
    return (seq_len - config.sink_size) // config.chunk_size


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _rewrite_nvfp4_kv_cache_cuda(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    attn_metadata: Any,
    config: KVQuantConfig,
) -> None:
    max_seq_len = int(getattr(attn_metadata, "max_seq_len", int(seq_lens.numel())))
    max_non_sink = max(0, max_seq_len - config.sink_size)
    max_chunks = triton.cdiv(max_non_sink, config.chunk_size)
    if max_chunks == 0:
        return

    head_dim = key_cache.shape[-1]
    if head_dim % config.group_size != 0:
        raise ValueError(
            f"head_dim {head_dim} must be divisible by group_size={config.group_size}"
        )
    block_size = key_cache.shape[1]
    num_reqs = int(seq_lens.numel())
    num_heads = key_cache.shape[-2]
    _rewrite_nvfp4_kv_cache_kernel[(num_reqs, max_chunks, num_heads)](
        key_cache,
        value_cache,
        query_start_loc,
        seq_lens,
        block_table,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        block_table.stride(0),
        config.sink_size,
        config.chunk_size,
        block_size,
        config.group_size,
        config.quant_k,
        config.quant_v,
        BLOCK_TOKENS=_next_power_of_2(config.chunk_size),
        BLOCK_DIMS=_next_power_of_2(head_dim),
        HEAD_DIM=head_dim,
        GROUP_SIZE=config.group_size,
    )


def _fake_quant_nvfp4_query_cuda(
    query: torch.Tensor,
    query_start_loc: torch.Tensor,
    config: KVQuantConfig,
    num_actual_tokens: int,
    max_query_len: int,
) -> torch.Tensor:
    if query.shape[-1] % config.group_size != 0:
        raise ValueError(
            f"head_dim {query.shape[-1]} must be divisible by group_size={config.group_size}"
        )
    out = query.clone()
    if num_actual_tokens == 0:
        return out
    num_reqs = int(query_start_loc.numel()) - 1
    _, num_heads, head_dim = query.shape
    _fake_quant_nvfp4_query_kernel[(num_reqs, num_heads)](
        query,
        out,
        query_start_loc,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        config.group_size,
        BLOCK_TOKENS=_next_power_of_2(max_query_len),
        BLOCK_DIMS=_next_power_of_2(head_dim),
        HEAD_DIM=head_dim,
        GROUP_SIZE=config.group_size,
    )
    return out


def _fake_quant_hif4_new_kv_cuda(
    x: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_metadata: Any,
    sink_size: int,
    hif4_1: bool,
) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [tokens, heads, head_dim], got {tuple(x.shape)}")
    out = torch.empty_like(x)
    num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", x.shape[0]))
    if x.shape[0] == 0:
        return out
    num_reqs = int(query_start_loc.numel()) - 1
    _, num_heads, head_dim = x.shape
    _fake_quant_hif4_new_kv_kernel[
        (x.shape[0] * num_heads, triton.cdiv(head_dim, 64))
    ](
        x,
        out,
        query_start_loc,
        seq_lens,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        num_actual_tokens,
        sink_size,
        BLOCK_DIMS=64,
        HEAD_DIM=head_dim,
        NUM_REQS=num_reqs,
        HIF4_1=hif4_1,
    )
    return out


@triton.jit
def _round_nearest_even_tl(x):
    floored = tl.floor(x)
    frac = x - floored
    half = frac == 0.5
    odd = (floored - 2.0 * tl.floor(floored * 0.5)) == 1.0
    return tl.where((frac > 0.5) | (half & odd), floored + 1.0, floored)


@triton.jit
def _hif4_fake_quant_block_tl(x, valid_mask, BLOCK_DIMS: tl.constexpr):
    offsets = tl.arange(0, BLOCK_DIMS)
    abs_x = tl.abs(tl.where(valid_mask, x, 0.0))
    max_lv1 = tl.max(abs_x, axis=0)

    max_lv2 = tl.zeros((BLOCK_DIMS,), tl.float32)
    for group_start in tl.static_range(0, BLOCK_DIMS, 8):
        group_mask = (offsets >= group_start) & (offsets < group_start + 8)
        group_max = tl.max(tl.where(group_mask & valid_mask, abs_x, 0.0), axis=0)
        max_lv2 = tl.where(group_mask, group_max, max_lv2)

    max_lv3 = tl.zeros((BLOCK_DIMS,), tl.float32)
    for group_start in tl.static_range(0, BLOCK_DIMS, 4):
        group_mask = (offsets >= group_start) & (offsets < group_start + 4)
        group_max = tl.max(tl.where(group_mask & valid_mask, abs_x, 0.0), axis=0)
        max_lv3 = tl.where(group_mask, group_max, max_lv3)

    div7 = tl.full((), 1.0 / 7.0, tl.float32).to(tl.bfloat16).to(tl.float32)
    scale_factor = (max_lv1 * div7).to(tl.bfloat16).to(tl.float32)
    scale_factor = tl.minimum(tl.maximum(scale_factor, 3.552713678800501e-15), 49152.0)
    exp_sf = tl.floor(tl.log2(scale_factor))
    mant_sf = scale_factor / tl.exp2(exp_sf) * 128.0
    scale_factor = _round_nearest_even_tl(mant_sf) / 128.0 * tl.exp2(exp_sf)
    exp_sf = tl.floor(tl.log2(scale_factor))
    scale_factor = (
        _round_nearest_even_tl(scale_factor * tl.exp2(2.0 - exp_sf))
        * tl.exp2(exp_sf - 2.0)
    )
    rec_sf = (1.0 / scale_factor).to(tl.bfloat16).to(tl.float32)

    scale_lv2 = tl.exp2(tl.floor(tl.minimum(tl.maximum(max_lv2 * rec_sf, 0.0), 4.0) / 4.0))
    scale_lv3 = tl.exp2(
        tl.floor(tl.minimum(tl.maximum(max_lv3 * rec_sf / scale_lv2, 0.0), 2.0) / 2.0)
    )
    mant = abs_x / scale_lv2 / scale_lv3 * rec_sf
    mant = tl.floor(mant * 4.0 + 0.5) / 4.0
    mant = tl.minimum(mant, 1.75)
    sign = tl.where(x > 0.0, 1.0, tl.where(x < 0.0, -1.0, 0.0))
    return sign * mant * scale_lv2 * scale_lv3 * scale_factor


@triton.jit
def _hif4_1_fake_quant_block_tl(x, valid_mask, BLOCK_DIMS: tl.constexpr):
    abs_x = tl.abs(tl.where(valid_mask, x, 0.0))
    max_abs = tl.max(abs_x, axis=0)
    div = tl.full((), 1.0 / 1.75, tl.float32).to(tl.bfloat16).to(tl.float32)
    scale_factor = (max_abs * div).to(tl.bfloat16).to(tl.float32)
    scale_factor = tl.minimum(tl.maximum(scale_factor, 3.552713678800501e-15), 49152.0)
    exp_sf = tl.floor(tl.log2(scale_factor))
    scale_factor = (
        _round_nearest_even_tl(scale_factor * tl.exp2(2.0 - exp_sf))
        * tl.exp2(exp_sf - 2.0)
    )
    rec_sf = (1.0 / scale_factor).to(tl.bfloat16).to(tl.float32)
    mant = abs_x * rec_sf
    mant = tl.floor(mant * 4.0 + 0.5) / 4.0
    mant = tl.minimum(mant, 1.75)
    sign = tl.where(x > 0.0, 1.0, tl.where(x < 0.0, -1.0, 0.0))
    return sign * mant * scale_factor


@triton.jit
def _fake_quant_hif4_tensor_kernel(
    x_ptr,
    out_ptr,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    num_heads,
    HIF4_1: tl.constexpr,
    BLOCK_DIMS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    token_idx = row // num_heads
    head_idx = row - token_idx * num_heads
    dims = block_idx * BLOCK_DIMS + tl.arange(0, BLOCK_DIMS)
    mask = dims < HEAD_DIM
    x = tl.load(
        x_ptr + token_idx * x_stride_0 + head_idx * x_stride_1 + dims * x_stride_2,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HIF4_1:
        out = _hif4_1_fake_quant_block_tl(x, mask, BLOCK_DIMS)
    else:
        out = _hif4_fake_quant_block_tl(x, mask, BLOCK_DIMS)
    tl.store(
        out_ptr + token_idx * out_stride_0 + head_idx * out_stride_1 + dims * out_stride_2,
        out,
        mask=mask,
    )


@triton.jit
def _fake_quant_hif4_new_kv_kernel(
    x_ptr,
    out_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    num_heads,
    num_actual_tokens,
    sink_size,
    BLOCK_DIMS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_REQS: tl.constexpr,
    HIF4_1: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    token_idx = row // num_heads
    head_idx = row - token_idx * num_heads
    dims = block_idx * BLOCK_DIMS + tl.arange(0, BLOCK_DIMS)
    mask = dims < HEAD_DIM
    x = tl.load(
        x_ptr + token_idx * x_stride_0 + head_idx * x_stride_1 + dims * x_stride_2,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    abs_pos = tl.full((), 0, tl.int64)
    found = token_idx < 0
    for req_idx in tl.static_range(0, NUM_REQS):
        query_start = tl.load(query_start_loc_ptr + req_idx)
        query_end = tl.load(query_start_loc_ptr + req_idx + 1)
        in_req = (token_idx >= query_start) & (token_idx < query_end)
        query_len = query_end - query_start
        context_len = tl.load(seq_lens_ptr + req_idx) - query_len
        req_abs_pos = context_len + token_idx - query_start
        abs_pos = tl.where(in_req, req_abs_pos, abs_pos)
        found = found | in_req

    should_quant = (token_idx < num_actual_tokens) & found & (abs_pos >= sink_size)
    if HIF4_1:
        quant = _hif4_1_fake_quant_block_tl(x, mask, BLOCK_DIMS)
    else:
        quant = _hif4_fake_quant_block_tl(x, mask, BLOCK_DIMS)
    out = tl.where(should_quant, quant, x)
    tl.store(
        out_ptr + token_idx * out_stride_0 + head_idx * out_stride_1 + dims * out_stride_2,
        out,
        mask=mask,
    )


@triton.jit
def _cast_to_fp8_e4m3fn_tl(x):
    sign = tl.where(x < 0.0, -1.0, 1.0)
    abs_x = tl.minimum(tl.abs(x), 448.0)
    safe_abs_x = tl.where(abs_x == 0.0, 1.0, abs_x)
    exponent = tl.floor(tl.log2(safe_abs_x))
    exponent = tl.minimum(tl.maximum(exponent, -6.0), 8.0)
    step = tl.exp2(exponent - 3.0)
    rounded = _round_nearest_even_tl(abs_x / step) * step
    rounded = tl.minimum(rounded, 448.0)
    return rounded * sign


@triton.jit
def _cast_to_fp4_e2m1_tl(x):
    sign = tl.where(x < 0.0, -1.0, 1.0)
    abs_x = tl.abs(x)
    out = tl.full(abs_x.shape, 6.0, tl.float32)
    out = tl.where(abs_x <= 0.25, 0.0, out)
    out = tl.where((abs_x > 0.25) & (abs_x < 0.75), 0.5, out)
    out = tl.where((abs_x >= 0.75) & (abs_x <= 1.25), 1.0, out)
    out = tl.where((abs_x > 1.25) & (abs_x < 1.75), 1.5, out)
    out = tl.where((abs_x >= 1.75) & (abs_x <= 2.5), 2.0, out)
    out = tl.where((abs_x > 2.5) & (abs_x < 3.5), 3.0, out)
    out = tl.where((abs_x >= 3.5) & (abs_x <= 5.0), 4.0, out)
    return out * sign


@triton.jit
def _quantize_cache_tensor_tl(
    cache_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    block_table_ptr,
    cache_stride_0,
    cache_stride_1,
    cache_stride_2,
    cache_stride_3,
    block_table_stride_0,
    sink_size,
    chunk_size,
    block_size,
    group_size: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIMS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)
    context_len = seq_len - query_len
    prev_chunks = tl.maximum(context_len - sink_size, 0) // chunk_size
    cur_chunks = tl.maximum(seq_len - sink_size, 0) // chunk_size
    process = (chunk_idx >= prev_chunks) & (chunk_idx < cur_chunks)

    positions = sink_size + chunk_idx * chunk_size + tl.arange(0, BLOCK_TOKENS)
    token_mask = positions < (sink_size + (chunk_idx + 1) * chunk_size)
    token_mask = token_mask & process
    block_ids = tl.load(
        block_table_ptr + req_idx * block_table_stride_0 + positions // block_size,
        mask=token_mask,
        other=0,
    )
    block_offsets = positions % block_size
    global_amax = tl.full((), 0.0, tl.float32)

    for group_start in tl.static_range(0, BLOCK_DIMS, GROUP_SIZE):
        dims = group_start + tl.arange(0, GROUP_SIZE)
        ptrs = (
            cache_ptr
            + block_ids[:, None] * cache_stride_0
            + block_offsets[:, None] * cache_stride_1
            + head_idx * cache_stride_2
            + dims[None, :] * cache_stride_3
        )
        mask = token_mask[:, None] & (dims[None, :] < HEAD_DIM)
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        group_abs = tl.max(tl.abs(tl.where(mask, x, 0.0)), axis=1)
        group_amax = tl.max(group_abs, axis=0)
        global_amax = tl.maximum(global_amax, group_amax)

    global_scale = tl.where(global_amax == 0.0, 0.0, 2688.0 / global_amax)

    for group_start in tl.static_range(0, BLOCK_DIMS, GROUP_SIZE):
        dims = group_start + tl.arange(0, GROUP_SIZE)
        ptrs = (
            cache_ptr
            + block_ids[:, None] * cache_stride_0
            + block_offsets[:, None] * cache_stride_1
            + head_idx * cache_stride_2
            + dims[None, :] * cache_stride_3
        )
        mask = token_mask[:, None] & (dims[None, :] < HEAD_DIM)
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        token_amax = tl.max(tl.abs(tl.where(mask, x, 0.0)), axis=1)
        block_scale = global_scale * (token_amax / 6.0)
        block_scale = tl.minimum(tl.maximum(block_scale, -448.0), 448.0)
        block_scale = _cast_to_fp8_e4m3fn_tl(block_scale)
        output_scale = tl.where(block_scale == 0.0, 0.0, global_scale / block_scale)
        scaled = tl.minimum(tl.maximum(x * output_scale[:, None], -6.0), 6.0)
        x_fp4 = _cast_to_fp4_e2m1_tl(scaled)
        dequant_scale = tl.where(global_scale == 0.0, 0.0, block_scale / global_scale)
        tl.store(ptrs, x_fp4 * dequant_scale[:, None], mask=mask)


@triton.jit
def _rewrite_nvfp4_kv_cache_kernel(
    key_cache_ptr,
    value_cache_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    block_table_ptr,
    key_stride_0,
    key_stride_1,
    key_stride_2,
    key_stride_3,
    value_stride_0,
    value_stride_1,
    value_stride_2,
    value_stride_3,
    block_table_stride_0,
    sink_size,
    chunk_size,
    block_size,
    group_size: tl.constexpr,
    quant_k: tl.constexpr,
    quant_v: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIMS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    if quant_k:
        _quantize_cache_tensor_tl(
            key_cache_ptr,
            query_start_loc_ptr,
            seq_lens_ptr,
            block_table_ptr,
            key_stride_0,
            key_stride_1,
            key_stride_2,
            key_stride_3,
            block_table_stride_0,
            sink_size,
            chunk_size,
            block_size,
            group_size,
            BLOCK_TOKENS,
            BLOCK_DIMS,
            HEAD_DIM,
            GROUP_SIZE,
        )
    if quant_v:
        _quantize_cache_tensor_tl(
            value_cache_ptr,
            query_start_loc_ptr,
            seq_lens_ptr,
            block_table_ptr,
            value_stride_0,
            value_stride_1,
            value_stride_2,
            value_stride_3,
            block_table_stride_0,
            sink_size,
            chunk_size,
            block_size,
            group_size,
            BLOCK_TOKENS,
            BLOCK_DIMS,
            HEAD_DIM,
            GROUP_SIZE,
        )


@triton.jit
def _fake_quant_nvfp4_query_kernel(
    query_ptr,
    out_ptr,
    query_start_loc_ptr,
    query_stride_0,
    query_stride_1,
    query_stride_2,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    group_size: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIMS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    token_offsets = tl.arange(0, BLOCK_TOKENS)
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    tokens = query_start + token_offsets
    token_mask = token_offsets < query_len
    global_amax = tl.full((), 0.0, tl.float32)

    for group_start in tl.static_range(0, BLOCK_DIMS, GROUP_SIZE):
        dims = group_start + tl.arange(0, GROUP_SIZE)
        ptrs = (
            query_ptr
            + tokens[:, None] * query_stride_0
            + head_idx * query_stride_1
            + dims[None, :] * query_stride_2
        )
        mask = token_mask[:, None] & (dims[None, :] < HEAD_DIM)
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        group_abs = tl.max(tl.abs(tl.where(mask, x, 0.0)), axis=1)
        group_amax = tl.max(group_abs, axis=0)
        global_amax = tl.maximum(global_amax, group_amax)

    global_scale = tl.where(global_amax == 0.0, 0.0, 2688.0 / global_amax)

    for group_start in tl.static_range(0, BLOCK_DIMS, GROUP_SIZE):
        dims = group_start + tl.arange(0, GROUP_SIZE)
        ptrs = (
            query_ptr
            + tokens[:, None] * query_stride_0
            + head_idx * query_stride_1
            + dims[None, :] * query_stride_2
        )
        out_ptrs = (
            out_ptr
            + tokens[:, None] * out_stride_0
            + head_idx * out_stride_1
            + dims[None, :] * out_stride_2
        )
        mask = token_mask[:, None] & (dims[None, :] < HEAD_DIM)
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        token_amax = tl.max(tl.abs(tl.where(mask, x, 0.0)), axis=1)
        block_scale = global_scale * (token_amax / 6.0)
        block_scale = tl.minimum(tl.maximum(block_scale, -448.0), 448.0)
        block_scale = _cast_to_fp8_e4m3fn_tl(block_scale)
        output_scale = tl.where(block_scale == 0.0, 0.0, global_scale / block_scale)
        scaled = tl.minimum(tl.maximum(x * output_scale[:, None], -6.0), 6.0)
        x_fp4 = _cast_to_fp4_e2m1_tl(scaled)
        dequant_scale = tl.where(global_scale == 0.0, 0.0, block_scale / global_scale)
        tl.store(out_ptrs, x_fp4 * dequant_scale[:, None], mask=mask)


def _fake_quant_nvfp4_tensor(
    x: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    original_shape = x.shape
    hidden_size = original_shape[-1]
    x_2d = x.reshape(-1, hidden_size).to(torch.float32)
    x_grouped = x_2d.reshape(x_2d.shape[0], hidden_size // group_size, group_size)

    global_amax = x_grouped.abs().amax()
    global_scale = torch.where(
        global_amax == 0,
        torch.zeros((), device=x.device, dtype=torch.float32),
        torch.tensor(
            FP8_E4M3FN_MAX * FP4_E2M1_MAX,
            device=x.device,
            dtype=torch.float32,
        )
        / global_amax,
    )

    amax = x_grouped.abs().amax(dim=-1, keepdim=True)
    block_scale = global_scale * (amax / FP4_E2M1_MAX)
    block_scale = torch.clamp(block_scale, min=-FP8_E4M3FN_MAX, max=FP8_E4M3FN_MAX)
    block_scale = cast_to_fp8_e4m3fn(block_scale).to(torch.float32)

    output_scale = torch.where(
        block_scale == 0,
        torch.zeros_like(block_scale),
        global_scale / block_scale,
    )
    scaled = torch.clamp(
        x_grouped * output_scale,
        min=-FP4_E2M1_MAX,
        max=FP4_E2M1_MAX,
    )
    x_fp4 = cast_to_fp4_e2m1(scaled)
    dequant_scale = torch.where(
        global_scale == 0,
        torch.zeros_like(block_scale),
        block_scale / global_scale,
    )
    return (x_fp4 * dequant_scale).reshape(original_shape).to(output_dtype)
