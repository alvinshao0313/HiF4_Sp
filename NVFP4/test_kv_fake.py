import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3rdparty" / "vllm"))

from vllm.model_executor.layers.attention.kv_fake_quant import (  # noqa: E402
    Nvfp4KVQuantConfig,
    fake_quant_hif4_tensor,
    fake_quant_hif4_new_kv,
    fake_quant_kv_tensor,
    fake_quant_mxfp8_tensor,
    fake_quant_nvfp4_query,
    fake_quant_nvfp4_per_head_chunk,
    get_kv_quant_config,
    rewrite_completed_kv_chunks,
)
from vllm.model_executor.layers.quantization import hif4_fake  # noqa: E402


@dataclass
class Metadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor


def make_metadata(context_len: int, query_len: int, block_size: int = 4) -> Metadata:
    seq_len = context_len + query_len
    num_blocks = (seq_len + block_size - 1) // block_size
    return Metadata(
        query_start_loc=torch.tensor([0, query_len], dtype=torch.long),
        seq_lens=torch.tensor([seq_len], dtype=torch.long),
        block_table=torch.arange(num_blocks, dtype=torch.long).reshape(1, num_blocks),
    )


def make_cache(seq_len: int, block_size: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(123)
    num_blocks = (seq_len + block_size - 1) // block_size
    key = torch.randn(num_blocks, block_size, 2, 16, dtype=torch.float32) * 7
    value = torch.randn(num_blocks, block_size, 2, 16, dtype=torch.float32) * 7
    return key, value


def make_noncontiguous_cache(
    seq_len: int,
    block_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(123)
    num_blocks = (seq_len + block_size - 1) // block_size
    key_base = torch.randn(num_blocks, block_size, 2, 32, dtype=torch.float32) * 7
    value_base = torch.randn(num_blocks, block_size, 2, 32, dtype=torch.float32) * 7
    key = key_base[..., ::2]
    value = value_base[..., ::2]
    assert not key.is_contiguous()
    assert not value.is_contiguous()
    return key, value


def flat(cache: torch.Tensor) -> torch.Tensor:
    return cache.view(-1, cache.shape[-2], cache.shape[-1])


def test_sink_tokens_are_not_quantized():
    config = Nvfp4KVQuantConfig(chunk_size=4, sink_size=4, target="kv")
    key, value = make_cache(seq_len=4)
    key_before = key.clone()
    value_before = value.clone()

    rewrite_completed_kv_chunks(key, value, make_metadata(0, 4), config)

    torch.testing.assert_close(key, key_before)
    torch.testing.assert_close(value, value_before)


def test_residual_tail_is_not_quantized():
    config = Nvfp4KVQuantConfig(chunk_size=4, sink_size=2, target="kv")
    key, value = make_cache(seq_len=5)
    key_before = key.clone()
    value_before = value.clone()

    rewrite_completed_kv_chunks(key, value, make_metadata(0, 5), config)

    torch.testing.assert_close(key, key_before)
    torch.testing.assert_close(value, value_before)


def test_prefill_quantizes_full_non_sink_chunks_only():
    config = Nvfp4KVQuantConfig(chunk_size=4, sink_size=2, target="kv")
    key, value = make_cache(seq_len=8)
    key_before = key.clone()
    value_before = value.clone()

    rewrite_completed_kv_chunks(key, value, make_metadata(0, 8), config)

    torch.testing.assert_close(flat(key)[:2], flat(key_before)[:2])
    torch.testing.assert_close(flat(value)[:2], flat(value_before)[:2])
    torch.testing.assert_close(flat(key)[6:8], flat(key_before)[6:8])
    torch.testing.assert_close(flat(value)[6:8], flat(value_before)[6:8])
    torch.testing.assert_close(
        flat(key)[2:6],
        fake_quant_nvfp4_per_head_chunk(flat(key_before)[2:6]),
    )
    torch.testing.assert_close(
        flat(value)[2:6],
        fake_quant_nvfp4_per_head_chunk(flat(value_before)[2:6]),
    )


def test_decode_rewrites_completed_residual_chunk():
    config = Nvfp4KVQuantConfig(chunk_size=4, sink_size=2, target="kv")
    key, value = make_cache(seq_len=6)
    key_before = key.clone()
    value_before = value.clone()

    rewrite_completed_kv_chunks(key, value, make_metadata(5, 1), config)

    torch.testing.assert_close(flat(key)[:2], flat(key_before)[:2])
    torch.testing.assert_close(flat(value)[:2], flat(value_before)[:2])
    torch.testing.assert_close(
        flat(key)[2:6],
        fake_quant_nvfp4_per_head_chunk(flat(key_before)[2:6]),
    )
    torch.testing.assert_close(
        flat(value)[2:6],
        fake_quant_nvfp4_per_head_chunk(flat(value_before)[2:6]),
    )


def test_rewrite_supports_noncontiguous_kv_cache():
    config = Nvfp4KVQuantConfig(chunk_size=4, sink_size=2, target="kv")
    key, value = make_noncontiguous_cache(seq_len=6)
    key_before = key.clone()
    value_before = value.clone()

    rewrite_completed_kv_chunks(key, value, make_metadata(0, 6), config)

    sink_block_ids = torch.tensor([0, 0], dtype=torch.long)
    sink_offsets = torch.tensor([0, 1], dtype=torch.long)
    chunk_block_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    chunk_offsets = torch.tensor([2, 3, 0, 1], dtype=torch.long)
    torch.testing.assert_close(
        key[sink_block_ids, sink_offsets],
        key_before[sink_block_ids, sink_offsets],
    )
    torch.testing.assert_close(
        value[sink_block_ids, sink_offsets],
        value_before[sink_block_ids, sink_offsets],
    )
    torch.testing.assert_close(
        key[chunk_block_ids, chunk_offsets],
        fake_quant_nvfp4_per_head_chunk(key_before[chunk_block_ids, chunk_offsets]),
    )
    torch.testing.assert_close(
        value[chunk_block_ids, chunk_offsets],
        fake_quant_nvfp4_per_head_chunk(value_before[chunk_block_ids, chunk_offsets]),
    )


def test_target_k_only_does_not_change_value_cache():
    config = Nvfp4KVQuantConfig(chunk_size=4, sink_size=2, target="k")
    key, value = make_cache(seq_len=6)
    value_before = value.clone()

    rewrite_completed_kv_chunks(key, value, make_metadata(0, 6), config)

    torch.testing.assert_close(value, value_before)


def test_target_v_only_does_not_change_key_cache():
    config = Nvfp4KVQuantConfig(chunk_size=4, sink_size=2, target="v")
    key, value = make_cache(seq_len=6)
    key_before = key.clone()

    rewrite_completed_kv_chunks(key, value, make_metadata(0, 6), config)

    torch.testing.assert_close(key, key_before)


def test_hif4_sink_tokens_are_not_quantized():
    config = Nvfp4KVQuantConfig(
        chunk_size=4,
        sink_size=4,
        target="kv",
        format="hif4",
    )
    key = torch.randn(4, 2, 16, dtype=torch.float32) * 7
    value = torch.randn(4, 2, 16, dtype=torch.float32) * 7
    key_before = key.clone()
    value_before = value.clone()

    key, value = fake_quant_hif4_new_kv(key, value, make_metadata(0, 4), config)

    torch.testing.assert_close(key, key_before)
    torch.testing.assert_close(value, value_before)


def test_hif4_quantizes_new_non_sink_tokens_immediately():
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=2,
        target="kv",
        format="hif4",
    )
    key = torch.randn(5, 2, 16, dtype=torch.float32) * 7
    value = torch.randn(5, 2, 16, dtype=torch.float32) * 7
    key_before = key.clone()
    value_before = value.clone()

    key, value = fake_quant_hif4_new_kv(key, value, make_metadata(0, 5), config)

    torch.testing.assert_close(key[:2], key_before[:2])
    torch.testing.assert_close(value[:2], value_before[:2])
    torch.testing.assert_close(
        key[2:5],
        hif4_fake.hif4_fake_quantize_hifx4(key_before[2:5]),
    )
    torch.testing.assert_close(
        value[2:5],
        hif4_fake.hif4_fake_quantize_hifx4(value_before[2:5]),
    )


@pytest.mark.parametrize("quant_format", ["hif4", "hif4-1"])
def test_hif4_protects_recent_tokens_during_prefill(quant_format: str):
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=2,
        target="kv",
        format=quant_format,
        recent_size=3,
    )
    key = torch.randn(8, 2, 16, dtype=torch.float32) * 7
    value = torch.randn(8, 2, 16, dtype=torch.float32) * 7
    key_before = key.clone()
    value_before = value.clone()

    key, value = fake_quant_hif4_new_kv(key, value, make_metadata(0, 8), config)

    torch.testing.assert_close(key[:2], key_before[:2])
    torch.testing.assert_close(value[:2], value_before[:2])
    torch.testing.assert_close(key[2:5], fake_quant_kv_tensor(key_before[2:5], config))
    torch.testing.assert_close(
        value[2:5], fake_quant_kv_tensor(value_before[2:5], config)
    )
    torch.testing.assert_close(key[5:], key_before[5:])
    torch.testing.assert_close(value[5:], value_before[5:])


def test_hif4_sink_and_recent_windows_can_overlap():
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=4,
        target="kv",
        format="hif4",
        recent_size=5,
    )
    key = torch.randn(8, 2, 16, dtype=torch.float32) * 7
    value = torch.randn(8, 2, 16, dtype=torch.float32) * 7
    key_before = key.clone()
    value_before = value.clone()

    key, value = fake_quant_hif4_new_kv(key, value, make_metadata(0, 8), config)

    torch.testing.assert_close(key, key_before)
    torch.testing.assert_close(value, value_before)


def test_hif4_recent_protection_handles_multiple_requests_and_padding():
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=1,
        target="kv",
        format="hif4",
        recent_size=2,
    )
    metadata = Metadata(
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.long),
        seq_lens=torch.tensor([6, 3], dtype=torch.long),
        block_table=torch.empty(0, dtype=torch.long),
    )
    metadata.num_actual_tokens = 5
    key = torch.randn(7, 2, 16, dtype=torch.float32) * 7
    value = torch.randn(7, 2, 16, dtype=torch.float32) * 7
    key_before = key.clone()
    value_before = value.clone()

    key, value = fake_quant_hif4_new_kv(key, value, metadata, config)

    torch.testing.assert_close(key[:1], fake_quant_kv_tensor(key_before[:1], config))
    torch.testing.assert_close(
        value[:1], fake_quant_kv_tensor(value_before[:1], config)
    )
    torch.testing.assert_close(key[1:], key_before[1:])
    torch.testing.assert_close(value[1:], value_before[1:])


def test_hif4_decode_quantizes_token_leaving_recent_window():
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=2,
        target="kv",
        format="hif4",
        recent_size=3,
    )
    key, value = make_cache(seq_len=7)
    key_before = key.clone()
    value_before = value.clone()

    rewrite_completed_kv_chunks(key, value, make_metadata(6, 1), config)

    torch.testing.assert_close(flat(key)[:3], flat(key_before)[:3])
    torch.testing.assert_close(flat(value)[:3], flat(value_before)[:3])
    torch.testing.assert_close(
        flat(key)[3:4], fake_quant_kv_tensor(flat(key_before)[3:4], config)
    )
    torch.testing.assert_close(
        flat(value)[3:4], fake_quant_kv_tensor(flat(value_before)[3:4], config)
    )
    torch.testing.assert_close(flat(key)[4:7], flat(key_before)[4:7])
    torch.testing.assert_close(flat(value)[4:7], flat(value_before)[4:7])


@pytest.mark.parametrize(("target", "unchanged"), [("k", "value"), ("v", "key")])
def test_hif4_recent_rewrite_respects_target(target: str, unchanged: str):
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=2,
        target=target,
        format="hif4",
        recent_size=3,
    )
    key, value = make_cache(seq_len=7)
    key_before = key.clone()
    value_before = value.clone()

    rewrite_completed_kv_chunks(key, value, make_metadata(6, 1), config)

    if unchanged == "key":
        torch.testing.assert_close(key, key_before)
    else:
        torch.testing.assert_close(value, value_before)


def test_hif4_ignores_padded_tokens():
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=2,
        target="kv",
        format="hif4",
    )
    metadata = make_metadata(0, 5)
    metadata.num_actual_tokens = 5
    key = torch.randn(8, 2, 16, dtype=torch.float32) * 7
    value = torch.randn(8, 2, 16, dtype=torch.float32) * 7
    key_before = key.clone()
    value_before = value.clone()

    key, value = fake_quant_hif4_new_kv(key, value, metadata, config)

    torch.testing.assert_close(key[5:], key_before[5:])
    torch.testing.assert_close(value[5:], value_before[5:])


def test_hif4_decode_quantizes_only_current_new_non_sink_token():
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=2,
        target="kv",
        format="hif4",
    )
    key = torch.randn(1, 2, 16, dtype=torch.float32) * 7
    value = torch.randn(1, 2, 16, dtype=torch.float32) * 7
    key_before = key.clone()
    value_before = value.clone()

    key, value = fake_quant_hif4_new_kv(key, value, make_metadata(5, 1), config)

    torch.testing.assert_close(
        key,
        hif4_fake.hif4_fake_quantize_hifx4(key_before),
    )
    torch.testing.assert_close(
        value,
        hif4_fake.hif4_fake_quantize_hifx4(value_before),
    )


def test_hif4_target_k_only_does_not_change_value_cache():
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=2,
        target="k",
        format="hif4",
    )
    key = torch.randn(5, 2, 16, dtype=torch.float32) * 7
    value = torch.randn(5, 2, 16, dtype=torch.float32) * 7
    value_before = value.clone()

    _, value = fake_quant_hif4_new_kv(key, value, make_metadata(0, 5), config)

    torch.testing.assert_close(value, value_before)


def test_hif4_target_v_only_does_not_change_key_cache():
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=2,
        target="v",
        format="hif4",
    )
    key = torch.randn(5, 2, 16, dtype=torch.float32) * 7
    value = torch.randn(5, 2, 16, dtype=torch.float32) * 7
    key_before = key.clone()

    key, _ = fake_quant_hif4_new_kv(key, value, make_metadata(0, 5), config)

    torch.testing.assert_close(key, key_before)


def test_hif4_quantization_direction_is_last_dimension():
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=0,
        target="kv",
        format="hif4",
    )
    x = torch.randn(3, 2, 16, dtype=torch.float32) * 7

    actual = fake_quant_kv_tensor(x, config)
    expected = hif4_fake.hif4_fake_quantize_hifx4(x)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hif4_triton_matches_reference_cuda():
    torch.manual_seed(321)
    x = (torch.randn(5, 3, 80, dtype=torch.float32, device="cuda") * 7).transpose(0, 1)
    assert not x.is_contiguous()

    actual = fake_quant_hif4_tensor(x)
    expected = hif4_fake.hif4_fake_quantize_hifx4(x.cpu()).cuda()

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("quant_format", ["hif4", "hif4-1"])
def test_hif4_new_kv_triton_matches_reference_cuda(quant_format: str):
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=2,
        target="kv",
        format=quant_format,
        recent_size=2,
    )
    torch.manual_seed(456)
    key_cpu = torch.randn(5, 2, 80, dtype=torch.float32) * 7
    value_cpu = torch.randn(5, 2, 80, dtype=torch.float32) * 7
    metadata_cpu = Metadata(
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.long),
        seq_lens=torch.tensor([6, 3], dtype=torch.long),
        block_table=torch.empty(0, dtype=torch.long),
    )
    metadata_cpu.num_actual_tokens = 5
    metadata_cuda = Metadata(
        query_start_loc=metadata_cpu.query_start_loc.cuda(),
        seq_lens=metadata_cpu.seq_lens.cuda(),
        block_table=torch.empty(0, dtype=torch.long, device="cuda"),
    )
    metadata_cuda.num_actual_tokens = 5

    expected_key, expected_value = fake_quant_hif4_new_kv(
        key_cpu,
        value_cpu,
        metadata_cpu,
        config,
    )
    actual_key, actual_value = fake_quant_hif4_new_kv(
        key_cpu.cuda(),
        value_cpu.cuda(),
        metadata_cuda,
        config,
    )

    torch.testing.assert_close(actual_key.cpu(), expected_key, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_value.cpu(), expected_value, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("quant_format", ["hif4", "hif4-1"])
def test_hif4_recent_rewrite_triton_matches_reference_cuda(quant_format: str):
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=2,
        target="kv",
        format=quant_format,
        recent_size=2,
    )
    torch.manual_seed(789)
    key_cpu = torch.randn(4, 4, 2, 16, dtype=torch.float32) * 7
    value_cpu = torch.randn(4, 4, 2, 16, dtype=torch.float32) * 7
    key_cuda = key_cpu.cuda()
    value_cuda = value_cpu.cuda()
    metadata_cpu = Metadata(
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.long),
        seq_lens=torch.tensor([7, 5], dtype=torch.long),
        block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
    )
    metadata_cuda = Metadata(
        query_start_loc=metadata_cpu.query_start_loc.cuda(),
        seq_lens=metadata_cpu.seq_lens.cuda(),
        block_table=metadata_cpu.block_table.cuda(),
    )
    metadata_cuda.max_query_len = 2

    rewrite_completed_kv_chunks(key_cpu, value_cpu, metadata_cpu, config)
    rewrite_completed_kv_chunks(key_cuda, value_cuda, metadata_cuda, config)

    torch.testing.assert_close(key_cuda.cpu(), key_cpu, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(value_cuda.cpu(), value_cpu, rtol=1e-3, atol=1e-3)


def test_query_quant_is_disabled_by_default():
    config = Nvfp4KVQuantConfig(chunk_size=4, sink_size=0, target="kv")
    query = torch.randn(4, 2, 16, dtype=torch.float32) * 7

    actual = fake_quant_nvfp4_query(query, make_metadata(0, 4), config)

    torch.testing.assert_close(actual, query)


def test_query_quant_enabled_changes_query():
    config = Nvfp4KVQuantConfig(
        chunk_size=4,
        sink_size=0,
        target="kv",
        query_format="kv",
    )
    query = torch.randn(4, 2, 16, dtype=torch.float32) * 7

    actual = fake_quant_nvfp4_query(query, make_metadata(0, 4), config)

    torch.testing.assert_close(actual, fake_quant_nvfp4_per_head_chunk(query))


def test_get_kv_quant_config_parses_query_flag():
    config = get_kv_quant_config(
        {
            "kv_quant_format": "nvfp4",
            "kv_quant_query": "enabled",
        }
    )

    assert config is not None
    assert config.query_format == "kv"


def test_get_kv_quant_config_parses_mxfp8_query():
    config = get_kv_quant_config(
        {
            "kv_quant_format": "hif4-1",
            "kv_quant_query": "mxfp8",
        }
    )

    assert config is not None
    assert config.query_format == "mxfp8"


@pytest.mark.parametrize("query_format", ["enabled", "mxfp8"])
def test_get_kv_quant_config_rejects_query_quant_without_kv(query_format: str):
    with pytest.raises(ValueError, match="requires KV quantization"):
        get_kv_quant_config(
            {
                "kv_quant_format": "none",
                "kv_quant_query": query_format,
            }
        )


@pytest.mark.parametrize("quant_format", ["hif4", "hif4-1"])
def test_query_quant_enabled_follows_hif4_format(quant_format: str):
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=16,
        target="kv",
        format=quant_format,
        recent_size=128,
        query_format="kv",
    )
    query = torch.randn(4, 2, 64, dtype=torch.float32) * 7

    actual = fake_quant_nvfp4_query(query, make_metadata(0, 4), config)

    torch.testing.assert_close(actual, fake_quant_hif4_tensor(query, quant_format))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mxfp8_query_preserves_shape_dtype_and_quantizes_per_block(dtype: torch.dtype):
    torch.manual_seed(2468)
    query = torch.randn(3, 2, 64, dtype=dtype) * 7
    query[..., :32] *= 128

    actual = fake_quant_mxfp8_tensor(query)
    first_block = fake_quant_mxfp8_tensor(query[..., :32])
    second_block = fake_quant_mxfp8_tensor(query[..., 32:])

    assert actual.shape == query.shape
    assert actual.dtype == query.dtype
    torch.testing.assert_close(actual[..., :32], first_block)
    torch.testing.assert_close(actual[..., 32:], second_block)


@pytest.mark.parametrize("quant_format", ["nvfp4", "hif4", "hif4-1"])
def test_mxfp8_query_ignores_kv_format_and_protection_windows(quant_format: str):
    config = Nvfp4KVQuantConfig(
        chunk_size=64,
        sink_size=16,
        target="kv",
        format=quant_format,
        recent_size=128,
        query_format="mxfp8",
    )
    query = torch.randn(4, 2, 64, dtype=torch.float32) * 7

    actual = fake_quant_nvfp4_query(query, make_metadata(0, 4), config)

    torch.testing.assert_close(actual, fake_quant_mxfp8_tensor(query))


def test_mxfp8_query_rejects_non_multiple_of_32_head_dim():
    query = torch.randn(4, 2, 48, dtype=torch.float32)

    with pytest.raises(ValueError, match="head_dim divisible by 32"):
        fake_quant_mxfp8_tensor(query)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mxfp8_triton_matches_reference_cuda(dtype: torch.dtype):
    torch.manual_seed(1357)
    base = torch.randn(3, 2, 128, dtype=dtype) * 7
    query_cpu = base[..., ::2]
    query_cuda = base.cuda()[..., ::2]
    assert not query_cpu.is_contiguous()
    assert not query_cuda.is_contiguous()

    expected = fake_quant_mxfp8_tensor(query_cpu)
    actual = fake_quant_mxfp8_tensor(query_cuda).cpu()

    torch.testing.assert_close(actual, expected)


def test_get_kv_quant_config_parses_recent_size():
    config = get_kv_quant_config(
        {
            "kv_quant_format": "hif4",
            "kv_quant_recent_size": 128,
        }
    )

    assert config is not None
    assert config.recent_size == 128


@pytest.mark.parametrize("quant_format", ["none", "nvfp4"])
def test_get_kv_quant_config_rejects_recent_size_for_non_hif4(quant_format: str):
    with pytest.raises(ValueError, match="only supports hif4/hif4-1"):
        get_kv_quant_config(
            {
                "kv_quant_format": quant_format,
                "kv_quant_recent_size": 1,
            }
        )
