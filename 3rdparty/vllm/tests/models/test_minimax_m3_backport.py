"""CPU contracts for the isolated MiniMax M3.1 backport; no engine stubs."""

import torch
import pytest

from vllm.model_executor.models.minimax_m3.attention import cache_views, insert_cache
from vllm.model_executor.models.minimax_m3.model import expert_parameter
from vllm.model_executor.models.minimax_m3.quantization import nvfp4_per_token_qdq, swiglu_oai
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant_dequant


def test_lighteval_sampling_keeps_vllm_top_k_disabled():
    from vllm import SamplingParams
    from vllm.model_executor.models.minimax_m3.evaluation import generation_parameters
    parameters = generation_parameters(1.0, 0.95, 38912)
    actual = SamplingParams(**parameters.to_vllm_dict())
    assert actual.top_k == 0
    assert (actual.temperature, actual.top_p, actual.min_p, actual.max_tokens) == (1.0, 0.95, 0.0, 38912)


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_main_and_index_pages_share_slots_without_corrupting_heads(heads):
    cache = torch.full((3, 128, (2 * heads + 1) * 128), -9., dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(19)
    k = torch.randn(4, heads * 128, generator=gen).bfloat16()
    v = torch.randn(4, heads * 128, generator=gen).bfloat16()
    ik = torch.randn(4, 128, generator=gen).bfloat16()
    slots = torch.tensor([0, 129, 258, -1])
    insert_cache(cache, k, v, ik, slots, heads)
    main, index = cache_views(cache, heads)
    for row, slot in enumerate(slots[:3].tolist()):
        page, offset = divmod(slot, 128)
        torch.testing.assert_close(main[page, :, offset, :128], k[row].reshape(heads, 128))
        torch.testing.assert_close(main[page, :, offset, 128:], v[row].reshape(heads, 128))
        torch.testing.assert_close(index[page, offset], ik[row])
    assert torch.all(cache[2, 127] == -9.)
    # Incremental decode updates a later slot while cached prefix remains intact.
    prefix = cache[0, 0].clone()
    insert_cache(cache, k[:1], v[:1], ik[:1], torch.tensor([1]), heads)
    torch.testing.assert_close(cache[0, 0], prefix)


def test_nvfp4_each_token_uses_its_own_global_scale():
    gen = torch.Generator().manual_seed(71)
    x = torch.randn(4, 64, generator=gen).bfloat16()
    x[0].zero_()
    x[1] *= 0.01
    x[3] *= 100
    actual = nvfp4_per_token_qdq(x)
    for row in range(1, len(x)):
        global_scale = 2688. / x[row].float().abs().max()
        reference = ref_nvfp4_quant_dequant(x[row:row+1], global_scale, 16)
        torch.testing.assert_close(actual[row:row+1], reference, rtol=0, atol=0)
    assert torch.equal(actual[0], x[0])
    torch.testing.assert_close(nvfp4_per_token_qdq(x[1:2]), actual[1:2], rtol=0, atol=0)


def test_swiglu_layout_and_clamp_are_not_gpt_oss_interleaved():
    x = torch.tensor([[9., -2., 9., -9.]], dtype=torch.bfloat16)
    gate = torch.tensor([[7., -2.]])
    up = torch.tensor([[7., -7.]])
    expected = (gate * torch.sigmoid(1.702 * gate) * (up + 1)).bfloat16()
    torch.testing.assert_close(swiglu_oai(x), expected, rtol=0, atol=0)


def test_packed_weight_multiplier_is_inverted_and_weights_stay_packed():
    name = "language_model.model.layers.3.block_sparse_moe.experts.0.w1."
    source = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    target, eid, shard, packed = expert_parameter(name + "weight_packed", source)
    assert target.endswith(".experts.w13_weight") and eid == 0 and shard == "w1"
    assert packed is source and packed.dtype == torch.uint8
    _, _, _, scale = expert_parameter(name + "weight_global_scale", torch.tensor(15123.6923828125))
    torch.testing.assert_close(scale, torch.tensor(1. / 15123.6923828125))


def test_opt_in_registration_does_not_replace_existing_quantizers():
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.models.minimax_m3.integration import register
    previous = {name: get_quantization_config(name) for name in ("mxfp8", "modelopt", "compressed-tensors")}
    register()
    assert all(get_quantization_config(name) is cls for name, cls in previous.items())


def test_serialized_mxfp8_keeps_block_scales_and_executes_bf16_linear():
    from vllm.model_executor.models.minimax_m3.quantization import MiniMaxM3MXFP8Linear
    method = MiniMaxM3MXFP8Linear()
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(32, 64, dtype=torch.float8_e4m3fn),
        requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.full((32, 2), 127, dtype=torch.uint8),
        requires_grad=False)
    method.process_weights_after_loading(layer)
    actual = method.apply(layer, torch.ones(2, 64, dtype=torch.bfloat16))
    torch.testing.assert_close(actual, torch.full((2, 32), 64., dtype=torch.bfloat16),
        rtol=0, atol=0)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.shape == (32, 2)


def test_mixed_prefill_decode_metadata_preserves_absolute_context_positions():
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.model_executor.models.minimax_m3.attention import MiniMaxM3MetadataBuilder
    starts = torch.tensor([0, 1, 3], dtype=torch.int32)
    common = CommonAttentionMetadata(query_start_loc=starts, query_start_loc_cpu=starts,
        seq_lens=torch.tensor([258, 130], dtype=torch.int32), num_reqs=2,
        num_actual_tokens=3, max_query_len=2, max_seq_len=258,
        block_table_tensor=torch.tensor([[2, 0, 3], [1, 4, 0]], dtype=torch.int32),
        slot_mapping=torch.tensor([386, 512, 513]))
    spec = MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=384,
        dtype=torch.bfloat16)
    builder = MiniMaxM3MetadataBuilder(spec, ["language_model.model.layers.0.self_attn.attn"],
        VllmConfig(), torch.device("cpu"))
    md = builder.build(0, common)
    assert (md.num_decodes, md.num_decode_tokens, md.num_prefills) == (1, 1, 1)
    assert md.prefix_lens.tolist() == [257, 128]
    assert md.query_start_loc.tolist() == [0, 1, 3]
    assert md.block_table is common.block_table_tensor
