"""GPU numerical checks for the isolated M3.1 SM80 backport."""

import pytest
import torch

from vllm.model_executor.models.minimax_m3.attention import cache_views
from vllm.model_executor.models.minimax_m3.ops.index_topk import (
    minimax_m3_index_decode, minimax_m3_index_score, minimax_m3_index_topk,
)
from vllm.model_executor.models.minimax_m3.ops.sparse_attn import (
    minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def reference(q, iq, main, index, table, q_lens, prefixes):
    """FP32 dense math, masked to independently selected sparse blocks."""
    heads, kv_heads = q.shape[1], main.shape[1]
    group = heads // kv_heads
    selected = torch.full((kv_heads, len(q), 16), -1, dtype=torch.int32, device=q.device)
    output = torch.empty_like(q)
    scores = []
    start = 0
    for req, (ql, prefix) in enumerate(zip(q_lens, prefixes)):
        length = prefix + ql
        pos = torch.arange(length, device=q.device)
        pages = table[req, pos // 128]
        ik = index[pages, pos % 128].float()
        kv = main[pages, :, pos % 128].float()
        for local in range(ql):
            row = start + local
            end = prefix + local + 1
            block_count = (end + 127) // 128
            dot = iq[row].float() @ ik[:end].T
            padded = torch.nn.functional.pad(dot, (0, block_count * 128 - end), value=-torch.inf)
            block_score = padded.reshape(kv_heads, block_count, 128).amax(-1)
            scores.append(block_score)
            # The last local block is always included, as in the checkpoint.
            ranking = block_score.clone()
            ranking[:, -1] = 1e29
            ids = ranking.topk(min(16, block_count), dim=-1).indices
            selected[:, row, :ids.shape[-1]] = ids
            for head in range(kv_heads):
                visible = (pos < end) & ((pos[:, None] // 128) == ids[head]).any(-1)
                key, value = kv[visible, head].split(128, -1)
                query = q[row, head * group:(head + 1) * group].float()
                prob = (query @ key.T * 128 ** -0.5).softmax(-1)
                output[row, head * group:(head + 1) * group] = (prob @ value).bfloat16()
        start += ql
    return selected, output, scores


@pytest.mark.parametrize("heads,kv_heads", [(64, 4), (8, 1)])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
@torch.inference_mode()
def test_real_head_shapes_paged_index_and_attention(heads, kv_heads, mode):
    torch.manual_seed(2031)
    # Match the combined cache's actual noncontiguous K/V and index views.
    q_lens = [9, 3] if mode == "prefill" else [1, 1]
    prefixes = [3600, 127]
    pages_per_req = 30
    table = torch.randperm(2 * pages_per_req, device="cuda", dtype=torch.int32).reshape(2, -1)
    cache = torch.randn(2 * pages_per_req, 128, (2 * kv_heads + 1) * 128,
                        dtype=torch.bfloat16, device="cuda") * 0.3
    main, index = cache_views(cache, kv_heads)
    cu = torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int32, device="cuda")
    prefix_tensor = torch.tensor(prefixes, dtype=torch.int32, device="cuda")
    q = torch.randn(sum(q_lens), heads, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    iq = torch.randn(sum(q_lens), kv_heads, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    repeats = 3 if mode == "decode" else 1
    for step in range(repeats):
        current_prefixes = [p + step for p in prefixes]
        seq = torch.tensor([p + ql for p, ql in zip(current_prefixes, q_lens)],
                           dtype=torch.int32, device="cuda")
        if mode == "prefill":
            score = minimax_m3_index_score(iq, index, table, cu, seq, prefix_tensor,
                max(q_lens), int(seq.max()), kv_heads)
            topk = minimax_m3_index_topk(score, cu, prefix_tensor, max(q_lens), 16, 0, 1)
        else:
            topk = minimax_m3_index_decode(iq, index, table, seq, int(seq.max()),
                                         16, 0, 1, kv_heads, 1, 1)
        expected_ids, expected, expected_scores = reference(
            q, iq, main, index, table, q_lens, current_prefixes)
        torch.testing.assert_close(topk.sort(-1).values, expected_ids.sort(-1).values,
                                   rtol=0, atol=0)
        if mode == "prefill":
            for row, expected_score in enumerate(expected_scores):
                torch.testing.assert_close(score[:, row, :expected_score.shape[-1]],
                                           expected_score, rtol=1e-4, atol=1e-4)
        actual = torch.empty_like(q)
        if mode == "prefill":
            minimax_m3_sparse_attn(q, main, topk, table, cu, seq, prefix_tensor,
                                  max(q_lens), kv_heads, 128 ** -0.5, actual)
        else:
            minimax_m3_sparse_attn_decode(q, main, topk, table, seq, kv_heads,
                                         128 ** -0.5, actual, 1)
        torch.cuda.synchronize()
        # BF16 softmax probabilities/partial outputs are rounded inside the
        # Triton kernel; the reference retains FP32 until its final cast.
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.003)
        error = (actual.float() - expected.float()).square().sum()
        nmse = error / expected.float().square().sum().clamp_min(1e-20)
        assert nmse < 1e-4
        print({"mode": mode, "q_heads": heads, "kv_heads": kv_heads, "step": step,
               "max_abs": (actual - expected).abs().max().item(), "nmse": nmse.item()})
