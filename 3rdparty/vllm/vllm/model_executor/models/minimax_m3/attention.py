# SPDX-License-Identifier: Apache-2.0
"""Paged MSA adapter for vLLM 0.19; kernels are from upstream v0.33.

K, V, and the shared index key occupy a single page allocation. This keeps
their page identities aligned without changing the old engine's cache manager.
The underlying attention and block-selection kernels are unchanged upstream.
"""

from dataclasses import dataclass

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.v1.attention.backend import (
    AttentionBackend, AttentionCGSupport, AttentionImplBase, AttentionMetadata,
    AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from .ops.index_topk import minimax_m3_index_decode, minimax_m3_index_score, minimax_m3_index_topk
from .ops.sparse_attn import minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode


class MiniMaxM3Backend(AttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16"]

    @staticmethod
    def get_name():
        return "MINIMAX_M3_TEXT"

    @staticmethod
    def get_impl_cls():
        return MiniMaxM3Impl

    @staticmethod
    def get_builder_cls():
        return MiniMaxM3MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        return num_blocks, block_size, head_size

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [128]

    @classmethod
    def is_sparse(cls):
        return True

    @classmethod
    def get_supported_head_sizes(cls):
        return [128, 384, 640, 1152]


@dataclass
class MiniMaxM3Metadata(AttentionMetadata):
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    prefix_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    max_query_len: int
    max_seq_len: int


class MiniMaxM3MetadataBuilder(AttentionMetadataBuilder[MiniMaxM3Metadata]):
    _cudagraph_support = AttentionCGSupport.NEVER
    reorder_batch_threshold = 1

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = common_attn_metadata
        nd, np, nt, _ = split_decodes_and_prefills(md, decode_threshold=1, require_uniform=True)
        return MiniMaxM3Metadata(
            md.query_start_loc, md.seq_lens, md.compute_num_computed_tokens(),
            md.block_table_tensor, md.slot_mapping, md.num_actual_tokens,
            nd, nt, np, md.max_query_len, md.max_seq_len,
        )


class MiniMaxM3Impl(AttentionImplBase[MiniMaxM3Metadata]):
    """Cache/backend marker; the owning layer runs both MSA kernels."""

    def forward(self, *args, **kwargs):
        raise RuntimeError("MiniMax M3 attention must run through its owning layer")


class MiniMaxGemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * (1.0 + self.weight.float())).to(x.dtype)


def cache_views(cache, num_kv_heads, head_dim=128):
    """Views of [page, token, (K/V heads + one index head) * dim]."""
    pages, block_size, width = cache.shape
    if block_size != 128 or width != (2 * num_kv_heads + 1) * head_dim:
        raise ValueError("Invalid MiniMax M3 combined KV/index page")
    main = cache[..., : 2 * num_kv_heads * head_dim].view(
        pages, block_size, num_kv_heads, 2 * head_dim
    ).transpose(1, 2)
    return main, cache[..., 2 * num_kv_heads * head_dim:]


def insert_cache(cache, k, v, index_k, slots, num_kv_heads):
    n = slots.numel()
    valid = slots >= 0
    kv = torch.cat((k.reshape(n, num_kv_heads, 128),
                    v.reshape(n, num_kv_heads, 128)), -1)
    values = torch.cat((kv.reshape(n, -1), index_k), -1)
    cache.view(-1, cache.shape[-1])[slots[valid]] = values[valid]


class MiniMaxM3Attention(nn.Module, AttentionLayerBase):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        vc = get_current_vllm_config()
        if not vc.model_config.enforce_eager:
            raise ValueError("MiniMax M3.1 backport currently requires enforce_eager=True")
        if vc.cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("MiniMax M3.1 SM80 backport currently requires BF16 KV/index cache")
        if vc.cache_config.block_size != 128:
            raise ValueError("MiniMax M3.1 requires block_size=128")
        tp = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % tp or not (
            config.num_key_value_heads % tp == 0 or tp % config.num_key_value_heads == 0
        ):
            raise ValueError("Tensor parallel size does not divide/replicate MiniMax heads")
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = max(1, config.num_key_value_heads // tp)
        self.head_dim = config.head_dim
        sparse = config.sparse_attention_config
        if (self.head_dim != 128 or sparse["sparse_index_dim"] != 128
                or sparse["sparse_num_index_heads"] != config.num_key_value_heads
                or sparse["sparse_block_size"] != 128
                or sparse.get("sparse_score_type", "max") != "max"):
            raise ValueError("Unsupported MiniMax M3 sparse attention configuration")
        self.topk = sparse["sparse_topk_blocks"]
        self.init_blocks = sparse.get("sparse_init_block", 0)
        self.local_blocks = sparse.get("sparse_local_block", 1)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim, config.num_attention_heads,
            config.num_key_value_heads, bias=False, quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # Keep this small projection replicated, then select the same head range
        # as QKVParallelLinear uses for K (including replicated heads at TP8).
        self.index_head_start = (get_tensor_model_parallel_rank() * self.num_kv_heads
            if tp <= config.num_key_value_heads else
            get_tensor_model_parallel_rank() // (tp // config.num_key_value_heads))
        self.index_q_proj = ReplicatedLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim,
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.index_q_proj",
        )
        self.index_k_proj = ReplicatedLinear(config.hidden_size, self.head_dim,
            bias=False, quant_config=quant_config, prefix=f"{prefix}.index_k_proj")
        self.o_proj = RowParallelLinear(config.num_attention_heads * self.head_dim,
            config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj")
        for name in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
            setattr(self, name, MiniMaxGemmaRMSNorm(self.head_dim, config.rms_norm_eps))
        self.rotary_emb = get_rope(self.head_dim, max_position=config.max_position_embeddings,
            rope_parameters={"rope_theta": config.rope_theta,
                             "partial_rotary_factor": config.partial_rotary_factor})
        self.layer_name = f"{prefix}.attn"
        self.kv_cache = torch.tensor([])
        self.impl = MiniMaxM3Impl()
        context = vc.compilation_config.static_forward_context
        if self.layer_name in context:
            raise ValueError(f"Duplicate attention layer: {self.layer_name}")
        context[self.layer_name] = self

    def get_attn_backend(self):
        return MiniMaxM3Backend

    def get_kv_cache_spec(self, vllm_config):
        return MLAAttentionSpec(block_size=128, num_kv_heads=1,
            head_size=(2 * self.num_kv_heads + 1) * self.head_dim, dtype=torch.bfloat16)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], -1)
        iq, _ = self.index_q_proj(hidden_states)
        start = self.index_head_start * self.head_dim
        iq = iq[:, start: start + self.kv_size]
        ik, _ = self.index_k_proj(hidden_states)
        q = self.q_norm(q.reshape(-1, self.num_heads, 128)).reshape(-1, self.q_size)
        k = self.k_norm(k.reshape(-1, self.num_kv_heads, 128)).reshape(-1, self.kv_size)
        iq = self.index_q_norm(iq.reshape(-1, self.num_kv_heads, 128)).reshape(-1, self.kv_size)
        ik = self.index_k_norm(ik.reshape(-1, 1, 128)).reshape(-1, 128)
        q, k = self.rotary_emb(positions, q, k)
        iq, ik = self.rotary_emb(positions, iq, ik)
        md = get_forward_context().attn_metadata
        if not isinstance(md, dict) or self.layer_name not in md:
            out = q.new_zeros(q.shape)
        else:
            md = md[self.layer_name]
            main, index = cache_views(self.kv_cache, self.num_kv_heads)
            n = md.num_actual_tokens
            slots = md.slot_mapping[:n]
            insert_cache(self.kv_cache, k[:n], v[:n], ik[:n], slots, self.num_kv_heads)
            out = torch.zeros_like(q).view(-1, self.num_heads, 128)
            query = q[:n].view(-1, self.num_heads, 128)
            index_query = iq[:n].view(-1, self.num_kv_heads, 128)
            nd, nt = md.num_decodes, md.num_decode_tokens
            if nd:
                topk = minimax_m3_index_decode(index_query[:nt], index, md.block_table[:nd],
                    md.seq_lens[:nd], md.max_seq_len, self.topk, self.init_blocks,
                    self.local_blocks, self.num_kv_heads, 1, 1)
                minimax_m3_sparse_attn_decode(query[:nt], main, topk, md.block_table[:nd],
                    md.seq_lens[:nd], self.num_kv_heads, 128 ** -0.5, out[:nt], 1)
            if md.num_prefills:
                cu = md.query_start_loc[nd:] - nt
                score = minimax_m3_index_score(index_query[nt:], index, md.block_table[nd:],
                    cu, md.seq_lens[nd:], md.prefix_lens[nd:], md.max_query_len,
                    md.max_seq_len, self.num_kv_heads)
                topk = minimax_m3_index_topk(score, cu, md.prefix_lens[nd:],
                    md.max_query_len, self.topk, self.init_blocks, self.local_blocks)
                minimax_m3_sparse_attn(query[nt:], main, topk, md.block_table[nd:], cu,
                    md.seq_lens[nd:], md.prefix_lens[nd:], md.max_query_len,
                    self.num_kv_heads, 128 ** -0.5, out[nt:n])
            out = out.reshape_as(q)
        return self.o_proj(out)[0]
