# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3.1 text backbone, adapted from upstream PR #64111.

Uses the existing v0.19 parallel layers and packed NVFP4 GEMMs. It deliberately
does not register a multimodal processor or modify the Qwen/HiF4/Ban paths.
"""

import re
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer, is_pp_missing_parameter, make_empty_intermediate_tensors_factory,
    make_layers, maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .attention import MiniMaxGemmaRMSNorm, MiniMaxM3Attention
from .quantization import swiglu_oai


class MiniMaxM3MLP(nn.Module):
    def __init__(self, config, intermediate_size, quant_config, prefix, reduce_results=True):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(config.hidden_size,
            [intermediate_size, intermediate_size], bias=False,
            quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size, config.hidden_size,
            bias=False, quant_config=quant_config, reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj")

    def forward(self, x):
        return self.down_proj(swiglu_oai(self.gate_up_proj(x)[0]))[0]


class MiniMaxM3MoE(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.e_score_correction_bias = nn.Parameter(torch.empty(config.num_local_experts,
            dtype=torch.float32))
        self.e_score_correction_bias.weight_loader = default_weight_loader
        self.gate = ReplicatedLinear(config.hidden_size, config.num_local_experts,
            bias=False, params_dtype=torch.float32, quant_config=None, prefix=f"{prefix}.gate")
        self.shared_experts = MiniMaxM3MLP(config, config.shared_intermediate_size,
            quant_config, f"{prefix}.shared_experts", reduce_results=False)
        self.experts = FusedMoE(num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok, hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size, scoring_func="sigmoid",
            e_score_correction_bias=self.e_score_correction_bias, renormalize=True,
            activation="swigluoai", routed_scaling_factor=config.routed_scaling_factor,
            router_logits_dtype=torch.float32, shared_experts=self.shared_experts,
            quant_config=quant_config, reduce_results=False, prefix=f"{prefix}.experts")

    def forward(self, x):
        logits = self.gate(x.float())[0]
        shared, routed = self.experts(hidden_states=x, router_logits=logits)
        return self.experts.maybe_all_reduce_tensor_model_parallel(shared + routed)


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(self, vllm_config, prefix):
        super().__init__()
        c = vllm_config.model_config.hf_text_config
        q = vllm_config.quant_config
        self.self_attn = MiniMaxM3Attention(c, q, f"{prefix}.self_attn")
        layer_id = int(prefix.rsplit(".", 1)[1])
        self.is_moe = bool(c.moe_layer_freq[layer_id])
        if self.is_moe:
            self.block_sparse_moe = MiniMaxM3MoE(c, q, f"{prefix}.block_sparse_moe")
        else:
            self.mlp = MiniMaxM3MLP(c, c.dense_intermediate_size, q, f"{prefix}.mlp")
        self.input_layernorm = MiniMaxGemmaRMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = MiniMaxGemmaRMSNorm(c.hidden_size, c.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        residual = hidden_states if residual is None else hidden_states + residual
        x = self.self_attn(positions, self.input_layernorm(residual))
        residual = residual + x
        x = self.post_attention_layernorm(residual)
        return (self.block_sparse_moe(x) if self.is_moe else self.mlp(x)), residual


class MiniMaxM3Model(nn.Module):
    def __init__(self, vllm_config, prefix):
        super().__init__()
        c = vllm_config.model_config.hf_text_config
        self.config = c
        self.embed_tokens = (VocabParallelEmbedding(c.vocab_size, c.hidden_size,
            prefix=f"{prefix}.embed_tokens") if get_pp_group().is_first_rank else PPMissingLayer())
        self.start_layer, self.end_layer, self.layers = make_layers(c.num_hidden_layers,
            lambda prefix: MiniMaxM3DecoderLayer(vllm_config, prefix), prefix=f"{prefix}.layers")
        self.norm = (MiniMaxGemmaRMSNorm(c.hidden_size, c.rms_norm_eps)
            if get_pp_group().is_last_rank else PPMissingLayer())
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], c.hidden_size)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if get_pp_group().is_first_rank:
            hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        return self.norm(hidden_states + residual)


class MiniMaxM3LanguageModel(nn.Module):
    def __init__(self, vllm_config, prefix):
        super().__init__()
        c = vllm_config.model_config.hf_text_config
        self.model = MiniMaxM3Model(vllm_config, f"{prefix}.model")
        self.lm_head = (ParallelLMHead(c.vocab_size, c.hidden_size, prefix=f"{prefix}.lm_head")
            if get_pp_group().is_last_rank else PPMissingLayer())


def expert_parameter(name: str, weight: torch.Tensor):
    """Convert compressed-tensors names/scales to the existing ModelOpt ABI."""
    match = re.fullmatch(r"(.*\.experts)\.(\d+)\.(w[123])\.(.+)", name)
    if match is None:
        return None
    prefix, expert, shard, suffix = match.groups()
    target = "w2" if shard == "w2" else "w13"
    suffixes = {"weight_packed": "weight", "weight_scale": "weight_scale",
                "weight_global_scale": "weight_scale_2", "input_global_scale": "input_scale"}
    if suffix not in suffixes:
        raise ValueError(f"Unrecognized NVFP4 expert tensor: {name}")
    if suffix == "weight_global_scale":
        if not torch.all(torch.isfinite(weight) & (weight > 0)):
            raise ValueError(f"Invalid NVFP4 weight multiplier: {name}")
        weight = weight.float().reciprocal()
    elif suffix == "input_global_scale":
        # M3.1 stores placeholders; the per-token QAT kernel derives both
        # activation scales at runtime instead of using these serialized values.
        weight = torch.ones_like(weight, dtype=torch.float32)
    return f"{prefix}.{target}_{suffixes[suffix]}", int(expert), shard, weight


class MiniMaxM3ForTextGeneration(nn.Module, SupportsPP):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"],
                              "gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        c = vllm_config.model_config.hf_text_config
        sparse = c.sparse_attention_config
        if (not all(sparse["sparse_attention_freq"])
                or not all(sparse["sparse_disable_index_value"])
                or c.hidden_act != "swigluoai" or c.swiglu_alpha != 1.702
                or c.swiglu_beta != 1.0 or c.swiglu_limit != 7.0
                or c.n_shared_experts != 1 or not c.use_gemma_norm
                or c.scoring_func != "sigmoid" or not c.use_routing_bias):
            raise ValueError("This backport targets the MiniMax-M3.1-Flash text architecture")
        if vllm_config.speculative_config is not None:
            raise ValueError("DSpark is not part of this text inference backport")
        self.language_model = MiniMaxM3LanguageModel(vllm_config,
            maybe_prefix(prefix, "language_model"))
        self.logits_processor = LogitsProcessor(c.vocab_size)
        self.make_empty_intermediate_tensors = self.language_model.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids):
        return self.language_model.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        return self.language_model.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.language_model.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        stacked = [(".q_proj.", ".qkv_proj.", "q"), (".k_proj.", ".qkv_proj.", "k"),
                   (".v_proj.", ".qkv_proj.", "v"), (".gate_proj.", ".gate_up_proj.", 0),
                   (".up_proj.", ".gate_up_proj.", 1)]
        for name, value in weights:
            if name.startswith(("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")):
                continue
            expert = expert_parameter(name, value)
            if expert is not None:
                target, eid, shard, value = expert
                if is_pp_missing_parameter(target, self):
                    continue
                param = params[target]
                param.weight_loader(param, value, target, shard_id=shard, expert_id=eid)
                loaded.add(target)
                continue
            name = name.replace(".weight_scale_inv", ".weight_scale")
            shard_id = None
            for original, replacement, shard in stacked:
                if original in name:
                    name, shard_id = name.replace(original, replacement), shard
                    break
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params:
                raise ValueError(f"Unrecognized MiniMax checkpoint tensor: {name}")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, value) if shard_id is None else loader(param, value, shard_id)
            loaded.add(name)
        return loaded
