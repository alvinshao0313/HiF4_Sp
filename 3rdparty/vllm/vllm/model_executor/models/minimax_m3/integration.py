# SPDX-License-Identifier: Apache-2.0
"""Opt-in registration used only by the MiniMax entrypoint and its workers."""


def register():
    from transformers import AutoConfig
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS, register_quantization_config,
    )
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from .config import MiniMaxM3Config
    from .quantization import MiniMaxM3QuantizationConfig

    AutoConfig.register("minimax_m3_vl", MiniMaxM3Config, exist_ok=True)
    _CONFIG_REGISTRY["minimax_m3_vl"] = MiniMaxM3Config
    if "minimax_m3_mixed" not in QUANTIZATION_METHODS:
        register_quantization_config("minimax_m3_mixed")(MiniMaxM3QuantizationConfig)
    architecture = "MiniMaxM3SparseForConditionalGeneration"
    if architecture not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(architecture,
            "vllm.model_executor.models.minimax_m3.model:MiniMaxM3ForTextGeneration")
