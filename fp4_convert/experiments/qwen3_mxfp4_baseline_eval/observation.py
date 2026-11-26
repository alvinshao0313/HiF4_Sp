"""Record the actual native method/backend and TP shapes after original loading."""
def install():
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
    from ..qwen3_mxfp4.runtime import audit

    original = ModelOptNvFp4Config.get_quant_method

    def observed(config, layer, prefix):
        method = original(config, layer, prefix)
        if isinstance(layer, FusedMoE):
            process = method.process_weights_after_loading

            def loaded(module):
                result = process(module)
                audit({"event": "loaded_native_experts", "prefix": prefix,
                       "tp_rank": get_tensor_model_parallel_rank(),
                       "tp_size": get_tensor_model_parallel_world_size(),
                       "method": type(method).__name__, "backend": str(method.nvfp4_backend),
                       "experts_class": method.experts_cls.__name__,
                       "tensors": {name: {"shape": list(getattr(module, name).shape),
                                          "dtype": str(getattr(module, name).dtype)}
                                   for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")}})
                return result

            method.process_weights_after_loading = loaded
        else:
            audit({"event": "non_expert_method", "prefix": prefix, "method": type(method).__name__})
        return method

    ModelOptNvFp4Config.get_quant_method = observed
