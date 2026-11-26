def register():
    from vllm.model_executor.layers.quantization import register_quantization_config
    from .runtime import Qwen3RoutedMXFP4Config
    register_quantization_config("qwen3_routed_mxfp4_r16x2")(Qwen3RoutedMXFP4Config)
