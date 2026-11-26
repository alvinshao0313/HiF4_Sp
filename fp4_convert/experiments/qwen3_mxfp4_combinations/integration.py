def register():
    from vllm.model_executor.layers.quantization import register_quantization_config
    from .runtime import Qwen3RoutedMXFP4Config
    from .format import MARKER
    register_quantization_config(MARKER)(Qwen3RoutedMXFP4Config)
