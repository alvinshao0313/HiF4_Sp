def register():
    from vllm.model_executor.models.minimax_m3.integration import register as original
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
    from .runtime import MiniMaxM3MXFP4Config
    original()
    if 'minimax_m3_mxfp4' not in QUANTIZATION_METHODS:
        register_quantization_config('minimax_m3_mxfp4')(MiniMaxM3MXFP4Config)
