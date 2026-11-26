"""Use -m for worker registration. Pass lm_eval as first argument for choice tasks."""
import sys

from vllm.model_executor.layers.quantization import register_quantization_config

from .native import NativeConfig
from .runtime import PaperConfig

register_quantization_config("paper_native")(NativeConfig)
register_quantization_config("paper_fp4")(PaperConfig)

if __name__ == "__main__":
    if sys.argv[1:2] == ["lm_eval"]:
        del sys.argv[1]
        from ..e2e_diag_reconstruction.evaluation.vllm_runner import _patch_lm_eval_transformers
        from ..e2e_diag_reconstruction.evaluation.lm_eval_vllm import patch_lm_eval_vllm_generate_compat
        _patch_lm_eval_transformers()
        patch_lm_eval_vllm_generate_compat()
        from .qa import evaluate
        evaluate()
    else:
        from main import main
        main()
