"""Real main.py evaluation with pinned inputs and the actual engine recorded."""
import json
import os
from pathlib import Path
import runpy

from .pinned import install


def install_engine_audit():
    from vllm.entrypoints.llm import LLM
    original = LLM.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        config = self.llm_engine.vllm_config
        print("MATRIX_EFFICIENCY_ENGINE " + json.dumps({
            "quantization": str(config.model_config.quantization),
            "tensor_parallel_size": config.parallel_config.tensor_parallel_size}), flush=True)

    LLM.__init__ = initialize


if __name__ == "__main__":
    install(Path(os.environ["HIF4_EFFICIENCY_RUN"]))
    install_engine_audit()
    runpy.run_path(str(Path(__file__).resolve().parents[3] / "main.py"), run_name="__main__")
