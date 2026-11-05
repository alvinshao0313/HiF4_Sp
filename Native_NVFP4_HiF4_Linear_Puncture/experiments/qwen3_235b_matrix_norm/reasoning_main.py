"""Run the real main.py with this RUN's pinned data and actual engine metadata."""
import json
import os
from pathlib import Path
import runpy

from .config import REPO
from .pinned import install


def install_engine_audit():
    from vllm.entrypoints.llm import LLM
    original = LLM.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        config = self.llm_engine.vllm_config
        print("MATRIX235_ENGINE_CONFIG " + json.dumps({
            "quantization": str(config.model_config.quantization),
            "tensor_parallel_size": config.parallel_config.tensor_parallel_size}), flush=True)

    LLM.__init__ = initialize


if __name__ == "__main__":
    install(Path(os.environ["HIF4_235B_RUN"]))
    install_engine_audit()
    runpy.run_path(str(REPO / "main.py"), run_name="__main__")
