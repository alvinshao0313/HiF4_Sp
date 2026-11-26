"""Explicit worker extension used only by learned-rotation exports."""
from vllm.v1.worker.gpu_worker import Worker as VllmWorker


class Worker(VllmWorker):
    def __init__(self, *args, **kwargs):
        from .runtime import install_runtime
        install_runtime()
        super().__init__(*args, **kwargs)
