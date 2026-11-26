"""Run lm_eval's CLI and close its vLLM engines before interpreter teardown."""
from functools import wraps
import math


def evaluate():
    from lm_eval.__main__ import cli_evaluate
    from lm_eval.models.vllm_causallms import VLLM

    engines = []
    original = VLLM.__init__
    original_loglikelihood = VLLM.loglikelihood

    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        engines.append(self.model.llm_engine.engine_core)

    VLLM.__init__ = initialize
    @wraps(original_loglikelihood)
    def loglikelihood(self, *args, **kwargs):
        values = original_loglikelihood(self, *args, **kwargs)
        if any(not math.isfinite(score) for score, _ in values):
            raise FloatingPointError("Non-finite choice loglikelihood; QA scores are invalid")
        return values

    VLLM.loglikelihood = loglikelihood
    try:
        cli_evaluate()
    finally:
        VLLM.__init__ = original
        VLLM.loglikelihood = original_loglikelihood
        for engine in engines:
            engine.shutdown(timeout=30)
