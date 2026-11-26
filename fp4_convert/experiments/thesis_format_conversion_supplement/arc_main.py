"""Retain lm_eval's already-computed sample records without changing its protocol."""
import json
from pathlib import Path
import runpy
import sys

from ..e2e_diag_reconstruction.evaluation.vllm_runner import _patch_lm_eval_transformers


if __name__ == "__main__":
    _patch_lm_eval_transformers()
    import lm_eval
    import numpy as np
    original = lm_eval.simple_evaluate
    output = Path(sys.argv[sys.argv.index("--output_dir") + 1])

    def serializable(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(f"unsupported sample field: {type(value)}")

    def evaluate_with_samples(*args, **kwargs):
        result = original(*args, **kwargs)
        samples = result.get("samples")
        if not samples: raise ValueError("lm_eval did not retain requested per-question evidence")
        (output / "arc_samples.json").write_text(json.dumps(samples, ensure_ascii=False, default=serializable))
        return result

    lm_eval.simple_evaluate = evaluate_with_samples
    runpy.run_module("fp4_convert.experiments.downstream_accuracy_recovery.evaluate", run_name="__main__")
