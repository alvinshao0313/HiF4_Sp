"""Source evidence for the isolated TP2 implementation and reused kernels."""
from pathlib import Path
from ..qwen3_mxfp4_dualscale.artifacts import code_hashes as original_hashes, save, tensor_record
from ..qwen3_mxfp4.artifacts import file_hash

def code_hashes():
    result = original_hashes()
    result.update({str(path.resolve()): file_hash(path) for path in Path(__file__).parent.glob("*.py")})
    waiter = Path(__file__).parent.parent / "qwen3_aisbench_gpqa/continue_gpu03.py"
    result[str(waiter.resolve())] = file_hash(waiter)
    return result
