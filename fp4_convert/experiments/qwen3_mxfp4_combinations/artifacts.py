"""Freeze conversion dependencies separately from later inference preparation."""
from pathlib import Path
from ..qwen3_mxfp4.artifacts import REPO, file_hash, save, tensor_record, check_hashes

PACKAGE = "fp4_convert.experiments.qwen3_mxfp4_combinations"


def conversion_hashes():
    parent = Path(__file__).parent.parent
    names = ("qwen3_mxfp4_combinations/__init__.py", "qwen3_mxfp4_combinations/format.py",
             "qwen3_mxfp4_combinations/convert.py", "qwen3_mxfp4_combinations/artifacts.py",
             "qwen3_mxfp4/format.py", "qwen3_mxfp4/artifacts.py", "qwen3_mxfp4/fht32.py",
             "diag_gradient/r64_transform.py", "qwen3_mxfp4_uos/format.py",
             "qwen3_mxfp4_dualscale/format.py", "qwen3_mxfp4_r16x2/transform.py")
    return {str(parent / name): file_hash(parent / name) for name in names}


def code_hashes():
    from ..qwen3_mxfp4.artifacts import code_hashes as shared_hashes
    result = shared_hashes()
    parent = Path(__file__).parent.parent
    for name in ("qwen3_mxfp4_combinations", "qwen3_mxfp4_dualscale", "qwen3_mxfp4_dualscale_tp2",
                 "qwen3_mxfp4_r16x2", "qwen3_mxfp4_uos", "qwen3_aisbench_gpqa", "nvfp4_gpqa"):
        result.update({str(p): file_hash(p) for p in (parent / name).glob("*.py")})
    return result
