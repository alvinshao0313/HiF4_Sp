"""Hash the new candidate and every unchanged shared numerical/evaluation helper."""
from pathlib import Path

from ..qwen3_mxfp4.artifacts import REPO, check_hashes, file_hash, save, tensor_record
from ..qwen3_mxfp4.artifacts import code_hashes as shared_code_hashes

PACKAGE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_uos"


def code_hashes():
    result = shared_code_hashes()
    result.update({str(path.relative_to(REPO)): file_hash(path)
                   for path in sorted(Path(__file__).parent.glob("*.py"))})
    return result
