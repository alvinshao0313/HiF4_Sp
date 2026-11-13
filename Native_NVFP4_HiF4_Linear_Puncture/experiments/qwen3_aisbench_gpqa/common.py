"""Fixed original run locations; no weight conversion or runtime modification."""
from pathlib import Path
from datetime import datetime, timezone

from ..qwen3_mxfp4.artifacts import REPO, check_hashes, file_hash, save

HERE = Path(__file__).resolve().parent
PACKAGE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_aisbench_gpqa"
RESULTS = REPO / "Native_NVFP4_HiF4_Linear_Puncture/results"
BASE = RESULTS / "qwen3_mxfp4"
NATIVE = RESULTS / "nvfp4_gpqa/native_tp4_fixedkv_20261109T081227Z"
OLD_RUNS = {
    "native": NATIVE,
    "direct": BASE / "direct_gpu03_formal_20261109T100620Z/direct",
    "r64": BASE / "direct_gpu03_formal_20261109T100620Z/r64",
    "fht32": BASE / "fht32_tp4_20261110T032038Z/fht32",
    "r16x2": BASE / "r16x2_gpu03_20261110T040449Z/r16x2",
    "direct_uos_c7p25": BASE / "uos_c7p25_gpu03_20261110T074810Z/direct_uos_c7p25",
    "dualscale_g256": BASE / "dualscale_gpu47_20261110T085025Z/dualscale_g256",
    "dualscale_g512": BASE / "dualscale_gpu47_20261110T085025Z/dualscale_g512",
}
LANES = {
    "0,1,2,3": ["native", "r64", "r16x2", "dualscale_g256"],
    "4,5,6,7": ["direct", "fht32", "direct_uos_c7p25", "dualscale_g512"],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    import json
    return json.loads(Path(path).read_text())


def command_protocol(command):
    """All original CLI arguments except paths and task identity must match."""
    values = list(command[3:])
    for flag in ("--model_path", "--output_dir", "--custom_tasks", "--datasets"):
        index = values.index(flag)
        values[index + 1] = "<identity>"
    return values
