"""Register both unchanged formats in the main process and vLLM spawn workers."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, ROOT / "3rdparty/lighteval/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4.integration import register
from Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_r16x2.integration import register as register_r16
register()
register_r16()


def run():
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_rotation_eval.common import (
        install_datasets, observe_generation,
    )
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_manifest", required=True)
    args, remaining = parser.parse_known_args()
    install_datasets(args.dataset_manifest)
    output = remaining[remaining.index("--output_dir") + 1]
    observe_generation(output)
    sys.argv = [sys.argv[0], *remaining]
    from main import main
    main()


if __name__ == "__main__":
    run()
