"""Register unchanged quantization in all ranks, pin data, and call main.py."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, ROOT / "3rdparty/lighteval/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from fp4_convert.experiments.qwen3_mxfp4_combinations.integration import register

register()


def run():
    from fp4_convert.experiments.qwen3_mxfp4_rotation_eval.common import (
        install_datasets, observe_generation,
    )
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_manifest", required=True)
    args, remaining = parser.parse_known_args()
    install_datasets(args.dataset_manifest)
    observe_generation(remaining[remaining.index("--output_dir") + 1])
    sys.argv = [sys.argv[0], *remaining]
    from main import main
    main()


if __name__ == "__main__":
    run()
