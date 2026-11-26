"""Register the combination quantizer in parent and spawn workers, then run main.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, ROOT / "3rdparty/lighteval/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from fp4_convert.experiments.qwen3_mxfp4_combinations.integration import register

register()

if __name__ == "__main__":
    from main import main
    main()
