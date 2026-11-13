"""Registration executes in both parent and spawn children before loading Qwen."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, ROOT / "3rdparty/lighteval/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4.integration import register
register()

if __name__ == "__main__":
    from main import main
    main()
